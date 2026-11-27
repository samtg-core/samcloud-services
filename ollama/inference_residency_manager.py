"""Bind exact manager model/lease links to durable rollover, not size guesses.

This adapter owns its metadata work beyond a renewal observer's lifetime. It
does not allocate/load a cold model or assert physical working-set sufficiency.
Initial allocation/capacity policy and production configuration remain explicit
launcher responsibilities. No legacy acquire/release fallback is permitted.
"""
import asyncio
import inspect
import logging

from .inference_arbiter import Conflict


class ManagerResidency:
    def __init__(self, manager, actor, models, *, ttl_seconds, on_change=None, require_size=True):
        if (type(ttl_seconds) is not int or ttl_seconds <= 0 or not models or
                any(not isinstance(value,str) or not value.strip()
                    for pair in models.items() for value in pair)):
            raise ValueError('explicit positive TTL and model/resource mapping required')
        self.manager, self.actor = manager, actor
        self.require_size = require_size
        self.models, self.ttl = dict(models), ttl_seconds
        self.tasks = {}
        self.operations = {}
        self.claims = {}
        self.on_change = on_change
        self.paused = False

    def pause(self):
        self.paused = True
        for model in self.models:
            self._notify(model)

    def resume(self):
        # A maintenance pause/handover is not proof old cached permission is
        # current. The next owned registry read must establish it again.
        self.paused = False
        for model in self.models:
            try:
                self.actor.store.forget_confirmation(self.allocation(model))
            except Conflict:
                pass
            self._notify(model)

    def _notify(self, model):
        if self.on_change is None:
            return
        try:
            result = self.on_change(model)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError('residency change callback must be synchronous')
        except Exception:
            # Dispatch must consult ready() and physical policy, not take a
            # notification as authorization. Failure can delay a pump, never
            # undo a committed receipt or grant stale capacity.
            logging.getLogger(__name__).warning('Manager residency notification incomplete')

    def allocation(self, model):
        # Model name and resource are fixed service configuration. Never choose
        # an identity-wide latest lease or match one by its byte count.
        if model not in self.models:
            raise Conflict('model has no configured residency allocation')
        return model

    def ready(self, model):
        if self.paused:
            return False
        allocation = self.allocation(model)
        mm = self.manager.models.get(model)
        if mm is None:
            return False
        try:
            snapshot = self.actor.store.snapshot(allocation)
        except Conflict:
            return False
        return (self.actor.has_current_receipt(allocation) and
                snapshot['lease_id'] == mm.lease_id and
                snapshot['resource'] == self.models[model] and
                snapshot['model'] == mm.name and
                (not self.require_size or snapshot['memory_mb'] >= mm.memory_mb))

    def begin(self, model):
        if self.paused:
            raise Conflict('residency maintenance is paused for handover')
        allocation = self.allocation(model)
        previous = self.tasks.get(model)
        if previous is not None and not previous.done():
            return previous
        if any(not operation.done() for operation in self.operations.get(model, ())):
            raise Conflict('previous metadata worker still owns this model')
        if previous is not None:
            self._release_ready(model)  # done callbacks may not have run yet
        mm = self.manager.models.get(model)
        if mm is None or not mm.lease_id:
            raise Conflict('model has no exact resident lease link')
        # Claim on the owning loop BEFORE any HTTP await, so cooldown cannot
        # unload the model while its replacement metadata is being committed.
        if self.manager.claim(model) is not mm:
            raise Conflict('model changed before residency claim')
        self.claims[model] = mm
        self.operations[model] = set()
        try:
            self.actor.store.forget_confirmation(allocation)
        except Conflict:
            pass  # initial model/lease link has not been registered yet
        task = asyncio.create_task(self._renew(model, mm))
        self.tasks[model] = task
        self._notify(model)
        def finished(done):
            self._release_ready(model)
            if not done.cancelled():
                done.exception()  # no unobserved raw HTTP exception/body logging
        task.add_done_callback(finished)
        return task

    def _release_ready(self, model):
        if not self.tasks[model].done() or any(not operation.done()
                                               for operation in self.operations[model]):
            return
        mm = self.claims.pop(model, None)
        if mm is None:
            return
        # Release exactly once, against the claimed object, not a replacement
        # that happens to occupy the same model name after a manual mutation.
        if self.manager.models.get(model) is mm:
            self.manager.release(model)
        else:
            mm.in_flight -= 1

    def _own(self, model, task):
        self.operations[model].add(task)
        def finished(done):
            self._release_ready(model)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    async def _renew(self, model, mm):
        store = self.actor.store
        allocation = self.allocation(model)
        try:
            snapshot = store.snapshot(allocation)
        except Conflict:
            snapshot = None
        try:
            if snapshot is None or snapshot['pending_id'] is None:
                predecessor = mm.lease_id
                # Claim keeps unload out; confirm object/link again after I/O.
                read = self._own(model,asyncio.create_task(asyncio.to_thread(
                    self.manager.sc.list_leases,resource_id=self.models[model],
                    service_id=store.service,status='active')))
                rows = await asyncio.shield(read)
                if self.manager.models.get(model) is not mm or mm.lease_id != predecessor:
                    raise Conflict('manager model/lease changed during ownership read')
                matches = [row for row in rows if row.get('id') == predecessor]
                if len(matches) != 1:
                    raise Conflict('current registry read does not prove exact lease')
                store.register(allocation,resource=self.models[model],model=mm.name,receipt=matches[0])
            else:
                # A lost reply/restart must retry the persisted predecessor/key,
                # not register an old pointer over it or make another allocation.
                if (snapshot['resource'] != self.models[model] or snapshot['model'] != mm.name
                        or snapshot['lease_id'] != mm.lease_id):
                    raise Conflict('pending handover differs from manager link')
            run = self.actor.begin(allocation,ttl_seconds=self.ttl)
            self._own(model,run.task)
            result = await self.actor.observe(run)
            current = store.snapshot(allocation)
            if (self.manager.models.get(model) is not mm or
                    mm.lease_id != run.operation.predecessor):
                store.forget_confirmation(allocation)
                raise Conflict('manager link changed before handover application')
            if current['pending_id'] is None:
                # Even a released/expired replacement is the exact linked ID;
                # ready() stays false. Never fall back to its retired predecessor.
                mm.lease_id = current['lease_id']
            self._notify(model)  # after manager pointer application, not just registry receipt
            return result
        except BaseException:
            if snapshot is not None or store.db.execute('SELECT 1 FROM allocations WHERE id=?',
                                                       (allocation,)).fetchone():
                store.forget_confirmation(allocation)
            self._notify(model)
            raise

    async def renew_models(self):
        tasks = [self.begin(model) for model in self.models if model in self.manager.models]
        if tasks:
            # The manager's periodic observer can be cancelled; these actual
            # metadata operations and claims must survive to their real boundary.
            return await asyncio.shield(asyncio.gather(*tasks))
        return []

    async def drain(self):
        while tasks := tuple(task for task in (*self.tasks.values(),
                *(operation for operations in self.operations.values() for operation in operations))
                if not task.done()):
            await asyncio.shield(asyncio.gather(*tasks,return_exceptions=True))
        await self.actor.drain()
        for model in self.tasks:
            self._release_ready(model)
