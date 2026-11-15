"""Explicit GatewayQueue/provider-capacity installation; no startup default.

The manager keeps its existing local capacity collector. EXO publishes covered
group eligibility through that same source, without creating a manager model,
initial reservation or second physical Slice lease.
"""

import asyncio

from ..inference_arbiter import CapacityDeferred, NotFound
from ..inference_events import source_call
from .acquire import GroupAcquisition
from .backend import EXOBackend
from .completion import GroupCompletion
from .demand import AcquisitionDemand
from .plan import HOST_RESOURCES


class GroupProvider:
    def __init__(self, installation, actor):
        self.installation, self.actor = installation, actor
        self.source, self.queue = installation.source, installation.runtime.queue
        self.model, self.resource = actor.plan.model_id, HOST_RESOURCES['slice']
        self.backend = None
        self.completion = GroupCompletion(actor)
        self.publication = None
        self.receipts = None
        self.readiness = None
        self._ready = False
        self._actor_epoch = None
        self._epoch = 0
        self._task = None
        self._closed = False
        self._manager_present = bool(installation.runtime.manager.models)
        previous = actor.notify

        def changed(report):
            # A received group event can change local physical ownership too.
            self.source.invalidate('exo_group_observation_changed')
            self.invalidate('group_observation_changed')
            previous(report)
            self.wake()

        actor.notify = changed

    def _saved(self):
        try:
            return self.actor.intents.snapshot(self.actor.plan.owner_id, self.actor.plan.request_key)
        except NotFound:
            return None

    def local_permission(self):
        """True means this group does not block local admission, never permission.

        In-flight metadata can acquire Slice before its reply. A lost outcome
        stays unknown; waiting only on the first host leaves Slice usable once
        the server confirms its member has not been requested.
        """
        saved = self._saved()
        if saved is None or not saved['create_started'] and saved['group_id'] is None:
            return True
        if (self.actor.ownership_uncertain or not self.actor._events_ready
                or self.actor._task is not None and not self.actor._task.done()):
            return None
        group = self.actor.group_snapshot
        if not isinstance(group, dict):
            return None
        if group.get('phase') in ('released', 'cancelled'):
            return True
        members = group.get('members')
        if not isinstance(members, list):
            return None
        matching = [row for row in members if isinstance(row, dict)
                    and row.get('resource_id') == self.resource]
        if len(matching) != 1:
            return None
        if matching[0].get('status') == 'not_requested' and matching[0].get('lease_id') is None:
            return True
        return False if matching[0].get('status') in (
            'queued', 'active', 'expired', 'released', 'cancelled') else None

    def owns_group(self):
        saved = self._saved()
        return bool(saved and (saved['create_started'] or saved['group_id']) and not (
            isinstance(self.actor.group_snapshot, dict)
            and self.actor.group_snapshot.get('phase') in ('released', 'cancelled')
            and not self.actor.ownership_uncertain))

    def bind_backend(self, backend):
        """Startup controller calls only after exact group/launch binding."""
        saved = self._saved()
        staged = self.actor.intents.staging(self.actor.plan.owner_id, self.actor.plan.request_key)
        if (not isinstance(backend, EXOBackend) or backend.actor is not self.actor
                or backend.bindings.journal is not self.queue.supervisor.journal
                or saved is None or saved['group_id'] != backend.contract.group_id
                or saved['generation'] != backend.contract.generation
                or staged is None or staged['runtime_sha256'] != backend.contract.runtime_sha256
                or self.backend is not None and self.backend is not backend):
            raise ValueError('exact frozen EXO backend required')
        self.actor.intents.bind_launches(self.actor.plan.owner_id,
                                        self.actor.plan.request_key, backend.launches)
        self.backend = backend
        self.wake()

    def _busy(self):
        return self.queue.supervisor.journal.db.execute('''SELECT 1 FROM executions x
            JOIN exo_group_demands d ON d.execution_id=x.id
            WHERE d.owner=? AND d.request_key=? AND x.state IN ('running','uncertain') LIMIT 1''',
            (self.actor.plan.owner_id, self.actor.plan.request_key)).fetchone() is not None

    async def release_group(self, *, generation):
        return await self.completion.begin_release(generation=generation)

    async def settle_member(self, lease_id, *, generation, file_id, sha256):
        return await self.completion.settle_member(lease_id, generation=generation,
                                                  file_id=file_id, sha256=sha256)

    def bind_publication(self, *, group_scope):
        """Trusted deployment configuration chooses the existing file audience."""
        from .publication import GroupPublication
        if self.publication is not None:
            if self.publication.group_scope != group_scope:
                raise ValueError('cleanup publication audience is already configured')
            return self.publication
        self.publication = GroupPublication(self.completion, group_scope=group_scope)
        return self.publication

    async def settle_observed_member(self, lease_id, raw, *, generation):
        if self.publication is None:
            raise RuntimeError('cleanup publication audience is unconfigured')
        return await self.publication.complete_member(lease_id, raw, generation=generation)

    def bind_receipts(self, routes):
        """Bind deployment-owned host routes; callers cannot supply them."""
        from .receipts import GroupReceipts
        if self.publication is None:
            raise RuntimeError('cleanup publication audience is unconfigured')
        if self.receipts is not None:
            if tuple(self.receipts.routes.values()) != routes:
                raise ValueError('host receipt routes are already configured')
            return self.receipts
        self.receipts = GroupReceipts(self.completion,self.publication,routes)
        return self.receipts

    def bind_readiness(self, startup):
        from .readiness import GroupReadiness
        if self.readiness is not None:
            if self.readiness.startup is not startup:
                raise ValueError('private readiness owner is already configured')
            return self.readiness
        self.readiness = GroupReadiness(self,startup)
        return self.readiness

    def permitted(self, resource):
        if (resource != self.resource or self.source.paused or self._closed
                or self.source.permission is not True):
            return None
        return True if (self._ready and not self._busy() and self.backend is not None
            and self._actor_epoch == self.actor._epoch and self.actor.may_serve_requests()
            and not self.installation.runtime.manager.models) else None

    def publish(self):
        available = self.permitted(self.resource) is True
        self.source.publish(self.resource, 'available' if available else 'unknown',
                            'exo_group_covered' if available else 'exo_group_unconfirmed', model=self.model)

    def observe_manager(self):
        present = bool(self.installation.runtime.manager.models)
        if present != self._manager_present:
            self._manager_present = present
            self.wake()  # Existing inventory transition, not a new poll loop.

    def invalidate(self, _reason='invalidated'):
        self._ready = False
        self._epoch += 1
        self.publish()

    def wake(self):
        if self._closed:
            return
        self.invalidate()
        if self._task is None or self._task.done():
            self._task = self.queue.supervisor.prepare(self._drive())

    async def _drive(self):
        while not self._closed:
            epoch = self._epoch
            try:
                await self.actor.wait()
                actor_epoch = self.actor._epoch
                backend = self.backend
                if (not self.source.paused and backend is not None and not self._busy()
                        and not self.installation.runtime.manager.models
                        and self.actor.may_serve_requests()):
                    runtime = await backend.verify_runtime()
                    state = await backend.private.state()
                    backend.contract.check_idle(self.actor.plan, backend.launches, state,
                                                runtime_sha256=runtime)
                    if (epoch == self._epoch and actor_epoch == self.actor._epoch
                            and not self.source.paused and not self._busy()
                            and self.actor.may_serve_requests()
                            and not self.installation.runtime.manager.models):
                        self._ready, self._actor_epoch = True, actor_epoch
                # Reuse the existing collector/publication revision. This is
                # one event-triggered refresh, never another monitoring loop.
                if not self.source.paused:
                    physical = await source_call(self.installation.runtime.manager._collect_stats)
                    await self.source.refresh(physical)
                self.publish()
            except (Exception, asyncio.CancelledError) as error:
                self._ready = False
                self.publish()
                if isinstance(error, asyncio.CancelledError):
                    raise
            if epoch == self._epoch:
                break

    async def execute(self, grant, call):
        self.invalidate('dispatch')
        if self.backend is None:
            raise CapacityDeferred('capacity_unavailable')
        return await self.backend.execute(grant, call)


class EXOInstallation:
    def __init__(self, runtime, actors):
        source = runtime.capacity_source
        supervisor = runtime.queue.supervisor
        if (source is None or source.external or supervisor.prepare_demand is not None
                or supervisor.workers or supervisor.preparations
                or getattr(runtime.manager, '_background_started', False)
                or not actors or getattr(runtime, 'exo', None) is not None):
            raise ValueError('fresh manager-bound runtime required for EXO installation')
        for model, actor in actors.items():
            if (not isinstance(actor, GroupAcquisition)
                    or runtime.settings.models.get(model) != HOST_RESOURCES['slice']
                    or actor.plan.model_id != model or actor.intents.journal is not supervisor.journal
                    or not callable(actor.verify_staging)
                    or actor._task is not None or actor._closed):
                raise ValueError('fresh exact queued Slice actors required')
        self.runtime, self.source = runtime, source
        self.closed = False
        self.providers = {model: GroupProvider(self, actor) for model, actor in actors.items()}
        self.demand = AcquisitionDemand(supervisor, actors)
        for model, provider in self.providers.items():
            runtime.queue.bind_provider(model, provider)
        source.external.update(self.providers)
        runtime.exo = self
        source.invalidate('exo_provider_installed')
        previous = supervisor.on_finished

        async def completed(grant, execution):
            try:
                if previous is not None:
                    await previous(grant, execution)
            finally:
                provider = self.providers.get(grant['model'])
                if provider is not None:
                    provider.wake()

        supervisor.on_finished = completed

    def owned(self):
        if self.closed:
            # close() reached this state only after fresh released-group
            # observations. Its deliberate event disconnect cannot undo them.
            return 0
        return sum(provider.owns_group() for provider in self.providers.values())

    async def close(self):
        if self.closed:
            return
        self.demand.paused = True
        self.source.pause()
        for provider in self.providers.values():
            await provider.completion.wait()
            if provider._task is not None:
                await asyncio.shield(provider._task)
            await provider.actor.wait()
        if self.owned():
            raise RuntimeError('EXO ownership requires observed cleanup and exact settlement before close')
        for provider in self.providers.values():
            provider._closed = True
            await provider.actor.close()
            if provider.backend is not None:
                await provider.backend.private.close()
        self.closed = True
