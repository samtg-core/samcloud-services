"""Local service capacity observations, not caller permission or compute timers.

Cold admission checks the configured reservation against fresh free memory.
Resident admission reuses its exact confirmed reservation, checks current host
headroom and aggregate device use, and does NOT credit cached provider sizes.
These are admission observations, not a guarantee of future allocation peaks.
"""
import asyncio
from datetime import datetime
import time

from . import capacity
from .inference_events import source_call
from .inference_bootstrap import cold_capacity


class ManagerCapacitySource:
    def __init__(self,runtime,binding,*,work_permitted=None):
        if work_permitted is not None and type(work_permitted) is not bool:
            raise ValueError('explicit work permission required')
        self.runtime,self.binding = runtime,binding
        self.permission = work_permitted
        self.revision = 0
        self.workers = set()
        self.paused = False
        self.last = {}
        self.last_model = {}
        self.observed_residency = {}
        self.epoch = 0
        self.lock = asyncio.Lock()
        runtime.capacity.guard = self.guard
        if hasattr(runtime,'queue'):
            runtime.queue.supervisor.permitted_model = self.permitted_model

    def guard(self,resource):
        """Recheck exact ownership/expiry at dispatch, without network I/O."""
        return any(self.permitted_model(resource,model) is True for model,target in
                   self.runtime.settings.models.items() if target==resource)

    def permitted_model(self,resource,model):
        """True permits dispatch, False confirms unavailability, None is unknown.

        Observation gaps withdraw admission but do not revoke an idle client's
        stint. Ownership drift likewise needs reconciliation, not a guessed
        capacity-loss event. Only an affirmative unavailable observation rotates
        affinity for capacity reasons.
        """
        key=(resource,model)
        if self.paused:
            return None
        state = self.last_model.get(key,{}).get('state')
        if state == 'unavailable':
            return False
        if state != 'available' or key not in self.observed_residency:
            return None
        model,observed,lease = self.observed_residency[key]
        current = self.runtime.manager.models
        if set(current)-set(self.runtime.settings.models):return None
        if observed is None:
            return True if model not in current else None
        return (True if current.get(model) is observed and
                observed.lease_id==lease and self.binding.ready(model) else None)

    def publish(self,resource,state,reason,*,model=None):
        self.revision += 1
        value={'state':state,'reason':reason,'revision':self.revision}
        names=[model] if model is not None else [name for name,target in
                    self.runtime.settings.models.items() if target==resource]
        queue=getattr(self.runtime,'queue',None)
        journal=getattr(getattr(queue,'supervisor',None),'journal',None)
        if journal is not None:
            for name in names:
                journal.capacity_event(resource,name,state,reason)
        for name in names:self.last_model[resource,name]=value
        self.last[resource]=value
        states=[self.last_model.get((resource,name),{}).get('state','unknown')
                for name,target in self.runtime.settings.models.items() if target==resource]
        aggregate=('available' if 'available' in states else
                   'unknown' if 'unknown' in states else 'unavailable')
        self.runtime.capacity.observe(resource,self.revision,aggregate)

    def pause(self):
        self.paused = True
        self.invalidate('source_paused')

    def invalidate(self,reason='observing'):
        """Withdraw admission before collection, including failed collectors."""
        self.epoch += 1
        for resource in set(self.runtime.settings.models.values()):
            self.publish(resource,'unknown',reason)

    def resume(self):
        self.paused = False  # permission stays unknown until a new observation

    async def refresh(self,physical):
        if self.paused:
            handoff = getattr(self.runtime, 'shared_resource', None)
            if (handoff is not None and handoff.snapshot()['phase'] != 'idle'
                    and not getattr(self.runtime, '_external_quiesce', False)):
                handoff.begin()  # reconcile owned metadata, not a compute retry
            return
        # The manager calls this after an actual collect, not its offer cache.
        self.invalidate()
        task = asyncio.create_task(self._serial_refresh(dict(physical),self.epoch))
        self.workers.add(task)
        def finished(done):
            self.workers.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def _serial_refresh(self,physical,epoch):
        async with self.lock:
            if epoch == self.epoch and not self.paused:
                await self._refresh(physical,epoch)

    async def drain(self):
        while pending := tuple(t for t in self.workers if not t.done()):
            await asyncio.shield(asyncio.gather(*pending,return_exceptions=True))

    async def _refresh(self,physical,epoch):
        manager,settings = self.runtime.manager,self.runtime.settings
        for model,resource in settings.models.items():
            publish=lambda state,reason:self.publish(resource,state,reason,model=model)
            publish('unknown','observing')
            if self.permission is not True:
                publish('unavailable' if self.permission is False else 'unknown',
                             'work_mode' if self.permission is False else 'work_permission_unknown')
                continue
            mm = manager.models.get(model)
            lease = getattr(mm,'lease_id',None)
            try:
                registry = await source_call(manager.sc.get_resource,resource)
                if self.paused or epoch != self.epoch:
                    return
                handoff = getattr(self.runtime, 'shared_resource', None)
                if handoff is not None:
                    foreign = handoff._foreign(registry)  # verifies hold policy and snapshot scope
                    own_debt = any(row.get('service_id') == handoff.initial.service for row in
                                   registry.get('unsettled', []) if isinstance(row, dict))
                    if (foreign is not None and manager.models) or own_debt:
                        handoff.begin()  # pause before any available publication
                        return
                    from .manager import VLM_MODELS
                    if mm is not None and any(info['default'] == model for info in VLM_MODELS.values()):
                        child = getattr(mm, 'vlm_process', None)
                        if not getattr(mm, 'managed', False) or child is None:
                            publish('unknown', 'vision_process_ownership_unknown')
                            continue
                        ended = await source_call(child.poll)
                        if self.paused or epoch != self.epoch:
                            return
                        if ended is not None:
                            handoff.begin()
                            return
                if mm is None:
                    from .inference_arbiter import Conflict
                    store=self.binding.actor.store
                    try:former=store.snapshot(self.binding.allocation(model))
                    except Conflict:former=None
                    initial = getattr(manager, 'initial_reservations', None)
                    prior_intent = initial.snapshot(model) if initial is not None else None
                    needs_initial_retirement = (handoff is not None and former is not None and prior_intent is not None
                        and prior_intent['state'] == 'handed_off')
                    if former is not None and (former['state']!='retired' or needs_initial_retirement):
                        ended=await source_call(lambda:manager.sc.list_leases(
                            resource_id=resource,service_id=store.service))
                        from .manager import VLM_MODELS
                        vision_model=any(info['default']==model for info in VLM_MODELS.values())
                        if vision_model:
                            from .inference_shared_resource import cold_vision_absent
                            absent=await source_call(cold_vision_absent)
                            running=[] if absent is True else None
                        else:
                            running=await source_call(manager.ollama.list_running)
                        if self.paused or epoch!=self.epoch:return
                        if (model in manager.models or not isinstance(running,list)
                                or any(not isinstance(item,dict) or not isinstance(item.get('name'),str)
                                    or item['name'].removesuffix(':latest')==model.removesuffix(':latest')
                                    for item in running)):
                            publish('unknown','previous_residency_unconfirmed')
                            continue
                        matches=[item for item in ended if isinstance(item,dict)
                                 and item.get('id')==former['lease_id']] if isinstance(ended,list) else []
                        if len(matches)!=1:
                            publish('unknown','previous_allocation_unconfirmed')
                            continue
                        store.retire(self.binding.allocation(model),matches[0])
                        if needs_initial_retirement:
                            # Renewal can replace the residency lease while the
                            # initial intent still names its original lease.
                            # Reconcile both exact pointers independently; the
                            # initial actor checks its own saved lease and sizing
                            # successors, not the residency receipt above. Fresh
                            # absence and exact registry evidence remain required; no
                            # foreign-service handoff or resident text drain is
                            # needed to reconcile an already released allocation.
                            await source_call(lambda:initial.reconcile_absent(
                                model, confirm_absent=handoff._absent))
                            if self.paused or epoch != self.epoch:
                                return
                    # Run 3 keeps the existing demand-driven model estimate.
                    # A deployment may explicitly override it, but starting a
                    # queue does not require a fixed whole-machine reservation.
                    required = settings.reservation_mb.get(model)
                    if required is None:
                        from .manager import VLM_MODELS
                        vlm=next((info for info in VLM_MODELS.values() if info['default']==model),None)
                        required = vlm['memory_mb'] if vlm is not None else await source_call(
                            manager.ollama.memory_estimate_mb,model)
                        if self.paused or epoch != self.epoch:
                            return
                        if type(required) is not int or required <= 0:
                            raise ValueError('model estimate unknown')
                    if model in manager.models:
                        publish('unknown','residency_changed')
                        continue
                    if handoff is not None:
                        initial = manager.initial_reservations
                        required = initial.required_memory(model, required)
                        pending = self.runtime.queue.supervisor.journal.pending_models(resource)
                        saved = initial.snapshot(model)
                        if model in pending:
                            if saved is not None and saved['state'] in ('unknown','handed_off','retiring'):
                                handoff.begin()
                                return
                            from .inference_arbiter import CapacityDeferred
                            from .manager import LEASE_TTL
                            try:
                                owned = await source_call(lambda:initial.stage(model,
                                    resource=resource, service_id=initial.service,
                                    memory_mb=required, ttl_seconds=LEASE_TTL))
                            except CapacityDeferred:
                                publish('unavailable','registry_queued')
                                continue
                            if self.paused or epoch != self.epoch:
                                return
                            # Specs-minus-leases includes our reserved-but-not-
                            # loaded grant. Credit only that exact fresh row,
                            # not cached model memory or another service's claim.
                            registry = await source_call(manager.sc.get_resource,resource)
                            handoff._foreign(registry)
                            matches = [row for row in registry['leases'] if row.get('id') == owned]
                            if len(matches) != 1:
                                raise ValueError('staged grant observation unknown')
                            row = matches[0]
                            expiry = datetime.fromisoformat(row['expires_at'].replace('Z','+00:00'))
                            if (row.get('agent_id') != initial.owner or row.get('service_id') != initial.service
                                    or row.get('resource_id') != resource or row.get('status') != 'active'
                                    or row.get('exclusive') != 0 or type(row.get('memory_mb')) is not int
                                    or row['memory_mb'] != required or expiry.tzinfo is None
                                    or expiry.timestamp() <= time.time()
                                    or type(registry.get('available_memory_mb')) is not int
                                    or registry['available_memory_mb'] < 0):
                                raise ValueError('staged grant is not current exact ownership')
                            registry = dict(registry,available_memory_mb=registry['available_memory_mb']+required)
                            physical = await source_call(manager._collect_stats)
                            if self.paused or epoch != self.epoch:
                                return
                        elif saved is not None and saved['state'] in ('queued','reserved'):
                            handoff.begin()  # unused metadata reservation; no idle default lease
                            return
                    verdict = cold_capacity(resource,required,
                        physical=physical,registry=registry,work_permitted=True)
                    if verdict['state'] == 'available':
                        self.observed_residency[resource,model] = (model,None,None)
                    publish(verdict['state'],verdict['reason'])
                    continue
                rows = await source_call(lambda: manager.sc.list_leases(
                    resource_id=resource,service_id=self.binding.actor.store.service,status='active'))
                if self.paused or epoch != self.epoch:
                    return
                if manager.models.get(model) is not mm or mm.lease_id != lease:
                    publish('unknown','residency_changed')
                    continue
                if (not isinstance(registry,dict) or registry.get('id') != resource or
                        registry.get('allocation_mode') != 'shared' or
                        not isinstance(registry.get('leases'),list)):
                    publish('unknown','registry_resource_unknown')
                    continue
                leases = registry['leases']
                if any(not isinstance(row,dict) or row.get('status') not in ('active','queued') or
                       row.get('exclusive') not in (0,1) for row in leases):
                    publish('unknown','registry_leases_unknown')
                    continue
                if any(row['status']=='active' and row['exclusive']==1 for row in leases):
                    publish('unavailable','exclusive_holder')
                    continue
                if not isinstance(rows,list):
                    raise ValueError('lease observation unknown')
                matches = [row for row in rows if isinstance(row,dict) and row.get('id') == lease]
                if len(matches) != 1:
                    raise ValueError('exact lease unknown')
                receipt = matches[0]
                self.binding.actor.store.register(model,resource=resource,model=mm.name,receipt=receipt)
                if not self.binding.ready(model):
                    publish('unknown','residency_unconfirmed')
                    continue
                if not settings.reservation_mb:
                    # Existing serving policy reuses a resident model. Keep
                    # measured use/free/reserved bytes separate in telemetry;
                    # do not add the draft's new resident physical-size gate.
                    self.observed_residency[resource,model] = (model,mm,lease)
                    publish('available','resident_existing_policy')
                    continue
                free,device = physical.get('memory_available_mb'),physical.get('memory_device_inuse_mb')
                if type(free) is not int or free < 0 or type(device) is not int or device < 0:
                    publish('unknown','physical_capacity_unknown')
                elif receipt['memory_mb'] < settings.reservation_mb.get(model,0):
                    publish('unknown','reservation_under_budget')
                elif capacity.usable_mb(free) <= 0:
                    publish('unavailable','host_headroom')
                elif device > receipt['memory_mb']:
                    # Aggregate includes foreign/OS device use: withdraw, but
                    # do not attribute the excess to a model or cancel its work.
                    publish('unavailable','device_usage_exceeds_reservation')
                else:
                    self.observed_residency[resource,model] = (model,mm,lease)
                    publish('available','resident_reserved_with_headroom')
            except Exception:
                if epoch == self.epoch:
                    publish('unknown','observation_unconfirmed')
