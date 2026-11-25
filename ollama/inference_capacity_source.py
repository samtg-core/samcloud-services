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


def _label(row):
    who = str(row.get('service_id') or row.get('agent_id') or 'another holder')
    mb = row.get('memory_mb')
    # An exclusive blocks the whole pool, whatever its size.
    return (f'{who} (exclusive)' if row.get('exclusive') else
            f'{who} ({mb} MB)' if type(mb) is int else who)


def _holders(registry,own_service):
    """Labels for what a queued lease of ours waits on: the active leases that
    are not ours, and the queued requests ahead of our first place in line."""
    active = [_label(row) for row in registry.get('leases') or []
              if isinstance(row,dict) and row.get('status') == 'active'
              and row.get('service_id') != own_service]
    ahead = []
    for row in registry.get('queue') or []:
        if not isinstance(row,dict):
            continue
        if row.get('service_id') == own_service:
            break
        ahead.append(_label(row))
    return active, ahead


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
        self.external = {}
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
        provider = self.external.get(model)
        if provider is not None:
            return provider.permitted(resource)
        host_rank = getattr(self.runtime, 'host_rank', None)
        if host_rank is not None and not host_rank.closed:
            return False  # Group-owned host cannot run an ordinary local load.
        boundaries = [provider.local_permission() for provider in self.external.values()]
        if any(value is False for value in boundaries):
            return False
        if any(value is None for value in boundaries):
            return None
        state = self.last_model.get(key,{}).get('state')
        if state == 'unavailable':
            return False
        if state != 'available' or key not in self.observed_residency:
            return None
        model,observed,lease = self.observed_residency[key]
        current = self.runtime.manager.models
        if set(current)-set(self.runtime.settings.models):
            return None
        if observed is None:
            return True if model not in current else None
        return (True if current.get(model) is observed and
                observed.lease_id==lease and self.binding.ready(model) else None)

    def publish(self,resource,state,reason,*,model=None):
        self.revision += 1
        names=[model] if model is not None else [name for name,target in
                    self.runtime.settings.models.items() if target==resource]
        queue=getattr(self.runtime,'queue',None)
        journal=getattr(getattr(queue,'supervisor',None),'journal',None)
        host_rank = getattr(self.runtime, 'host_rank', None)
        value={'state':state,'reason':reason,'revision':self.revision}
        for name in names:
            selected_state, selected_reason = state, reason
            if (state == 'available' and host_rank is not None and not host_rank.closed
                    and name not in self.external):
                selected_state, selected_reason = 'unavailable', 'exo_host_rank_owned'
            value={'state':selected_state,'reason':selected_reason,'revision':self.revision}
            if journal is not None:
                journal.capacity_event(resource,name,selected_state,selected_reason)
            self.last_model[resource,name]=value
        self.last[resource]=value
        states=[self.last_model.get((resource,name),{}).get('state','unknown')
                for name,target in self.runtime.settings.models.items() if target==resource]
        aggregate=('available' if 'available' in states else
                   'unknown' if 'unknown' in states else 'unavailable')
        self.runtime.capacity.observe(resource,self.revision,aggregate)

    def pause(self):
        self.paused = True
        for provider in self.external.values():
            provider.invalidate('source_paused')
        self.invalidate('source_paused')

    def invalidate(self,reason='observing'):
        """Withdraw admission before collection, including failed collectors."""
        self.epoch += 1
        for resource in set(self.runtime.settings.models.values()):
            self.publish(resource,'unknown',reason)

    def resume(self):
        self.paused = False  # permission stays unknown until a new observation
        for provider in self.external.values():
            provider.wake()

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
            def publish(state,reason):
                self.publish(resource,state,reason,model=model)
            publish('unknown','observing')
            if self.permission is not True:
                publish('unavailable' if self.permission is False else 'unknown',
                             'work_mode' if self.permission is False else 'work_permission_unknown')
                continue
            provider = self.external.get(model)
            if provider is not None:
                provider.observe_manager()
                provider.publish()
                continue
            boundaries = [provider.local_permission() for provider in self.external.values()]
            if any(value is None for value in boundaries) and not any(value is False for value in boundaries):
                publish('unknown', 'exo_group_ownership_unconfirmed')
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
                if any(value is False for value in boundaries):
                    # Still run the existing foreign-queue handoff above: a
                    # queued group member must let local residency drain.
                    publish('unavailable', 'exo_group_ownership')
                    continue
                if mm is None:
                    from .inference_arbiter import Conflict
                    store=self.binding.actor.store
                    try:
                        former=store.snapshot(self.binding.allocation(model))
                    except Conflict:
                        former=None
                    initial = getattr(manager, 'initial_reservations', None)
                    prior_intent = initial.snapshot(model) if initial is not None else None
                    needs_initial_retirement = (handoff is not None and former is not None and prior_intent is not None
                        and prior_intent['state'] == 'handed_off'
                        and prior_intent['lease_id'] == former['lease_id'])
                    if former is not None and (former['state']!='retired' or needs_initial_retirement):
                        ended=await source_call(lambda:manager.sc.list_leases(
                            resource_id=resource,service_id=store.service))
                        from .manager import VLM_MODELS, match_whisper_model
                        vision_model=any(info['default']==model for info in VLM_MODELS.values())
                        if vision_model:
                            from .inference_shared_resource import cold_vision_absent
                            absent=await source_call(cold_vision_absent)
                            running=[] if absent is True else None
                        elif match_whisper_model(model) is not None:
                            from .whisper_lifecycle import cold_whisper_absent
                            absent=await source_call(cold_whisper_absent)
                            running=[] if absent is True else None
                        else:
                            running=await source_call(manager.ollama.list_running)
                        if self.paused or epoch!=self.epoch:
                            return
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
                            # Ordinary cooldown ended this exact allocation. Retire
                            # its initial pointer as well, including after restart
                            # between the two commits. A fresh absence observation
                            # and exact terminal receipt are still required; no
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
                        journal = self.runtime.queue.supervisor.journal
                        if model in pending:
                            if saved is not None and saved['state'] in ('unknown','handed_off','retiring'):
                                handoff.begin()
                                return
                            from .inference_arbiter import CapacityDeferred
                            from .manager import LEASE_TTL
                            if saved is None or saved['state'] not in ('queued','reserved'):
                                # No reservation for bytes this host cannot hold.
                                # When the registry has room but the host does
                                # not, a grant would be active bytes nobody uses
                                # until its TTL. The call waits here instead and
                                # says so. A registry that is short queues the
                                # request below: a place in line holds no bytes.
                                early = cold_capacity(resource,required,
                                    physical=physical,registry=registry,work_permitted=True)
                                if early['reason'] == 'physical_capacity':
                                    journal.memory_wait(resource,model,required)
                                    publish('unavailable','physical_capacity')
                                    continue
                            try:
                                owned = await source_call(lambda:initial.stage(model,
                                    resource=resource, service_id=initial.service,
                                    memory_mb=required, ttl_seconds=LEASE_TTL))
                            except CapacityDeferred:
                                # The lease waits in the resource queue. Say
                                # whose leases hold the memory, from the same
                                # registry read; ours are not a reason.
                                active,ahead = _holders(registry,initial.service)
                                journal.memory_wait(resource,model,required,active,ahead)
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
                    if handoff is not None:
                        if (verdict['reason'] == 'physical_capacity' and model in pending
                                and initial.snapshot(model)['state'] == 'reserved'):
                            # Promoted, never handed to a provider, and the
                            # host cannot hold it: give the exact lease back
                            # now. The check above keeps it from re-staging
                            # until the memory is there.
                            await source_call(lambda:initial.release_unused(model))
                            if self.paused or epoch != self.epoch:
                                return
                        if verdict['reason'] == 'physical_capacity':
                            journal.memory_wait(resource,model,required)
                        elif verdict['state'] == 'available':
                            journal.memory_wait(resource,model,None)
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
