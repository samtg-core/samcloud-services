"""Explicit event-driven root/model preparation. Not installed by bootstrap.

The service owns the task; queue/host/registry events are hints, never grants.
Only the shared registry allocates. No idle retry loop, job timer or inferred
backend termination. Qualification of real host/process sensors is separate.
"""
import asyncio
import logging
from uuid import uuid4

from . import capacity
from .inference_arbiter import Conflict
from .inference_events import source_call
from .inference_root_models import _physical_headroom,RootAdmissionDeferred
from .inference_root_scheduler import RootPreparingArbiter,_unexpired


class RootController:
    def __init__(self,supervisor,binding,*,work_permitted):
        if type(work_permitted) is not bool:
            raise ValueError('Explicit service work permission required')
        if (not isinstance(supervisor.journal,RootPreparingArbiter)
            or supervisor.journal is not binding.scheduler
            or supervisor.on_resource_demand is not None or supervisor.permitted_model is not None
            or supervisor.workers or supervisor.preparations
            or binding.manager.inference_capacity_source is not None
            or binding.on_health_observed is not None or binding._source_owner is not None):
            raise Conflict('Explicit idle root supervisor/binding required')
        self.supervisor,self.binding=supervisor,binding
        self.scheduler,self.manager=supervisor.journal,binding.manager
        self.resource=binding.resource;self.permission=work_permitted
        self.paused=False;self._host_known=True;self._stopped=False
        self._ready=None;self._task=None;self._dirty=False;self._epoch=0;self._publishing=False
        self.last={'state':'unknown','reason':'unobserved'}
        binding.withdraw()
        supervisor.permitted=self.guard
        supervisor.permitted_model=self.permitted_model
        supervisor.on_resource_demand=self.request
        # Reuse actual manager stats and root-health maintenance. No new timer
        # or second allocator/watcher is constructed here.
        binding._source_owner=self
        self.manager.inference_capacity_source=self
        binding.on_health_observed=lambda:self.request(self.resource)

    def retains_model(self,model):
        return model in self.binding.models and self.scheduler.owns_job(self.resource)

    def invalidate(self,reason='observing'):
        if self._stopped:return
        # Existing stats_loop calls this BEFORE collecting. Failed/cancelled
        # collection cannot leave an old positive readiness cached.
        self._epoch+=1;self._ready=None;self._host_known=False
        self.binding.withdraw()
        self.last={'state':'unknown','reason':reason}

    async def refresh(self,physical):
        if self._stopped:return
        self.invalidate()
        values=[physical.get(name) if isinstance(physical,dict) else None
                for name in ('memory_available_mb','memory_device_inuse_mb')]
        self._host_known=all(type(value) is int and value>=0 for value in values)
        if self.paused:return
        # Stats collection/reporting must not wait behind a long model preload.
        # The supervisor owns preparation; drain() is the explicit waiter.
        self.request(self.resource)

    def pause(self):
        self.paused=True
        self.invalidate('source_paused')

    def resume(self):
        if self._stopped:raise Conflict('Stopped root source requires fresh startup')
        # Like the legacy source, resume needs a new positive observation.
        self.paused=False
        self.invalidate('awaiting_observation')

    def snapshot(self):
        return {'resource':self.resource,**self.last,'paused':self.paused,'stopped':self._stopped,
                'host_known':self._host_known,
                'preparing':self._task is not None and not self._task.done()}

    def _stop_observations(self):
        # Called only by binding.begin_shutdown AFTER its positive audit.
        self.paused=True
        self.invalidate('manager_stopped')
        self._stopped=True

    def set_work_permitted(self,value):
        if type(value) is not bool:raise ValueError('Explicit service work permission required')
        self.permission=value
        return self.request(self.resource)

    def guard(self,resource):
        ready=self._ready
        return ready is not None and self.permitted_model(resource,ready['model']) is True

    def permitted_model(self,resource,model):
        """Cached loop-local dispatch check: no SQL, network or model load."""
        r=self._ready
        if self.paused or not self._host_known or not self.permission or resource!=self.resource or r is None or r['model']!=model:
            return None
        if (self.manager.inference_capacity_source is not self or self.manager.root_binding is not self.binding
            or self.manager.sc is not self.binding._proxy or not self.manager.durable_inference
            or self.manager.initial_reservations is not None or self.manager.residency_renewal is not None):
            return None
        with self.binding._lock:current=self.binding._permission
        if (current is None or current[0]!=r['preparation_id']
            or current[1].get('current_lease_id')!=r['lease_id'] or not _unexpired(current[1].get('lease',{}))
            or set(self.manager.models)-set(self.binding.models)):
            return None
        mm=self.manager.models.get(model)
        if (mm is not r['handle'] or not mm.managed or mm.lease_id!=r['lease_id']
            or mm.root_model_intent!=r['model_intent']):
            return None
        return True

    def request(self,resource):
        """Queue/actual host/registry-change hint; coalesced and service-owned.

        Binding observations remain authoritative. Call after actual external
        change or a completed root heartbeat, never from a periodic retry loop.
        Publishing our own readiness suppresses its synchronous pump hint.
        """
        if self._stopped or resource!=self.resource or self._publishing:return None
        self._epoch+=1;self._ready=None;self._dirty=True
        self.binding.withdraw()
        if self._task is None or self._task.done():
            self._task=self.supervisor.prepare(self._drive())
        return self._task

    async def drain(self):
        task=self._task
        if task is not None:await asyncio.shield(task)

    async def _drive(self):
        while self._dirty:
            self._dirty=False;epoch=self._epoch
            try:
                await self._step(epoch)
            except asyncio.CancelledError:
                self._ready=None;self.binding.withdraw()
                self.last={'state':'unknown','reason':'controller_observation_interrupted'}
                raise
            except Exception as error:
                self._ready=None;self.binding.withdraw()
                # A lost control reply or provider error retains exact intent.
                # No self-scheduled retry or raw input/exception detail in logs.
                self.last={'state':'unknown','reason':type(error).__name__}
                logging.getLogger(__name__).warning('Root preparation observation incomplete (%s)',type(error).__name__)

    async def _step(self,epoch):
        if self.paused and self.scheduler._current_preparation(self.resource) is None:
            self.last={'state':'unknown','reason':'source_paused'}
            return  # Preserve queued work; don't select a new parent while stopping.
        prep=self.scheduler.prepare_root(self.resource)
        if prep is None:
            self.last={'state':'idle','reason':'no_accepted_demand'}
            return
        if prep['state']=='ending':
            await self.binding.end_drained_root(prep['id'],observation='root-controller-cleanup:'+uuid4().hex)
            # An actual physical completion exposes the next saved requester.
            # No busy retry: failed/unknown cleanup exits above until an event.
            self._dirty=True
            return
        if self.paused or not self.permission or not self._host_known:
            reason='source_paused' if self.paused else 'work_not_permitted' if not self.permission else 'host_observation_unknown'
            self.last={'state':'unknown','reason':reason}
            return
        await self.binding.prepare_root(prep['id'])
        if epoch!=self._epoch:
            self.binding.withdraw()
            return
        prep=self.scheduler.preparation(prep['id'])
        if prep['state']=='ending':
            self._dirty=True
            return
        if prep['state']!='admitted':
            self.last={'state':'waiting','reason':'root_not_admitted'}
            return
        child=self.scheduler.next_child(prep['id'])
        if child is None:
            self.last={'state':'held','reason':'no_ready_child_or_work_unresolved'}
            return
        from .manager import Backend
        model=child['model'];backend=self.binding.models.get(model)
        if backend==Backend.OLLAMA:loader=self.manager.load_ollama_model
        elif backend==Backend.VLM:loader=self.manager.load_vlm_model
        else:raise Conflict('Unsupported root child model')
        mm=await source_call(loader,model)
        if epoch!=self._epoch:
            self.binding.withdraw()
            return
        # Preload completed under its exact retained claim. Before dispatch we
        # still need current physical headroom and fresh root/model ownership.
        available,device=await source_call(_physical_headroom)
        if epoch!=self._epoch:
            self.binding.withdraw()
            return
        ident,active=self.binding._admission()
        if ident!=prep['id'] or device>active['lease']['memory_mb'] or capacity.usable_mb(available)<=0:
            raise RootAdmissionDeferred('host_headroom_unavailable')
        self.binding.validate_resident(mm)
        self._ready={'preparation_id':ident,'model':model,'handle':mm,
            'lease_id':mm.lease_id,'model_intent':mm.root_model_intent}
        self.last={'state':'available','reason':'fresh_root_model_and_headroom'}
        self._publishing=True
        try:self.supervisor.capacity_available(self.resource)
        finally:self._publishing=False
