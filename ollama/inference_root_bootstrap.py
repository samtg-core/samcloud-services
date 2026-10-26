"""Trusted, explicit root runtime construction; no launcher/default activation.

No legacy claim adoption or fallback. Fresh physical absence and current own
registry identity/capability are required before mounting one dispatcher.
"""
import asyncio
import sqlite3
import json
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Callable,Mapping
from uuid import uuid4

from .inference_arbiter import Conflict
from .inference_bootstrap import QueueSettings
from .inference_events import source_call
from .inference_gateway import GatewayQueue
from .inference_root_client import RegistryRootClient,_base
from .inference_root_controller import RootController
from .inference_root_models import RootModelBinding
from .inference_root_scheduler import RootPreparingArbiter
from .manager import Backend


@dataclass(frozen=True)
class RootQueueStartup:
    settings: QueueSettings
    client: RegistryRootClient
    models: Mapping[str,Backend]
    confirm_cold: Callable
    work_permitted: bool

    def __post_init__(self):
        if (not isinstance(self.settings,QueueSettings) or not isinstance(self.client,RegistryRootClient)
            or type(self.work_permitted) is not bool or not callable(self.confirm_cold)):
            raise ValueError('Explicit root settings, own client, cold sensor and work permission required')
        models=dict(self.models)
        if (set(models)!=set(self.settings.models) or len(set(self.settings.models.values()))!=1
            or any(value not in (Backend.OLLAMA,Backend.VLM) or not isinstance(value,Backend) for value in models.values())
            or self.settings.reservation_mb or self.settings.shared_queue_wait_s is not None):
            raise ValueError('Exact single-resource root model map without legacy budgets/deadlines required')
        object.__setattr__(self,'models',MappingProxyType(models))

    async def start(self,service):
        manager=service.mgr;store=self.client.store
        resource=next(iter(self.settings.models.values()))
        if (getattr(service,'inference_dispatcher',None) is not None
            or getattr(service,'inference_runtime',None) is not None
            or any(getattr(route,'path','').startswith('/v1/inference') for route in service.app.routes)
            or getattr(service.config,'EXO_ENABLED',False)
            or resource!=service.config.SC_RESOURCE_ID or store.service!=service.config.SC_SERVICE_ID
            or _base(manager.sc.base)!=store.registry):
            raise Conflict('Fresh exact local root service binding required')
        if (manager.durable_inference or manager.models or manager.root_binding is not None
            or manager.inference_capacity_source is not None or manager.initial_reservations is not None
            or manager.residency_renewal is not None or manager._background_started
            or getattr(manager,'_pool_leases',set()) or getattr(manager,'_unreleased',{})
            or any((task:=getattr(manager,name,None)) is not None and not task.done()
                   for name in manager.MAINTENANCE_TASKS)):
            raise Conflict('Root startup cannot adopt legacy accounting or live maintenance')
        identity=await source_call(manager.sc.verify_token)
        if (not isinstance(identity,dict) or identity.get('valid') is not True
            or identity.get('owner_id')!=store.actor):
            raise Conflict('Root provider credential identity differs')
        await source_call(lambda:self.client.negotiate(refresh=True))
        physical=await source_call(manager.sc.get_resource,resource)
        if (not isinstance(physical,dict) or physical.get('id')!=resource
            or physical.get('allocation_mode')!='shared' or physical.get('expiry_policy')!='hold'):
            raise Conflict('Root resource requires exact shared hold accounting')
        if await source_call(self.confirm_cold) is not True:
            raise Conflict('Fresh backend/cache absence required before root startup')
        with store._transaction() as conn:
            for row in conn.execute('SELECT resource,receipt FROM root_client_intents WHERE resource!=?',(resource,)):
                receipt=json.loads(row['receipt']) if row['receipt'] else None
                if not isinstance(receipt,dict) or receipt.get('state')!='complete':
                    raise Conflict('Unfinished physical root outside this runtime scope')
        # A cold process is not permission to migrate a legacy accepted queue.
        # Check before RootPreparingArbiter could stamp its provider identity.
        if self.settings.journal.exists():
            conn=sqlite3.connect('file:'+str(self.settings.journal)+'?mode=ro',uri=True)
            try:
                tables={row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if {'executions','grants'} <= tables:
                    calls=conn.execute("SELECT g.resource,g.model FROM executions e JOIN grants g ON g.id=e.grant_id WHERE e.state IN ('queued','running','uncertain')")
                    if any(row[0]!=resource or row[1] not in self.models for row in calls):
                        raise Conflict('Accepted model/resource outside startup scope requires explicit migration')
                if 'requester_root_identity' not in tables:
                    debt=(('executions' in tables and conn.execute("SELECT 1 FROM executions WHERE state IN ('queued','running','uncertain')").fetchone())
                          or ('grants' in tables and conn.execute("SELECT 1 FROM grants WHERE state IN ('granted','uncertain')").fetchone())
                          or ('requester_jobs' in tables and conn.execute("SELECT 1 FROM requester_jobs WHERE state='owned'").fetchone()))
                    if debt:raise Conflict('Explicit legacy queue migration/drain required before root startup')
            finally:conn.close()
        original=(manager.sc,manager.durable_inference,manager.root_binding,manager.inference_capacity_source)
        routes=list(service.app.router.routes);queue=None
        try:
            factory=partial(RootPreparingArbiter,registry=store.registry,
                            registry_actor=store.actor,service_id=store.service)
            queue=GatewayQueue(self.settings.journal,service,self.settings.models,lambda _:False,
                stint_policy=self.settings.stint_policy,journal_factory=factory)
            manager.durable_inference=True
            binding=RootModelBinding(manager,queue.supervisor.journal,self.client,resource=resource,
                models=self.models,confirm_cold=self.confirm_cold)
            controller=RootController(queue.supervisor,binding,work_permitted=self.work_permitted)
            controller.invalidate('startup_awaiting_host_observation')
            runtime=RootQueueRuntime(queue,manager,self.settings,binding,controller)
            service.app.include_router(queue.router())
        except BaseException:
            service.app.router.routes[:]=routes
            manager.sc,manager.durable_inference,manager.root_binding,manager.inference_capacity_source=original
            if queue is not None:queue.supervisor.close()
            raise  # Never fall back to direct dispatch or legacy accounting.
        service.inference_dispatcher=queue;service.inference_runtime=runtime
        return runtime


class RootQueueRuntime:
    def __init__(self,queue,manager,settings,binding,controller):
        self.queue,self.manager,self.settings=queue,manager,settings
        self.binding,self.capacity_source=binding,controller
        self.shared_resource=None;self._quiesce_task=None;self._closed=False

    def root_snapshot(self):
        """Bounded loop-owned diagnostics; saved IDs are not admission proof."""
        prep=None if self._closed else self.queue.supervisor.journal._current_preparation(self.binding.resource)
        return {'mode':'requester_root','source':self.capacity_source.snapshot(),
            'preparation':None if prep is None else {key:prep[key] for key in
                ('id','state','root_id','initial_lease_id','current_lease_id')},
            'quiescing':self._quiesce_task is not None and not self._quiesce_task.done(),
            'closed':self._closed}

    async def quiesce(self):
        """Owned unbounded drain plus explicit cold suspension, never parent end."""
        if self._closed:raise Conflict('Root runtime is closed')
        task=self._quiesce_task
        if task is not None and task.done():
            if task.cancelled() or task.exception() is not None or task.result().get('drained') is not True:
                task=None  # An explicit caller may retry after reconciliation.
        if task is None:
            task=asyncio.create_task(self._quiesce());self._quiesce_task=task
        return await asyncio.shield(task)

    async def _quiesce(self):
        source=self.capacity_source;source.pause();await source.drain()
        supervisor=self.queue.supervisor
        while tasks:=tuple(t for t in (*supervisor.workers.values(),*supervisor.preparations) if not t.done()):
            await asyncio.shield(asyncio.gather(*tasks,return_exceptions=True))
        counts={row[0]:row[1] for row in supervisor.journal.db.execute('SELECT state,COUNT(*) FROM executions GROUP BY state')}
        state={'running':counts.get('running',0),'uncertain':counts.get('uncertain',0),
               'queued':counts.get('queued',0),'held_claims':len(self.queue.held_claims),
               'manager_claims':sum(max(0,mm.in_flight) for mm in self.manager.models.values())}
        if any(state[key] for key in ('running','uncertain','held_claims','manager_claims')):
            return {'drained':False,**state}
        try:
            await self.binding.suspend_for_restart(observation='root-runtime-cold-suspension:'+uuid4().hex)
        except Exception as error:
            return {'drained':False,**state,'reason':type(error).__name__}
        return {'drained':True,**state,'ownership_retained':True}

    def resume(self):
        if self._closed or self._quiesce_task is not None and not self._quiesce_task.done():
            raise Conflict('Root shutdown drain must finish before explicit resume')
        self.capacity_source.resume();self._quiesce_task=None

    def close(self):
        task=getattr(self.manager,'_shutdown_drain_task',None)
        if (task is None or not task.done() or task.cancelled() or task.exception() is not None
            or not self.binding._health_stopped):
            raise Conflict('Manager maintenance shutdown must finish before root runtime close')
        if not self._closed:
            self.binding.assert_shutdown_ready()
            self.queue.supervisor.close()
            self.binding.client.close();self._closed=True
