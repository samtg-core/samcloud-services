"""Real local queue/capacity/handoff objects with an in-memory registry adapter.

No provider process, network, production journal or real resource is touched.
"""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .inference_bootstrap import CapacityEvents, QueueRuntime, QueueSettings
from .inference_capacity_source import ManagerCapacitySource
from .inference_reservation import InitialReservations
from .inference_residency import ResidencyStore
from .inference_residency_worker import ResidencyRollover
from .inference_residency_manager import ManagerResidency
from .inference_shared_resource import ResourceHandoff
from .inference_supervisor import Supervisor
from .manager import LEASE_TTL


class SharedResourceCapacity(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows, self.requests = [], []
        self.physical = {'memory_available_mb':20000,'memory_device_inuse_mb':0}
        self.client = SimpleNamespace(get_resource=self.view, list_leases=self.list_rows,
            request_lease=self.request,release_lease=self.release,
            renew_lease=lambda *args:None)
        self.initial = InitialReservations(self.root/'initial.sqlite',self.client,
            registry_owner='owner',service_id='model-service',models={'qwen':'gpu'},wait_s=3600)
        self.running = []
        self.manager = SimpleNamespace(sc=self.client,models={},initial_reservations=self.initial,
            ollama=SimpleNamespace(list_running=lambda:self.running),_collect_stats=lambda:dict(self.physical),
            unload=self.unload)
        runtime = self.runtime = QueueRuntime()
        runtime.manager, runtime.capacity = self.manager, CapacityEvents({'gpu'})
        runtime.settings = QueueSettings(self.root/'queue.sqlite',{'qwen':'gpu'},
            reservation_mb={'qwen':100},shared_queue_wait_s=3600)
        async def execute(*args):
            raise AssertionError('test must not dispatch provider work')
        supervisor = Supervisor(runtime.settings.journal,execute,runtime.capacity.permitted)
        runtime.queue = SimpleNamespace(supervisor=supervisor,held_claims={})
        runtime.residency_actor = ResidencyRollover(ResidencyStore(self.root/'residency.sqlite',
            registry_owner='owner',service_id='model-service'),self.client)
        binding = ManagerResidency(self.manager,runtime.residency_actor,{'qwen':'gpu'},ttl_seconds=LEASE_TTL)
        self.manager.residency_renewal = binding
        runtime.capacity_source = self.source = ManagerCapacitySource(runtime,binding,work_permitted=True)
        runtime.shared_resource = ResourceHandoff(runtime)

    async def asyncTearDown(self):
        await self.source.drain()
        if self.runtime.shared_resource.task is not None:
            await self.runtime.shared_resource.task
        self.runtime.close()
        self.tmp.cleanup()

    def list_rows(self, **filters):
        return [dict(row) for row in self.rows if all(row.get(k)==v for k,v in filters.items())]

    def view(self, resource):
        self.assertEqual(resource,'gpu')
        occupying=[r for r in self.rows if r['status']=='active' or
            (r['status']=='expired' and r.get('expiry_reason')=='ttl_expired' and not r.get('settled_at'))]
        return {'id':'gpu','allocation_mode':'shared','expiry_policy':'hold',
            'available_memory_mb':max(0,100-sum(r['memory_mb'] for r in occupying)),
            'leases':[dict(r) for r in self.rows if r['status'] in ('active','queued')],
            'unsettled':[dict(r) for r in occupying if r['status']=='expired']}

    def request(self, **body):
        self.requests.append(body)
        status='active' if self.view('gpu')['available_memory_mb']>=body['memory_mb'] else 'queued'
        row={'id':f'own-{len(self.requests)}','status':status,'agent_id':'owner',
             'service_id':body['service_id'],'resource_id':body['resource_id'],'memory_mb':body['memory_mb'],
             'exclusive':0,'max_total_s':body['ttl_seconds'],'expires_at':'2099-01-01T00:00:00Z'}
        self.rows.append(row)
        return dict(row,lease_id=row['id'],status_code=200)

    def release(self, lease_id):
        row=next(r for r in self.rows if r['id']==lease_id)
        row['status']='released'
        for queued in self.rows:
            if queued['status']=='queued' and self.view('gpu')['available_memory_mb']>=queued['memory_mb']:
                queued['status']='active'
        return {'status':'released'}

    def unload(self, model, *, require_confirmed):
        self.assertTrue(require_confirmed)
        mm=self.manager.models.pop(model)
        self.running=[]
        self.release(mm.lease_id)
        return {'status':'unloaded'}

    def demand(self):
        journal=self.runtime.queue.supervisor.journal
        grant=journal.submit('requester','turn','gpu','qwen')
        journal.enqueue(grant['id'],'requester','call',{'messages':[]})
        return grant

    def unity(self, status='active'):
        row={'id':'unity','status':status,'service_id':'unity-service','resource_id':'gpu',
             'agent_id':'unity-owner','exclusive':0,'memory_mb':100,'expires_at':'2099-01-01T00:00:00Z'}
        self.rows.append(row)
        return row

    async def test_idle_configuration_does_not_lease_and_pending_demand_does(self):
        await self.source.refresh(self.physical)
        self.assertEqual(self.requests, [])
        self.demand()
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(self.requests[0]['wait_s'],3600)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.initial.snapshot('qwen')['state'],'reserved')

    async def test_vision_demand_stages_and_consumes_one_exact_claim(self):
        from .manager import VLM_MODELS
        vision = next(iter(VLM_MODELS.values()))['default']
        self.runtime.settings = QueueSettings(self.root/'queue.sqlite',
            {'qwen':'gpu', vision:'gpu'}, reservation_mb={'qwen':100},
            shared_queue_wait_s=3600)
        self.initial.models[vision] = 'gpu'
        self.source.binding.models[vision] = 'gpu'
        journal = self.runtime.queue.supervisor.journal
        grant = journal.submit('creative', 'vision-turn', 'gpu', vision)
        execution = journal.enqueue(grant['id'], 'creative', 'vision-call', {'messages':[]})
        with patch('ollama.manager.VLM_MODELS',
                   {'vision':{'default':vision,'memory_mb':100}}):
            await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests), 1)
        self.assertIs(self.source.permitted_model('gpu', vision), True)
        exact = self.initial.snapshot(vision)['lease_id']
        self.assertEqual(self.initial.reserve(vision, resource='gpu',
            service_id='model-service', memory_mb=100, ttl_seconds=LEASE_TTL), exact)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(journal.execution(execution['id'], 'creative')['state'], 'queued')

    async def test_cold_vision_retires_only_exact_terminal_coverage(self):
        from .manager import VLM_MODELS
        vision = next(iter(VLM_MODELS.values()))['default']
        self.runtime.settings = QueueSettings(self.root/'queue.sqlite',
            {'qwen':'gpu', vision:'gpu'}, reservation_mb={'qwen':100},
            shared_queue_wait_s=3600)
        self.source.binding.models[vision] = 'gpu'
        row = dict(id='vision-ended', resource_id='gpu', agent_id='owner',
            service_id='model-service', memory_mb=100, exclusive=0,
            status='active', expires_at='2099-01-01T00:00:00Z', max_total_s=3600)
        store = self.runtime.residency_actor.store
        store.register(vision, resource='gpu', model=vision, receipt=row)
        self.rows.append(row)
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=True):
            await self.source.refresh(self.physical)
        self.assertEqual(store.snapshot(vision)['state'], 'covered')  # active is not retired
        row['status'] = 'released'
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=False):
            await self.source.refresh(self.physical)
        self.assertEqual(store.snapshot(vision)['state'], 'covered')  # terminal alone insufficient
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=True):
            await self.source.refresh(self.physical)
        self.assertEqual(store.snapshot(vision)['state'], 'retired')
        self.assertFalse(self.source.permitted_model('gpu', vision))
        self.assertEqual(self.requests, [])

    async def test_cooled_vision_retires_initial_pointer_without_draining_text(self):
        from .manager import VLM_MODELS
        vision=next(iter(VLM_MODELS.values()))['default']
        self.runtime.settings=QueueSettings(self.root/'queue.sqlite',
            {'qwen':'gpu',vision:'gpu'},reservation_mb={'qwen':100},shared_queue_enabled=True)
        self.initial.models[vision]='gpu'
        self.source.binding.models[vision]='gpu'
        old=self.initial.reserve(vision,resource='gpu',service_id='model-service',
                                 memory_mb=100,ttl_seconds=LEASE_TTL)
        store=self.runtime.residency_actor.store
        store.register(vision,resource='gpu',model=vision,receipt=self.rows[0])
        self.release(old)
        store.retire(vision,self.rows[0])  # prior refresh already retired coverage
        text=self.initial.reserve('qwen',resource='gpu',service_id='model-service',
                                  memory_mb=100,ttl_seconds=LEASE_TTL)
        self.manager.models['qwen']=SimpleNamespace(name='qwen',lease_id=text,memory_mb=100,in_flight=0)
        self.running=[{'name':'qwen'}]
        journal=self.runtime.queue.supervisor.journal
        grant=journal.submit('creative','next-vision','gpu',vision)
        journal.enqueue(grant['id'],'creative','call',{'messages':[]})
        with patch('ollama.manager.VLM_MODELS',{'vision':{'default':vision,'memory_mb':100}}):
            with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=False):
                await self.source.refresh(self.physical)
            self.assertEqual(self.initial.snapshot(vision)['state'],'handed_off')
            self.assertEqual(len(self.requests),2)
            self.rows[0]['agent_id']='another-owner'
            with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
                await self.source.refresh(self.physical)
            self.assertEqual(self.initial.snapshot(vision)['state'],'handed_off')
            self.assertEqual(len(self.requests),2)
            self.rows[0]['agent_id']='owner'
            with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
                await self.source.refresh(self.physical)
        self.assertIsNone(self.runtime.shared_resource.task)
        self.assertIn('qwen',self.manager.models)
        self.assertEqual(self.initial.snapshot(vision)['state'],'queued')
        self.assertEqual(len(self.requests),3)
        self.assertNotEqual(self.initial.snapshot(vision)['lease_id'],old)

    async def test_cooled_vision_retires_pre_rollover_initial_without_draining_text(self):
        from .manager import VLM_MODELS
        vision=next(iter(VLM_MODELS.values()))['default']
        self.runtime.settings=QueueSettings(self.root/'queue.sqlite',
            {'qwen':'gpu',vision:'gpu'},reservation_mb={'qwen':100},shared_queue_enabled=True)
        self.initial.models[vision]='gpu'
        self.source.binding.models[vision]='gpu'
        old=self.initial.reserve(vision,resource='gpu',service_id='model-service',
                                 memory_mb=100,ttl_seconds=LEASE_TTL)
        store=self.runtime.residency_actor.store
        store.register(vision,resource='gpu',model=vision,receipt=self.rows[0])
        self.release(old)
        store.retire(vision,self.rows[0])  # prior refresh already retired coverage
        # The initial reservation retains its original ID after renewal rolls
        # residency onto a successor; ordinary cooldown ends the successor.
        successor=dict(self.rows[0], id='vision-renewal-successor', status='active')
        self.rows.append(successor)
        store.register(vision,resource='gpu',model=vision,receipt=successor)
        self.release(successor['id'])
        store.retire(vision,successor)
        self.assertNotEqual(store.snapshot(vision)['lease_id'],old)
        text=self.initial.reserve('qwen',resource='gpu',service_id='model-service',
                                  memory_mb=100,ttl_seconds=LEASE_TTL)
        self.manager.models['qwen']=SimpleNamespace(name='qwen',lease_id=text,memory_mb=100,in_flight=0)
        self.running=[{'name':'qwen'}]
        journal=self.runtime.queue.supervisor.journal
        grant=journal.submit('creative','next-vision','gpu',vision)
        journal.enqueue(grant['id'],'creative','call',{'messages':[]})
        with patch('ollama.manager.VLM_MODELS',{'vision':{'default':vision,'memory_mb':100}}):
            with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=False):
                await self.source.refresh(self.physical)
            self.assertEqual(self.initial.snapshot(vision)['state'],'handed_off')
            self.assertEqual(len(self.requests),2)
            self.rows[0]['agent_id']='another-owner'
            with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
                await self.source.refresh(self.physical)
            self.assertEqual(self.initial.snapshot(vision)['state'],'handed_off')
            self.assertEqual(len(self.requests),2)
            self.rows[0]['agent_id']='owner'
            with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
                await self.source.refresh(self.physical)
        self.assertIsNone(self.runtime.shared_resource.task)
        self.assertIn('qwen',self.manager.models)
        self.assertEqual(self.initial.snapshot(vision)['state'],'queued')
        self.assertEqual(len(self.requests),3)
        self.assertNotEqual(self.initial.snapshot(vision)['lease_id'],old)

    async def test_busy_pool_stages_once_then_consumes_same_promoted_row_after_fresh_fit(self):
        unity=self.unity()
        self.demand()
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),1)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.initial.snapshot('qwen')['lease_id'],'own-1')
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),1)
        unity['status']='released'
        self.rows[-1]['status']='active'
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.initial.reserve('qwen',resource='gpu',service_id='model-service',
            memory_mb=100,ttl_seconds=LEASE_TTL),'own-1')
        self.assertEqual(len(self.requests),1)

    async def queued_text_with_resident_vision(self, *, foreign_first=False):
        from .manager import VLM_MODELS
        absence=patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True)
        absence.start()
        self.addCleanup(absence.stop)
        vision=next(iter(VLM_MODELS.values()))['default']
        self.runtime.settings=QueueSettings(self.root/'queue.sqlite',
            {'qwen':'gpu',vision:'gpu'},reservation_mb={'qwen':100,vision:20},
            shared_queue_enabled=True)
        self.initial.models[vision]='gpu'
        self.source.binding.models[vision]='gpu'
        big=self.unity()
        big['memory_mb']=50
        lease=self.initial.reserve(vision,resource='gpu',service_id='model-service',
                                   memory_mb=20,ttl_seconds=LEASE_TTL)
        self.manager.models[vision]=SimpleNamespace(name=vision,lease_id=lease,
                                                   memory_mb=20,in_flight=0)
        grant=self.demand()
        await self.source.refresh(self.physical)
        wanted=self.initial.snapshot('qwen')['lease_id']
        self.assertEqual(self.initial.snapshot('qwen')['state'],'queued')
        foreign=dict(big,id='later-render',status='queued',memory_mb=25)
        index=next(i for i,row in enumerate(self.rows) if row['id']==wanted)
        self.rows.insert(index if foreign_first else len(self.rows),foreign)
        released=[]

        def strict_release(lease_id):
            row=next(row for row in self.rows if row['id']==lease_id)
            row['status']='cancelled' if row['status']=='queued' else 'released'
            released.append(lease_id)
            # The registry's strict head of line, not the older fixture's
            # fit-any-waiter shortcut: later small requests cannot pass.
            for queued in self.rows:
                if queued['status']!='queued':
                    continue
                if self.view('gpu')['available_memory_mb']<queued['memory_mb']:
                    break
                queued['status']='active'
            return {'status':row['status']}

        self.release=strict_release
        self.client.release_lease=strict_release
        return grant,wanted,foreign,released

    async def test_handoff_preserves_earlier_queued_text_demand_and_registry_order(self):
        grant,wanted,foreign,released=await self.queued_text_with_resident_vision()
        before=dict(self.runtime.queue.supervisor.journal.get(grant['id'],'requester'))
        await self.source.refresh(self.physical)
        self.assertEqual((await self.runtime.shared_resource.task)['phase'],'idle')
        self.assertEqual(self.initial.snapshot('qwen')['lease_id'],wanted)
        self.assertEqual(self.initial.snapshot('qwen')['state'],'queued')
        self.assertNotIn(wanted,released)
        self.assertEqual(foreign['status'],'queued')
        self.assertEqual(len(self.requests),2)
        self.assertEqual(dict(self.runtime.queue.supervisor.journal.get(grant['id'],'requester')),before)
        with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
            await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),2)  # same queue claim, no re-acquisition
        # Explicit caller withdrawal makes the reservation unused and releasable.
        self.runtime.queue.supervisor.journal.request_release(grant['id'],'requester')
        with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
            await self.source.refresh(self.physical)
            await self.runtime.shared_resource.task
        self.assertEqual(self.initial.snapshot('qwen')['state'],'retired')
        self.assertIn(wanted,released)
        self.assertEqual(foreign['status'],'active')

    async def test_handoff_still_serves_earlier_foreign_waiter_without_rekeying_text(self):
        grant,wanted,foreign,released=await self.queued_text_with_resident_vision(foreign_first=True)
        await self.source.refresh(self.physical)
        self.assertEqual((await self.runtime.shared_resource.task)['phase'],'idle')
        self.assertEqual(foreign['status'],'active')
        self.assertEqual(self.manager.models,{})
        self.assertEqual(self.initial.snapshot('qwen')['state'],'queued')
        self.assertEqual(self.initial.snapshot('qwen')['lease_id'],wanted)
        self.assertNotIn(wanted,released)
        self.assertEqual(len(self.requests),2)

    async def test_saved_handoff_reentry_retains_exact_queued_demand(self):
        self.unity()
        grant=self.demand()
        await self.source.refresh(self.physical)
        wanted=self.initial.snapshot('qwen')['lease_id']
        call=dict(self.runtime.queue.supervisor.journal.db.execute(
            'SELECT * FROM executions WHERE grant_id=?',(grant['id'],)).fetchone())
        self.runtime.shared_resource._save('unloading',allocations=[
            {'model':'qwen','lease_id':wanted,'memory_mb':100}])
        # Recover a saved handoff rather than building fresh allocation pointers.
        self.runtime.shared_resource=ResourceHandoff(self.runtime)
        result=await self.runtime.shared_resource.begin()
        self.assertEqual(result['phase'],'idle')
        self.assertEqual(self.rows[-1]['status'],'queued')
        self.assertEqual(self.initial.snapshot('qwen')['lease_id'],wanted)
        self.assertEqual(self.initial.snapshot('qwen')['state'],'queued')
        self.assertEqual(len(self.requests),1)
        self.assertEqual(dict(self.runtime.queue.supervisor.journal.execution(call['id'],'requester')),call)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))

    async def test_queued_demand_preservation_refuses_foreign_receipt(self):
        self.unity()
        self.demand()
        await self.source.refresh(self.physical)
        wanted=self.initial.snapshot('qwen')['lease_id']
        self.runtime.shared_resource._save('unloading',allocations=[
            {'model':'qwen','lease_id':wanted,'memory_mb':100}])
        self.rows[-1]['agent_id']='different-owner'
        result=await self.runtime.shared_resource.begin()
        self.assertEqual(result['phase'],'uncertain')
        self.assertEqual(self.rows[-1]['status'],'queued')
        self.assertEqual(self.initial.snapshot('qwen')['state'],'queued')
        self.assertEqual(len(self.requests),1)

    async def test_promoted_grant_does_not_override_failed_physical_fit(self):
        self.demand()
        self.manager._collect_stats=lambda:{}
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.initial.snapshot('qwen')['state'],'reserved')

    async def test_resident_yields_then_actual_demand_requeues_behind_unity(self):
        lease=self.initial.reserve('qwen',resource='gpu',service_id='model-service',
            memory_mb=100,ttl_seconds=LEASE_TTL)
        self.manager.models['qwen']=SimpleNamespace(name='qwen',lease_id=lease,memory_mb=100,in_flight=0)
        self.running=[{'name':'qwen'}]
        await self.source.refresh(self.physical)  # bind resident receipt
        unity=self.unity('queued')
        self.demand()
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        await self.runtime.shared_resource.task
        self.assertEqual(self.manager.models,{})
        self.assertEqual(unity['status'],'active')
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),2)
        self.assertEqual(self.rows[-1]['status'],'queued')
        self.assertEqual(self.initial.snapshot('qwen')['lease_id'],'own-2')
        self.assertFalse(self.runtime.capacity.permitted('gpu'))

    async def test_explicitly_cancelled_demand_releases_unused_staged_reservation(self):
        grant=self.demand()
        await self.source.refresh(self.physical)
        self.assertEqual(self.initial.snapshot('qwen')['state'],'reserved')
        self.runtime.queue.supervisor.journal.request_release(grant['id'],'requester')
        await self.source.refresh(self.physical)
        await self.runtime.shared_resource.task
        self.assertEqual(self.initial.snapshot('qwen')['state'],'retired')
        self.assertEqual(self.rows[0]['status'],'released')
        self.assertEqual(len(self.requests),1)

    async def test_shutdown_waits_for_owned_handoff_and_cannot_be_reopened_by_it(self):
        lease=self.initial.reserve('qwen',resource='gpu',service_id='model-service',
            memory_mb=100,ttl_seconds=LEASE_TTL)
        self.manager.models['qwen']=SimpleNamespace(name='qwen',lease_id=lease,memory_mb=100,in_flight=0)
        self.running=[{'name':'qwen'}]
        self.unity('queued')
        barrier=asyncio.Event()
        async def prepare():await barrier.wait()
        self.runtime.queue.supervisor.prepare(prepare())
        owned=self.runtime.shared_resource.begin()
        await asyncio.sleep(0)
        with self.assertRaisesRegex(RuntimeError,'must finish'):
            self.runtime.close()
        stopping=asyncio.create_task(self.runtime.quiesce())
        await asyncio.sleep(0)
        self.assertFalse(stopping.done())
        barrier.set()
        await owned
        self.assertTrue((await stopping)['drained'])
        self.assertTrue(self.source.paused)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))


if __name__ == '__main__':
    unittest.main()
