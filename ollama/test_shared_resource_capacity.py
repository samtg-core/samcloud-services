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
            'available_memory_mb':max(0,100-sum(r['memory_mb'] or 0 for r in occupying)),
            'leases':[dict(r) for r in self.rows if r['status'] in ('active','queued')],
            'queue':[{'lease_id':r['id'],'service_id':r.get('service_id'),'agent_id':r.get('agent_id'),
                      'memory_mb':r.get('memory_mb'),'exclusive':bool(r.get('exclusive'))}
                     for r in self.rows if r['status']=='queued'],
            'unsettled':[dict(r) for r in occupying if r['status']=='expired']}

    def request(self, **body):
        self.requests.append(body)
        exclusive = body.get('exclusive', False)
        if exclusive:
            status = 'queued' if any(r['status'] == 'active' for r in self.rows) else 'active'
        else:
            status='active' if self.view('gpu')['available_memory_mb']>=body['memory_mb'] else 'queued'
        row={'id':f'own-{len(self.requests)}','status':status,'agent_id':'owner',
             'service_id':body['service_id'],'resource_id':body['resource_id'],'memory_mb':body.get('memory_mb'),
             'exclusive':int(exclusive),'max_total_s':body['ttl_seconds'],'expires_at':'2099-01-01T00:00:00Z'}
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

    async def test_exclusive_admission_uses_physical_fit_and_recognizes_only_its_own_grant(self):
        import json
        config = self.root/'queue.json'
        config.write_text(json.dumps(dict(journal=str(self.root/'queue.sqlite'), models={'qwen':'gpu'},
            reservation_mb={'qwen': 61440}, exclusive_models=['qwen'], shared_queue_enabled=True)))
        settings = self.runtime.settings = QueueSettings.from_file(config)
        self.initial.exclusive_models = frozenset(settings.exclusive_models)
        self.initial.budgets = dict(settings.reservation_mb)
        self.runtime.residency_actor.store.exclusive_budgets = dict(settings.reservation_mb)
        self.demand()
        await self.source.refresh(self.physical)
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'], 'physical_capacity')
        self.assertEqual(self.requests, [])
        self.physical['memory_available_mb'] = 80000
        peer = self.unity()
        await self.source.refresh(self.physical)
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'], 'registry_queued')
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(self.requests[0]['exclusive'])
        self.assertNotIn('memory_mb', self.requests[0])
        peer['status'] = 'released'
        self.rows[-1]['status'] = 'active'
        await self.source.refresh(self.physical)
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'], 'cold_fit')
        lease = self.initial.reserve('qwen', resource='gpu', service_id='model-service',
                                     memory_mb=61440, ttl_seconds=LEASE_TTL)
        self.manager.models['qwen'] = SimpleNamespace(name='qwen', lease_id=lease, memory_mb=40000)
        self.running = [{'name':'qwen'}]
        await self.source.refresh(self.physical)
        self.assertTrue(self.source.permitted_model('gpu','qwen'))
        peer['status'] = 'active'
        await self.source.refresh(self.physical)
        self.assertIsNone(self.source.permitted_model('gpu','qwen'))
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'], 'exclusive_ownership_unconfirmed')
        self.assertEqual(len(self.requests), 1)

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

    async def test_promoted_grant_does_not_override_failed_physical_fit(self):
        self.demand()
        self.manager._collect_stats=lambda:{}
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.initial.snapshot('qwen')['state'],'reserved')

    def queued_call(self):
        journal=self.runtime.queue.supervisor.journal
        grant=journal.submit('requester','turn','gpu','qwen')
        return journal.enqueue(grant['id'],'requester','call',{'messages':[]})

    def wait_reason(self, execution):
        return self.runtime.queue.supervisor.journal.execution(execution['id'],'requester')['wait_reason']

    async def test_registry_size_refusal_is_distinct_from_waiting_in_line(self):
        self.queued_call()
        refusal = {'status_code': 409, 'detail': {'code': 'exceeds_capacity'}}
        with patch.object(self.initial.client, 'request_lease', return_value=refusal):
            await self.source.refresh(self.physical)
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'], 'reservation_exceeds_capacity')
        self.assertIsNone(self.initial.snapshot('qwen')['lease_id'])
        self.assertFalse(self.runtime.capacity.permitted('gpu'))

    async def test_host_misfit_takes_no_lease_and_names_the_wait(self):
        call=self.queued_call()
        self.physical['memory_available_mb']=50  # registry has room, host does not
        await self.source.refresh(self.physical)
        self.assertEqual(self.requests,[])
        self.assertIsNone(self.initial.snapshot('qwen'))
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'],'physical_capacity')
        self.assertTrue(self.wait_reason(call).startswith('waiting for memory: needs 100 MB'))
        await self.source.refresh(self.physical)
        self.assertEqual(self.requests,[])  # waiting re-asks nothing
        self.physical['memory_available_mb']=20000
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(self.initial.snapshot('qwen')['state'],'reserved')
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertIsNone(self.wait_reason(call))

    async def test_promoted_reservation_the_host_cannot_hold_is_given_back(self):
        unity=self.unity()
        call=self.queued_call()
        await self.source.refresh(self.physical)
        self.assertEqual(self.rows[-1]['status'],'queued')  # a place in line, no bytes
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'],'registry_queued')
        unity['status']='released'
        self.rows[-1]['status']='active'
        self.physical['memory_available_mb']=50
        await self.source.refresh(self.physical)
        self.assertEqual(self.rows[-1]['status'],'released')
        self.assertEqual(self.initial.snapshot('qwen')['state'],'retired')
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertTrue(self.wait_reason(call).startswith('waiting for memory'))
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),1)  # no re-stage while the host is short
        self.physical['memory_available_mb']=20000
        await self.source.refresh(self.physical)
        self.assertEqual(len(self.requests),2)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertIsNone(self.wait_reason(call))

    async def test_a_registry_wait_names_who_holds_the_memory(self):
        unity=self.unity()
        call=self.queued_call()
        await self.source.refresh(self.physical)
        self.assertEqual(self.rows[-1]['status'],'queued')
        self.assertEqual(self.source.last_model['gpu','qwen']['reason'],'registry_queued')
        self.assertEqual(self.wait_reason(call),
            'waiting for memory: needs 100 MB of gpu, held now by unity-service (100 MB); '
            'the model is in the resource queue and holds no memory while it waits')
        await self.source.refresh(self.physical)  # still queued: same words, no second lease
        self.assertEqual(len(self.requests),1)
        self.assertIn('held now by unity-service (100 MB)',self.wait_reason(call))
        unity['status']='released'
        self.rows[-1]['status']='active'
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertIsNone(self.wait_reason(call))

    async def test_a_registry_wait_names_the_queue_ahead_and_exclusive_wins(self):
        self.unity()
        self.rows.append({'id':'excl','status':'queued','service_id':None,'agent_id':'user:36',
            'resource_id':'gpu','exclusive':1,'memory_mb':50,'expires_at':'2099-01-01T00:00:00Z'})
        call=self.queued_call()
        await self.source.refresh(self.physical)
        self.assertEqual(self.wait_reason(call),
            'waiting for memory: needs 100 MB of gpu, held now by unity-service (100 MB); '
            'behind 1 queued request (user:36 (exclusive)); '
            'the model is in the resource queue and holds no memory while it waits')
        await self.source.refresh(self.physical)  # our own place in line is not "ahead"
        self.assertIn('behind 1 queued request (user:36 (exclusive))',self.wait_reason(call))

    async def test_holder_labels_are_checked_capped_and_empty_means_earlier_requests(self):
        call=self.queued_call()
        journal=self.runtime.queue.supervisor.journal
        journal.memory_wait('gpu','qwen',100,[],[])
        self.assertIn('of gpu, behind earlier requests;',self.wait_reason(call))
        journal.memory_wait('gpu','qwen',100,[f's{i} (1 MB)' for i in range(5)],['q (exclusive)','r'])
        self.assertIn('held now by s0 (1 MB), s1 (1 MB), s2 (1 MB) and 2 more; '
                      'behind 2 queued requests (q (exclusive), r);',self.wait_reason(call))
        for bad in ('unity',[''],[None]):
            with self.assertRaises(ValueError):
                journal.memory_wait('gpu','qwen',100,bad)
        with self.assertRaises(ValueError):
            journal.memory_wait('gpu','qwen',100,None,['q'])

    async def test_a_deferral_reason_is_not_overwritten_by_the_memory_note(self):
        call=self.queued_call()
        journal=self.runtime.queue.supervisor.journal
        journal.db.execute("UPDATE executions SET wait_reason='capacity_unavailable' WHERE id=?",(call['id'],))
        journal.memory_wait('gpu','qwen',100)
        self.assertEqual(self.wait_reason(call),'capacity_unavailable')
        journal.memory_wait('gpu','qwen',None)
        self.assertEqual(self.wait_reason(call),'capacity_unavailable')

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
