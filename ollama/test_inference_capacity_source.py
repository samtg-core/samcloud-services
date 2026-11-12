"""Fresh registry/headroom admission; real reservation receipts, no live compute."""
import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from .inference_bootstrap import CapacityEvents,QueueSettings
from .inference_capacity_source import ManagerCapacitySource
from .inference_residency import ResidencyStore
from .inference_residency_worker import ResidencyRollover
from .inference_residency_manager import ManagerResidency


class CapacitySource(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.receipt = {'id':'owned','agent_id':'service-owner','service_id':'service',
            'resource_id':'gpu','memory_mb':100,'max_total_s':3600,'exclusive':0,
            'status':'active','expires_at':'2099-01-01T00:00:00+00:00'}
        self.resource = {'id':'gpu','allocation_mode':'shared','available_memory_mb':10000,'leases':[]}
        self.rows = [self.receipt]
        self.client = SimpleNamespace(get_resource=lambda resource:self.resource,
                                     list_leases=lambda **kw:self.rows)
        self.manager = SimpleNamespace(sc=self.client,models={})
        store = ResidencyStore(Path(self.tmp.name)/'residency.sqlite',
                               registry_owner='service-owner',service_id='service')
        self.actor = ResidencyRollover(store,self.client)
        self.binding = ManagerResidency(self.manager,self.actor,{'qwen':'gpu'},ttl_seconds=3600)
        self.runtime = SimpleNamespace(manager=self.manager,
            settings=QueueSettings(Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu'},
                                   reservation_mb={'qwen':100}),capacity=CapacityEvents({'gpu'}))
        self.source = ManagerCapacitySource(self.runtime,self.binding,work_permitted=True)
        self.physical = {'memory_available_mb':20000,'memory_device_inuse_mb':80}

    async def asyncTearDown(self):
        await self.source.drain()
        self.actor.close()
        self.tmp.cleanup()

    def resident(self):
        self.manager.models['qwen'] = SimpleNamespace(name='qwen',lease_id='owned',memory_mb=80)
        self.resource.update(available_memory_mb=0,leases=[self.receipt])

    async def test_cold_uses_real_fit_without_assuming_a_fixed_hardware_total(self):
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.resource['available_memory_mb'] = 1
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'registry_capacity')

    async def test_normal_unload_and_reload_reconciles_ended_residency(self):
        self.resident()
        await self.source.refresh(self.physical)
        self.assertTrue(self.binding.ready('qwen'))
        self.manager.models.clear()
        self.manager.ollama=SimpleNamespace(list_running=lambda:[])
        self.rows=[self.receipt|{'status':'released'}]
        self.resource.update(available_memory_mb=10000,leases=[])
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.actor.store.snapshot(self.binding.allocation('qwen'))['state'],'retired')
        self.receipt=self.receipt|{'id':'reload'}
        self.rows=[self.receipt]
        self.resident()
        self.manager.models['qwen'].lease_id='reload'
        await self.source.refresh(self.physical)
        self.assertTrue(self.binding.ready('qwen'))

    async def test_unload_does_not_guess_backend_absence_or_registry_release(self):
        self.resident()
        await self.source.refresh(self.physical)
        self.manager.models.clear()
        for running,rows in (([],[self.receipt]),([{'name':'qwen'}],
                            [self.receipt|{'status':'released'}]),(None,[]),([],[])):
            with self.subTest(running=running,rows=rows):
                self.manager.ollama=SimpleNamespace(list_running=lambda:running)
                self.rows=rows
                await self.source.refresh(self.physical)
                self.assertFalse(self.runtime.capacity.permitted('gpu'))

    async def test_existing_sizing_reads_model_estimate_without_allocating_or_loading(self):
        from unittest.mock import Mock
        self.runtime.settings = QueueSettings(Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu'})
        self.manager.ollama = SimpleNamespace(memory_estimate_mb=Mock(return_value=100),load_model=Mock())
        self.client.request_lease = Mock()
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.manager.ollama.memory_estimate_mb.assert_called_once_with('qwen')
        self.manager.ollama.load_model.assert_not_called()
        self.client.request_lease.assert_not_called()
        self.assertEqual(self.manager.models,{})
        self.manager.ollama.memory_estimate_mb.return_value = 30000
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'registry_capacity')

    async def test_smaller_vision_can_fit_without_the_large_text_estimate_fitting(self):
        vision='mlx-community/Qwen2.5-VL-7B-Instruct-4bit'
        self.runtime.settings=QueueSettings(Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu',vision:'gpu'})
        self.binding.models={'qwen':'gpu',vision:'gpu'}
        self.binding.require_size=False
        self.manager.ollama=SimpleNamespace(memory_estimate_mb=lambda model:20000)
        await self.source.refresh({'memory_available_mb':10000,'memory_device_inuse_mb':0})
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertFalse(self.source.permitted_model('gpu','qwen'))
        self.assertTrue(self.source.permitted_model('gpu',vision))
        self.assertEqual(self.manager.models,{})
        from .inference_arbiter import Arbiter
        journal=Arbiter(Path(self.tmp.name)/'dispatch.sqlite')
        try:
            first=journal.submit('text','text','gpu','qwen')
            journal.enqueue(first['id'],'text','call',{})
            second=journal.submit('vision','vision','gpu',vision)
            journal.enqueue(second['id'],'vision','call',{})
            item=journal.dispatch_next('gpu',permitted_model=self.source.permitted_model)
            self.assertEqual(item['grant']['id'],second['id'])
        finally:journal.close()

    async def test_existing_resident_text_and_vision_have_separate_exact_receipts(self):
        vision='mlx-community/Qwen2.5-VL-7B-Instruct-4bit'
        self.runtime.settings=QueueSettings(Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu',vision:'gpu'})
        self.binding.models={'qwen':'gpu',vision:'gpu'}
        self.binding.require_size=False
        self.resident()
        extra=self.receipt|{'id':'vision-owned','memory_mb':5700}
        self.manager.models[vision]=SimpleNamespace(name=vision,lease_id='vision-owned',memory_mb=5700)
        self.resource['leases'].append(extra)
        self.rows=[self.receipt,extra]
        await self.source.refresh(self.physical)
        self.assertTrue(self.source.permitted_model('gpu','qwen'))
        self.assertTrue(self.source.permitted_model('gpu',vision))
        self.rows=[self.receipt,extra|{'agent_id':'other'}]
        await self.source.refresh(self.physical)
        self.assertTrue(self.source.permitted_model('gpu','qwen'))
        self.assertFalse(self.source.permitted_model('gpu',vision))

    async def test_vision_first_refresh_preserves_text_stint_and_blocks_unknown_dispatch(self):
        from .inference_arbiter import Arbiter, StintPolicy
        from unittest.mock import patch
        vision='mlx-community/Qwen2.5-VL-7B-Instruct-4bit'
        self.runtime.settings=QueueSettings(Path(self.tmp.name)/'queue.sqlite',
                                           {vision:'gpu','qwen':'gpu'})
        self.binding.models={vision:'gpu','qwen':'gpu'}
        self.binding.require_size=False
        self.resident()
        self.resource['available_memory_mb']=10000
        await self.source.refresh(self.physical)
        journal=Arbiter(Path(self.tmp.name)/'stints.sqlite',
                        stint_policy=StintPolicy(idle_grace_seconds=300))
        with patch('ollama.inference_arbiter.time.time',return_value=100):
            try:
                a=journal.submit('a','conversation-a','gpu','qwen')
                first=journal.enqueue(a['id'],'a','first',{})
                journal.dispatch_next('gpu',permitted_model=self.source.permitted_model)
                journal.finish(first['id'],{})
                b=journal.submit('b','conversation-b','gpu','qwen')
                journal.enqueue(b['id'],'b','waiting',{})
                journal.enqueue(a['id'],'a','quick-next-round',{})
                dispatched=[]
                unknown_checks=[]
                def pump(resource):
                    permission=self.source.permitted_model(resource,'qwen')
                    item=journal.dispatch_next(resource,permitted_model=self.source.permitted_model)
                    if permission is None:
                        unknown_checks.append(item)
                        self.assertEqual(journal.get(a['id'],'a')['state'],'granted')
                    if item is not None:dispatched.append(item['grant']['owner'])
                self.runtime.capacity._pump=pump
                await self.source.refresh(self.physical)
                self.assertTrue(unknown_checks)  # real partial publication reached dispatch
                self.assertTrue(all(item is None for item in unknown_checks))
                self.assertEqual(dispatched,['a'])
                self.assertIsNone(journal.get(a['id'],'a')['handoff_reason'])
                self.assertEqual(journal.get(b['id'],'b')['state'],'queued')
            finally:
                self.runtime.capacity._pump=None
                journal.close()

    async def test_unknown_is_distinct_from_affirmative_unavailability(self):
        await self.source.refresh(self.physical)
        self.assertIs(self.source.permitted_model('gpu','qwen'),True)
        self.source.invalidate()
        self.assertIsNone(self.source.permitted_model('gpu','qwen'))
        self.source.publish('gpu','unavailable','host_headroom',model='qwen')
        self.assertIs(self.source.permitted_model('gpu','qwen'),False)
        self.source.pause()
        self.assertIsNone(self.source.permitted_model('gpu','qwen'))

    async def test_unknown_existing_model_estimate_withdraws_dispatch(self):
        self.runtime.settings = QueueSettings(Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu'})
        self.manager.ollama = SimpleNamespace(memory_estimate_mb=lambda model:None)
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'observation_unconfirmed')

    async def test_existing_resident_policy_does_not_add_a_new_memory_gate(self):
        self.runtime.settings = QueueSettings(Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu'})
        self.binding.require_size = False
        self.resident()
        self.manager.models['qwen'].memory_mb = 110  # existing sizing tolerance
        await self.source.refresh({'memory_available_mb':0,'memory_device_inuse_mb':200})
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'resident_existing_policy')
        self.rows = [self.receipt|{'agent_id':'other'}]
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))

    async def test_resident_reuses_exact_owned_reservation_without_charging_it_twice(self):
        self.resident()
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.assertTrue(self.binding.ready('qwen'))
        self.assertEqual(self.source.last['gpu']['reason'],'resident_reserved_with_headroom')

    async def test_lost_headroom_or_device_excess_withdraws_not_unloads(self):
        self.resident()
        mm = self.manager.models['qwen']
        for physical,reason in ((self.physical|{'memory_available_mb':0},'host_headroom'),
                                (self.physical|{'memory_device_inuse_mb':101},'device_usage_exceeds_reservation')):
            await self.source.refresh(physical)
            self.assertFalse(self.runtime.capacity.permitted('gpu'))
            self.assertEqual(self.source.last['gpu']['reason'],reason)
            self.assertIs(self.manager.models['qwen'],mm)

    async def test_foreign_receipt_and_missing_measurements_never_credit_cached_model_size(self):
        self.resident()
        self.rows = [self.receipt|{'agent_id':'other'}]
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.rows = [self.receipt]
        await self.source.refresh({'memory_available_mb':20000})
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'physical_capacity_unknown')

    async def test_permission_and_pause_are_not_inferred_from_availability(self):
        self.source.permission = None
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.source.permission = True
        await self.source.refresh(self.physical)
        self.source.pause()
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.source.resume()
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))

    async def test_registry_failure_withdraws_old_permission_without_compute_retry(self):
        await self.source.refresh(self.physical)
        self.client.get_resource = lambda resource: (_ for _ in ()).throw(TimeoutError('PRIVATE'))
        await self.source.refresh(self.physical)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'observation_unconfirmed')

    async def test_dispatch_guard_rechecks_residency_change_and_exact_receipt_expiry(self):
        await self.source.refresh(self.physical)
        self.resident()
        self.assertFalse(self.runtime.capacity.permitted('gpu'))  # old cold permission
        await self.source.refresh(self.physical)
        self.assertTrue(self.runtime.capacity.permitted('gpu'))
        self.actor.store.clock = lambda:4102444800  # 2100: receipt expired
        self.assertFalse(self.runtime.capacity.permitted('gpu'))

    async def test_late_observation_cannot_reinstate_earlier_headroom(self):
        entered,release = threading.Event(),threading.Event()
        def blocked(resource):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('synthetic metadata read')
            return self.resource
        self.client.get_resource = blocked
        first = asyncio.create_task(self.source.refresh(self.physical))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            latest = asyncio.create_task(self.source.refresh(self.physical|{'memory_available_mb':0}))
            await asyncio.sleep(0)
            self.assertFalse(self.runtime.capacity.permitted('gpu'))
        finally:
            release.set()
        await asyncio.gather(first,latest)
        self.assertFalse(self.runtime.capacity.permitted('gpu'))
        self.assertEqual(self.source.last['gpu']['reason'],'physical_capacity')

    async def test_observer_cancel_does_not_abandon_metadata_worker_or_lift_pause(self):
        entered,release = threading.Event(),threading.Event()
        def blocked(resource):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('synthetic metadata read')
            return self.resource
        self.client.get_resource = blocked
        observer = asyncio.create_task(self.source.refresh(self.physical))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            observer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await observer
            self.assertTrue(self.source.workers)
            self.source.pause()
        finally:
            release.set()
        await self.source.drain()
        self.assertFalse(self.runtime.capacity.permitted('gpu'))


if __name__ == '__main__':
    unittest.main()
