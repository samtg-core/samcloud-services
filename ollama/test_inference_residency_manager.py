"""Real manager claim/link semantics; synthetic registry, no fleet operations."""
import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .inference_residency import ResidencyStore
from .inference_residency_worker import ResidencyRollover
from .inference_residency_manager import ManagerResidency, Conflict
from .manager import ModelManager


class Registry:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.calls = []
        self.lost = False
        self.receipt = {'id':'old','resource_id':'gpu','agent_id':'owner','service_id':'svc',
            'memory_mb':100,'exclusive':0,'status':'active','max_total_s':3600,
            'expires_at':'2099-01-01T00:00:00+00:00'}

    def list_leases(self, *, resource_id, service_id, status):
        assert resource_id == 'gpu' and service_id == 'svc' and status == 'active'
        return [self.receipt.copy()]

    def rollover_lease(self, predecessor, *, key, ttl_seconds):
        self.calls.append((predecessor,key,ttl_seconds))
        self.entered.set()
        if not self.release.wait(5):
            raise AssertionError('synthetic registry was not released')
        if self.lost:
            raise TimeoutError('synthetic lost reply')
        return {'lease_id':'new','replaces_lease_id':predecessor,'resource_id':'gpu',
                'memory_mb':100,'max_total_s':3600,'status':'active',
                'expires_at':'2099-01-01T00:00:00+00:00'}


class Binding(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = Registry()
        self.manager = object.__new__(ModelManager)
        self.manager.sc = self.client
        self.mm = SimpleNamespace(name='qwen',lease_id='old',memory_mb=100,
                                  in_flight=0,last_used=0,request_count=0)
        self.manager.models = {'qwen':self.mm}
        store = ResidencyStore(Path(self.tmp.name)/'residency.sqlite',
                               registry_owner='owner',service_id='svc')
        self.actor = ResidencyRollover(store,self.client)
        self.binding = ManagerResidency(self.manager,self.actor,{'qwen':'gpu'},ttl_seconds=3600)

    async def asyncTearDown(self):
        self.client.release.set()
        await self.binding.drain()
        self.actor.close()
        self.tmp.cleanup()

    async def entered(self):
        async with asyncio.timeout(2):
            while not self.client.entered.is_set():
                await asyncio.sleep(.005)

    async def test_exact_receipt_updates_real_manager_pointer_without_old_swap(self):
        self.assertFalse(self.binding.ready('qwen'))
        result = await self.binding.renew_models()
        await self.binding.drain()
        self.assertEqual(result[0]['outcome'],'covered')
        self.assertEqual(self.mm.lease_id,'new')
        self.assertEqual(self.mm.in_flight,0)
        self.assertTrue(self.binding.ready('qwen'))
        self.assertGreater(self.mm.last_used,0)

    async def test_cancelled_renewal_observer_does_not_unclaim_actual_worker(self):
        self.client.release.clear()
        observer = asyncio.create_task(self.binding.renew_models())
        await self.entered()
        self.assertEqual(self.mm.in_flight,1)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertEqual(self.mm.in_flight,1)
        self.assertFalse(self.binding.ready('qwen'))
        self.client.release.set()
        await self.binding.drain()
        self.assertEqual(self.mm.in_flight,0)
        self.assertEqual(self.mm.lease_id,'new')

    async def test_cancelled_parent_task_still_holds_until_child_http_ends(self):
        self.client.release.clear()
        task = self.binding.begin('qwen')
        await self.entered()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.mm.in_flight,1)
        with self.assertRaises(Conflict):
            self.binding.begin('qwen')
        self.client.release.set()
        await self.binding.drain()
        self.assertEqual(self.mm.in_flight,0)
        self.assertFalse(self.binding.ready('qwen'))  # pointer application remains unresolved

    async def test_lost_reply_retries_same_persisted_operation_not_new_lease(self):
        self.client.lost = True
        await self.binding.renew_models()
        self.assertEqual(self.mm.lease_id,'old')
        self.assertFalse(self.binding.ready('qwen'))
        self.client.lost = False
        await self.binding.renew_models()
        await self.binding.drain()
        self.assertEqual(len(self.client.calls),2)
        self.assertEqual(self.client.calls[0],self.client.calls[1])
        self.assertEqual(self.mm.lease_id,'new')

    async def test_wrong_holder_or_exact_id_never_rolls_over_another_allocation(self):
        for field,value in (('agent_id','other'),('id','same-size-other-lease')):
            before = self.client.receipt.copy()
            self.client.receipt[field] = value
            with self.assertRaises(Conflict):
                await self.binding.renew_models()
            await self.binding.drain()
            self.assertEqual(self.client.calls,[])
            self.assertEqual(self.mm.in_flight,0)
            self.client.receipt = before

    async def test_manual_object_replacement_cannot_inherit_old_pointer_or_claim(self):
        self.client.release.clear()
        task = self.binding.begin('qwen')
        await self.entered()
        other = SimpleNamespace(name='qwen',lease_id='other',memory_mb=100,in_flight=0)
        self.manager.models['qwen'] = other
        self.client.release.set()
        with self.assertRaises(Conflict):
            await task
        await self.binding.drain()
        self.assertEqual(other.lease_id,'other')
        self.assertEqual(other.in_flight,0)
        self.assertEqual(self.mm.in_flight,0)
        self.assertFalse(self.binding.ready('qwen'))

    async def test_current_receipt_is_not_working_set_permission(self):
        await self.binding.renew_models()
        await self.binding.drain()
        self.mm.memory_mb = 101
        self.assertFalse(self.binding.ready('qwen'))

    async def test_pause_blocks_new_maintenance_but_drains_existing_work(self):
        self.client.release.clear()
        task = self.binding.begin('qwen')
        await self.entered()
        self.binding.pause()
        with self.assertRaises(Conflict):
            self.binding.begin('qwen')
        self.assertFalse(self.binding.ready('qwen'))
        self.assertFalse(task.done())
        self.client.release.set()
        await self.binding.drain()
        self.assertEqual(self.mm.lease_id,'new')
        self.assertFalse(self.binding.ready('qwen'))
        self.binding.resume()
        self.assertFalse(self.binding.ready('qwen'))

    async def test_ready_event_follows_manager_pointer_not_just_registry_commit(self):
        events = []
        self.binding.on_change = lambda model: events.append((self.mm.lease_id,self.binding.ready(model)))
        await self.binding.renew_models()
        await self.binding.drain()
        self.assertEqual(events[0],('old',False))
        self.assertEqual(events[-1],('new',True))

    async def test_callback_failure_does_not_undo_a_confirmed_handover(self):
        def broken(model):
            raise RuntimeError('PRIVATE observer error')
        self.binding.on_change = broken
        with self.assertLogs('ollama.inference_residency_manager',level='WARNING') as logs:
            await self.binding.renew_models()
        await self.binding.drain()
        self.assertTrue(self.binding.ready('qwen'))
        self.assertEqual(self.mm.lease_id,'new')
        self.assertNotIn('PRIVATE',str(logs.output))

    async def test_durable_manager_refuses_legacy_swap(self):
        self.manager.durable_inference = True
        with self.assertRaises(RuntimeError), patch.object(self.client,'list_leases') as read:
            self.manager._renew_leases()
        read.assert_not_called()

    async def test_actual_manager_maintenance_uses_bound_controller_only(self):
        self.manager.durable_inference = True
        self.manager.residency_renewal = self.binding
        with patch.object(self.manager,'_renew_leases') as legacy:
            await self.manager.renew_model_leases()
        await self.binding.drain()
        legacy.assert_not_called()
        self.assertTrue(self.binding.ready('qwen'))

    async def test_runtime_quiesce_pauses_and_drains_actual_lease_worker(self):
        from .inference_bootstrap import QueueRuntime, QueueSettings
        from .test_inference_bootstrap import service
        host = service()
        host.mgr = self.manager
        host.config.SC_SERVICE_ID = 'svc'
        # Install before residency exists, then model the post-admission bind.
        self.manager.models = {}
        runtime = QueueRuntime.install(host,QueueSettings(
            Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu'},reservation_mb={'qwen':100}),registry_owner='owner')
        self.manager.models = {'qwen':self.mm}
        self.manager.residency_renewal = self.binding
        self.client.release.clear()
        self.binding.begin('qwen')
        await self.entered()
        observer = asyncio.create_task(runtime.quiesce())
        try:
            await asyncio.sleep(.01)
            self.assertTrue(self.binding.paused)
            self.assertFalse(observer.done())
            self.assertEqual(self.mm.in_flight,1)
            self.client.release.set()
            result = await asyncio.wait_for(observer,2)
            self.assertTrue(result['drained'])
            self.assertEqual(result['manager_claims'],0)
            self.assertEqual(self.mm.lease_id,'new')
        finally:
            self.client.release.set()
            await self.binding.drain()
            if not observer.done():
                await observer
            runtime.queue.supervisor.close()

    async def test_unknown_compute_does_not_pause_residency_renewal(self):
        from .inference_bootstrap import QueueRuntime, QueueSettings
        from .test_inference_bootstrap import service
        host = service()
        host.mgr = self.manager
        host.config.SC_SERVICE_ID = 'svc'
        self.manager.models = {}
        runtime = QueueRuntime.install(host,QueueSettings(
            Path(self.tmp.name)/'queue.sqlite',{'qwen':'gpu'},reservation_mb={'qwen':100}),registry_owner='owner')
        self.manager.models = {'qwen':self.mm}
        self.manager.residency_renewal = self.binding
        try:
            runtime.queue.held_claims['unreconciled-provider-call'] = 'qwen'
            result = await runtime.quiesce()
            self.assertFalse(result['drained'])
            self.assertFalse(self.binding.paused)
            await self.manager.renew_model_leases()  # reservation can still be continued
            await self.binding.drain()
            self.assertEqual(self.mm.lease_id,'new')
        finally:
            runtime.queue.held_claims.clear()
            runtime.queue.supervisor.close()

    async def test_unbound_enabled_manager_fails_without_legacy_fallback(self):
        self.manager.durable_inference = True
        with patch.object(self.manager,'_renew_leases') as legacy:
            with self.assertRaises(RuntimeError):
                await self.manager.renew_model_leases()
        legacy.assert_not_called()

    async def test_legacy_maintenance_contract_is_unchanged_when_disabled(self):
        with patch.object(self.manager,'_renew_leases',return_value='legacy-result') as legacy:
            self.assertEqual(await self.manager.renew_model_leases(),'legacy-result')
        legacy.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
