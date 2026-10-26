"""Owned drain/unload/settlement against a fake ledger; no live inference."""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .inference_bootstrap import CapacityEvents
from .inference_reservation import InitialReservations
from .inference_arbiter import Arbiter
from .inference_shared_resource import ResourceHandoff, cold_vision_absent


class SharedResourceHandoff(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.events = []
        self.row = {'id':'owned','resource_id':'gpu','agent_id':'owner','service_id':'model-service',
            'memory_mb':100,'exclusive':0,'status':'active','expires_at':'2099-01-01T00:00:00Z'}
        self.foreign = {'id':'unity','resource_id':'gpu','agent_id':'unity-owner',
                        'service_id':'unity-service','status':'queued','memory_mb':200,'exclusive':0}
        self.client = SimpleNamespace(request_lease=Mock(return_value={'status_code':200,
            'lease_id':'owned','status':'active','expires_at':'2099-01-01T00:00:00Z'}),
            get_resource=lambda _:dict(id='gpu',expiry_policy='hold',leases=[self.row,self.foreign]),
            list_leases=lambda **_: [dict(self.row)],release_lease=Mock(side_effect=self.release))
        self.initial = InitialReservations(self.root/'initial.sqlite', self.client,
            registry_owner='owner', service_id='model-service', models={'qwen':'gpu'})
        self.initial.reserve('qwen', resource='gpu',service_id='model-service',memory_mb=100,ttl_seconds=3600)
        self.models = {'qwen':SimpleNamespace(lease_id='owned')}
        self.running = [{'name':'qwen'}]
        self.unload_error = False
        self.manager = SimpleNamespace(sc=self.client,models=self.models,
            initial_reservations=self.initial,ollama=SimpleNamespace(list_running=lambda:self.running),
            unload=Mock(side_effect=self.unload),residency_renewal=SimpleNamespace(resume=Mock()))
        self.journal = Arbiter(self.root/'queue.sqlite')
        self.db = self.journal.db
        self.drain = asyncio.Event()
        self.drain.set()
        self.uncertain = False
        async def quiesce(**kwargs):
            self.events.append('draining')
            await self.drain.wait()
            return {'drained':not self.uncertain}
        self.capacity = CapacityEvents({'gpu'})
        self.capacity.observe('gpu',1,'available')
        self.runtime = SimpleNamespace(manager=self.manager,
            settings=SimpleNamespace(models={'qwen':'gpu'}),capacity=self.capacity,capacity_source=None,
            queue=SimpleNamespace(supervisor=SimpleNamespace(journal=self.journal)),
            residency_actor=None,quiesce=quiesce,resume=Mock(side_effect=self.capacity.resume))
        self.actor = ResourceHandoff(self.runtime)

    async def asyncTearDown(self):
        self.drain.set()
        if self.actor.task is not None:
            await self.actor.task
        self.journal.close()
        self.tmp.cleanup()

    def unload(self, model, *, require_confirmed):
        self.assertTrue(require_confirmed)
        self.events.append('unload')
        self.models.pop(model)
        if self.unload_error:
            raise RuntimeError('lost unload reply; provider state unknown')
        self.running = []

    def release(self, lease_id):
        self.assertEqual(lease_id, 'owned')
        self.assertEqual(self.running, [])
        self.events.append('release')
        if self.row['status'] == 'expired':
            self.row['settled_at'] = '2099-01-01T00:00:00Z'
            return {'status':'settled'}
        self.row['status'] = 'released'
        return {'status':'released'}

    async def test_current_work_drains_before_unload_and_no_direct_dispatch_on_resume(self):
        self.drain.clear()
        task = self.actor.begin()
        self.assertFalse(self.capacity.permitted('gpu'))
        await asyncio.sleep(0)
        self.assertEqual(self.events, ['draining'])
        self.manager.unload.assert_not_called()
        self.drain.set()
        result = await task
        self.assertEqual(result['phase'], 'idle')
        self.assertEqual(self.events, ['draining','unload','release'])
        self.assertFalse(self.capacity.permitted('gpu'))  # a new snapshot is required
        self.client.request_lease.assert_called_once()   # never lease on affinity/startup

    async def test_cold_vision_recovery_preserves_queued_work_and_checks_absence(self):
        from .manager import VLM_MODELS
        vision = next(iter(VLM_MODELS.values()))['default']
        grant = self.journal.submit('owner', 'waiting', 'gpu', 'qwen')
        execution = self.journal.enqueue(grant['id'], 'owner', 'input', {'messages':[]})
        self.models.clear()
        self.running = []
        self.row['status'] = 'released'
        self.actor._save('uncertain', allocations=[
            {'model':vision, 'lease_id':'owned', 'memory_mb':100}])
        # A restart observes the persisted state; it does not clear it.
        self.actor = ResourceHandoff(self.runtime)
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=False):
            self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        with patch('ollama.inference_shared_resource.cold_vision_absent', return_value=True):
            self.assertEqual((await self.actor.begin())['phase'], 'idle')
        self.assertEqual(self.journal.execution(execution['id'], 'owner')['state'], 'queued')
        self.client.release_lease.assert_not_called()
        self.manager.unload.assert_not_called()

    async def test_live_manager_vision_never_uses_cold_absence_override(self):
        from .manager import VLM_MODELS
        vision = next(iter(VLM_MODELS.values()))['default']
        self.models[vision] = SimpleNamespace(lease_id='vision')
        with patch('ollama.inference_shared_resource.cold_vision_absent') as observe:
            self.assertFalse(self.actor._absent(vision))
            observe.assert_not_called()

    async def test_vision_handoff_uses_real_manager_stop_and_exact_staged_claim(self):
        from .manager import ModelManager, ManagedModel, Backend, VLM_MODELS, VLM_HOST, VLM_PORT
        vision=next(iter(VLM_MODELS.values()))['default']
        self.client.request_lease.reset_mock()  # exclude the fixture's unrelated qwen setup
        initial=InitialReservations(self.root/'vision.sqlite',self.client,
            registry_owner='owner',service_id='model-service',models={vision:'gpu'})
        initial.stage(vision,resource='gpu',service_id='model-service',memory_mb=100,ttl_seconds=3600)
        self.client.list_leases=lambda **_: [dict(self.row)]
        initial.reserve(vision,resource='gpu',service_id='model-service',memory_mb=100,ttl_seconds=3600)
        child=Mock(pid=1234)
        child.args=['python','-m','mlx_vlm.server','--model',vision,
                    '--port',str(VLM_PORT),'--host',VLM_HOST]
        child.poll.return_value=None
        child.terminate.side_effect=lambda:setattr(child.poll,'return_value',0)
        manager=object.__new__(ModelManager)
        manager.durable_inference=True
        manager.initial_reservations=initial
        manager.sc=self.client
        manager.models={vision:ManagedModel(name=vision,backend=Backend.VLM,
            memory_mb=100,lease_id='owned',port=VLM_PORT,managed=True,vlm_process=child,
            loaded_at=0,last_used=0)}
        manager.residency_renewal=SimpleNamespace(resume=Mock())
        self.running=[]
        manager._release_lease_quietly=self.client.release_lease
        self.runtime.manager=manager
        self.runtime.settings=SimpleNamespace(models={vision:'gpu'})
        self.actor=ResourceHandoff(self.runtime)
        with patch('ollama.inference_shared_resource.cold_vision_absent',return_value=True):
            result=await self.actor.begin()
        self.assertEqual(result['phase'],'idle')
        self.assertEqual(manager.models,{})
        self.assertEqual(initial.snapshot(vision)['state'],'retired')
        self.client.request_lease.assert_called_once()
        self.client.release_lease.assert_called_once_with('owned')
        child.wait.assert_called_once_with(timeout=10)

    async def test_cold_vision_inventory_failure_or_live_process_refuses_absence(self):
        with patch('ollama.inference_shared_resource.subprocess.run',
                   return_value=SimpleNamespace(returncode=1, stdout='')):
            self.assertFalse(cold_vision_absent())
        with patch('ollama.inference_shared_resource.subprocess.run',
                   return_value=SimpleNamespace(returncode=0,
                       stdout='/python -m mlx_vlm.server --port 8801\n')):
            self.assertFalse(cold_vision_absent())
        with patch('ollama.inference_shared_resource.subprocess.run',
                   return_value=SimpleNamespace(returncode=0,
                       stdout="/sbin/launchd\n/unrelated --title=Sam's\n")), \
             patch('ollama.inference_bootstrap.vision_backend_absent', return_value=True):
            self.assertTrue(cold_vision_absent())

    async def test_uncertain_native_work_keeps_residency_and_allocation(self):
        self.uncertain = True
        result = await self.actor.begin()
        self.assertEqual(result['phase'], 'uncertain')
        self.manager.unload.assert_not_called()
        self.client.release_lease.assert_not_called()
        self.assertFalse(self.capacity.permitted('gpu'))

    async def test_canceled_foreign_wait_does_not_unload(self):
        self.foreign['status'] = 'cancelled'
        self.assertEqual((await self.actor.begin())['phase'], 'idle')
        self.manager.unload.assert_not_called()
        self.client.release_lease.assert_not_called()

    async def test_observer_cancellation_cannot_cancel_owned_handoff(self):
        self.drain.clear()
        task = self.actor.begin()
        async def observe():
            return await asyncio.shield(task)
        waiter = asyncio.create_task(observe())
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(task.done())
        self.client.release_lease.assert_not_called()
        self.drain.set()
        self.assertEqual((await task)['phase'], 'idle')

    async def test_lost_unload_reply_persists_ids_after_manager_inventory_disappears(self):
        self.unload_error = True
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.assertEqual(self.models, {})
        self.assertEqual(self.actor.snapshot()['allocations'][0]['lease_id'], 'owned')
        self.client.release_lease.assert_not_called()
        self.actor = ResourceHandoff(self.runtime)  # same journal, no recreated job
        self.assertFalse(self.capacity.permitted('gpu'))
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.client.release_lease.assert_not_called()  # backend still reports the model
        self.running = []                          # affirmative later provider observation
        self.assertEqual((await self.actor.begin())['phase'], 'idle')
        self.manager.unload.assert_called_once()
        self.client.release_lease.assert_called_once_with('owned')

    async def test_expired_orphan_settles_only_after_absence_and_lost_reply_is_reconciled(self):
        self.models.clear()
        self.running = []
        self.row.update(status='expired',expiry_reason='ttl_expired',settled_at=None)
        def lost(lease_id):
            self.release(lease_id)
            raise TimeoutError('lost settlement acknowledgment')
        self.client.release_lease.side_effect = lost
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.assertEqual(self.actor.snapshot()['allocations'][0]['lease_id'], 'owned')
        self.actor = ResourceHandoff(self.runtime)
        self.assertEqual((await self.actor.begin())['phase'], 'idle')
        self.manager.unload.assert_not_called()
        self.client.release_lease.assert_called_once_with('owned')
        self.client.request_lease.assert_called_once()

    async def test_foreign_ownership_or_backend_visibility_failure_keeps_unknown(self):
        self.row['agent_id'] = 'somebody-else'
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.manager.unload.assert_not_called()
        self.client.release_lease.assert_not_called()

    async def test_duplicate_hints_share_one_owned_drain(self):
        self.drain.clear()
        first = self.actor.begin()
        self.assertIs(self.actor.begin(), first)
        self.drain.set()
        self.assertEqual((await first)['phase'], 'idle')
        self.manager.unload.assert_called_once()
        self.client.release_lease.assert_called_once()

    async def test_handoff_transitions_are_committed_replayable_metadata(self):
        await self.actor.begin()
        rows=self.db.execute("SELECT state,reason FROM metadata_events WHERE kind='service_handoff' ORDER BY seq").fetchall()
        self.assertEqual([row['state'] for row in rows],['draining','unloading','idle'])
        self.assertEqual(rows[1]['reason'],'unity')
        self.actor = ResourceHandoff(self.runtime)
        self.assertEqual(self.actor.snapshot()['phase'],'idle')


if __name__ == '__main__':
    unittest.main()
