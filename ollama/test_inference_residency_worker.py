"""Owned rollover requests and queue coupling; no registry/provider writes."""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
import unittest

import httpx

from .inference_residency import ResidencyStore, Conflict
from .inference_residency_worker import ResidencyRollover
from .inference_supervisor import Supervisor


class Registry:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.calls = []
        self.mode = 'success'

    def rollover_lease(self, predecessor, *, key, ttl_seconds):
        self.calls.append((predecessor,key,ttl_seconds))
        self.entered.set()
        if not self.release.wait(5):
            raise AssertionError('test must release its synthetic registry')
        if self.mode == 'transport':
            raise httpx.ReadTimeout('synthetic lost reply')
        if self.mode == 'bad':
            return {'lease_id':'not-a-complete-receipt'}
        return {'lease_id':'lease-new','replaces_lease_id':predecessor,
                'resource_id':'gpu','memory_mb':17530,'max_total_s':3600,
                'expires_at':'2033-05-18T04:33:20+00:00',
                'status':'released' if self.mode == 'ended' else 'active'}


class OwnedResidencyRequest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResidencyStore(Path(self.temp.name)/'residency.sqlite',
            registry_owner='owner',service_id='model-service',clock=lambda:2_000_000_000)
        expiry = datetime.fromtimestamp(2_000_003_600,timezone.utc).isoformat()
        self.store.register('allocation',resource='gpu',model='qwen',receipt={
            'id':'lease-old','resource_id':'gpu','agent_id':'owner','service_id':'model-service',
            'memory_mb':17530,'exclusive':0,'status':'active','expires_at':expiry,'max_total_s':3600})
        self.client = Registry()
        self.events = []
        self.actor = ResidencyRollover(self.store,self.client,on_state=self.events.append)

    async def asyncTearDown(self):
        self.client.release.set()
        await self.actor.drain()
        self.actor.close()
        self.temp.cleanup()

    async def entered(self):
        async with asyncio.timeout(2):
            while not self.client.entered.is_set():
                await asyncio.sleep(.005)

    def begin(self):
        return self.actor.begin('allocation',ttl_seconds=3600)

    async def test_observer_cancellation_preserves_owned_request_and_receipt_application(self):
        self.client.release.clear()
        run = self.begin()
        self.assertFalse(self.actor.has_current_receipt('allocation'))
        await self.entered()
        observer = asyncio.create_task(self.actor.observe(run))
        await asyncio.sleep(0)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(run.task.done())
        with self.assertRaises(RuntimeError):
            self.actor.close()
        self.client.release.set()
        result = await self.actor.observe(run)
        self.assertEqual(result['outcome'],'covered')
        self.assertTrue(self.actor.has_current_receipt('allocation'))
        self.assertEqual([event['outcome'] for event in self.events],['handover_pending','covered'])

    async def test_same_running_intent_does_not_create_second_network_request(self):
        self.client.release.clear()
        run = self.begin()
        await self.entered()
        self.assertIs(self.begin(),run)
        with self.assertRaises(Conflict):
            self.actor.begin('allocation',ttl_seconds=1800)
        self.assertEqual(len(self.client.calls),1)
        self.client.release.set()
        await self.actor.observe(run)

    async def test_unknown_response_keeps_exact_persisted_key_for_retry(self):
        self.client.mode = 'transport'
        first = self.begin()
        self.assertEqual((await self.actor.observe(first))['outcome'],'unknown')
        self.assertFalse(self.actor.has_current_receipt('allocation'))
        self.client.mode = 'success'
        second = self.begin()
        self.assertEqual(second.operation,first.operation)
        self.assertEqual((await self.actor.observe(second))['outcome'],'covered')
        self.assertEqual(self.client.calls,[self.client.calls[0]]*2)

    async def test_malformed_receipt_stays_unknown_without_fresh_allocation_fallback(self):
        self.client.mode = 'bad'
        run = self.begin()
        self.assertEqual((await self.actor.observe(run))['outcome'],'unknown')
        self.assertEqual(self.store.snapshot('allocation')['pending_id'],run.operation.id)
        self.assertEqual(len(self.client.calls),1)

    async def test_terminal_replacement_does_not_authorize_dispatch(self):
        self.client.mode = 'ended'
        result = await self.actor.observe(self.begin())
        self.assertEqual(result['outcome'],'uncovered')
        self.assertFalse(self.actor.has_current_receipt('allocation'))
        with self.assertRaises(Conflict):
            self.begin()

    async def test_diagnostic_callback_failure_does_not_relabel_valid_receipt(self):
        def broken(event):
            raise RuntimeError('synthetic diagnostic failure')
        self.actor.on_state = broken
        with self.assertLogs('ollama.inference_residency_worker',level='WARNING'):
            result = await self.actor.observe(self.begin())
        self.assertEqual(result['outcome'],'covered')
        self.assertTrue(self.actor.has_current_receipt('allocation'))

    async def test_drain_observation_cancellation_does_not_cancel_owned_request(self):
        self.client.release.clear()
        run = self.begin()
        await self.entered()
        drain = asyncio.create_task(self.actor.drain())
        await asyncio.sleep(0)
        drain.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await drain
        self.assertFalse(run.task.done())
        self.client.release.set()
        await self.actor.drain()
        self.assertEqual(run.task.result()['outcome'],'covered')

    async def test_unknown_handover_blocks_next_dispatch_not_existing_compute(self):
        provider_entered, provider_release = asyncio.Event(), asyncio.Event()
        dispatched = []
        async def execute(grant, call):
            dispatched.append(grant['owner'])
            provider_entered.set()
            if len(dispatched) == 1:
                await provider_release.wait()
            return {'ok':True}
        scheduler = Supervisor(Path(self.temp.name)/'queue.sqlite',execute,
                               lambda resource:self.actor.has_current_receipt('allocation'))
        self.actor.on_state = lambda event:scheduler.pump('gpu')
        def enqueue(owner):
            grant = scheduler.journal.submit(owner,owner,'gpu','qwen','execution')
            return scheduler.journal.enqueue(grant['id'],owner,'call',{})
        try:
            first = enqueue('first')
            second = enqueue('peer')
            scheduler.pump('gpu')
            await asyncio.wait_for(provider_entered.wait(),2)
            self.client.release.clear()
            self.client.mode = 'transport'
            run = self.begin()
            await self.entered()
            first_worker = scheduler.workers[first['id']]
            self.assertFalse(first_worker.done())
            provider_release.set()
            await first_worker
            self.assertEqual(dispatched,['first'])
            self.client.release.set()
            self.assertEqual((await self.actor.observe(run))['outcome'],'unknown')
            self.assertEqual(scheduler.journal.execution(second['id'],'peer')['state'],'queued')
            self.client.mode = 'success'
            retry = self.begin()
            self.assertEqual(retry.operation,run.operation)
            await self.actor.observe(retry)
            await asyncio.gather(*list(scheduler.workers.values()))
            self.assertEqual(dispatched,['first','peer'])
        finally:
            provider_release.set()
            self.client.release.set()
            await self.actor.drain()
            await asyncio.gather(*list(scheduler.workers.values()))
            self.actor.on_state = None
            scheduler.close()


if __name__ == '__main__':
    unittest.main()
