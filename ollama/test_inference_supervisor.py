"""Blocked fake compute, no real provider, registry, credentials or network."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from .inference_arbiter import Conflict, NotFound
from .inference_supervisor import BackendEnded, Supervisor


class ExecutionLifetime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'jobs.sqlite'
        self.entered = []
        self.gate = asyncio.Event()
        self.allowed = True
        self.failure = None

        async def execute(grant, body):
            self.entered.append((grant['owner'], body))
            await self.gate.wait()
            if self.failure:
                raise self.failure
            return {'answer': body['prompt']}

        self.execute = execute
        self.s = Supervisor(self.path, execute, lambda _: self.allowed)

    async def asyncTearDown(self):
        self.gate.set()
        await asyncio.gather(*list(self.s.workers.values()))
        self.s.close()
        self.temp.cleanup()

    def queued(self, owner, key='one'):
        grant = self.s.journal.submit(owner, key, 'slice', 'qwen')
        call = self.s.journal.enqueue(grant['id'], owner, key, {'prompt': owner})
        return grant, call

    async def entered_count(self, count):
        async with asyncio.timeout(2):
            while len(self.entered) < count:
                await asyncio.sleep(0.001)

    async def test_waiting_observer_can_expire_without_losing_or_duplicating_work(self):
        first, a = self.queued('creative')
        second, b = self.queued('assistant')
        self.s.pump('slice')
        self.s.release(first['id'], 'creative')
        self.s.release(second['id'], 'assistant')  # explicit queued cancellation
        self.assertEqual(self.s.journal.execution(b['id'], 'assistant')['state'], 'cancelled')
        second, b = self.queued('assistant', 'retry-new-work')
        await self.entered_count(1)
        self.assertIsNone(self.s.pump('slice'))
        # A deliberately tiny observer budget expires while real work stays held.
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(self.s.observe(a['id'], 'creative'), 0.02)
        self.assertEqual(self.s.journal.get(first['id'], 'creative')['state'], 'granted')
        self.assertEqual(self.s.journal.execution(b['id'], 'assistant')['state'], 'queued')
        same = self.s.journal.enqueue(first['id'], 'creative', 'one', {'prompt': 'creative'})
        self.assertEqual(same['id'], a['id'])
        self.assertEqual(len(self.entered), 1)
        self.gate.set()
        await self.entered_count(2)
        saved = await self.s.observe(a['id'], 'creative')
        self.assertEqual(json.loads(saved['result']), {'answer': 'creative'})
        self.assertEqual([v[0] for v in self.entered], ['creative', 'assistant'])
        self.assertEqual((await self.s.observe(b['id'], 'assistant'))['state'], 'complete')

    async def test_work_session_retains_grant_between_calls_until_explicit_handoff(self):
        first, a = self.queued('creative')
        self.queued('assistant')
        self.s.pump('slice')
        self.gate.set()
        await self.s.observe(a['id'], 'creative')
        self.assertEqual(len(self.entered), 1)
        next_call = self.s.journal.enqueue(first['id'], 'creative', 'two', {'prompt': 'next'})
        self.s.pump('slice')
        await self.s.observe(next_call['id'], 'creative')
        self.assertEqual([v[0] for v in self.entered], ['creative', 'creative'])
        self.s.release(first['id'], 'creative')
        await self.entered_count(3)
        self.assertEqual(self.entered[-1][0], 'assistant')

    async def test_capacity_permission_unknown_keeps_durable_queue(self):
        grant, call = self.queued('creative')
        self.allowed = False
        self.assertIsNone(self.s.pump('slice'))
        self.assertEqual(self.s.journal.execution(call['id'], 'creative')['state'], 'queued')
        self.assertEqual(self.entered, [])
        self.allowed = True
        self.s.pump('slice')
        await self.entered_count(1)

    async def test_transport_failure_is_uncertain_and_pins_capacity(self):
        grant, call = self.queued('creative')
        self.queued('assistant')
        self.failure = TimeoutError('secret must not enter saved result')
        self.s.pump('slice')
        self.s.release(grant['id'], 'creative')
        self.gate.set()
        row = await self.s.observe(call['id'], 'creative')
        self.assertEqual(row['state'], 'uncertain')
        self.assertIsNone(row['result'])
        self.assertIsNone(self.s.pump('slice'))
        self.assertEqual(len(self.entered), 1)

    async def test_confirmed_backend_failure_can_drain_and_handoff(self):
        grant, call = self.queued('creative')
        self.queued('assistant')
        self.failure = BackendEnded({'error': 'backend terminated'})
        self.s.pump('slice')
        self.s.release(grant['id'], 'creative')
        self.gate.set()
        self.assertEqual((await self.s.observe(call['id'], 'creative'))['state'], 'failed')
        await self.entered_count(2)

    async def test_supervisor_cancellation_is_not_backend_cancellation_ack(self):
        grant, call = self.queued('creative')
        self.queued('assistant')
        self.s.pump('slice')
        await self.entered_count(1)
        self.s.workers[call['id']].cancel()
        row = await self.s.observe(call['id'], 'creative')
        self.assertEqual(row['state'], 'uncertain')
        self.s.release(grant['id'], 'creative')
        self.assertIsNone(self.s.pump('slice'))
        self.assertEqual(len(self.entered), 1)

    async def test_invalid_adapter_output_is_uncertain_not_forever_running(self):
        async def bad_result(grant, body):
            return object()
        self.s.execute = bad_result
        grant, call = self.queued('creative')
        self.s.pump('slice')
        row = await self.s.observe(call['id'], 'creative')
        self.assertEqual(row['state'], 'uncertain')
        self.assertIsNone(row['result'])

    async def test_cleanup_runs_after_saved_result_without_blocking_next_compute(self):
        cleanup_entered, cleanup_release = asyncio.Event(),asyncio.Event()
        async def cleanup(grant,execution_id):
            self.assertEqual(self.s.journal.execution(execution_id,grant['owner'])['state'],'complete')
            cleanup_entered.set()
            await cleanup_release.wait()
        self.s.on_finished = cleanup
        first,call = self.queued('creative')
        self.queued('assistant')
        self.s.pump('slice')
        self.s.release(first['id'],'creative')
        self.gate.set()
        try:
            await asyncio.wait_for(cleanup_entered.wait(),2)
            await self.entered_count(2)
            self.assertEqual((await self.s.observe(call['id'],'creative'))['state'],'complete')
        finally:
            cleanup_release.set()

    async def test_cleanup_failure_cannot_relabel_saved_completion(self):
        async def cleanup(grant,execution_id):
            raise RuntimeError('PRIVATE cleanup detail')
        self.s.on_finished = cleanup
        first,call = self.queued('creative')
        self.queued('assistant')
        self.s.pump('slice')
        self.s.release(first['id'],'creative')
        self.gate.set()
        await self.entered_count(2)
        row = await self.s.observe(call['id'],'creative')
        self.assertEqual(row['state'],'complete')
        self.assertNotIn('PRIVATE',str(row['result']))

    async def test_second_supervisor_cannot_steal_live_scheduler(self):
        grant, call = self.queued('creative')
        self.s.pump('slice')
        await self.entered_count(1)
        with self.assertRaises(BlockingIOError):
            Supervisor(self.path, self.execute, lambda _: True)
        self.assertEqual(self.s.journal.execution(call['id'], 'creative')['state'], 'running')
        with self.assertRaises(RuntimeError):
            self.s.close()

    async def test_restart_recovers_pending_payload_but_does_not_replay_unknown(self):
        grant, call = self.queued('creative')
        self.s.journal.dispatch_next('slice')  # committed intent; scheduler vanished
        second, pending = self.queued('assistant')
        self.s.close()
        self.s = Supervisor(self.path, self.execute, lambda _: True)
        self.assertEqual(self.s.journal.execution(call['id'], 'creative')['state'], 'uncertain')
        self.assertIsNone(self.s.pump('slice'))
        self.assertEqual(self.entered, [])
        self.assertEqual(json.loads(self.s.journal.execution(pending['id'], 'assistant')['request']),
                         {'prompt': 'assistant'})
        # Only after separate affirmative reconciliation of backend termination.
        self.s.release(grant['id'], 'creative')
        self.s.journal.finish(call['id'], {'error': 'confirmed stopped'}, failed=True)
        self.s.pump('slice')
        await self.entered_count(1)
        self.assertEqual(self.entered[0][0], 'assistant')

    async def test_pending_call_cannot_be_bypassed_or_finished_without_dispatch(self):
        grant, call = self.queued('creative')
        self.s.journal.admit_next('slice')
        with self.assertRaises(Conflict):
            self.s.journal.start(grant['id'], 'creative', 'bypass', {})
        with self.assertRaises(Conflict):
            self.s.journal.finish(call['id'], {})
        with self.assertRaises(NotFound):
            self.s.journal.execution(call['id'], 'assistant')


if __name__ == '__main__':
    unittest.main()
