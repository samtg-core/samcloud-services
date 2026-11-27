"""Pre-provider refusal is a saved wait, never a new inference attempt on a timer."""
import asyncio
from pathlib import Path
import tempfile
import unittest

from .inference_arbiter import CapacityDeferred
from .inference_bootstrap import CapacityEvents
from .inference_supervisor import Supervisor


class CapacityWait(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'queue.sqlite'
        self.capacity = CapacityEvents(['slice', 'other'])
        self.calls = []
        self.mode = 'defer'
        self.cleaned = []

        async def execute(grant, body):
            self.calls.append((grant['resource'], body))
            if grant['resource'] == 'slice':
                if self.mode == 'partial':
                    row = self.s.journal.get(grant['id'], grant['owner'])
                    self.s.journal.append_event(row['active_execution'], 'partial')
                if self.mode == 'timeout':
                    raise TimeoutError('unknown provider outcome')
                if self.mode != 'ready':
                    raise CapacityDeferred('capacity_unavailable')
            return {'answer': 'saved result'}

        async def cleanup(grant, execution):
            self.cleaned.append(execution)

        self.s = Supervisor(self.path, execute, self.capacity.permitted, cleanup)
        self.capacity._pump = self.s.capacity_available
        self.execute, self.cleanup = execute, cleanup

    async def asyncTearDown(self):
        await self.settle()
        self.s.close()
        self.temp.cleanup()

    async def settle(self):
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.gather(*list(self.s.workers.values()))
        await asyncio.sleep(0)

    def enqueue(self, resource='slice', key='one'):
        grant = self.s.journal.submit('owner', key, resource, 'qwen')
        call = self.s.journal.enqueue(grant['id'], 'owner', key, {'input': 'same'})
        return grant, call

    async def blocked(self):
        grant, call = self.enqueue()
        self.capacity.observe('slice', 1, 'available')
        await self.settle()
        self.assertEqual(self.s.journal.execution(call['id'], 'owner')['state'], 'queued')
        return grant, call

    async def test_same_request_waits_for_fresh_event_without_cleanup_or_retry(self):
        grant, call = await self.blocked()
        saved = self.s.journal.execution(call['id'], 'owner')
        self.assertEqual(saved['wait_reason'], 'capacity_unavailable')
        self.assertEqual(saved['request'], call['request'])
        self.assertIsNone(saved['result'])
        self.assertEqual(self.cleaned, [])
        for _ in range(3):
            self.s.pump('slice')
        self.assertFalse(self.capacity.observe('slice', 1, 'available'))
        await self.settle()
        self.assertEqual(len(self.calls), 1)
        self.mode = 'ready'
        self.capacity.observe('slice', 2, 'available')
        await self.settle()
        saved = self.s.journal.execution(call['id'], 'owner')
        self.assertEqual(saved['state'], 'complete')
        self.assertIsNone(saved['wait_reason'])
        self.assertEqual(self.cleaned, [call['id']])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.s.journal.enqueue(grant['id'], 'owner', 'one',
                                             {'input': 'same'})['id'], call['id'])

    async def test_wait_survives_reopen_and_does_not_block_another_resource(self):
        _, call = await self.blocked()
        self.s.close()
        self.s = Supervisor(self.path, self.execute, self.capacity.permitted, self.cleanup)
        self.capacity._pump = self.s.capacity_available
        self.s.pump('slice')
        _, other = self.enqueue('other', 'other')
        self.capacity.observe('other', 1, 'available')
        await self.settle()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.s.journal.execution(call['id'], 'owner')['state'], 'queued')
        self.assertEqual(self.s.journal.execution(other['id'], 'owner')['state'], 'complete')

    async def test_partial_output_cannot_be_requeued(self):
        self.mode = 'partial'
        _, call = self.enqueue()
        self.capacity.observe('slice', 1, 'available')
        await self.settle()
        self.assertEqual(self.s.journal.execution(call['id'], 'owner')['state'], 'uncertain')
        self.capacity.observe('slice', 2, 'available')
        await self.settle()
        self.assertEqual(len(self.calls), 1)

    async def test_timeout_remains_unknown_not_capacity_wait(self):
        self.mode = 'timeout'
        _, call = self.enqueue()
        self.capacity.observe('slice', 1, 'available')
        await self.settle()
        row = self.s.journal.execution(call['id'], 'owner')
        self.assertEqual(row['state'], 'uncertain')
        self.assertIsNone(row['wait_reason'])

    async def test_explicit_cancellation_of_wait_is_not_resurrected(self):
        grant, call = await self.blocked()
        self.s.release(grant['id'], 'owner')
        self.mode = 'ready'
        self.capacity.observe('slice', 2, 'available')
        await self.settle()
        self.assertEqual(self.s.journal.execution(call['id'], 'owner')['state'], 'cancelled')
        self.assertEqual(len(self.calls), 1)

    async def test_release_after_active_call_still_releases_after_deferred_completion(self):
        grant, call = self.enqueue()
        self.capacity.observe('slice', 1, 'available')
        self.s.release(grant['id'], 'owner')
        await self.settle()
        self.assertEqual(self.s.journal.execution(call['id'], 'owner')['state'], 'queued')
        self.mode = 'ready'
        self.capacity.observe('slice', 2, 'available')
        await self.settle()
        self.assertEqual(self.s.journal.execution(call['id'], 'owner')['state'], 'complete')
        self.assertEqual(self.s.journal.get(grant['id'], 'owner')['state'], 'released')
