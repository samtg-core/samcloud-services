"""Real HTTP consumer/API contract; fault injection, never live inference."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI, Request
import httpx

from .inference_api import build_router
from .inference_consumer import DurableConsumer, Job, ReconciliationRequired
from .inference_supervisor import Supervisor


class LoseResponses(httpx.AsyncBaseTransport):
    def __init__(self, app):
        self.inner = httpx.ASGITransport(app=app)
        self.drop = set()
        self.requests = []

    async def handle_async_request(self, request):
        self.requests.append((request.method, request.url.path))
        response = await self.inner.handle_async_request(request)
        if request.url.path in self.drop:
            self.drop.remove(request.url.path)
            await response.aclose()
            raise httpx.ReadError('synthetic lost response after commit', request=request)
        return response


class ConsumerContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gate = asyncio.Event()
        self.entered = []
        self.fail = False
        async def execute(grant, call):
            self.entered.append(grant['owner'])
            self.s.journal.append_event(grant['active_execution'], 'early output')
            await self.gate.wait()
            if self.fail:
                raise ConnectionError('unknown backend state')
            self.s.journal.append_event(grant['active_execution'], 'finished', terminal=True)
            return {'answer': call['prompt']}
        self.s = Supervisor(Path(self.temp.name) / 'jobs.sqlite', execute, lambda _: True)
        app = FastAPI()
        # Service auth middleware is exercised separately in test_inference_api.
        # Here synthetic identity injection keeps fault-injection tests isolated.
        @app.middleware('http')
        async def identity(request: Request, next_handler):
            request.state.caller = {'username': request.headers.get('x-test-owner', 'creative')}
            return await next_handler(request)
        app.include_router(build_router(self.s, lambda m: ('gpu', m), lambda m, b: b))
        self.transport = LoseResponses(app)
        self.http = httpx.AsyncClient(transport=self.transport, base_url='http://test')
        self.states = []
        self.consumer = DurableConsumer(self.http, poll_interval=.005, progress=self.states.append)

    async def asyncTearDown(self):
        self.gate.set()
        await asyncio.gather(*list(self.s.workers.values()))
        await self.http.aclose()
        self.s.close()
        self.temp.cleanup()

    async def submit(self, consumer=None, key='turn'):
        return await (consumer or self.consumer).submit('qwen', grant_key=key,
                     call_key='inference-1', request={'prompt': key})

    async def wait_for(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    async def test_lost_grant_and_execution_responses_find_same_committed_work(self):
        self.transport.drop.add('/v1/inference/grants')
        # Obtain the persisted id, then lose the execution response after commit.
        grant = self.s.journal.submit('creative', 'turn', 'gpu', 'qwen')
        self.transport.drop.add('/v1/inference/grants/' + grant['id'] + '/executions')
        job = await self.submit()
        self.assertEqual(self.entered, ['creative'])
        self.assertEqual(self.states.count('transport_unavailable'), 2)
        self.assertEqual(self.s.journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0], 1)
        self.gate.set()
        self.assertEqual(await self.consumer.result(job), {'answer': 'turn'})

    async def test_cancel_observer_then_new_client_resumes_same_job(self):
        job = await self.submit()
        reader = asyncio.create_task(self.consumer.result(job))
        await self.wait_for(lambda: 'running' in self.states)
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader
        # Caller persisted handle or logical keys; both recover the same work.
        restored = Job(**json.loads(json.dumps(job.as_dict())))
        new = DurableConsumer(self.http, poll_interval=.005)
        self.assertEqual(await self.submit(new), restored)
        self.assertEqual(self.entered, ['creative'])
        self.assertEqual(self.s.journal.execution(job.execution_id, 'creative')['state'], 'running')
        self.gate.set()
        self.assertEqual(await new.result(restored), {'answer': 'turn'})

    async def test_transport_recovery_is_visible_without_execution_state_change(self):
        job = await self.submit()
        reader = asyncio.create_task(self.consumer.result(job))
        try:
            await self.wait_for(lambda: 'running' in self.states)
            self.transport.drop.add('/v1/inference/executions/' + job.execution_id)
            await self.wait_for(lambda: 'transport_available' in self.states)
            self.assertEqual(self.states, ['running', 'transport_unavailable', 'transport_available'])
            self.assertFalse(reader.done())
            self.assertEqual(self.s.journal.execution(job.execution_id, 'creative')['state'], 'running')
            self.assertEqual(self.entered, ['creative'])
            self.gate.set()
            self.assertEqual(await asyncio.wait_for(reader, 2), {'answer': 'turn'})
            self.assertEqual(self.states[-1], 'complete')
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def test_busy_wait_does_not_fail_or_dispatch_until_explicit_handoff(self):
        first = await self.submit()
        async with httpx.AsyncClient(transport=self.transport, base_url='http://test',
                                    headers={'x-test-owner': 'assistant'}) as http:
            states = []
            second_client = DurableConsumer(http, poll_interval=.005, progress=states.append)
            second = await self.submit(second_client, key='other')
            reader = asyncio.create_task(second_client.result(second))
            await self.wait_for(lambda: 'queued' in states)
            await asyncio.sleep(.05)  # many observations; no queue-age failure
            self.assertFalse(reader.done())
            self.assertEqual(self.entered, ['creative'])
            self.gate.set()
            await self.consumer.result(first)
            self.assertEqual(self.entered, ['creative'])  # work-session grant remains held
            await self.consumer.release(first)
            self.assertEqual(await asyncio.wait_for(reader, 2), {'answer': 'other'})
            self.assertEqual(self.entered, ['creative', 'assistant'])

    async def test_cursor_reconnect_and_lost_event_read_never_repeat_inference(self):
        job = await self.submit()
        await self.wait_for(lambda: len(self.entered) == 1)
        reader = self.consumer.events(job)
        first = await anext(reader)
        self.assertEqual(first['events'][0]['data'], 'early output')
        cursor = first['next_cursor']
        await reader.aclose()
        self.transport.drop.add('/v1/inference/executions/' + job.execution_id + '/events')
        self.gate.set()
        new = DurableConsumer(self.http, poll_interval=.005)
        pages = [p async for p in new.events(job, after=cursor)]
        self.assertEqual([e['data'] for p in pages for e in p['events']], ['finished'])
        self.assertEqual(self.entered, ['creative'])

    async def test_uncertain_is_exposed_not_replaced_by_another_job(self):
        self.fail = True
        job = await self.submit()
        self.gate.set()
        result = asyncio.create_task(self.consumer.result(job))
        async with asyncio.timeout(2):
            while 'uncertain' not in self.states:
                await asyncio.sleep(.001)
        self.assertFalse(result.done())
        reader = self.consumer.events(job)
        page = await anext(reader)
        self.assertEqual(page['state'], 'uncertain')
        await reader.aclose()
        self.assertEqual(self.entered, ['creative'])
        self.assertIsNone(self.s.pump('gpu'))
        # Only explicit supervisor reconciliation can end this saved job.
        self.s.journal.finish(job.execution_id, {'answer': 'reconciled'})
        self.assertEqual(await asyncio.wait_for(result, 2), {'answer': 'reconciled'})
        self.assertEqual(self.entered, ['creative'])

    async def test_conflicting_key_fails_instead_of_retrying_or_mutating(self):
        await self.submit()
        with self.assertRaises(httpx.HTTPStatusError) as error:
            await self.consumer.submit('qwen', grant_key='turn', call_key='inference-1',
                                       request={'prompt': 'different'})
        self.assertEqual(error.exception.response.status_code, 409)
        self.assertNotIn('transport_unavailable', self.states)


if __name__ == '__main__':
    unittest.main()
