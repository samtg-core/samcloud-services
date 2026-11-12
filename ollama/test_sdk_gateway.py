"""SDK -> durable HTTP API -> real gateway handlers -> blocked fake provider.

This joins separately tested pieces. Authentication is synthetic here; verified
identity enforcement has its own API tests. No live service or model calls.
"""
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Request
import httpx
from openai import BadRequestError, ConflictError

from . import test_gateway_queue as fixture
from .inference_supervisor import BackendEnded
from .inference_sdk import QueuedClient
from .inference_identity import CallKeys, IdentityStore
from .inference_arbiter import StintPolicy
from .ollama_client import OllamaClient
from .test_inference_sdk import arguments


class SDKGateway(unittest.IsolatedAsyncioTestCase):
    wait_for = fixture.GatewayLifetime.wait_for

    async def asyncSetUp(self):
        await fixture.GatewayLifetime.asyncSetUp(self)
        self.accepted_submissions = []
        self.clients = []
        self.install_app()

    def install_app(self):
        self.app = FastAPI()
        @self.app.middleware('http')
        async def identity(request: Request, next_handler):
            request.state.caller = {'username': request.headers['x-test-owner']}
            response = await next_handler(request)
            if request.method == 'POST' and request.url.path.endswith('/executions') and response.status_code == 202:
                self.accepted_submissions.append(request.state.caller['username'])
            return response
        self.app.include_router(self.queue.router())

    def reopen_service(self, *, stint_policy=None):
        self.queue.supervisor.close()
        self.queue = fixture.GatewayQueue(Path(self.temp.name) / 'jobs.sqlite', fixture.server,
                                           {'qwen': 'gpu'}, lambda _: True,
                                           stint_policy=stint_policy)
        self.dispatch.stop()
        self.dispatch = patch.object(fixture.server, 'inference_dispatcher', self.queue)
        self.dispatch.start()
        self.install_app()

    async def asyncTearDown(self):
        await fixture.GatewayLifetime.asyncTearDown(self)
        for client in self.clients:
            await client.aclose()

    def http(self, owner):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                base_url='http://test', headers={'x-test-owner': owner})

    def client(self, owner, **policy):
        client = QueuedClient(api_key='synthetic', base_url='http://test/v1',
                              http_factory=lambda: self.http(owner), poll_interval=.005, **policy)
        self.clients.append(client)
        return client

    def args(self, owner, **kwargs):
        return arguments(messages=[{'role': 'user', 'content': owner}], **kwargs)

    async def test_stint_affinity_then_idle_handoff_through_sdk_http_and_gateway(self):
        self.reopen_service(stint_policy=StintPolicy(1, 1800))
        self.backend.gate.set()
        original = self.backend.answer
        self.backend.answer = lambda: {**original(), 'message': {'content': 'done', 'tool_calls': [
            {'id': 'tool', 'type': 'function', 'function': {'name': 'terminal', 'arguments': {}}}]}}
        creative = self.client('creative', grant_lifetime='work_session')
        assistant = self.client('assistant', grant_lifetime='work_session')
        first = await creative.create(**self.args('creative', key='first'))
        self.assertEqual(first.choices[0].message.content, 'done')
        waiting = asyncio.create_task(assistant.create(**self.args('assistant')))
        try:
            await self.wait_for(lambda: len(self.accepted_submissions) == 2)
            second = await creative.create(**self.args('creative', key='second'))
            self.assertEqual(second.choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'creative'])
            # No new submitting HTTP request is needed to trigger the handoff.
            self.assertEqual((await asyncio.wait_for(waiting, 3)).choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'creative', 'assistant'])
            replay = await creative.create(**self.args('creative', key='first'))
            self.assertEqual(replay.choices[0].message.content, 'done')
            self.assertEqual(len(self.backend.entered), 3)
            self.assertEqual(self.model.in_flight, 0)
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

    async def test_dependency_handoff_without_elapsed_time_through_sdk_and_gateway(self):
        self.reopen_service(stint_policy=StintPolicy())
        self.backend.gate.set()
        original = self.backend.answer
        self.backend.answer = lambda: ({**original(), 'message': {'content': 'done', 'tool_calls': [
            {'id': 'tool', 'type': 'function', 'function': {'name': 'terminal', 'arguments': {}}}]}}
            if self.backend.entered[-1] == 'main' else original())
        main = self.client('creative', require_stint_policy=True)
        await main.create(**self.args('main', key='first'))
        parent = self.queue.supervisor.journal.db.execute(
            "SELECT state FROM grants WHERE client_key='work-session'").fetchone()
        self.assertEqual(parent['state'], 'granted')
        auxiliary = self.client('creative', grant_lifetime='execution',
            key_provider=lambda _: CallKeys('summary', 'summary-call', ('work-session',)))
        args = self.args('summary')
        args.pop('extra_headers')
        answer = await asyncio.wait_for(auxiliary.create(**args), 2)
        self.assertEqual(answer.choices[0].message.content, 'done')
        self.assertEqual(self.backend.entered, ['main', 'summary'])
        await main.create(**self.args('main', key='second'))
        self.assertEqual(self.backend.entered, ['main', 'summary', 'main'])
        # Replaying a saved dependency cannot yield the later main stint.
        await auxiliary.create(**args)
        self.assertEqual(len(self.backend.entered), 3)
        self.assertEqual(self.model.in_flight, 0)

    async def test_explicit_native_turn_event_yields_without_a_clock_not_provider_stop(self):
        self.reopen_service(stint_policy=StintPolicy())
        self.backend.gate.set()
        first = self.client('creative', require_stint_policy=True)
        second = self.client('assistant', require_stint_policy=True)
        await first.create(**self.args('creative'))
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            "SELECT state FROM grants WHERE owner='creative'").fetchone()['state'], 'granted')
        async with self.http('creative') as http:
            event = await http.post('/v1/inference/yield', json={
                'grant_key': 'work-session', 'expected_call_key': 'call'})
            self.assertEqual(event.status_code, 200)
        await asyncio.wait_for(second.create(**self.args('assistant')), 2)
        self.assertEqual(self.backend.entered, ['creative', 'assistant'])
        states = self.queue.supervisor.journal.db.execute('SELECT state,handoff_reason FROM grants').fetchall()
        self.assertEqual(states[0]['handoff_reason'], 'explicit_yield')

    async def test_durable_native_outbox_yields_through_service_and_dispatches_peer(self):
        from .hermes_queue_identity import HostBindings
        from .inference_consumer import DurableConsumer
        from .inference_events import relay_once
        self.reopen_service(stint_policy=StintPolicy())
        self.backend.gate.set()
        host = HostBindings(Path(self.temp.name)/'identity.sqlite', 'own-profile')
        host.pre_api_request(session_id='native', turn_id='native-turn',
            conversation_history=[{'role': 'user', '_row_id': 1}])
        main = self.client('creative', key_provider=host.main_stint_keys, require_stint_policy=True)
        args = self.args('creative')
        args.pop('extra_headers')
        await main.create(**args)
        assistant = self.client('assistant')
        waiting = asyncio.create_task(assistant.create(**self.args('assistant')))
        try:
            await self.wait_for(lambda: len(self.accepted_submissions)==2)
            self.assertEqual(self.backend.entered, ['creative'])
            host.finish_turn('native', 'native-turn')
            async with self.http('creative') as http:
                self.assertEqual(await relay_once(DurableConsumer(http), host), 1)
            await asyncio.wait_for(waiting, 2)
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            self.assertEqual(host.pending_events(), [])
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

    async def test_provider_usage_survives_queue_capture_and_sdk_stream(self):
        self.backend.gate.set()
        answer = self.backend.answer
        self.backend.answer = lambda: {**answer(),'prompt_eval_count':1234,'eval_count':56}
        client = self.client('creative',grant_lifetime='execution',state_frames=True)
        args = self.args('creative',stream=True,stream_options={'include_usage':True})
        response = await client.create(**args)
        async with response:
            chunks = [chunk async for chunk in response]
        usage = [chunk.usage for chunk in chunks if chunk.usage is not None]
        self.assertEqual(len(usage),1)
        self.assertEqual((usage[0].prompt_tokens,usage[0].completion_tokens,usage[0].total_tokens),
                         (1234,56,1290))
        self.assertEqual(self.backend.entered,['creative'])

    async def test_confirmed_failure_is_saved_typed_and_not_redispatched(self):
        entered = []
        async def confirmed_refusal(grant, call):
            entered.append(grant['id'])
            raise BackendEnded({'kind': 'error', 'status': 400,
                'detail': 'context length exceeded', 'code': 'context_length_exceeded'})
        self.queue.supervisor.execute = confirmed_refusal
        client = self.client('creative', grant_lifetime='execution')
        for stream in (False, True, False, True):
            args = self.args('creative', key=str(stream), stream=stream)
            args['extra_headers']['X-Inference-Grant-Key'] = 'mode:' + str(stream)
            with self.subTest(stream=stream), self.assertRaises(BadRequestError) as caught:
                response = await client.create(**args)
                if stream:
                    async with response:
                        _ = [chunk async for chunk in response]
            self.assertEqual(caught.exception.code, 'context_length_exceeded')
        self.assertEqual(len(entered), 2)
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            "SELECT COUNT(*) FROM grants WHERE state='released'").fetchone()[0], 2)

    async def test_actual_ollama_refusal_crosses_gateway_journal_and_sdk_once(self):
        posts = []
        def refuse(request):
            posts.append(request.url.path)
            return httpx.Response(400, json={'error': 'PRIVATE backend validation text'})
        backend = OllamaClient(base_url='http://fake')
        backend._http.close()
        backend._http = httpx.Client(base_url='http://fake', transport=httpx.MockTransport(refuse))
        client = self.client('creative', grant_lifetime='execution')
        try:
            with patch.object(self.manager, 'ollama', backend):
                for _ in range(2):
                    with self.assertRaises(BadRequestError) as caught:
                        await client.create(**self.args('creative'))
                    self.assertEqual(caught.exception.code, 'ollama_refused')
                    self.assertNotIn('PRIVATE', str(caught.exception))
            self.assertEqual(posts, ['/api/chat'])
            self.assertEqual(self.model.in_flight, 0)
            self.assertEqual(self.queue.held_claims, {})
            self.assertEqual(self.queue.supervisor.journal.db.execute(
                "SELECT COUNT(*) FROM grants WHERE state='released'").fetchone()[0], 1)
        finally:
            backend._http.close()

    async def test_execution_grants_handoff_after_disconnect_and_recover_after_restart(self):
        path = Path(self.temp.name) / 'execution-identities.sqlite'
        body = self.args('creative')
        body.pop('extra_headers')
        def keys(tail):
            store = IdentityStore(path)
            try:
                return store.bind_execution('creative/conversation', 'input:1', tail, body)
            finally:
                store.close()
        first_keys = keys('tail:1')
        first = asyncio.create_task(self.client('creative', grant_lifetime='execution').create(
            **body, extra_headers=first_keys.headers()))
        await self.wait_for(lambda: self.backend.entered == ['creative'])
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(self.client('assistant', grant_lifetime='execution').with_options().create(
            **self.args('assistant')))
        try:
            await self.wait_for(lambda: len(self.accepted_submissions) == 2)
            self.assertEqual(self.backend.entered, ['creative'])
            self.assertGreater(self.model.in_flight, 0)
            self.backend.gate.set()
            self.assertEqual((await asyncio.wait_for(second, 2)).choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            self.assertEqual(self.queue.supervisor.journal.db.execute(
                "SELECT COUNT(*) FROM grants WHERE state='released'").fetchone()[0], 2)
            await asyncio.gather(*list(self.queue.supervisor.workers.values()))
            self.reopen_service()
            replay = await self.client('creative', grant_lifetime='execution').create(
                **body, extra_headers=keys('tail:1').headers())
            self.assertEqual(replay.choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            # Same conversation/input, new tool boundary, even identical prompt:
            # distinct intent gets a new grant rather than reopening a released one.
            await self.client('creative', grant_lifetime='execution').create(
                **body, extra_headers=keys('tail:2').headers())
            self.assertEqual(self.backend.entered, ['creative', 'assistant', 'creative'])
            self.assertEqual(self.queue.supervisor.journal.db.execute(
                'SELECT COUNT(*) FROM executions').fetchone()[0], 3)
        finally:
            self.backend.gate.set()
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)

    async def test_closed_stream_does_not_own_execution_grant_handoff(self):
        stream = await self.client('creative', grant_lifetime='execution').create(
            **self.args('creative', stream=True))
        await self.wait_for(lambda: self.backend.entered == ['creative'])
        await stream.close()
        second = asyncio.create_task(self.client('assistant', grant_lifetime='execution').create(
            **self.args('assistant')))
        try:
            await self.wait_for(lambda: len(self.accepted_submissions) == 2)
            self.assertEqual(self.backend.entered, ['creative'])
            self.assertGreater(self.model.in_flight, 0)
            self.backend.gate.set()
            await asyncio.wait_for(second, 2)
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            replay = await self.client('creative', grant_lifetime='execution').create(
                **self.args('creative', stream=True))
            async with replay:
                chunks = [chunk async for chunk in replay]
            self.assertEqual(chunks[-1].choices[0].finish_reason, 'stop')
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
        finally:
            self.backend.gate.set()
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)

    async def test_completed_result_survives_both_caller_and_service_reopen(self):
        path = Path(self.temp.name) / 'caller.sqlite'
        body = self.args('creative')
        body.pop('extra_headers')
        def keys():
            store = IdentityStore(path)
            try:
                return store.bind('conversation', 'input:1', 'tail:2', body)
            finally:
                store.close()
        first_keys = keys()
        self.backend.gate.set()
        answer = await self.client('creative').create(**body, extra_headers=first_keys.headers())
        await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        self.assertEqual(answer.choices[0].message.content, 'done')
        self.assertEqual(self.backend.entered, ['creative'])
        self.reopen_service()
        recovered_keys = keys()
        replay = await self.client('creative').create(**body, extra_headers=recovered_keys.headers())
        self.assertEqual(recovered_keys, first_keys)
        self.assertEqual(replay.choices[0].message.content, 'done')
        self.assertEqual(self.backend.entered, ['creative'])
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM executions').fetchone()[0], 1)

    async def test_uncertain_restart_is_exposed_to_sdk_without_redispatch(self):
        body = self.args('creative', stream=True)
        body.pop('extra_headers')
        store = IdentityStore(Path(self.temp.name) / 'caller.sqlite')
        try:
            keys = store.bind('conversation', 'input:1', 'tail:2', body)
        finally:
            store.close()
        journal = self.queue.supervisor.journal
        self.queue.supervisor.permitted = lambda _: False
        original = await self.client('creative').create(**body, extra_headers=keys.headers())
        await original.close()
        execution = journal.db.execute('SELECT id FROM executions').fetchone()
        journal.dispatch_next('gpu')  # persisted admission; process outcome unknown
        self.reopen_service()
        stream = await self.client('creative', state_frames=True).create(**body, extra_headers=keys.headers())
        async with stream:
            accepted = await asyncio.wait_for(anext(stream), 2)
            self.assertEqual(accepted.model_service['kind'], 'job_accepted')
            unknown = await asyncio.wait_for(anext(stream), 2)
            self.assertEqual(unknown.choices, [])
            self.assertEqual(unknown.model_service['state'], 'uncertain')
        self.assertEqual(self.backend.entered, [])
        self.assertEqual(self.queue.supervisor.journal.execution(
            execution[0], 'creative')['state'], 'uncertain')
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_reopened_identity_store_recovers_running_execution(self):
        path = Path(self.temp.name) / 'caller.sqlite'
        body = self.args('creative')
        body.pop('extra_headers', None)
        def recover_keys():
            store = IdentityStore(path)
            try:
                return store.bind('profile/conversation', 'input:42', 'tail:42', body)
            finally:
                store.close()
        keys = recover_keys()
        original = asyncio.create_task(self.client('creative').create(
            **body, extra_headers=keys.headers()))
        replay = None
        try:
            await self.wait_for(lambda: self.backend.entered == ['creative'])
            original.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await original
            recovered = recover_keys()
            self.assertEqual(recovered, keys)
            replay = asyncio.create_task(self.client('creative').create(
                **body, extra_headers=recovered.headers()))
            await self.wait_for(lambda: self.accepted_submissions.count('creative') == 2)
            self.assertEqual(self.backend.entered, ['creative'])
            self.assertEqual(self.queue.supervisor.journal.db.execute(
                'SELECT COUNT(*) FROM executions').fetchone()[0], 1)
            self.assertGreater(self.model.in_flight, 0)
            self.backend.gate.set()
            self.assertEqual((await asyncio.wait_for(replay, 2)).choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative'])
        finally:
            tasks = [task for task in (original, replay) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_cancel_reconnect_and_explicit_handoff_keep_one_real_runner(self):
        creative, assistant = self.client('creative'), self.client('assistant')
        first = asyncio.create_task(creative.create(**self.args('creative')))
        await self.wait_for(lambda: len(self.backend.entered) == 1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertGreater(self.model.in_flight, 0)
        second = asyncio.create_task(assistant.create(**self.args('assistant')))
        try:
            await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                'SELECT COUNT(*) FROM executions').fetchone()[0] == 2)
            self.assertEqual(self.backend.entered, ['creative'])
            # A new SDK instance and the same keys recover existing execution.
            recovered = self.client('creative')
            replay = asyncio.create_task(recovered.create(**self.args('creative')))
            self.backend.gate.set()
            self.assertEqual((await asyncio.wait_for(replay, 2)).choices[0].message.content, 'done')
            self.assertFalse(second.done())  # work-session grant, not a hidden request-sized lease
            self.assertEqual(self.backend.entered, ['creative'])
            grant = self.queue.supervisor.journal.db.execute(
                "SELECT id FROM grants WHERE owner='creative'").fetchone()[0]
            async with self.http('creative') as http:
                (await http.post('/v1/inference/grants/' + grant + '/release')).raise_for_status()
            self.assertEqual((await asyncio.wait_for(second, 2)).choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            self.assertEqual(self.model.in_flight, 0)
        finally:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)

    async def test_stream_partial_then_uncertain_is_not_success_or_redispatch(self):
        client = self.client('creative')
        self.backend.omit_done = True
        stream = await client.create(**self.args('creative', stream=True))
        self.assertEqual((await asyncio.wait_for(anext(stream), 2)).choices[0].delta.content, 'first token')
        self.assertGreater(self.model.in_flight, 0)
        await stream.close()
        self.backend.gate.set()
        replay = await self.client('creative', state_frames=True).create(**self.args('creative', stream=True))
        async with replay:
            async with asyncio.timeout(2):
                async for chunk in replay:
                    if chunk.choices:
                        self.assertIsNone(chunk.choices[0].finish_reason)
                    elif getattr(chunk, 'model_service', {}).get('state') == 'uncertain':
                        break
        self.assertEqual(self.backend.entered, ['creative'])
        self.assertGreater(self.model.in_flight, 0)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_six_observer_disconnects_do_not_exhaust_or_duplicate_queued_work(self):
        running = asyncio.create_task(self.client('creative').create(**self.args('creative')))
        observers = []
        journal = self.queue.supervisor.journal
        try:
            await self.wait_for(lambda: self.backend.entered == ['creative'])
            for attempt in range(6):
                # Each transport is new; logical keys belong to the caller and
                # remain unchanged. No live clock, provider or inference used.
                observer = asyncio.create_task(self.client('assistant').create(**self.args('assistant')))
                observers.append(observer)
                await self.wait_for(lambda: self.accepted_submissions.count('assistant') == attempt + 1)
                self.assertFalse(observer.done())
                observer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await observer
                self.assertEqual(self.backend.entered, ['creative'])
                self.assertEqual(journal.db.execute(
                    'SELECT COUNT(*) FROM executions').fetchone()[0], 2)
            recovered = asyncio.create_task(self.client('assistant').create(**self.args('assistant')))
            observers.append(recovered)
            self.backend.gate.set()
            await asyncio.wait_for(running, 2)
            self.assertFalse(recovered.done())
            grant = journal.db.execute("SELECT id FROM grants WHERE owner='creative'").fetchone()[0]
            async with self.http('creative') as http:
                (await http.post('/v1/inference/grants/' + grant + '/release')).raise_for_status()
            self.assertEqual((await asyncio.wait_for(recovered, 2)).choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            self.assertEqual(journal.db.execute(
                'SELECT COUNT(*) FROM executions').fetchone()[0], 2)
        finally:
            for task in [running, *observers]:
                task.cancel()
            await asyncio.gather(running, *observers, return_exceptions=True)

    async def test_queued_observer_can_leave_and_reconnect_without_new_job(self):
        creative = self.client('creative')
        running = asyncio.create_task(creative.create(**self.args('creative')))
        queued = None
        replay = None
        try:
            await self.wait_for(lambda: self.backend.entered == ['creative'])
            queued = asyncio.create_task(self.client('assistant').create(**self.args('assistant')))
            journal = self.queue.supervisor.journal
            await self.wait_for(lambda: journal.db.execute(
                'SELECT COUNT(*) FROM executions').fetchone()[0] == 2)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            # Reconstruct the SDK observer with the same caller-owned keys.
            # This models reconnect, NOT persistence of keys by the native host.
            replay = asyncio.create_task(self.client('assistant').create(**self.args('assistant')))
            await asyncio.sleep(.03)
            self.assertFalse(replay.done())
            self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0], 2)
            self.assertEqual(self.backend.entered, ['creative'])
            self.backend.gate.set()
            await asyncio.wait_for(running, 2)
            self.assertFalse(replay.done())
            grant = journal.db.execute("SELECT id FROM grants WHERE owner='creative'").fetchone()[0]
            async with self.http('creative') as http:
                (await http.post('/v1/inference/grants/' + grant + '/release')).raise_for_status()
            result = await asyncio.wait_for(replay, 2)
            self.assertEqual(result.choices[0].message.content, 'done')
            self.assertEqual(self.backend.entered, ['creative', 'assistant'])
            self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0], 2)
        finally:
            tasks = [t for t in (running, queued, replay) if t is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    unittest.main()
