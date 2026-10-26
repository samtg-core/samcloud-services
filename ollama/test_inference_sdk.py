"""SDK wire contract using an in-memory queue API, never live inference.

The real journal/API/scheduler are exercised separately by consumer tests.
Run with the isolated Hermes PM environment (contains the production SDK).
"""
import asyncio
from concurrent.futures import CancelledError as FutureCancelled, ThreadPoolExecutor
from contextvars import ContextVar
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, AuthenticationError, BadRequestError, ConflictError

from .inference_sdk import QueuedClient
from .inference_identity import IdentityStore


class QueueFixture:
    def __init__(self):
        self.requests = []
        self.calls = {}
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.ready.set()
        self.observed = threading.Event()

    def http(self):
        return httpx.AsyncClient(base_url='https://queue.invalid',
                                transport=httpx.MockTransport(self.handle))

    def client(self):
        return QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                            http_factory=self.http, poll_interval=.001)

    def handle(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        with self.lock:
            self.requests.append((request.method, path, body))
            if path == '/v1/inference/capabilities':
                return httpx.Response(200, json={'stints': {'version': 1, 'enabled': True,
                                                         'dependency_yield': True, 'lifecycle_yield': True}})
            if path == '/v1/inference/yield':
                return httpx.Response(200, json={'client_key': body['grant_key'], 'state': 'queued'})
            if path == '/v1/inference/grants':
                return httpx.Response(200, json={'id': 'grant'})
            if path == '/v1/inference/grants/grant/executions':
                previous = self.calls.setdefault(body['key'], body['request'])
                if previous != body['request']:
                    return httpx.Response(409, json={'detail': 'conflicting call'})
                return httpx.Response(200, json={'id': body['key']})
        self.observed.set()
        if path.endswith('/events'):
            events = []
            if self.ready.is_set() and int(request.url.params.get('after', 0)) == 0:
                chunks = [
                    {'index': 0, 'delta': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': None},
                    {'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'tool-1', 'type': 'function',
                        'function': {'name': 'terminal', 'arguments': '{"cmd":"pwd"}'}}]}, 'finish_reason': None},
                    {'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'},
                ]
                frames = ['data: ' + json.dumps({'id': 'answer', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'qwen', 'choices': [c]}) + '\n\n' for c in chunks]
                frames.append('data: [DONE]\n\n')
                events = [{'seq': i + 1, 'data': frame} for i, frame in enumerate(frames)]
            return httpx.Response(200, json={'state': 'complete' if self.ready.is_set() else 'running',
                'events': events, 'next_cursor': 4 if events else int(request.url.params.get('after', 0)),
                'has_more': False})
        if path.startswith('/v1/inference/executions/'):
            return httpx.Response(200, json={'state': 'complete' if self.ready.is_set() else 'running',
                'result': {'kind': 'json', 'body': {'id': 'answer', 'object': 'chat.completion',
                    'created': 1, 'model': 'qwen', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'hello'}}]}}})
        raise AssertionError('unexpected endpoint: ' + path)


def arguments(key='call', **extra):
    return {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'synthetic'}],
            'extra_headers': {'X-Inference-Grant-Key': 'work-session', 'Idempotency-Key': key}, **extra}


class StintCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_detached_work_can_only_reduce_stint_lifetime_to_one_execution(self):
        from .inference_identity import CallKeys
        api=QueueFixture()
        client=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
            http_factory=api.http,grant_lifetime='work_session',
            key_provider=lambda _:CallKeys('fork','one',execution_only=True))
        body=arguments();body.pop('extra_headers')
        try:
            self.assertEqual((await client.create(**body)).choices[0].message.content,'hello')
            grants=[row[2] for row in api.requests if row[0]=='POST' and row[1].endswith('/grants')]
            self.assertEqual(grants[0]['lifetime'],'execution')
        finally:
            await client.aclose()

    async def test_dependency_keys_are_explicit_and_need_capability_before_submission(self):
        from .inference_identity import CallKeys
        for enabled in (True, False):
            api = QueueFixture()
            original = api.handle
            def handle(request):
                if not enabled and request.url.path.endswith('/capabilities'):
                    api.requests.append((request.method, request.url.path, None))
                    return httpx.Response(200, json={'stints': {'version': 1, 'enabled': True}})
                return original(request)
            api.handle = handle
            client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                http_factory=api.http, grant_lifetime='execution',
                key_provider=lambda _: CallKeys('aux', 'summary', ('parent',)))
            args = arguments()
            args.pop('extra_headers')
            try:
                if enabled:
                    self.assertEqual((await client.create(**args)).id, 'answer')
                    writes = [r[2] for r in api.requests if r[0] == 'POST']
                    self.assertEqual(writes[-1]['yield_grant_keys'], ['parent'])
                else:
                    with self.assertRaises(APIConnectionError):
                        await client.create(**args)
                    self.assertFalse(any(r[0] == 'POST' for r in api.requests))
            finally:
                await client.aclose()

    async def test_stint_mode_requires_capability_before_any_write_sync_and_async(self):
        for advertised in ({}, {'stints': {'version': 1, 'enabled': False}},
                           {'stints': {'version': 1, 'enabled': True}},
                           {'stints': {'version': 2, 'enabled': True}},
                           {'stints': {'version': True, 'enabled': True}}):
            for asynchronous in (False, True):
                with self.subTest(advertised=advertised, asynchronous=asynchronous):
                    requests = []
                    def handle(request):
                        requests.append((request.method, request.url.path))
                        return httpx.Response(200, json=advertised)
                    client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                        http_factory=lambda: httpx.AsyncClient(base_url='https://queue.invalid',
                            transport=httpx.MockTransport(handle)), require_stint_policy=True)
                    try:
                        with self.assertRaises(APIConnectionError):
                            if asynchronous:
                                await client.with_options().create(**arguments())
                            else:
                                await asyncio.to_thread(lambda: client.with_options().create(**arguments()))
                        self.assertEqual(requests, [('GET', '/v1/inference/capabilities')])
                    finally:
                        await client.aclose()

    async def test_enabled_stints_preflight_then_submit_on_both_transports(self):
        for asynchronous in (False, True):
            api = QueueFixture()
            client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                http_factory=api.http, require_stint_policy=True)
            try:
                if asynchronous:
                    answer = await client.with_options().create(**arguments())
                else:
                    answer = await asyncio.to_thread(lambda: client.with_options().create(**arguments()))
                self.assertEqual(answer.id, 'answer')
                self.assertEqual(api.requests[0][:2], ('GET', '/v1/inference/capabilities'))
                self.assertEqual(api.requests[1][2]['lifetime'], 'work_session')
            finally:
                await client.aclose()


class FailedQueueFixture(QueueFixture):
    """Saved terminal evidence, not an SDK-side timeout or guessed refusal."""
    state = 'failed'
    failure = {'kind': 'error', 'status': 400, 'detail': 'context length exceeded',
               'code': 'context_length_exceeded'}
    reject_auth = False

    def handle(self, request):
        if self.reject_auth:
            return httpx.Response(401, json={'error': {'message': 'invalid credential',
                                                     'code': 'invalid_api_key'}})
        path = request.url.path
        if path.startswith('/v1/inference/executions/'):
            self.observed.set()
        if path.endswith('/events'):
            return httpx.Response(200, json={'state': self.state, 'events': [],
                'next_cursor': 0, 'has_more': False})
        if path.startswith('/v1/inference/executions/'):
            return httpx.Response(200, json={'state': self.state, 'result': self.failure})
        return super().handle(request)


class FailureSDK(unittest.TestCase):
    def test_sync_saved_overflow_is_bad_request_for_json_and_stream(self):
        api = FailedQueueFixture()
        client = api.client()
        try:
            for stream in (False, True, False, True):
                with self.subTest(stream=stream), self.assertRaises(BadRequestError) as caught:
                    response = client.create(**arguments(key=str(stream), stream=stream))
                    if stream:
                        with response:
                            list(response)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertEqual(caught.exception.code, 'context_length_exceeded')
                self.assertIn('context length exceeded', str(caught.exception))
            self.assertEqual(len(api.calls), 2)  # reopen each mode, no replacement inference
        finally:
            client.close()

    def test_service_auth_rejection_keeps_its_status_without_creating_work(self):
        api = FailedQueueFixture()
        api.reject_auth = True
        client = api.client()
        try:
            with self.assertRaises(AuthenticationError):
                client.create(**arguments())
            self.assertEqual(api.calls, {})
        finally:
            client.close()

    def test_uncertain_does_not_trust_an_error_shaped_saved_result(self):
        api = FailedQueueFixture()
        api.state = 'uncertain'
        client = api.client()
        try:
            for stream in (False, True):
                api.state = 'uncertain'
                api.observed.clear()
                def observe():
                    response = client.create(**arguments(key=str(stream), stream=stream))
                    if stream:
                        with response:
                            list(response)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(observe)
                    try:
                        self.assertTrue(api.observed.wait(2))
                        threading.Event().wait(.02)
                        self.assertFalse(pending.done())  # no false overflow from uncertain result
                    finally:
                        api.state = 'failed'  # affirmative later terminal evidence
                    with self.assertRaises(BadRequestError):
                        pending.result(timeout=2)
            self.assertEqual(len(api.calls), 2)
        finally:
            client.close()

    def test_async_saved_failure_retains_type_for_json_and_stream(self):
        async def run():
            api = FailedQueueFixture()
            client = api.client()
            try:
                for stream in (False, True, False, True):
                    with self.subTest(stream=stream), self.assertRaises(BadRequestError) as caught:
                        response = await client.create(**arguments(key=str(stream), stream=stream))
                        if stream:
                            async with response:
                                _ = [chunk async for chunk in response]
                    self.assertEqual(caught.exception.code, 'context_length_exceeded')
                self.assertEqual(len(api.calls), 2)
            finally:
                await client.aclose()
        asyncio.run(run())


class HostIdentitySDK(unittest.TestCase):
    def test_scoped_persisted_identity_crosses_sync_portal_without_mixing_callers(self):
        api = QueueFixture()
        boundary = ContextVar('host_boundary')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calls.sqlite'
            store = IdentityStore(path)
            store.close()
            def resolve(body):
                scope, work, call = boundary.get()  # absence must refuse
                store = IdentityStore(path)
                try:
                    return store.bind(scope, work, call, body)
                finally:
                    store.close()
            client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                                  http_factory=api.http, key_provider=resolve)
            body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'same content'}]}
            def run(scope):
                token = boundary.set((scope, 'input:1', 'tail:1'))
                try:
                    return client.with_options().create(**body).choices[0].message.content
                finally:
                    boundary.reset(token)
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    self.assertEqual(list(pool.map(run, ['a', 'b', 'a'])), ['hello'] * 3)
                self.assertEqual(len(api.calls), 2)
                before = len(api.requests)
                with self.assertRaises(APIConnectionError):
                    client.create(**body)
                self.assertEqual(len(api.requests), before)  # no service write/fallback
                token = boundary.set(('a', 'input:1', 'tail:1'))
                try:
                    with self.assertRaises(APIConnectionError):
                        client.create(**body, extra_headers={'Idempotency-Key': 'conflicting'})
                    self.assertEqual(len(api.requests), before)
                finally:
                    boundary.reset(token)
            finally:
                client.close()

    def test_async_key_provider_keeps_task_context_and_refuses_changed_body(self):
        async def exercise(path):
            api = QueueFixture()
            boundary = ContextVar('async_host_boundary')
            async def resolve(body):
                await asyncio.sleep(0)
                store = IdentityStore(path)
                try:
                    return store.bind(boundary.get(), 'input:1', 'tail:1', body)
                finally:
                    store.close()
            client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                                  http_factory=api.http, key_provider=resolve)
            async def run(scope, text='same'):
                token = boundary.set(scope)
                try:
                    return await client.create(model='qwen', messages=[{'role': 'user', 'content': text}])
                finally:
                    boundary.reset(token)
            try:
                results = await asyncio.gather(run('a'), run('b'), run('a'))
                self.assertTrue(all(r.choices[0].message.content == 'hello' for r in results))
                self.assertEqual(len(api.calls), 2)
                before = len(api.requests)
                with self.assertRaises(APIConnectionError):
                    await run('a', 'changed')
                self.assertEqual(len(api.requests), before)
            finally:
                await client.aclose()
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory) / 'calls.sqlite'))


class SyncSDK(unittest.TestCase):
    def setUp(self):
        self.api = QueueFixture()
        self.client = self.api.client()

    def tearDown(self):
        self.client.close()

    def test_json_and_reconnect_reuse_logical_call(self):
        self.assertEqual(self.client.chat.completions.create(**arguments()).choices[0].message.content, 'hello')
        self.client.close()
        self.client = self.api.client()
        self.assertEqual(self.client.chat.completions.create(**arguments()).id, 'answer')
        self.assertEqual(len(self.api.calls), 1)

    def test_state_observer_is_out_of_band_for_sync_stream(self):
        states = []
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                              http_factory=self.api.http, on_state=states.append)
        try:
            with client.create(**arguments(stream=True)) as stream:
                chunks = list(stream)
            self.assertEqual(len(chunks), 3)
            self.assertEqual(states, [{'state': 'complete', 'kind': 'execution', 'grant_key': 'work-session',
                'call_key': 'call', 'grant_id': 'grant', 'execution_id': 'call'}])
            self.assertEqual(chunks[0].choices[0].delta.content, 'hello')
        finally:
            client.close()

    def test_stream_preserves_tools_and_terminal(self):
        with self.client.chat.completions.create(**arguments(stream=True)) as stream:
            chunks = list(stream)
        self.assertEqual(chunks[0].choices[0].delta.content, 'hello')
        self.assertEqual(chunks[1].choices[0].delta.tool_calls[0].function.name, 'terminal')
        self.assertEqual(chunks[-1].choices[0].finish_reason, 'tool_calls')

    def test_missing_keys_fails_without_any_request(self):
        args = arguments()
        del args['extra_headers']
        with self.assertRaises(APIConnectionError):
            self.client.chat.completions.create(**args)
        self.assertEqual(self.api.requests, [])

    def test_with_options_retains_transport_and_headers(self):
        client = self.client.with_options(default_headers=arguments()['extra_headers'])
        args = arguments()
        del args['extra_headers']
        self.assertEqual(client.chat.completions.create(**args).id, 'answer')
        self.assertEqual(len(self.api.calls), 1)

    def test_sync_clone_preserves_explicit_execution_grant_policy(self):
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                              http_factory=self.api.http, grant_lifetime='execution')
        try:
            self.assertEqual(client.with_options().create(**arguments()).id, 'answer')
            asks = [body for method, path, body in self.api.requests
                    if method == 'POST' and path.endswith('/grants')]
            self.assertEqual(asks[0]['lifetime'], 'execution')
        finally:
            client.close()

    def test_close_waiting_sync_stream_stops_observer_not_work(self):
        self.api.ready.clear()
        stream = self.client.chat.completions.create(**arguments(stream=True))
        with ThreadPoolExecutor(max_workers=1) as pool:
            reader = pool.submit(lambda: next(stream, None))
            try:
                self.assertTrue(self.api.observed.wait(2))
                stream.close()
                self.assertIsNone(reader.result(timeout=2))
                self.assertEqual(len(self.api.calls), 1)
                self.assertFalse(any(path.endswith('/release') for _, path, _ in self.api.requests))
            finally:
                # A failing implementation must not leave the test worker hung.
                self.api.ready.set()
        replay = self.client.chat.completions.create(**arguments(stream=True))
        with replay:
            self.assertEqual(list(replay)[-1].choices[0].finish_reason, 'tool_calls')
        self.assertEqual(len(self.api.calls), 1)

    def test_close_auxiliary_wait_reconnects_to_saved_execution(self):
        self.api.ready.clear()
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=self.api.http, poll_interval=.001, complete_responses=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            reader = pool.submit(lambda: client.create(**arguments(stream=True)))
            try:
                self.assertTrue(self.api.observed.wait(2))
                client.close()
                with self.assertRaises((APIConnectionError, FutureCancelled)):
                    reader.result(timeout=2)
                self.assertEqual(len(self.api.calls), 1)
                self.assertFalse(any(path.endswith('/release') for _, path, _ in self.api.requests))
            finally:
                self.api.ready.set()
                client.close()
        replay = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=self.api.http, complete_responses=True)
        try:
            self.assertEqual(replay.create(**arguments(stream=True)).id, 'answer')
            self.assertEqual(len(self.api.calls), 1)
        finally:
            replay.close()

    def test_default_factory_uses_one_v1_prefix_and_seat_credential(self):
        original = httpx.AsyncClient
        seen = []
        def record(request):
            seen.append((request.url.path, request.headers.get('Authorization')))
            return self.api.handle(request)
        class IsolatedHTTP(original):
            def __init__(self, **kwargs):
                kwargs.setdefault('transport', httpx.MockTransport(record))
                super().__init__(**kwargs)
        # Intercept only HTTP transport construction: exercise the REAL default
        # factory instead of the shorter base URL supplied by our fixture.
        with patch('ollama.inference_sdk.httpx.AsyncClient', IsolatedHTTP):
            client = QueuedClient(api_key='synthetic-seat', base_url='https://queue.invalid/v1/',
                                  poll_interval=.001)
        try:
            self.assertEqual(client.chat.completions.create(**arguments()).id, 'answer')
            self.assertTrue(seen)
            self.assertTrue(all(path.startswith('/v1/inference/') for path, _ in seen))
            self.assertTrue(all(auth == 'Bearer synthetic-seat' for _, auth in seen))
        finally:
            client.close()


class AsyncSDK(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api = QueueFixture()
        self.client = self.api.client()

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_json_and_stream_are_awaitable(self):
        result = await self.client.chat.completions.create(**arguments('json'))
        self.assertEqual(result.choices[0].message.content, 'hello')
        stream = await self.client.chat.completions.create(**arguments('stream', stream=True))
        async with stream:
            chunks = [chunk async for chunk in stream]
        self.assertEqual(chunks[-1].choices[0].finish_reason, 'tool_calls')

    async def test_opt_in_waiting_frames_do_not_consume_provider_cursor(self):
        self.api.ready.clear()
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=self.api.http, poll_interval=.001, state_frames=True)
        try:
            stream = await client.create(**arguments(stream=True))
            async with stream:
                accepted = await asyncio.wait_for(anext(stream), 2)
                self.assertEqual(accepted.model_service['kind'], 'job_accepted')
                self.assertEqual(accepted.model_service['execution_id'], 'call')
                for _ in range(3):
                    frame = await asyncio.wait_for(anext(stream), 2)
                    self.assertEqual(frame.choices, [])
                    self.assertIsNone(frame.usage)
                    self.assertEqual(frame.model_service, {'kind': 'queue_status',
                        'state': 'running', 'execution_id': 'call'})
                self.api.ready.set()
                chunks = [c async for c in stream]
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].choices[0].delta.content, 'hello')
            self.assertEqual(chunks[-1].choices[0].finish_reason, 'tool_calls')
            self.assertEqual(len(self.api.calls), 1)
        finally:
            await client.aclose()

    async def test_unreachable_service_reports_unknown_not_healthy_waiting_or_model_text(self):
        self.api.ready.clear()
        failed = asyncio.Event()
        async def broken(request):
            if request.url.path.endswith('/events'):
                failed.set()
                raise httpx.ReadError('synthetic offline', request=request)
            return self.api.handle(request)
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=lambda: httpx.AsyncClient(base_url='https://queue.invalid',
                transport=httpx.MockTransport(broken)), poll_interval=.005, state_frames=True)
        stream = await client.create(**arguments(stream=True))
        read = asyncio.create_task(anext(stream))
        try:
            accepted = await read
            self.assertEqual(accepted.model_service['kind'], 'job_accepted')
            read = asyncio.create_task(anext(stream))
            await asyncio.wait_for(failed.wait(), 2)
            unknown = await asyncio.wait_for(read, 2)
            self.assertEqual(unknown.choices, [])
            self.assertIsNone(unknown.usage)
            self.assertEqual(unknown.model_service['kind'], 'adapter_status')
            self.assertEqual(unknown.model_service['source'], 'local_adapter')
            self.assertEqual(unknown.model_service['state'], 'contact_unknown')
            self.assertEqual(unknown.model_service['service_confirmation'], 'unknown')
            self.assertEqual(len(self.api.calls), 1)
            self.assertFalse(any(p.endswith('/release') for _, p, _ in self.api.requests))
        finally:
            read.cancel()
            await asyncio.gather(read, return_exceptions=True)
            await stream.close()
            await client.aclose()

    async def test_lost_contact_before_acceptance_does_not_time_out_or_invent_a_service_job(self):
        attempted = []
        offline = True
        async def handle(request):
            attempted.append((request.method, request.url.path))
            if offline:
                raise httpx.ReadError('synthetic offline', request=request)
            return self.api.handle(request)
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=lambda: httpx.AsyncClient(base_url='https://queue.invalid',
                transport=httpx.MockTransport(handle)), poll_interval=.005, state_frames=True,
            require_stint_policy=True)
        try:
            stream = await client.create(**arguments(stream=True))
            self.assertNotIn('X-Inference-Execution', stream.response.headers)
            async with stream:
                for _ in range(2):
                    pulse = await asyncio.wait_for(anext(stream), 2)
                    self.assertEqual(pulse.choices, [])
                    self.assertEqual(pulse.model_service['state'], 'contact_unknown')
                    self.assertNotIn('execution_id', pulse.model_service)
                    self.assertEqual(pulse.model_service['call_key'], 'call')
                self.assertFalse(any(method=='POST' for method,path in attempted))
                offline = False
                chunks = [c async for c in stream]
            acknowledged = [c for c in chunks if getattr(c, 'model_service', {}).get('kind') == 'job_accepted']
            self.assertEqual(len(acknowledged), 1)
            self.assertEqual(acknowledged[0].model_service['execution_id'], 'call')
            answer = [c for c in chunks if c.choices]
            self.assertEqual(answer[0].choices[0].delta.content, 'hello')
            self.assertEqual(len(self.api.calls), 1)
        finally:
            await client.aclose()

    async def test_lost_submit_response_retries_same_intent_then_acknowledges_one_execution(self):
        lost = True
        attempts = []
        async def handle(request):
            response = self.api.handle(request)
            if request.method=='POST' and request.url.path.endswith('/executions'):
                attempts.append(request.content)
                if lost:
                    raise httpx.ReadError('saved but response lost', request=request)
            return response
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=lambda: httpx.AsyncClient(base_url='https://queue.invalid',
                transport=httpx.MockTransport(handle)), poll_interval=.005, state_frames=True)
        try:
            stream = await client.create(**arguments(stream=True))
            async with stream:
                unknown = await asyncio.wait_for(anext(stream), 2)
                self.assertEqual(unknown.model_service['kind'], 'adapter_status')
                self.assertEqual(unknown.model_service['state'], 'contact_unknown')
                self.assertEqual(len(self.api.calls), 1)
                lost = False
                chunks = [c async for c in stream]
            self.assertEqual(len(set(attempts)), 1)
            self.assertEqual(len(self.api.calls), 1)
            self.assertEqual(sum(getattr(c, 'model_service', {}).get('kind')=='job_accepted' for c in chunks), 1)
            self.assertFalse(any(path.endswith('/release') for method,path,body in self.api.requests))
        finally:
            await client.aclose()

    async def test_close_before_stream_iteration_cancels_observation_not_saved_submit(self):
        entered = asyncio.Event()
        blocked = True
        cancelled = asyncio.Event()
        async def handle(request):
            response = self.api.handle(request)
            if request.method=='POST' and request.url.path.endswith('/executions') and blocked:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return response
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=lambda: httpx.AsyncClient(base_url='https://queue.invalid',
                transport=httpx.MockTransport(handle)), poll_interval=.005, state_frames=True)
        try:
            stream = await client.create(**arguments(stream=True))
            await asyncio.wait_for(entered.wait(), 2)
            await stream.close()  # deliberately never iterated
            self.assertTrue(cancelled.is_set())
            self.assertEqual(len(self.api.calls), 1)
            blocked = False
            replay = await client.create(**arguments(stream=True))
            async with replay:
                chunks = [c async for c in replay]
            self.assertEqual(len(self.api.calls), 1)
            self.assertEqual([c for c in chunks if c.choices][0].choices[0].delta.content, 'hello')
            self.assertFalse(any(path.endswith('/release') for method,path,body in self.api.requests))
        finally:
            await client.aclose()

    async def test_unknown_observation_does_not_hide_affirmative_authentication_failure(self):
        def denied(request):
            return httpx.Response(401, json={'error': {'message': 'invalid credential'}})
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=lambda: httpx.AsyncClient(base_url='https://queue.invalid',
                transport=httpx.MockTransport(denied)), state_frames=True)
        try:
            stream = await client.create(**arguments(stream=True))
            with self.assertRaises(AuthenticationError):
                async with stream:
                    await anext(stream)
        finally:
            await client.aclose()

    async def test_concurrent_calls_keep_state_identity_separate(self):
        self.api.ready.clear()
        states = []
        ready = asyncio.Event()
        async def report(event):
            states.append(event)
            if {e['call_key'] for e in states if e['state'] == 'running'} == {'one', 'two'}:
                ready.set()
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                              http_factory=self.api.http, poll_interval=.001, on_state=report)
        calls = [asyncio.create_task(client.create(**arguments(key))) for key in ('one', 'two')]
        try:
            await asyncio.wait_for(ready.wait(), 2)
            self.assertTrue(all(not call.done() for call in calls))
            self.api.ready.set()
            await asyncio.wait_for(asyncio.gather(*calls), 2)
            for key in ('one', 'two'):
                events = [e for e in states if e['call_key'] == key]
                self.assertEqual([e['state'] for e in events], ['running', 'complete'])
                self.assertTrue(all(e['execution_id'] == key for e in events))
                self.assertTrue(all(set(e) == {'state', 'kind', 'call_key', 'grant_key', 'execution_id', 'grant_id'} for e in events))
                self.assertTrue(all(e['kind'] == 'execution' for e in events))
        finally:
            for call in calls:
                call.cancel()
            await asyncio.gather(*calls, return_exceptions=True)
            await client.aclose()

    async def test_failed_state_sink_does_not_fail_accepted_work(self):
        def broken(event):
            raise ValueError('private sink detail')
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                              http_factory=self.api.http, on_state=broken)
        try:
            with self.assertLogs('ollama.inference_sdk', level='WARNING') as logs:
                result = await client.create(**arguments())
            self.assertEqual(result.choices[0].message.content, 'hello')
            self.assertNotIn('private sink detail', str(logs.output))
            self.assertEqual(len(self.api.calls), 1)
        finally:
            await client.aclose()

    async def test_cancel_observation_reconnects_without_release_or_new_work(self):
        self.api.ready.clear()
        reader = asyncio.create_task(self.client.chat.completions.create(**arguments()))
        self.assertTrue(await asyncio.to_thread(self.api.observed.wait, 2))
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader
        self.assertEqual(len(self.api.calls), 1)
        self.assertFalse(any(path.endswith('/release') for _, path, _ in self.api.requests))
        self.api.ready.set()
        result = await self.client.chat.completions.create(**arguments())
        self.assertEqual(result.id, 'answer')
        self.assertEqual(len(self.api.calls), 1)

    async def test_cancel_auxiliary_wait_reconnects_to_saved_execution(self):
        self.api.ready.clear()
        client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=self.api.http, poll_interval=.001, complete_responses=True)
        reader = asyncio.create_task(client.create(**arguments(stream=True)))
        try:
            self.assertTrue(await asyncio.to_thread(self.api.observed.wait, 2))
            reader.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await reader
            await client.aclose()
            self.assertFalse(any(path.endswith('/release') for _, path, _ in self.api.requests))
            self.api.ready.set()
            replay = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                http_factory=self.api.http, complete_responses=True)
            try:
                self.assertEqual((await replay.create(**arguments(stream=True))).id, 'answer')
                self.assertEqual(len(self.api.calls), 1)
            finally:
                await replay.aclose()
        finally:
            self.api.ready.set()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await client.aclose()

    async def test_cancel_stream_wait_retains_call(self):
        self.api.ready.clear()
        stream = await self.client.chat.completions.create(**arguments(stream=True))
        reader = asyncio.create_task(anext(stream))
        self.assertTrue(await asyncio.to_thread(self.api.observed.wait, 2))
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader
        await stream.close()
        self.api.ready.set()
        replay = await self.client.chat.completions.create(**arguments(stream=True))
        async with replay:
            chunks = [chunk async for chunk in replay]
        self.assertEqual(chunks[-1].choices[0].finish_reason, 'tool_calls')
        self.assertEqual(len(self.api.calls), 1)

    async def test_close_waiting_async_stream_stops_observer_not_work(self):
        self.api.ready.clear()
        stream = await self.client.chat.completions.create(**arguments(stream=True))
        reader = asyncio.create_task(anext(stream, None))
        try:
            self.assertTrue(await asyncio.to_thread(self.api.observed.wait, 2))
            await stream.close()
            self.assertIsNone(await asyncio.wait_for(reader, 2))
            self.assertFalse(any(path.endswith('/release') for _, path, _ in self.api.requests))
        finally:
            self.api.ready.set()
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        replay = await self.client.chat.completions.create(**arguments(stream=True))
        async with replay:
            self.assertEqual([chunk async for chunk in replay][-1].choices[0].finish_reason, 'tool_calls')
        self.assertEqual(len(self.api.calls), 1)


if __name__ == '__main__':
    unittest.main()
