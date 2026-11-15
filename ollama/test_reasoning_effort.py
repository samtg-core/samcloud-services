"""Opt-in Ollama thinking: SDK -> durable HTTP API -> gateway -> real OllamaClient.

Task 965. Synthetic transports only; no live inference. The absent-effort wire
bytes are compared against the unpatched release by capture_absent_wire.py.
"""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import httpx
from openai import UnprocessableEntityError

from . import server
from .reasoning_request import reasoning_request_error
from . import test_gateway_queue as fixture
from .manager import Backend
from .ollama_client import OllamaClient
from . import test_sdk_gateway as sdk

DONE = {'model': 'qwen', 'message': {'role': 'assistant', 'content': 'done'}, 'done': True,
        'done_reason': 'stop', 'prompt_eval_count': 3, 'eval_count': 2}


def recording_backend():
    """Real OllamaClient payload construction; both transports recorded as sent."""
    posts = []
    def answer(request):
        posts.append(('collect', request.url.path, request.content))
        return httpx.Response(200, content=(json.dumps(DONE) + '\n').encode())
    backend = OllamaClient(base_url='http://fake')
    backend._http.close()
    backend._http = httpx.Client(base_url='http://fake', transport=httpx.MockTransport(answer))
    async def deadlines(path, payload, prompt_tokens=None, *, service_owned=False):
        posts.append(('stream', path, json.dumps(payload).encode()))
        yield DONE
    backend._stream_with_deadlines = deadlines
    backend.num_ctx_for = lambda name: 65536
    return backend, posts


class ReasoningRequest(unittest.TestCase):
    def test_absent_and_none_are_never_refused(self):
        for body in ({}, {'reasoning_effort': None}, {'reasoning_effort': 'none'},
                     {'reasoning_effort': 'none', 'max_tokens': 'x'}, {'max_tokens': -1}):
            with self.subTest(body=body):
                self.assertIsNone(reasoning_request_error(body))

    def test_level_needs_a_positive_integer_budget(self):
        for effort in ('low', 'medium', 'high'):
            self.assertIsNone(reasoning_request_error({'reasoning_effort': effort, 'max_tokens': 1}))
            for budget in (None, 0, -1, 1.5, 64.0, '64', True, False, [64]):
                with self.subTest(effort=effort, budget=budget):
                    body = {'reasoning_effort': effort}
                    if budget is not None:
                        body['max_tokens'] = budget
                    self.assertIn('max_tokens', reasoning_request_error(body))

    def test_only_supported_levels(self):
        for effort in ('max', 'minimal', 'xhigh', 'ultra', 'HIGH', '', True, 1, ['high']):
            with self.subTest(effort=effort):
                self.assertIn('reasoning_effort must be',
                              reasoning_request_error({'reasoning_effort': effort, 'max_tokens': 64}))

    def test_request_model_is_unchanged_so_exo_forwarding_is_untouched(self):
        # Exo already forwards reasoning_effort without a budget (server.py exo
        # branch); the request model must keep accepting it.
        req = server.ChatRequest(model='think', messages=[], reasoning_effort='high')
        self.assertEqual((req.reasoning_effort, req.max_tokens), ('high', None))
        self.assertEqual(server.ChatRequest(model='q', messages=[], max_tokens='64').max_tokens, 64)
        for budget in ('64', 64.0, True):
            req = server.ChatRequest.model_validate({'model': 'think', 'messages': [],
                                                     'reasoning_effort': 'high', 'max_tokens': budget})
            self.assertEqual(req.max_tokens, int(budget))
            self.assertIn('max_tokens', req.reasoning_problem())
        self.assertIsNone(server.ChatRequest(model='q', messages=[], reasoning_effort='low',
                                             max_tokens=8).reasoning_problem())


class ReasoningWire(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = sdk.SDKGateway.asyncSetUp
    asyncTearDown = sdk.SDKGateway.asyncTearDown
    install_app = sdk.SDKGateway.install_app
    reopen_service = sdk.SDKGateway.reopen_service
    http = sdk.SDKGateway.http
    client = sdk.SDKGateway.client
    args = sdk.SDKGateway.args
    wait_for = fixture.GatewayLifetime.wait_for

    def executions(self):
        return self.queue.supervisor.journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0]

    async def send(self, owner, stream, **extra):
        backend, posts = recording_backend()
        client = self.client(owner, grant_lifetime='execution')
        try:
            with patch.object(self.manager, 'ollama', backend):
                response = await client.create(**self.args(owner, key=owner, stream=stream, **extra))
                if stream:
                    async with response:
                        _ = [chunk async for chunk in response]
                else:
                    self.assertEqual(response.choices[0].message.content, 'done')
        finally:
            backend._http.close()
        self.assertEqual(len(posts), 1)
        mode, path, raw = posts[0]
        self.assertEqual((mode, path), ('stream' if stream else 'collect', '/api/chat'))
        return json.loads(raw)

    async def test_absent_and_none_send_think_false_and_no_budget(self):
        for stream in (False, True):
            absent = await self.send('absent-%s' % stream, stream)
            none = await self.send('none-%s' % stream, stream, reasoning_effort='none')
            with self.subTest(stream=stream):
                self.assertIs(absent['think'], False)
                self.assertEqual(absent['options'], {'num_ctx': 65536})
                absent['messages'] = none['messages'] = None
                self.assertEqual(absent, none)

    async def test_level_reaches_ollama_as_think_with_num_predict_bound(self):
        for stream in (False, True):
            for effort in ('low', 'medium', 'high'):
                payload = await self.send('%s-%s' % (effort, stream), stream,
                                          reasoning_effort=effort, max_tokens=96)
                with self.subTest(stream=stream, effort=effort):
                    self.assertEqual(payload['think'], effort)
                    self.assertEqual(payload['options'], {'num_predict': 96, 'num_ctx': 65536})

    async def test_harmony_model_gets_the_level_form(self):
        # GPT-OSS reads Think as a level (harmony Reasoning field: low/medium/high),
        # so the string is passed, never coerced to True.
        oss = SimpleNamespace(name='gpt-oss:120b', backend=Backend.OLLAMA, in_flight=0,
                              last_used=0, request_count=0)
        self.manager.models['gpt-oss:120b'] = oss
        self.queue.models['gpt-oss:120b'] = 'gpu'
        async def resolve(name, **kwargs):
            return self.manager.models[name]
        with patch.object(server, '_resolve_model', resolve):
            for stream in (False, True):
                payload = await self.send('oss-%s' % stream, stream, model='gpt-oss:120b',
                                          reasoning_effort='medium', max_tokens=128)
                with self.subTest(stream=stream):
                    self.assertEqual(payload['model'], 'gpt-oss:120b')
                    self.assertEqual(payload['think'], 'medium')
                    self.assertEqual(payload['options']['num_predict'], 128)

    async def test_refusal_is_before_the_journal_and_takes_no_admission(self):
        backend, posts = recording_backend()
        client = self.client('refused', grant_lifetime='execution')
        bad = ({'reasoning_effort': 'high'}, {'reasoning_effort': 'high', 'max_tokens': 0},
               {'reasoning_effort': 'high', 'max_tokens': -5}, {'reasoning_effort': 'high', 'max_tokens': 1.5},
               {'reasoning_effort': 'high', 'max_tokens': '64'}, {'reasoning_effort': 'high', 'max_tokens': True},
               {'reasoning_effort': 'max', 'max_tokens': 64}, {'reasoning_effort': 'minimal', 'max_tokens': 64})
        try:
            with patch.object(self.manager, 'ollama', backend):
                for i, extra in enumerate(bad):
                    for stream in (False, True):
                        with self.subTest(extra=extra, stream=stream), \
                                self.assertRaises(UnprocessableEntityError):
                            response = await client.create(**self.args('refused', key='bad%d%s' % (i, stream),
                                                                        stream=stream, **extra))
                            if stream:
                                async with response:
                                    _ = [chunk async for chunk in response]
        finally:
            backend._http.close()
        self.assertEqual(posts, [])
        self.assertEqual(self.executions(), 0)
        self.assertEqual(self.manager.models['qwen'].in_flight, 0)
        self.assertEqual(self.queue.held_claims, {})
        # Nothing refused holds the resource: the next valid call runs at once.
        self.assertEqual((await self.send('after', False))['think'], False)

    def test_queue_validation_reads_the_raw_body(self):
        for extra in ({'max_tokens': '64'}, {'max_tokens': 64.0}, {'max_tokens': True}, {}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.queue.validate_call('qwen', {'model': 'qwen', 'messages': [],
                                                  'reasoning_effort': 'high', **extra})
        call = self.queue.validate_call('qwen', {'model': 'qwen', 'messages': [],
                                                 'reasoning_effort': 'high', 'max_tokens': 64})
        self.assertEqual((call['body']['reasoning_effort'], call['body']['max_tokens']), ('high', 64))

    async def test_compat_route_refuses_before_the_journal(self):
        request = fixture.GatewayLifetime.request(self, 'compat', 'k')
        with self.assertRaises(HTTPException) as caught:
            await server.chat_completions(server.ChatRequest(
                model='qwen', messages=[{'role': 'user', 'content': 'x'}], reasoning_effort='high'), request)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(self.executions(), 0)
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM grants').fetchone()[0], 0)

    async def test_direct_route_refuses_before_any_backend_call(self):
        backend, posts = recording_backend()
        try:
            with patch.object(self.manager, 'ollama', backend), \
                    patch.object(server.prompt_size, 'count', side_effect=AssertionError('sized')):
                for stream in (False, True):
                    with self.subTest(stream=stream), self.assertRaises(HTTPException) as caught:
                        await server._chat_completions_direct(server.ChatRequest(
                            model='qwen', messages=[], stream=stream, reasoning_effort='low'),
                            resolved=self.manager.models['qwen'])
                    self.assertEqual(caught.exception.status_code, 400)
        finally:
            backend._http.close()
        self.assertEqual(posts, [])
        self.assertEqual(self.manager.models['qwen'].in_flight, 0)

    def route_app(self):
        """The real /v1/chat/completions endpoint, so FastAPI parses the raw JSON."""
        from fastapi import FastAPI, Request
        app = FastAPI()
        @app.middleware('http')
        async def identity(request: Request, next_handler):
            request.state.caller = {'username': 'route'}
            return await next_handler(request)
        app.post('/v1/chat/completions')(server.chat_completions)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://route')

    COERCIBLE = ('64', 64.0, True)

    async def route(self, budget, *, queued):
        backend, posts = recording_backend()
        body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False,
                'reasoning_effort': 'high', 'max_tokens': budget}
        try:
            with patch.object(self.manager, 'ollama', backend), \
                    patch.object(server, 'inference_dispatcher', self.queue if queued else None):
                async with self.route_app() as http:
                    response = await http.post('/v1/chat/completions', json=body,
                                               headers={'Idempotency-Key': 'route-%r' % (budget,)})
                    if queued and response.status_code == 200:
                        await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        finally:
            backend._http.close()
        return response, posts

    def rows(self, table):
        return self.queue.supervisor.journal.db.execute('SELECT COUNT(*) FROM %s' % table).fetchone()[0]

    async def test_compat_route_refuses_coercible_budgets_before_submit(self):
        for budget in self.COERCIBLE:
            response, posts = await self.route(budget, queued=True)
            with self.subTest(budget=budget):
                self.assertEqual(response.status_code, 422)
                self.assertEqual(posts, [])
        self.assertEqual((self.rows('grants'), self.rows('executions')), (0, 0))
        self.assertEqual(self.queue.held_claims, {})
        response, posts = await self.route(64, queued=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(posts[0][2])['options']['num_predict'], 64)

    async def test_direct_route_refuses_coercible_budgets_before_any_work(self):
        with patch.object(server.prompt_size, 'count', side_effect=AssertionError('sized')):
            for budget in self.COERCIBLE:
                response, posts = await self.route(budget, queued=False)
                with self.subTest(budget=budget):
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(posts, [])
        self.assertEqual(self.manager.models['qwen'].in_flight, 0)
        self.assertEqual(self.rows('executions'), 0)
        response, posts = await self.route(64, queued=False)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(posts[0][2])
        self.assertEqual((payload['think'], payload['options']['num_predict']), ('high', 64))

    async def test_effort_on_uncertainty_is_kept_and_not_redispatched(self):
        posts = []
        def broken(request):
            posts.append(request.content)
            raise httpx.ReadError('synthetic cut')
        backend = OllamaClient(base_url='http://fake')
        backend._http.close()
        backend._http = httpx.Client(base_url='http://fake', transport=httpx.MockTransport(broken))
        backend.num_ctx_for = lambda name: 65536
        async with self.http('uncertain') as http:
            with patch.object(self.manager, 'ollama', backend):
                grant = (await http.post('/v1/inference/grants', json={
                    'model': 'qwen', 'key': 'g', 'lifetime': 'execution'})).json()
                call = await http.post('/v1/inference/grants/%s/executions' % grant['id'], json={
                    'key': 'c', 'request': {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'x'}],
                                            'reasoning_effort': 'high', 'max_tokens': 32}})
                self.assertEqual(call.status_code, 202)
                await asyncio.gather(*list(self.queue.supervisor.workers.values()))
                await asyncio.sleep(.05)
        backend._http.close()
        self.assertEqual(len(posts), 1)
        self.assertEqual(json.loads(posts[0])['think'], 'high')
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT state FROM executions').fetchone()[0], 'uncertain')
        self.assertEqual(self.manager.models['qwen'].in_flight, 1)


if __name__ == '__main__':
    unittest.main()
