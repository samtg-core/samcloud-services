"""Queue against real handlers/claims; blocked fake provider, no live calls."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import httpx
from starlette.requests import Request

from . import server
from .inference_gateway import GatewayQueue, StreamCapture
from .inference_arbiter import Arbiter, CapacityDeferred
from .manager import Backend, ModelManager
from .test_service_admission import BlockedBackend
from .ollama_client import OllamaRefused

REAL_RESOLVE_MODEL = server._resolve_model


class FakeProvider(BlockedBackend):
    omit_done = False

    def __init__(self):
        super().__init__()
        self.owned_policies = []

    def chat(self, model, messages, **kwargs):
        self.owned_policies.append(kwargs.get('service_owned', False))
        yield from super().chat(model, messages, **kwargs)

    def answer(self):
        return {'message': {'content': 'done'}, 'response': 'done', 'done': not self.omit_done}

    def generate(self, model, prompt, **kwargs):
        yield from self.chat(model, [{'content': prompt}], **kwargs)

    async def chat_stream(self, model, messages, **kwargs):
        self.owned_policies.append(kwargs.get('service_owned', False))
        self.enter(messages)
        yield {'message': {'content': 'first token'}, 'done': False}
        while not self.gate.is_set():
            await asyncio.sleep(.005)
        yield self.answer()


class GatewayLifetime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = FakeProvider()
        self.model = SimpleNamespace(name='qwen', backend=Backend.OLLAMA, in_flight=0,
                                     last_used=0, request_count=0)
        self.manager = object.__new__(ModelManager)
        self.manager.models = {'qwen': self.model}
        self.manager.ollama = self.backend
        async def resolve(name, **kwargs):
            return self.model
        self.patches = [patch.object(server, 'mgr', self.manager),
                        patch.object(server, '_resolve_model', resolve),
                        patch.object(server.prompt_size, 'count', return_value=(10, 'stub'))]
        for p in self.patches:
            p.start()
        self.queue = GatewayQueue(Path(self.temp.name) / 'jobs.sqlite', server, {'qwen': 'gpu'}, lambda _: True)
        self.dispatch = patch.object(server, 'inference_dispatcher', self.queue)
        self.dispatch.start()

    async def asyncTearDown(self):
        self.backend.gate.set()
        await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        self.queue.supervisor.close()
        self.dispatch.stop()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def request(self, owner='creative', key='one'):
        return Request({'type': 'http', 'headers': [(b'idempotency-key', key.encode())],
                        'state': {'caller': {'username': owner}}})

    async def call(self, owner='creative', stream=False, completion=False):
        if completion:
            return await server.completions(server.CompletionRequest(model='qwen', prompt=owner, stream=stream),
                                            self.request(owner))
        return await server.chat_completions(server.ChatRequest(model='qwen', stream=stream,
                           messages=[{'role': 'user', 'content': owner}]), self.request(owner))

    async def wait_for(self, predicate, timeout=2):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(.005)

    async def test_owned_vision_worker_outlives_cancelled_observer_and_keeps_claim(self):
        self.model.backend=Backend.VLM
        self.model.name='vision'
        self.model.port=9999
        text=SimpleNamespace(name='qwen',backend=Backend.OLLAMA,in_flight=0,last_used=0,request_count=0)
        self.manager.models={'vision':self.model,'qwen':text}
        self.queue.models['vision']='gpu'
        async def resolve(name,**kwargs):return self.manager.models[name]
        entered,release=asyncio.Event(),asyncio.Event()
        seen=[]
        async def provider(request):
            seen.append(request.extensions['timeout'])
            entered.set()
            await release.wait()
            return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'vision'},
                'finish_reason':'stop'}],'usage':{'input_tokens':1,'output_tokens':1}})
        real_client=httpx.AsyncClient
        def client(**kwargs):return real_client(transport=httpx.MockTransport(provider),**kwargs)
        with patch.object(server.httpx,'AsyncClient',client),patch.object(server,'_resolve_model',resolve):
            observer=asyncio.create_task(server.chat_completions(server.ChatRequest(model='vision',
                messages=[{'role':'user','content':'synthetic image'}]),self.request()))
            second=None
            try:
                await asyncio.wait_for(entered.wait(),2)
                self.assertEqual(self.model.in_flight,1)
                observer.cancel()
                with self.assertRaises(asyncio.CancelledError):await observer
                self.assertEqual(self.model.in_flight,1)
                self.assertTrue(all(value is None for value in seen[0].values()))
                second=asyncio.create_task(self.call('assistant'))
                await asyncio.sleep(.03)
                self.assertEqual(self.backend.entered,[])
                self.backend.gate.set()
            finally:release.set()
            if second is not None:await asyncio.wait_for(second,2)
            await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        self.assertEqual(self.model.in_flight,0)
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT state FROM executions').fetchone()[0],'complete')

    async def test_vision_transport_or_terminal_ambiguity_keeps_the_claim(self):
        self.model.backend=Backend.VLM
        self.model.port=9999
        real_client=httpx.AsyncClient
        async def provider(request):return httpx.Response(200,json={'choices':[]})
        def client(**kwargs):return real_client(transport=httpx.MockTransport(provider),**kwargs)
        with patch.object(server.httpx,'AsyncClient',client):
            self.queue.enqueue_compat('chat',server.ChatRequest(model='qwen',messages=[]),self.request())
            await asyncio.gather(*list(self.queue.supervisor.workers.values()))
        self.assertEqual(self.model.in_flight,1)
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT state FROM executions').fetchone()[0],'uncertain')

    async def test_cancelled_http_waiter_retains_claim_and_idempotent_result(self):
        first = asyncio.create_task(self.call())
        await self.wait_for(lambda: len(self.backend.entered) == 1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertGreater(self.model.in_flight, 0)
        second = asyncio.create_task(self.call('assistant'))
        await asyncio.sleep(.03)
        self.assertEqual(self.backend.entered, ['creative'])
        self.backend.gate.set()
        await asyncio.wait_for(second, 2)
        replay = await self.call()
        self.assertEqual(replay['choices'][0]['message']['content'], 'done')
        self.assertEqual(self.backend.entered, ['creative', 'assistant'])
        self.assertEqual(self.model.in_flight, 0)

    async def test_unkeyed_compat_request_cannot_create_detached_duplicate_work(self):
        request=Request({'type':'http','headers':[],
                         'state':{'caller':{'username':'creative'}}})
        with self.assertRaises(HTTPException) as caught:
            await server.chat_completions(server.ChatRequest(model='qwen',
                messages=[{'role':'user','content':'same text is not an intent key'}]),request)
        self.assertEqual(caught.exception.status_code,428)
        self.assertEqual(caught.exception.detail['code'],'idempotency_key_required')
        self.assertEqual(self.backend.entered,[])
        self.assertEqual(self.queue.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM grants').fetchone()[0],0)

    async def check_resolver_liveness(self, stage, owned=True):
        entered, release = threading.Event(), threading.Event()
        def blocked(value):
            entered.set()
            if not release.wait(3):
                raise AssertionError('resolver cleanup did not release worker')
            return value
        def load(name):
            result = blocked(self.model) if stage=='load' else self.model
            self.manager.models[name] = result
            return result
        if stage!='ensure':
            self.manager.models.clear()
        fallback = threading.Timer(2, release.set)
        jobs = []
        async def resolve(name, **kwargs):
            return await REAL_RESOLVE_MODEL(name, queue_owned=owned)
        with patch.object(server, '_resolve_model', resolve), \
             patch.object(server, 'match_vlm_model', return_value=None), \
             patch.object(server, 'match_exo_tier', return_value=None), \
             patch.object(self.manager, 'ensure_running', lambda name: blocked(True)), \
             patch.object(self.manager, 'load_ollama_model', load), \
             patch.object(self.backend, 'list_models',
                lambda: blocked([{'name':'qwen'}]) if stage=='list' else [{'name':'qwen'}],
                create=True):
            fallback.start()  # a bad on-loop implementation fails instead of hanging the suite
            try:
                jobs.append(asyncio.create_task(self.call()))
                await self.wait_for(entered.is_set, timeout=3)
                self.assertFalse(release.is_set(), 'resolution returned before the loop could observe it')
                health = await asyncio.wait_for(server.health(), .5)
                self.assertEqual(health['status'], 'ok')
                self.assertFalse(release.is_set())
                if stage == 'ensure' and owned:
                    self.assertGreater(self.model.in_flight, 0, 'residency/reload worker must be claimed')
                jobs.append(asyncio.create_task(self.call('assistant')))
                await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                    'SELECT COUNT(*) FROM grants').fetchone()[0]==2)
                self.assertEqual(self.backend.entered, [])
                self.assertFalse(jobs[0].done())
                release.set()
                await self.wait_for(lambda: len(self.backend.entered)==1)
                self.backend.gate.set()
                await asyncio.wait_for(asyncio.gather(*jobs), 2)
                self.assertEqual(self.backend.entered, ['creative','assistant'])
            finally:
                release.set()
                self.backend.gate.set()
                fallback.cancel()
                for job in jobs:
                    if not job.done():
                        job.cancel()  # observers only; service-owned workers are joined in teardown
                await asyncio.gather(*jobs, return_exceptions=True)

    async def test_resident_health_resolution_does_not_block_queue_or_health(self):
        await self.check_resolver_liveness('ensure')

    async def test_cancelled_residency_observer_keeps_claim_until_reload_worker_ends(self):
        entered, release = threading.Event(), threading.Event()
        def ensure(name):
            entered.set()
            if not release.wait(3):
                raise AssertionError('test must release fake reload worker')
            return True
        with patch.object(self.manager,'ensure_running',ensure):
            task = asyncio.create_task(REAL_RESOLVE_MODEL('qwen',queue_owned=True))
            try:
                await self.wait_for(entered.is_set)
                self.assertEqual(self.model.in_flight,1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(self.model.in_flight,1)
                release.set()
                await self.wait_for(lambda:self.model.in_flight==0)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task,return_exceptions=True)

    async def test_model_list_resolution_does_not_block_queue_or_health(self):
        await self.check_resolver_liveness('list')

    async def test_model_load_resolution_does_not_block_queue_or_health(self):
        await self.check_resolver_liveness('load')

    async def check_cold_capacity_wait(self, refusal, reason):
        self.manager.models.clear()
        loads = []
        ready = False
        def load(name):
            loads.append(name)
            if not ready:
                raise refusal
            self.manager.models[name] = self.model
            return self.model
        with patch.object(server, '_resolve_model', REAL_RESOLVE_MODEL), \
             patch.object(server, 'match_vlm_model', return_value=None), \
             patch.object(server, 'match_exo_tier', return_value=None), \
             patch.object(self.backend, 'list_models', return_value=[{'name':'qwen'}], create=True), \
             patch.object(self.manager, 'load_ollama_model', load):
            observer = asyncio.create_task(self.call())
            try:
                await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                    'SELECT COUNT(*) FROM executions WHERE wait_reason=?', (reason,)).fetchone()[0] == 1)
                row = self.queue.supervisor.journal.db.execute('SELECT * FROM executions').fetchone()
                execution_id = row['id']
                self.assertEqual(row['state'], 'queued')
                self.assertFalse(observer.done())
                self.assertEqual(self.backend.entered, [])
                self.assertEqual(self.model.in_flight, 0)
                self.queue.supervisor.pump('gpu')
                await asyncio.sleep(.01)
                self.assertEqual(loads, ['qwen'], 'caller/clock must not retry a known capacity wait')
                ready = True
                self.queue.supervisor.capacity_available('gpu')
                await self.wait_for(lambda: self.backend.entered == ['creative'])
                self.backend.gate.set()
                await asyncio.wait_for(observer, 2)
                self.assertEqual(loads, ['qwen', 'qwen'])
                rows = self.queue.supervisor.journal.db.execute('SELECT * FROM executions').fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['id'], execution_id)
                self.assertEqual(rows[0]['state'], 'complete')
            finally:
                if not observer.done():
                    observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)

    async def test_cold_fit_refusal_waits_and_fresh_capacity_runs_same_execution(self):
        await self.check_cold_capacity_wait(server.capacity.InsufficientCapacity('synthetic fit'),
                                            'capacity_unavailable')

    async def test_cold_work_mode_refusal_waits_without_marking_provider_uncertain(self):
        await self.check_cold_capacity_wait(server.capacity.DeviceInUse('synthetic work mode'),
                                            'work_mode')

    async def test_cold_reservation_conflict_reaches_saved_capacity_wait(self):
        await self.check_cold_capacity_wait(CapacityDeferred('capacity_unavailable'),
                                            'capacity_unavailable')

    async def test_legacy_cold_fit_refusal_keeps_http_capacity_answer(self):
        self.manager.models.clear()
        with patch.object(server, 'match_vlm_model', return_value=None), \
             patch.object(server, 'match_exo_tier', return_value=None), \
             patch.object(self.backend, 'list_models', return_value=[{'name':'qwen'}], create=True), \
             patch.object(self.manager, 'load_ollama_model',
                          side_effect=server.capacity.InsufficientCapacity('synthetic fit')):
            with self.assertRaises(HTTPException) as caught:
                await REAL_RESOLVE_MODEL('qwen')
            self.assertEqual(caught.exception.status_code, 503)

    async def test_resolution_liveness_detector_rejects_legacy_on_loop_control(self):
        with self.assertRaisesRegex(AssertionError, 'resolution returned before'):
            await self.check_resolver_liveness('ensure', owned=False)

    async def test_cancelled_scheduler_keeps_capacity_while_load_thread_survives(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        loads = []
        self.manager.models.clear()
        def load(name):
            loads.append(name)
            entered.set()
            if not release.wait(3):
                raise AssertionError('load cleanup did not release worker')
            self.manager.models[name] = self.model
            finished.set()
            return self.model
        jobs = []
        with patch.object(server, '_resolve_model', REAL_RESOLVE_MODEL), \
             patch.object(server, 'match_vlm_model', return_value=None), \
             patch.object(server, 'match_exo_tier', return_value=None), \
             patch.object(self.manager, 'load_ollama_model', load), \
             patch.object(self.backend, 'list_models', return_value=[{'name':'qwen'}], create=True):
            try:
                jobs.append(asyncio.create_task(self.call()))
                await self.wait_for(entered.is_set)
                execution, worker = next(iter(self.queue.supervisor.workers.items()))
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                with self.assertRaises(HTTPException) as caught:
                    await jobs[0]
                self.assertEqual(caught.exception.detail['state'], 'uncertain')
                self.assertFalse(finished.is_set())
                jobs.append(asyncio.create_task(self.call('assistant')))
                await asyncio.sleep(.03)
                self.assertEqual(loads, ['qwen'])
                self.assertIsNone(self.queue.supervisor.pump('gpu'))
                release.set()
                await self.wait_for(finished.is_set)
                await asyncio.sleep(.03)
                self.assertIs(self.manager.models['qwen'], self.model)
                self.assertEqual(self.backend.entered, [])
                self.assertEqual(self.model.in_flight, 0)  # no generation was started
                self.assertIsNone(self.queue.supervisor.pump('gpu'))
                self.assertFalse(jobs[1].done())
                self.assertEqual(self.queue.supervisor.journal.execution(execution,'creative')['state'],
                                 'uncertain')
            finally:
                release.set()
                for job in jobs:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)

    async def test_all_route_modes_select_owned_policy_after_admission(self):
        self.backend.gate.set()
        for index, (stream, completion) in enumerate(((False, False), (True, False),
                                                     (False, True), (True, True))):
            response = await self.call('owned-' + str(index), stream=stream, completion=completion)
            if stream:
                _ = [f async for f in response.body_iterator]
        self.assertEqual(self.backend.owned_policies, [True] * 4)
        self.assertEqual(self.model.in_flight, 0)

    async def test_unarbitrated_request_body_cannot_select_owned_lifetime(self):
        self.backend.gate.set()
        req = server.ChatRequest(model='qwen', messages=[{'role':'user','content':'legacy'}],
                                 service_owned=True, queue_owned=True)
        with patch.object(server, 'inference_dispatcher', None):
            await server.chat_completions(req, self.request())
        self.assertEqual(self.backend.owned_policies, [False])

    async def test_scheduler_cancellation_quarantines_while_thread_still_owns_compute(self):
        first = asyncio.create_task(self.call())
        await self.wait_for(lambda: self.backend.entered == ['creative'])
        execution_id, worker = next(iter(self.queue.supervisor.workers.items()))
        worker.cancel()
        await worker
        with self.assertRaises(HTTPException) as caught:
            await first
        self.assertEqual(caught.exception.detail['state'], 'uncertain')
        self.assertEqual(self.model.in_flight, 2)  # durable owner plus live thread
        self.assertEqual(self.queue.supervisor.journal.execution(execution_id, 'creative')['state'], 'uncertain')
        self.assertIsNone(self.queue.supervisor.pump('gpu'))
        self.backend.gate.set()
        await self.wait_for(lambda: self.model.in_flight == 1)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))  # no automatic release on thread exit

    async def test_owned_stream_transport_loss_is_unknown_without_saved_error_text(self):
        async def broken(model, messages, **kwargs):
            self.backend.enter(messages)
            yield {'message': {'content': 'partial'}, 'done': False}
            raise httpx.ReadError('PRIVATE backend exception')
        with patch.object(self.backend, 'chat_stream', broken):
            response = await self.call(stream=True)
            execution = response.headers['X-Inference-Execution']
            await self.wait_for(lambda: self.queue.supervisor.journal.execution(
                execution, 'creative')['state'] == 'uncertain')
            frames = ''
            async with asyncio.timeout(2):
                while 'model-service uncertain' not in frames:
                    frames += await anext(response.body_iterator)
            await response.body_iterator.aclose()  # observer only, not a backend cancellation
        execution = response.headers['X-Inference-Execution']
        row = self.queue.supervisor.journal.execution(execution, 'creative')
        page = self.queue.supervisor.journal.events(execution, 'creative')
        self.assertEqual(row['state'], 'uncertain')
        self.assertIsNone(row['result'])
        self.assertIn('partial', frames)
        self.assertNotIn('inference_uncertain', frames)
        self.assertNotIn('[DONE]', frames)
        self.assertNotIn('PRIVATE', frames + json.dumps(page))
        self.assertEqual(self.model.in_flight, 1)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_confirmed_refusal_releases_all_route_modes_without_reexecution(self):
        refused = []
        def sync_refusal(model, *args, **kwargs):
            refused.append(model)
            raise OllamaRefused(400)
            yield  # preserve the provider's generator shape
        async def async_refusal(model, *args, **kwargs):
            refused.append(model)
            raise OllamaRefused(404)
            yield
        for index, (stream, completion) in enumerate(((False, False), (True, False),
                                                     (False, True), (True, True))):
            name = 'refusal-' + str(index)
            with self.subTest(stream=stream, completion=completion), \
                    patch.object(self.backend, 'chat', sync_refusal), \
                    patch.object(self.backend, 'generate', sync_refusal), \
                    patch.object(self.backend, 'chat_stream', async_refusal):
                for replay in (False, True):
                    if stream:
                        response = await self.call(name, stream=True, completion=completion)
                        frames = ''.join([f async for f in response.body_iterator])
                        self.assertIn('inference_failed', frames)
                    else:
                        with self.assertRaises(HTTPException) as error:
                            await self.call(name, completion=completion)
                        self.assertEqual(error.exception.detail['state'], 'failed')
                    self.assertEqual(len(refused), index + 1)
                self.assertEqual(self.model.in_flight, 0)
                self.assertEqual(self.queue.held_claims, {})
            # Actual next work can acquire the resource; refusal isn't quarantine.
            self.backend.gate.set()
            await self.call('next-' + str(index))

    async def test_generic_http_exception_during_generation_stays_uncertain(self):
        def unqualified(*args, **kwargs):
            raise HTTPException(400, 'PRIVATE not a typed backend refusal')
            yield
        with patch.object(self.backend, 'chat', unqualified):
            with self.assertRaises(HTTPException) as error:
                await self.call()
        self.assertEqual(error.exception.detail['state'], 'uncertain')
        self.assertNotIn('PRIVATE', str(error.exception.detail))
        self.assertGreater(self.model.in_flight, 0)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_legacy_stream_reports_sanitized_refusal_without_queue_mode(self):
        async def refusal(*args, **kwargs):
            raise OllamaRefused(400)
            yield
        with patch.object(self.backend, 'chat_stream', refusal):
            response = await server._chat_completions_direct(server.ChatRequest(
                model='qwen', messages=[{'role':'user', 'content':'PRIVATE prompt'}], stream=True),
                resolved=self.model)
            frames = ''.join([f async for f in response.body_iterator])
        self.assertIn('provider_refused', frames)
        self.assertIn('[DONE]', frames)
        self.assertNotIn('PRIVATE', frames)
        self.assertEqual(self.model.in_flight, 0)

    async def test_closing_stream_observer_does_not_release_worker(self):
        response = await self.call(stream=True)
        execution = response.headers['X-Inference-Execution']
        await anext(response.body_iterator)
        await self.wait_for(lambda: len(self.backend.entered) == 1)
        await response.body_iterator.aclose()
        self.assertGreater(self.model.in_flight, 0)
        self.backend.gate.set()
        result = await self.queue.supervisor.observe(execution, 'creative')
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(self.model.in_flight, 0)

    async def test_partial_output_precedes_completion_and_reconnect_does_not_redispatch(self):
        response = await self.call(stream=True)
        execution = response.headers['X-Inference-Execution']
        async with asyncio.timeout(2):
            while 'first token' not in await anext(response.body_iterator):
                pass
        self.assertFalse(self.backend.gate.is_set())
        page = self.queue.supervisor.journal.events(execution, 'creative')
        self.assertEqual(page['state'], 'running')
        self.assertEqual(len(page['events']), 1)
        cursor = page['next_cursor']
        await response.body_iterator.aclose()
        self.backend.gate.set()
        await self.queue.supervisor.observe(execution, 'creative')
        remainder = self.queue.supervisor.journal.events(execution, 'creative', after=cursor)
        self.assertEqual(remainder['state'], 'complete')
        self.assertNotIn('first token', ''.join(e['data'] for e in remainder['events']))
        self.assertIn('[DONE]', ''.join(e['data'] for e in remainder['events']))
        replay = await self.call(stream=True)
        self.assertEqual(replay.headers['X-Inference-Execution'], execution)
        replayed = ''.join([frame async for frame in replay.body_iterator])
        self.assertEqual(replayed.count('first token'), 1)
        self.assertEqual(self.backend.entered, ['creative'])

    async def test_nonstreaming_missing_terminal_keeps_claim_uncertain(self):
        self.backend.omit_done = True
        self.backend.gate.set()
        with self.assertRaises(HTTPException) as error:
            await self.call()
        self.assertEqual(error.exception.detail['state'], 'uncertain')
        self.assertGreater(self.model.in_flight, 0)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_streaming_missing_terminal_keeps_claim_uncertain(self):
        self.backend.omit_done = True
        self.backend.gate.set()
        response = await self.call(stream=True)
        await self.wait_for(lambda: self.queue.supervisor.journal.execution(
            response.headers['X-Inference-Execution'], 'creative')['state'] == 'uncertain')
        frames = []
        async def collect():
            async for frame in response.body_iterator:
                frames.append(frame)
        observer = asyncio.create_task(collect())
        try:
            await self.wait_for(lambda: any('uncertain' in frame for frame in frames))
            self.assertFalse(observer.done(), 'unknown backend state must not end the stream')
            self.assertNotIn('[DONE]', ''.join(frames))
            self.assertNotIn('inference_uncertain', ''.join(frames))
            self.assertGreater(self.model.in_flight, 0)
            self.assertIsNone(self.queue.supervisor.pump('gpu'))
            execution = response.headers['x-inference-execution']
            # Explicit operator reconciliation, after this fake backend ended.
            # Silence or an observer timeout cannot perform this transition.
            self.queue.supervisor.journal.finish(execution, {'kind':'stream'})
            self.manager.release(self.queue.held_claims.pop(execution))
            await asyncio.wait_for(observer, 2)
        finally:
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)

    async def test_completion_route_uses_same_runner_as_chat(self):
        first = asyncio.create_task(self.call())
        await self.wait_for(lambda: len(self.backend.entered) == 1)
        second = asyncio.create_task(self.call('assistant', completion=True))
        await asyncio.sleep(.03)
        self.assertEqual(self.backend.entered, ['creative'])
        self.backend.gate.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        self.assertEqual(self.backend.entered, ['creative', 'assistant'])

    async def test_unresolved_load_does_not_release_capacity_as_not_found(self):
        async def unresolved(name, **kwargs):
            return None
        with patch.object(server, '_resolve_model', unresolved):
            with self.assertRaises(HTTPException) as error:
                await self.call()
        self.assertEqual(error.exception.detail['state'], 'uncertain')
        self.assertEqual(self.backend.entered, [])
        self.assertIsNone(self.queue.supervisor.pump('gpu'))
        second = asyncio.create_task(self.call('assistant'))
        try:
            await asyncio.sleep(.03)
            self.assertFalse(second.done())
            self.assertEqual(self.backend.entered, [])
        finally:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)

    async def test_resolution_http_error_is_not_proof_of_ended_load(self):
        async def unresolved(name, **kwargs):
            raise HTTPException(404, 'PRIVATE provider failure')
        with patch.object(server, '_resolve_model', unresolved):
            with self.assertRaises(HTTPException) as error:
                await self.call()
        self.assertEqual(error.exception.detail['state'], 'uncertain')
        self.assertNotIn('PRIVATE',str(error.exception.detail))
        self.assertEqual(self.backend.entered, [])
        self.assertIsNone(self.queue.supervisor.pump('gpu'))

    async def test_missing_residency_claim_never_dispatches_untracked_work(self):
        with patch.object(self.manager, 'claim', return_value=None):
            with self.assertRaises(HTTPException) as error:
                await self.call()
        self.assertEqual(error.exception.detail['state'], 'uncertain')
        self.assertEqual(self.backend.entered, [])
        self.assertEqual(self.queue.held_claims, {})
        self.assertEqual(self.model.in_flight, 0)
        self.assertIsNone(self.queue.supervisor.pump('gpu'))
        self.assertEqual(self.model.in_flight, 0)

    async def test_streaming_completion_uses_same_runner_and_claim(self):
        first = asyncio.create_task(self.call())
        await self.wait_for(lambda: len(self.backend.entered) == 1)
        response = await self.call('assistant', stream=True, completion=True)
        await asyncio.sleep(.03)
        self.assertEqual(self.backend.entered, ['creative'])
        self.backend.gate.set()
        frames = ''.join([frame async for frame in response.body_iterator])
        await first
        self.assertIn('"done": true', frames)
        self.assertEqual(self.backend.entered, ['creative', 'assistant'])
        self.assertEqual(self.model.in_flight, 0)


class CaptureDurability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'journal.sqlite'
        self.a = Arbiter(self.path)
        grant = self.a.submit('creative', 'turn', 'gpu', 'qwen')
        self.a.enqueue(grant['id'], 'creative', 'call', {})
        self.execution = self.a.dispatch_next('gpu')['execution']['id']

    def tearDown(self):
        self.a.close()
        self.temp.cleanup()

    def test_split_utf8_sse_and_terminal_frames_are_durable_but_not_premature(self):
        capture = StreamCapture(self.a, self.execution, 'text/event-stream')
        first = 'data: ' + json.dumps({'choices': [{'delta': {'content': 'café'},
                                                  'finish_reason': None}]}, ensure_ascii=False) + '\r\n\r\n'
        finish = 'data: {"choices":[{"finish_reason":"stop"}]}\r\n\r\n'
        for byte in (first + finish + 'data: [DONE]\r\n\r\n').encode():
            capture.feed(bytes([byte]))
        capture.finish()
        page = self.a.events(self.execution, 'creative')
        self.assertEqual(len(page['events']), 1)
        self.assertIn('café', page['events'][0]['data'])
        cursor = page['next_cursor']
        self.a.finish(self.execution, {'kind': 'stream'})
        self.a.close()
        self.a = Arbiter(self.path)
        page = self.a.events(self.execution, 'creative', after=cursor, limit=1)
        self.assertTrue(page['has_more'])
        self.assertIn('stop', page['events'][0]['data'])
        page = self.a.events(self.execution, 'creative', after=page['next_cursor'])
        self.assertFalse(page['has_more'])
        self.assertEqual(page['events'][0]['data'], 'data: [DONE]\n\n')

    def test_error_after_finish_does_not_publish_success(self):
        capture = StreamCapture(self.a, self.execution, 'text/event-stream')
        capture.feed('data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        with self.assertRaises(RuntimeError):
            capture.feed('data: {"error":{"message":"private provider exception"}}\n\n')
        self.a.uncertain(self.execution)
        self.assertEqual(self.a.events(self.execution, 'creative')['events'], [])
        self.assertIsNone(self.a.dispatch_next('gpu'))

    def test_unterminated_sse_and_done_without_finish_are_not_success(self):
        for body in ('data: {"choices":[{"finish_reason":"stop"}]}', 'data: [DONE]\n\n'):
            capture = StreamCapture(self.a, self.execution, 'text/event-stream')
            capture.feed(body)
            with self.assertRaises(RuntimeError):
                capture.finish()

    def test_ndjson_terminal_without_trailing_newline(self):
        capture = StreamCapture(self.a, self.execution, 'application/x-ndjson')
        capture.feed('{"response":"hello","done":false}\n{"done":true}')
        capture.finish()
        self.assertEqual(len(self.a.events(self.execution, 'creative')['events']), 1)
        self.a.finish(self.execution, {})
        self.assertEqual(len(self.a.events(self.execution, 'creative')['events']), 2)


if __name__ == '__main__':
    unittest.main()
