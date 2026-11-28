"""Durable, scoped lifecycle metadata; no live client/provider or credentials."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace

import httpx

from .hermes_queue_identity import HostBindings
from .inference_consumer import DurableConsumer
from .inference_events import relay_once, source_call
from .inference_identity import IdentityStore
from .inference_sdk import QueuedClient


class LifecycleOutbox(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'identity.sqlite'
        self.host = HostBindings(self.path, 'own-profile')

    async def asyncTearDown(self):
        self.temp.cleanup()

    def turn(self, session='session', turn='turn', row=1, api='', tools=('tool',)):
        self.host.pre_api_request(session_id=session, turn_id=turn,
            api_request_id=api,
            conversation_history=[{'role': 'user', '_row_id': row, 'content': 'private text'}])
        keys = self.host.main_stint_keys({'model': 'qwen', 'messages': ['private text']})
        # This fixture models a successfully accepted API call before its response.
        keys.on_accepted(SimpleNamespace(grant_id='fixture-grant', execution_id=keys.call_key))
        self.host.post_api_request()
        if api:
            self.host.complete_api(session,turn,api,tools)
        return keys

    async def test_refused_attempt_can_recover_same_api_but_cannot_yield_before_success(self):
        self.host.pre_api_request(session_id='session',turn_id='turn',api_request_id='api',
            conversation_history=[{'role':'user','_row_id':1,'content':'synthetic'}])
        first = self.host.main_stint_keys({'model':'qwen'})
        self.host.post_api_request()
        self.assertEqual(self.host.dependency_wait('session','turn','api','tool'),0)
        final = self.turn(row=2,api='api')
        self.assertNotEqual(first.call_key,final.call_key)
        self.assertEqual(self.host.dependency_wait('session','turn','api','tool'),1)
        self.assertEqual(self.host.pending_events()[0]['expected_call_key'],final.call_key)

    async def test_dependency_uses_originating_api_not_latest_call_or_sibling(self):
        first = self.turn(api='api-one')
        latest = self.turn(row=2,api='api-two')
        self.turn(session='sibling',api='api-one')
        self.assertEqual(self.host.dependency_wait('session','turn','unknown-api','tool'),0)
        self.assertEqual(self.host.dependency_wait('session','other-turn','api-one','tool'),0)
        self.assertEqual(self.host.pending_events(),[])
        self.assertEqual(self.host.dependency_wait('session','turn','api-one','tool'),1)
        event = self.host.pending_events()[0]
        self.assertEqual(event['grant_key'],first.grant_key)
        self.assertEqual(event['expected_call_key'],first.call_key)
        self.assertNotEqual(event['expected_call_key'],latest.call_key)

    async def test_dependency_replay_reopen_and_later_native_finish_are_independent(self):
        first = self.turn(api='api-one',tools=('tool-one',))
        self.host.dependency_wait('session','turn','api-one','tool-one')
        original = self.host.pending_events()[0]
        reopened = HostBindings(self.path,'own-profile')
        reopened.dependency_wait('session','turn','api-one','tool-one')
        self.assertEqual(reopened.pending_events(),[original])
        latest = self.turn(row=2,api='api-two',tools=('tool-two',))
        self.host.dependency_wait('session','turn','api-two','tool-two')
        self.host.finish_turn('session','turn')
        events = self.host.pending_events()
        self.assertEqual(len(events),3)
        self.assertEqual([event['expected_call_key'] for event in events],
                         [first.call_key,latest.call_key,latest.call_key])
        self.assertNotIn(b'private text',self.path.read_bytes())

    async def test_returned_tool_identity_cannot_be_rebound_to_a_later_call(self):
        from .inference_arbiter import Conflict
        first = self.turn(api='one-api')
        with self.assertRaises(Conflict):
            self.turn(row=2,api='one-api')
        self.host.dependency_wait('session','turn','one-api','tool')
        self.assertEqual(self.host.pending_events()[0]['expected_call_key'],first.call_key)

    async def test_reused_api_counter_keeps_each_successful_tools_own_call(self):
        first = self.turn(api='api',tools=('first-tool',))
        later = self.turn(row=2,api='api',tools=('later-tool',))
        self.assertEqual(self.host.dependency_wait('session','turn','api','invented-tool'),0)
        self.host.dependency_wait('session','turn','api','first-tool')
        self.host.dependency_wait('session','turn','api','later-tool')
        self.assertEqual([e['expected_call_key'] for e in self.host.pending_events()],
                         [first.call_key,later.call_key])

    async def test_dependency_without_ids_never_guesses_current_turn(self):
        self.turn(api='api')
        for ids in (('session','turn','','tool'),('session','turn','api','')):
            with self.assertRaises(ValueError):
                self.host.dependency_wait(*ids)
        self.assertEqual(self.host.pending_events(),[])

    async def test_dependency_relay_hands_off_compute_without_finishing_native_turn(self):
        from fastapi import FastAPI
        from .inference_api import build_router
        from .inference_arbiter import StintPolicy
        from .inference_supervisor import Supervisor
        entered = []
        release = asyncio.Event()
        async def execute(grant, body):
            entered.append(grant['owner'])
            if grant['owner'] == 'assistant':
                await release.wait()
            return {'answer': 'synthetic'}
        supervisor = Supervisor(Path(self.temp.name)/'service.sqlite', execute,
                                lambda _: True, stint_policy=StintPolicy())
        app = FastAPI()
        @app.middleware('http')
        async def verified_test_owner(request, call_next):
            request.state.caller = {'username': 'creative'}
            return await call_next(request)
        app.include_router(build_router(supervisor,lambda model: ('gpu',model),
                                        lambda model,body: body))
        keys = self.turn(api='origin',tools=('build',))
        first = supervisor.journal.submit('creative',keys.grant_key,'gpu','qwen')
        call = supervisor.journal.enqueue(first['id'],'creative',keys.call_key,{})
        supervisor.pump('gpu')
        await asyncio.gather(*list(supervisor.workers.values()))
        other = supervisor.journal.submit('assistant','other','gpu','qwen')
        supervisor.journal.enqueue(other['id'],'assistant','other-call',{})
        supervisor.pump('gpu')
        self.assertEqual(entered,['creative'])
        try:
            self.host.dependency_wait('session','turn','origin','build')
            async with httpx.AsyncClient(base_url='http://test',
                    transport=httpx.ASGITransport(app=app)) as http:
                self.assertEqual(await relay_once(DurableConsumer(http),self.host),1)
            async with asyncio.timeout(2):
                while entered != ['creative','assistant']:
                    await asyncio.sleep(0)
            self.assertEqual(self.host.pending_events(),[])
            self.assertEqual(supervisor.journal.execution(call['id'],'creative')['state'],'complete')
            # Native turn remains open: its later API call uses the same grant,
            # waits behind the active assistant, then returns after its handoff.
            later = self.turn(row=2,api='after-build')
            self.assertEqual(later.grant_key,keys.grant_key)
            pending = supervisor.journal.enqueue(first['id'],'creative',later.call_key,{})
            supervisor.pump('gpu')
            self.assertEqual(supervisor.journal.execution(pending['id'],'creative')['state'],'queued')
            release.set()
            await asyncio.gather(*list(supervisor.workers.values()))
            supervisor.yield_by_key('assistant','other','other-call')
            await asyncio.gather(*list(supervisor.workers.values()))
            self.assertEqual(entered,['creative','assistant','creative'])
            self.assertEqual(supervisor.journal.execution(pending['id'],'creative')['state'],'complete')
            self.host.finish_turn('session','turn')
            self.assertEqual(self.host.pending_events()[0]['expected_call_key'],later.call_key)
        finally:
            release.set()
            await asyncio.gather(*list(supervisor.workers.values()))
            supervisor.close()

    def http(self, handler):
        return httpx.AsyncClient(base_url='https://synthetic.invalid', transport=httpx.MockTransport(handler))

    @staticmethod
    def capabilities():
        return httpx.Response(200, json={'stints': {'lifecycle_yield': True}})

    async def test_exact_native_turn_and_reopen_preserve_pending_event_without_text(self):
        first = self.turn()
        sibling = self.turn('sibling', 'turn')
        self.assertEqual(self.host.finish_turn('session', 'other-turn'), 0)
        self.assertEqual(self.host.pending_events(), [])
        self.assertEqual(self.host.finish_turn('session', 'turn'), 1)
        self.host.finish_turn('session', 'turn')
        reopened = HostBindings(self.path, 'own-profile')
        events = reopened.pending_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['grant_key'], first.grant_key)
        self.assertEqual(events[0]['expected_call_key'], first.call_key)
        self.assertNotEqual(events[0]['grant_key'], sibling.grant_key)
        self.assertNotIn(b'private text', self.path.read_bytes())

    async def test_latest_call_within_exact_turn_is_used_not_later_turn(self):
        first = self.turn()
        latest = self.turn(row=2)
        later = self.turn(turn='later-turn', row=3)
        self.host.finish_turn('session', 'turn')
        event = self.host.pending_events()[0]
        self.assertNotEqual(first.call_key, latest.call_key)
        self.assertEqual(event['expected_call_key'], latest.call_key)
        self.assertNotEqual(event['expected_call_key'], later.call_key)

    async def test_unsubmitted_or_other_scope_keys_cannot_bind_a_native_turn(self):
        keys = self.turn()
        store = IdentityStore(self.path)
        try:
            with self.assertRaises(ValueError):
                store.record_turn_call('scope', '', keys)
            from .inference_arbiter import Conflict
            with self.assertRaises(Conflict):
                store.record_turn_call('other-scope', 'turn', keys)
        finally:
            store.close()

    async def test_transport_failure_and_missing_grant_retain_then_same_event_is_accepted(self):
        keys = self.turn()
        self.host.finish_turn('session', 'turn')
        event = self.host.pending_events()[0]
        attempts = []
        mode = 'timeout'
        def handle(request):
            if request.url.path.endswith('/capabilities'):
                return self.capabilities()
            attempts.append(request.content)
            if mode == 'timeout':
                raise httpx.ReadTimeout('synthetic', request=request)
            if mode == 'missing':
                return httpx.Response(404)
            return httpx.Response(200, json={'client_key': keys.grant_key, 'state': 'queued'})
        async with self.http(handle) as http:
            consumer = DurableConsumer(http)
            self.assertEqual(await relay_once(consumer, self.host), -1)
            mode = 'missing'
            self.assertEqual(await relay_once(consumer, self.host), -1)
            self.assertEqual(self.host.pending_events(), [event])
            mode = 'accepted'
            self.assertEqual(await relay_once(consumer, self.host), 1)
            self.assertEqual(self.host.pending_events(), [])
        self.assertEqual(len(set(attempts)), 1)

    async def test_only_typed_stale_event_is_acknowledged_not_arbitrary_conflict(self):
        self.turn()
        self.host.finish_turn('session', 'turn')
        typed = False
        def handle(request):
            if request.url.path.endswith('/capabilities'):
                return self.capabilities()
            return httpx.Response(409, json={'detail': {'code': 'stale_lifecycle_event'}
                                           if typed else 'some other conflict'})
        async with self.http(handle) as http:
            consumer = DurableConsumer(http)
            self.assertEqual(await relay_once(consumer, self.host), -1)
            self.assertEqual(len(self.host.pending_events()), 1)
            typed = True
            self.assertEqual(await relay_once(consumer, self.host), 1)
            self.assertEqual(self.host.pending_events(), [])

    async def test_missing_capability_and_wrong_success_body_do_not_acknowledge(self):
        self.turn()
        self.host.finish_turn('session', 'turn')
        for capable in (False, True):
            writes = []
            def handle(request):
                if request.url.path.endswith('/capabilities'):
                    return self.capabilities() if capable else httpx.Response(200, json={})
                writes.append(request)
                return httpx.Response(200, json={'status': 'ok'})
            async with self.http(handle) as http:
                self.assertEqual(await relay_once(DurableConsumer(http), self.host), -1)
            self.assertEqual(len(writes), int(capable))
            self.assertEqual(len(self.host.pending_events()), 1)

    async def test_malformed_capability_is_unknown_not_an_acknowledgement(self):
        self.turn()
        self.host.finish_turn('session', 'turn')
        writes = []
        def handle(request):
            writes.append(request.method)
            return httpx.Response(200, json=[])
        async with self.http(handle) as http:
            self.assertEqual(await relay_once(DurableConsumer(http), self.host), -1)
        self.assertEqual(writes, ['GET'])
        self.assertEqual(len(self.host.pending_events()), 1)

    async def test_cancelled_metadata_read_drains_actual_sqlite_worker(self):
        entered, release = threading.Event(), threading.Event()
        def operation():
            entered.set()
            release.wait(3)
            return 1
        task = asyncio.create_task(source_call(operation))
        while not entered.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_old_unacknowledged_page_cannot_starve_newer_turn_events(self):
        for n in range(35):
            self.turn(turn='turn-'+str(n), row=n+1)
            self.host.finish_turn('session', 'turn-'+str(n))
        last = self.host.pending_events(after=32)[-1]['expected_call_key']
        def handle(request):
            if request.url.path.endswith('/capabilities'):
                return self.capabilities()
            body = json.loads(request.content)
            if body['expected_call_key'] == last:
                return httpx.Response(200, json={'client_key': body['grant_key'], 'state': 'queued'})
            return httpx.Response(404)
        cursor = [0]
        async with self.http(handle) as http:
            consumer = DurableConsumer(http)
            self.assertEqual(await relay_once(consumer, self.host, cursor=cursor), -1)
            self.assertEqual(await relay_once(consumer, self.host, cursor=cursor), 1)
        self.assertEqual(len(self.host.pending_events()), 32)  # first page still retained

    async def test_sdk_relay_delivers_persisted_event_without_an_inference_request(self):
        keys = self.turn()
        self.host.finish_turn('session', 'turn')
        delivered = threading.Event()
        def handle(request):
            if request.url.path.endswith('/capabilities'):
                return self.capabilities()
            self.assertEqual(request.url.path, '/v1/inference/yield')
            delivered.set()
            return httpx.Response(200, json={'client_key': keys.grant_key, 'state': 'queued'})
        client = QueuedClient(api_key='synthetic', base_url='https://synthetic.invalid/v1',
            http_factory=lambda: self.http(handle), event_source=self.host)
        try:
            self.assertTrue(await asyncio.to_thread(delivered.wait, 3))
            async with asyncio.timeout(3):
                while self.host.pending_events():
                    await asyncio.sleep(.001)
        finally:
            await client.aclose()
        self.assertEqual(self.host.pending_events(), [])
