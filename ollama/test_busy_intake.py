"""Busy intake across an observer replacement; ASGI/journal, no provider or sockets."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from fastapi import FastAPI
import httpx

from .inference_api import build_router
from .inference_arbiter import Arbiter, Conflict, GrantBusy, StintPolicy
from .inference_consumer import DurableConsumer
from .inference_sdk import _open_intent


class Capture(httpx.AsyncBaseTransport):
    def __init__(self, app):
        self.inner = httpx.ASGITransport(app=app)
        self.requests = []
        self.responses = []
        self.drop_next_intake = False
        self.observed = asyncio.Event()

    async def handle_async_request(self, request):
        self.requests.append((request.method, request.url.path,
                              json.loads(request.content) if request.content else None))
        response = await self.inner.handle_async_request(request)
        self.responses.append((request.method, request.url.path, response.status_code))
        if request.method == 'GET' and '/executions/' in request.url.path:
            self.observed.set()
        if (self.drop_next_intake and request.method == 'POST'
                and request.url.path.endswith('/executions') and response.status_code == 202):
            self.drop_next_intake = False
            await response.aclose()
            raise httpx.ReadError('synthetic lost intake reply', request=request)
        return response


class BusyIntake(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.journal = Arbiter(Path(self.temp.name)/'journal.sqlite', stint_policy=StintPolicy())
        self.pumps = []
        app = FastAPI()

        @app.middleware('http')
        async def identity(request, next_handler):
            request.state.caller = {'username': request.headers.get('x-owner', 'owner')}
            return await next_handler(request)

        app.include_router(build_router(
            SimpleNamespace(journal=self.journal, pump=self.pumps.append),
            lambda model: ('gpu', model), lambda model, body: body))
        self.transport = Capture(app)
        self.http = httpx.AsyncClient(transport=self.transport, base_url='http://fixture')
        self.states = []
        self.consumer = DurableConsumer(self.http, poll_interval=.001, progress=self.states.append)
        self.grant = self.journal.submit('owner', 'session', 'gpu', 'model')
        self.old = self.journal.enqueue(self.grant['id'], 'owner', 'old-call', {'prompt': 'old'})
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.http.aclose()
        self.journal.db.close()
        self.temp.cleanup()

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def until(self, predicate):
        async with asyncio.timeout(2):  # fixture observation bound, never a queue policy
            while not predicate():
                for task in self.tasks:
                    if task.done() and not task.cancelled() and task.exception():
                        raise task.exception()
                await asyncio.sleep(.001)

    def submit(self, call_key='new-call', request=None):
        return self.consumer.submit('model', grant_key='session', call_key=call_key,
                                    request=request or {'prompt': 'new'})

    def start_old(self):
        self.assertEqual(self.journal.dispatch_next('gpu')['execution']['id'], self.old['id'])

    def intakes(self):
        return [row for row in self.transport.requests
                if row[0] == 'POST' and row[1].endswith('/executions')]

    def count_calls(self):
        return self.journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0]

    async def test_replaced_observer_waits_for_exact_old_call_then_submits_once(self):
        self.start_old()
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.assertFalse(opening.done())
        self.assertEqual(self.count_calls(), 1)
        self.assertEqual(len(self.intakes()), 1)
        self.assertEqual(self.pumps, [])
        self.journal.finish(self.old['id'], {'private_old_result': 'not the new answer'})
        job = await opening
        self.assertNotEqual(job.execution_id, self.old['id'])
        self.assertEqual(job.grant_id, self.grant['id'])
        self.assertEqual(self.count_calls(), 2)
        self.assertEqual(len(self.intakes()), 2)
        self.assertEqual(self.intakes()[0], self.intakes()[1])
        self.assertEqual(self.pumps, ['gpu'])
        self.assertIn('waiting_for_predecessor', self.states)

    async def test_queued_and_uncertain_predecessor_do_not_admit_or_dispatch(self):
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain):
                if uncertain:
                    self.start_old()
                    self.journal.uncertain(self.old['id'])
                self.transport.observed.clear()
                opening = self.spawn(self.submit())
                await self.until(lambda: self.transport.observed.is_set())
                await asyncio.sleep(.01)
                self.assertFalse(opening.done())
                self.assertEqual(self.count_calls(), 1)
                self.assertEqual(len(self.intakes()), int(uncertain) + 1)
                opening.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await opening
                self.assertFalse(self.journal.get(self.grant['id'], 'owner')['release_requested'])
                self.assertEqual(self.pumps, [])

    async def test_same_keys_reattach_but_changed_request_is_not_retried(self):
        self.start_old()
        job = await self.submit('old-call', {'prompt': 'old'})
        self.assertEqual(job.execution_id, self.old['id'])
        with self.assertRaises(httpx.HTTPStatusError) as caught:
            await self.submit('old-call', {'prompt': 'changed'})
        self.assertEqual(caught.exception.response.status_code, 409)
        self.assertIsInstance(caught.exception.response.json()['detail'], str)
        self.assertEqual(len(self.intakes()), 2)
        self.assertFalse(self.transport.observed.is_set())
        self.assertEqual(self.count_calls(), 1)

    async def test_busy_is_typed_and_owner_scoped(self):
        self.start_old()
        path = '/v1/inference/grants/'+self.grant['id']+'/executions'
        body = {'key': 'new-call', 'request': {'prompt': 'new'}}
        reply = await self.http.post(path, json=body)
        self.assertEqual(reply.status_code, 409)
        self.assertEqual(reply.json(), {'detail': {'code': 'grant_busy',
            'grant_id': self.grant['id'], 'execution_id': self.old['id']}})
        denied = await self.http.post(path, json=body, headers={'x-owner': 'other'})
        self.assertEqual(denied.status_code, 404)
        self.assertNotIn(self.old['id'], denied.text)

    async def test_release_requested_and_execution_lifetime_do_not_offer_busy_wait(self):
        self.start_old()
        self.journal.request_release(self.grant['id'], 'owner')
        with self.assertRaises(httpx.HTTPStatusError):
            await self.submit()
        one = self.journal.submit('owner', 'one-shot', 'another-gpu', 'model', 'execution')
        self.journal.enqueue(one['id'], 'owner', 'one', {})
        with self.assertRaises(Conflict) as caught:
            self.journal.enqueue(one['id'], 'owner', 'two', {})
        self.assertNotIsInstance(caught.exception, GrantBusy)
        self.assertFalse(self.transport.observed.is_set())
        self.assertEqual(len(self.intakes()), 1)

    async def test_release_while_observing_does_not_revive_grant(self):
        self.start_old()
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.journal.request_release(self.grant['id'], 'owner')
        self.journal.finish(self.old['id'], {})
        with self.assertRaises(httpx.HTTPStatusError):
            await opening
        self.assertEqual(self.journal.get(self.grant['id'], 'owner')['state'], 'released')
        self.assertEqual(self.count_calls(), 1)

    async def test_lost_new_intake_reply_reattaches_without_duplicate(self):
        self.start_old()
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.transport.drop_next_intake = True
        self.journal.finish(self.old['id'], {})
        job = await opening
        self.assertEqual(self.count_calls(), 2)
        self.assertEqual(self.journal.execution(job.execution_id, 'owner')['state'], 'queued')
        self.assertEqual(len(self.intakes()), 3)
        self.assertTrue(all(row == self.intakes()[0] for row in self.intakes()))

    async def test_new_body_is_frozen_before_wait_and_not_old_output(self):
        self.start_old()
        body = {'messages': [{'role': 'user', 'content': 'new'}]}
        opening = self.spawn(self.submit(request=body))
        await self.until(lambda: self.transport.observed.is_set())
        body['messages'][0]['content'] = 'mutated while waiting'
        self.journal.finish(self.old['id'], {'old': 'never substitute as new output'})
        job = await opening
        saved = json.loads(self.journal.execution(job.execution_id, 'owner')['request'])
        self.assertEqual(saved, {'messages': [{'role': 'user', 'content': 'new'}]})
        self.assertEqual(self.intakes()[0], self.intakes()[1])

    async def test_sdk_intake_callback_is_for_new_call_only_after_acceptance(self):
        self.start_old()
        accepted = []
        opening = self.spawn(_open_intent(self.consumer,
            ({'model': 'model', 'stream': True}, 'session', 'new-call', (),
             'work_session', accepted.append)))
        await self.until(lambda: self.transport.observed.is_set())
        self.assertEqual(accepted, [])
        self.journal.finish(self.old['id'], {})
        headers, pages, _ = await opening
        self.assertEqual(len(accepted), 1)
        self.assertNotEqual(accepted[0].execution_id, self.old['id'])
        self.assertEqual(headers['X-Inference-Execution'], accepted[0].execution_id)
        await pages.aclose()

    async def test_cancel_new_observer_then_restart_same_input_changes_no_owned_work(self):
        self.start_old()
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        opening.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await opening
        self.assertEqual(self.journal.execution(self.old['id'], 'owner')['state'], 'running')
        self.assertEqual(self.count_calls(), 1)
        self.transport.observed.clear()
        resumed = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.journal.finish(self.old['id'], {})
        job = await resumed
        self.assertEqual(await self.submit(), job)
        self.assertEqual(self.count_calls(), 2)
        self.assertFalse(any(row[1].endswith(('/release', '/yield'))
                             for row in self.transport.requests))

    async def test_failed_predecessor_does_not_fail_distinct_new_input(self):
        self.start_old()
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.journal.finish(self.old['id'], {'old_error': 'synthetic'}, failed=True)
        job = await opening
        self.assertNotEqual(job.execution_id, self.old['id'])
        self.assertEqual(self.journal.execution(job.execution_id, 'owner')['state'], 'queued')

    async def test_another_new_call_can_win_race_without_overlap_or_replacing_input(self):
        self.start_old()
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.journal.finish(self.old['id'], {})
        competitor = self.journal.enqueue(self.grant['id'], 'owner', 'competing-call', {})
        self.journal.dispatch_next('gpu')
        await self.until(lambda: sum(row[2] == 409 for row in self.transport.responses) == 2)
        self.assertFalse(opening.done())
        self.assertEqual(self.count_calls(), 2)
        self.journal.finish(competitor['id'], {})
        job = await opening
        self.assertNotIn(job.execution_id, (self.old['id'], competitor['id']))
        self.assertEqual(self.count_calls(), 3)
        self.assertEqual(len(self.intakes()), 3)

    async def test_waiting_does_not_yield_or_bank_ahead_of_another_ready_grant(self):
        self.start_old()
        other = self.journal.submit('another-owner', 'other-session', 'gpu', 'model')
        theirs = self.journal.enqueue(other['id'], 'another-owner', 'their-call', {})
        opening = self.spawn(self.submit())
        await self.until(lambda: self.transport.observed.is_set())
        self.journal.finish(self.old['id'], {})
        self.journal.yield_by_key('owner', 'session', 'old-call')
        self.assertEqual(self.journal.dispatch_next('gpu')['execution']['id'], theirs['id'])
        job = await opening
        self.assertEqual(self.journal.execution(job.execution_id, 'owner')['state'], 'queued')
        self.assertIsNone(self.journal.dispatch_next('gpu'))
        self.journal.finish(theirs['id'], {})
        self.journal.yield_by_key('another-owner', 'other-session', 'their-call')
        self.assertEqual(self.journal.dispatch_next('gpu')['execution']['id'], job.execution_id)

    async def test_untyped_or_invalid_busy_replies_are_never_retried(self):
        valid = {'code': 'grant_busy', 'grant_id': self.grant['id'], 'execution_id': self.old['id']}
        details = ['grant cannot accept another execution', {'code': 'stale_lifecycle_event'},
                   None, [], {**valid, 'grant_id': 'foreign'}, {**valid, 'execution_id': '../x'},
                   {**valid, 'execution_id': ''}, {**valid, 'code': 'other_conflict'}]
        for detail in details:
            with self.subTest(detail=detail):
                calls = []
                def respond(request):
                    calls.append(request.url.path)
                    return httpx.Response(202, json={'id': self.grant['id']}) if len(calls) == 1 else \
                           httpx.Response(409, json={'detail': detail})
                async with httpx.AsyncClient(base_url='http://fixture',
                        transport=httpx.MockTransport(respond)) as http:
                    with self.assertRaises(httpx.HTTPStatusError):
                        await DurableConsumer(http).submit('model', grant_key='session',
                                                          call_key='new', request={})
                self.assertEqual(len(calls), 2)

    async def test_missing_foreign_or_unknown_predecessor_receipt_refuses_new_intake(self):
        good = {'id': self.old['id'], 'grant_id': self.grant['id'], 'state': 'complete'}
        for status, receipt in ((404, {'detail': 'not found'}),
                                (200, {**good, 'grant_id': 'foreign'}),
                                (200, {**good, 'id': 'another-call'}),
                                (200, {**good, 'state': 'unknown'})):
            with self.subTest(status=status, receipt=receipt):
                calls = []
                def respond(request):
                    calls.append((request.method, request.url.path))
                    if len(calls) == 1:
                        return httpx.Response(202, json={'id': self.grant['id']})
                    if len(calls) == 2:
                        return httpx.Response(409, json={'detail': {'code': 'grant_busy',
                            'grant_id': self.grant['id'], 'execution_id': self.old['id']}})
                    return httpx.Response(status, json=receipt)
                async with httpx.AsyncClient(base_url='http://fixture',
                        transport=httpx.MockTransport(respond)) as http:
                    with self.assertRaises((ValueError, httpx.HTTPStatusError)):
                        await DurableConsumer(http).submit('model', grant_key='session',
                                                          call_key='new', request={})
                self.assertEqual(len(calls), 3)

    async def test_repeated_busy_for_known_terminal_predecessor_refuses_instead_of_spinning(self):
        calls = []
        def respond(request):
            calls.append((request.method, request.url.path))
            if request.url.path.endswith('/grants'):
                return httpx.Response(202, json={'id': self.grant['id']})
            if request.method == 'POST':
                return httpx.Response(409, json={'detail': {'code': 'grant_busy',
                    'grant_id': self.grant['id'], 'execution_id': self.old['id']}})
            return httpx.Response(200, json={'id': self.old['id'], 'grant_id': self.grant['id'],
                                           'state': 'complete'})
        async with httpx.AsyncClient(base_url='http://fixture',
                transport=httpx.MockTransport(respond)) as http:
            with self.assertRaisesRegex(ValueError, 'already terminal'):
                await DurableConsumer(http).submit('model', grant_key='session',
                                                  call_key='new', request={})
        self.assertEqual(len(calls), 4)


if __name__ == '__main__':
    unittest.main()
