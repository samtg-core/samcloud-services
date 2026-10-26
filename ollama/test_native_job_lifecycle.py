"""Durable native-attempt recovery and lifecycle delivery, synthetic HTTP only."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import httpx

from .hermes_queue_identity import HostBindings
from .inference_consumer import DurableConsumer
from .inference_events import relay_once
from .inference_identity import IdentityStore, LocalSourceRefused


class NativeParentRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'identity.sqlite'
        self.store = IdentityStore(self.path)
        self.body = {'model': 'qwen', 'messages': ['synthetic']}

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_recoverable_attempt_resumes_parent_after_process_reopen(self):
        original = self.store.begin_native_job('scope', 'original')
        first = self.store.bind('scope', 'main', 'one', self.body, job_turn='original')
        self.store.observe_native_outcome('scope', 'original', completed=False, failed=True, interrupted=False)
        self.assertEqual(self.store.pending_events(), [])
        self.store.close()
        self.store = IdentityStore(self.path)
        self.assertEqual(self.store.begin_native_job('scope', 'retry'), original)
        resumed = self.store.bind('scope', 'main', 'two', self.body, job_turn='retry')
        self.assertEqual(first.job_key, resumed.job_key)
        self.assertNotEqual(first.call_key, resumed.call_key)
        self.assertEqual(first, self.store.bind('scope', 'main', 'one', self.body, job_turn='retry'))

    def test_interruption_and_incomplete_budget_are_not_completion(self):
        key = self.store.begin_native_job('scope', 'turn')
        for failed, interrupted in ((False, True), (False, False), (True, False)):
            self.store.observe_native_outcome('scope', 'turn', completed=False,
                                             failed=failed, interrupted=interrupted)
            self.assertEqual(self.store.pending_events(), [])
            self.assertEqual(self.store.begin_native_job('scope', 'turn'), key)

    def test_success_queues_one_parent_finish_instead_of_per_child_yields(self):
        key = self.store.begin_native_job('scope', 'turn')
        for role in ('main', 'compression', 'tool'):
            keys = self.store.bind_execution('scope', role, role, self.body, job_turn='turn')
            self.store.record_turn_call('scope', 'turn', keys,
                job=SimpleNamespace(grant_id='grant-' + role, execution_id='execution-' + role))
        self.assertEqual(self.store.observe_native_outcome('scope', 'turn',
            completed=True, failed=False, interrupted=False), 1)
        events = self.store.pending_events()
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]['kind'], events[0]['job_key']), ('job_finish', key))
        self.assertNotEqual(self.store.begin_native_job('scope', 'next-job'), key)

    def test_delayed_old_success_does_not_close_live_recovery_attempt(self):
        key = self.store.begin_native_job('scope', 'original')
        self.store.bind('scope', 'main', 'one', self.body, job_turn='original')
        self.store.observe_native_outcome('scope', 'original', completed=False, failed=True, interrupted=False)
        self.assertEqual(self.store.begin_native_job('scope', 'retry'), key)
        self.store.observe_native_outcome('scope', 'original', completed=True, failed=False, interrupted=False)
        self.assertEqual(self.store.pending_events(), [])
        active = self.store.db.execute('SELECT job_key,turn_id,state FROM native_requester_job').fetchone()
        self.assertEqual(active, (key, 'retry', 'active'))
        self.store.observe_native_outcome('scope', 'retry', completed=True, failed=False, interrupted=False)
        self.assertEqual(self.store.pending_events()[0]['job_key'], key)

    def test_pre_ack_finish_survives_reopen_and_late_intake_remains_same_parent(self):
        key = self.store.begin_native_job('scope', 'turn')
        keys = self.store.bind('scope', 'main', 'call', self.body, job_turn='turn')
        self.store.observe_native_outcome('scope', 'turn', completed=True, failed=False, interrupted=False)
        event = self.store.pending_events()[0]
        self.store.close()
        self.store = IdentityStore(self.path)
        self.assertEqual(self.store.pending_events(), [event])
        self.store.record_turn_call('scope', 'turn', keys,
                                   job=SimpleNamespace(grant_id='grant', execution_id='execution'))
        self.assertTrue(all(e['kind'] == 'job_finish' and e['job_key'] == key
                            for e in self.store.pending_events()))

    def test_late_old_intake_cannot_finish_resumed_parent(self):
        key = self.store.begin_native_job('scope', 'original')
        keys = self.store.bind('scope', 'main', 'one', self.body, job_turn='original')
        self.store.observe_native_outcome('scope', 'original', completed=False, failed=True, interrupted=False)
        self.assertEqual(self.store.begin_native_job('scope', 'retry'), key)
        self.store.observe_native_outcome('scope', 'original', completed=True, failed=False, interrupted=False)
        self.store.record_turn_call('scope', 'original', keys,
            job=SimpleNamespace(grant_id='grant', execution_id='execution'))
        self.assertEqual(self.store.pending_events(), [])
        self.assertEqual(self.store.db.execute(
            'SELECT job_key,turn_id,state FROM native_requester_job').fetchone(),
            (key, 'retry', 'active'))

    def test_legacy_tool_annotation_cannot_silently_yield_linked_parent(self):
        self.store.begin_native_job('scope', 'turn')
        keys = self.store.bind('scope', 'main', 'call', self.body, job_turn='turn')
        self.store.record_turn_call('scope', 'turn', keys, api_request_id='api',
                                   job=SimpleNamespace(grant_id='grant', execution_id='execution'))
        self.store.complete_api('scope', 'turn', 'api', ('tool',))
        with self.assertRaises(LocalSourceRefused):
            self.store.dependency_wait('scope', 'turn', 'api', 'tool')
        self.assertEqual(self.store.pending_events(), [])

    def test_missing_or_contradictory_verdict_does_not_finish(self):
        self.store.begin_native_job('scope', 'turn')
        for completed, failed, interrupted in ((None, False, False), (True, True, False)):
            with self.assertRaises(LocalSourceRefused):
                self.store.observe_native_outcome('scope', 'turn', completed=completed,
                                                 failed=failed, interrupted=interrupted)
        self.assertEqual(self.store.pending_events(), [])


class JobLifecycleRelay(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_old_server_never_receives_job_end_as_grant_yield(self):
        await self.exercise(qualified=False)

    async def test_missing_or_wrong_ack_retains_event_until_exact_named_ack(self):
        await self.exercise(qualified=True)

    async def exercise(self, qualified):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.sqlite'
            store = IdentityStore(path)
            key = store.begin_native_job('scope', 'turn')
            store.observe_native_outcome('scope', 'turn', completed=True, failed=False, interrupted=False)
            store.close()
            source = HostBindings(path, 'profile')
            posts, correct = [], [False]
            def http(request):
                if request.url.path.endswith('/capabilities'):
                    body = {'stints': {'lifecycle_yield': True}}
                    if qualified:
                        body['jobs'] = {'version': 1, 'explicit_finish': True}
                    return httpx.Response(200, json=body)
                posts.append(request.url.path)
                self.assertEqual(request.url.path, '/v1/inference/jobs/end')
                self.assertEqual(json.loads(request.content), {'key': key, 'action': 'finish'})
                return httpx.Response(202, json={'key': key if correct[0] else 'wrong',
                                               'action': 'finish', 'jobs': []})
            async with httpx.AsyncClient(base_url='https://queue.invalid',
                                         transport=httpx.MockTransport(http)) as client:
                consumer = DurableConsumer(client)
                await relay_once(consumer, source)
                self.assertEqual(len(source.pending_events()), 1)
                if not qualified:
                    self.assertEqual(posts, [])
                else:
                    correct[0] = True
                    self.assertEqual(await relay_once(consumer, source), 1)
                    self.assertEqual(source.pending_events(), [])


if __name__ == '__main__':
    unittest.main()
