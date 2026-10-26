"""Explicit foreground/source/profile contracts; no native process or compute."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .hermes_queue_identity import HostBindings
from .hermes_queue_provider import _host_for_home, host_for_home
from .hermes_queue_source import owner_scope
from .inference_identity import IdentityStore, LocalSourceRefused


class HostJobBindings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'identity.sqlite'
        self.host = HostBindings(self.path, 'profile', job_continuity=True)
        self.messages = [{'role': 'user', '_row_id': 1, 'content': 'synthetic'}]
        self.request = {'model': 'text', 'messages': [{'role': 'user', 'content': 'synthetic'}]}

    def tearDown(self):
        _host_for_home.cache_clear()
        self.tmp.cleanup()

    def main(self, host=None, turn='turn', row=1, api='api'):
        host = host or self.host
        host.begin_foreground('session', turn)
        host.pre_api_request(session_id='session', turn_id=turn, api_request_id=api,
            conversation_history=[{'role': 'user', '_row_id': row, 'content': 'synthetic'}])
        keys = host.main_stint_keys(self.request)
        keys.on_accepted(SimpleNamespace(grant_id='grant-' + keys.grant_key,
                                         execution_id='execution-' + keys.call_key))
        host.complete_api('session', turn, api, ('tool',))
        return keys

    def synthetic_tool_owner(self, host):
        # Metadata unit seam only. Real Relay/context/middleware ownership is
        # covered independently by test_native_tool_authority and native loops.
        turn = SimpleNamespace(turn_id='turn', lease=SimpleNamespace(
            profile_key=host.scope, session_id='session'))
        return patch('agent.relay_runtime.active_turn', return_value=turn)

    def test_same_foreground_job_covers_main_compression_and_tool_for_every_role(self):
        for role in ('creative', 'producer', 'assistant', 'samcloud'):
            host = HostBindings(self.root / (role + '.sqlite'), role, job_continuity=True)
            main = self.main(host)
            with host.compression_scope('session', self.messages,
                                        owner_provider=lambda: ('turn', 'text')):
                summary = host.auxiliary_keys(self.request | {'model': 'other-summary-model'})
            with self.synthetic_tool_owner(host), host.tool_scope('session', 'turn', 'api', 'tool'), host.tool_api_scope():
                tool = host.tool_keys(self.request | {'model': 'vision'})
            self.assertEqual((summary.job_key, tool.job_key), (main.job_key, main.job_key))
            self.assertEqual(summary.yield_grant_keys, ())
            self.assertEqual(tool.yield_grant_keys, ())
            self.assertTrue(callable(summary.on_accepted) and callable(tool.on_accepted))
            host.observe_native_outcome('session', 'turn', completed=True, failed=False, interrupted=False)
            events = host.pending_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['kind'], 'job_finish')

    def test_recovery_attempt_keeps_parent_and_success_starts_next_job(self):
        first = self.main()
        self.host.observe_native_outcome('session', 'turn', completed=False, failed=True, interrupted=False)
        self.host = HostBindings(self.path, 'profile', job_continuity=True)
        retry = self.main(turn='retry', row=2, api='retry-api')
        self.assertEqual(first.job_key, retry.job_key)
        self.assertEqual(self.host.pending_events(), [])
        self.host.observe_native_outcome('session', 'retry', completed=True, failed=False, interrupted=False)
        next_job = self.main(turn='next', row=3, api='next-api')
        self.assertNotEqual(next_job.job_key, first.job_key)

    def test_positive_compaction_owner_can_begin_before_first_main_call(self):
        with self.host.compression_scope('session', self.messages,
                                        owner_provider=lambda: ('turn', 'text')):
            summary = self.host.auxiliary_keys(self.request)
        main = self.main()
        self.assertEqual(main.job_key, summary.job_key)

    def test_background_sources_do_not_borrow_foreground_or_yield_it(self):
        main = self.main()
        self.host.pre_api_request(session_id='session', turn_id='background', api_request_id='background',
            conversation_history=[{'role': 'user', '_row_id': 2, 'content': 'background'}])
        background = self.host.main_stint_keys(self.request)
        self.assertIsNone(background.job_key)
        self.assertTrue(background.execution_only)
        with self.host.compression_scope('session', self.messages + [
                {'role': 'assistant', '_row_id': 3, 'content': 'background'}]):
            aux = self.host.auxiliary_keys(self.request)
        self.assertIsNone(aux.job_key)
        self.assertEqual(aux.yield_grant_keys, ())
        self.assertEqual(self.host.observe_native_outcome('session', 'background',
            completed=True, failed=False, interrupted=False), 0)
        self.assertEqual(self.host.pending_events(), [])
        store = IdentityStore(self.path)
        try:
            self.assertEqual(store.db.execute('SELECT job_key FROM native_requester_job').fetchone()[0], main.job_key)
        finally:
            store.close()

    def test_tool_with_legacy_parent_keeps_legacy_identity_in_job_mode(self):
        old = HostBindings(self.path, 'profile')
        main = self.main(old)
        self.assertIsNone(main.job_key)
        with self.synthetic_tool_owner(self.host), self.host.tool_scope('session', 'turn', 'api', 'tool'), self.host.tool_api_scope():
            child = self.host.tool_keys(self.request | {'model': 'vision'})
        self.assertIsNone(child.job_key)
        self.assertEqual(child.yield_grant_keys, (main.grant_key,))

    def test_mixed_frozen_tool_parent_linkages_refuse_without_new_intent(self):
        self.main()
        scope = owner_scope('profile', 'session')
        store = IdentityStore(self.path)
        try:
            old = store.bind(scope, 'conversation-stint:legacy-model', 'legacy', self.request)
            store.record_turn_call(scope, 'turn', old, api_request_id='api',
                                   job=SimpleNamespace(grant_id='old-grant', execution_id='old-execution'))
            store.complete_api(scope, 'turn', 'api', ('tool',))
        finally:
            store.close()
        with self.host.tool_scope('session', 'turn', 'api', 'tool'), self.host.tool_api_scope():
            with self.assertRaises(LocalSourceRefused):
                self.host.tool_keys(self.request | {'model': 'vision'})

    def test_job_mode_and_legacy_default_are_explicit_profile_local_settings(self):
        one, two = self.root / 'one', self.root / 'two'
        one.mkdir(); two.mkdir()
        (one / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
        (two / 'config.yaml').write_text('{"samcloud_queue":{"job_continuity":false}}')
        self.assertTrue(host_for_home(one).job_continuity)
        self.assertFalse(host_for_home(two).job_continuity)
        self.assertNotEqual(host_for_home(one).scope, host_for_home(two).scope)

    def test_invalid_policy_does_not_silently_disable_or_enable_continuity(self):
        for index, config in enumerate(('[]', 'false', 'samcloud_queue: null',
                                       'samcloud_queue: {job_continuity: 1}',
                                       'samcloud_queue: {job_continuity: "true"}')):
            home = self.root / str(index)
            home.mkdir()
            (home / 'config.yaml').write_text(config)
            with self.assertRaises(ValueError):
                host_for_home(home)


if __name__ == '__main__':
    unittest.main()
