"""Explicit host boundaries persist; content equality never deduplicates intent."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from .inference_identity import IdentityStore, Conflict


class IdentityContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'identity.sqlite'

    def tearDown(self):
        self.temp.cleanup()

    def bind(self, scope='profile/conversation', work='input:1', call='tail:1', body=None):
        store = IdentityStore(self.path)
        try:
            return store.bind(scope, work, call, body or {'model': 'qwen', 'messages': ['private content']})
        finally:
            store.close()

    def test_reopen_recovers_keys_and_identical_new_intent_is_distinct(self):
        first = self.bind()
        self.assertEqual(first, self.bind())
        self.assertNotEqual(first.call_key, self.bind(call='tail:2').call_key)
        self.assertNotEqual(first.grant_key, self.bind(work='input:2').grant_key)
        self.assertNotEqual(first.grant_key, self.bind(scope='other/conversation').grant_key)
        self.assertEqual(set(first.headers()), {'X-Inference-Grant-Key', 'Idempotency-Key'})

    def test_content_conflict_does_not_replace_or_leak_existing_call(self):
        first = self.bind()
        with self.assertRaises(Conflict) as error:
            self.bind(body={'messages': ['different private content']})
        self.assertNotIn('private content', str(error.exception))
        self.assertEqual(first, self.bind())
        self.assertNotIn(b'private content', self.path.read_bytes())

    def test_execution_boundaries_reopen_and_release_between_tool_rounds(self):
        def bind(call, work='input:1'):
            store = IdentityStore(self.path)
            try:
                return store.bind_execution('profile/conversation', work, call, {'prompt': 'same'})
            finally:
                store.close()
        first = bind('tail:1')
        self.assertEqual(first, bind('tail:1'))
        self.assertNotEqual(first.grant_key, bind('tail:2').grant_key)
        self.assertNotEqual(first.call_key, bind('tail:2').call_key)
        self.assertNotEqual(first.grant_key, bind('tail:1', work='input:2').grant_key)

    def test_only_a_terminal_failure_gives_a_retry_a_fresh_call(self):
        def bind(fresh=True, body=None):
            store = IdentityStore(self.path)
            try:
                return store.bind('profile/conversation', 'stint', 'rows:1', body or {'prompt': 'same'},
                                  fresh_after_failure=fresh)
            finally:
                store.close()
        def fail(keys):
            store = IdentityStore(self.path)
            try:
                return store.record_terminal_failure(keys.grant_key, keys.call_key)
            finally:
                store.close()
        first = bind()
        self.assertEqual(first, bind())  # nothing recorded (running, uncertain, unknown): replay
        self.assertEqual(fail(first), 1)
        second = bind()
        self.assertEqual(second.grant_key, first.grant_key)
        self.assertNotEqual(second.call_key, first.call_key)
        self.assertEqual(second, bind())  # the new call replays in turn until it is failed too
        self.assertEqual(first, bind(fresh=False))  # a caller that doesn't opt in still replays
        self.assertEqual(fail(first), 0)
        fail(second)
        third = bind()
        self.assertNotIn(third.call_key, (first.call_key, second.call_key))
        # A failed call can't be replayed, so different content after it is a new call, not a conflict.
        fail(third)
        self.assertNotIn(bind(body={'prompt': 'changed'}).call_key,
                         (first.call_key, second.call_key, third.call_key))

    def test_terminal_failure_records_only_this_stores_own_call(self):
        first = self.bind()
        store = IdentityStore(self.path)
        try:
            self.assertEqual(store.record_terminal_failure('other-grant', first.call_key), 0)
            self.assertEqual(store.record_terminal_failure(first.grant_key, 'unknown-call'), 0)
            self.assertEqual(store.bind('profile/conversation', 'input:1', 'tail:1',
                {'model': 'qwen', 'messages': ['private content']}, fresh_after_failure=True), first)
        finally:
            store.close()

    def test_concurrent_connections_allocate_one_identity(self):
        self.bind()  # initialize schema before racing allocations
        with ThreadPoolExecutor(max_workers=4) as pool:
            keys = list(pool.map(lambda _: self.bind(call='concurrent'), range(12)))
        self.assertEqual(len(set(keys)), 1)

    def test_public_or_symlink_store_is_rejected(self):
        self.path.touch(mode=0o644)
        self.path.chmod(0o644)
        with self.assertRaises(PermissionError):
            IdentityStore(self.path)
        link = self.path.with_name('link')
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            IdentityStore(link)


if __name__ == '__main__':
    unittest.main()
