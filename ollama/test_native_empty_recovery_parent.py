"""The job-aware lineage keeps recovery calls under the accepted parent."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .inference_identity import IdentityStore
from .test_native_empty_recovery import capture, history


class NativeEmptyParent(unittest.TestCase):
    def test_later_recovery_intake_retains_parent_and_exact_api_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'identity.sqlite'
            source = capture(history())
            store = IdentityStore(path)
            try:
                parent = store.begin_native_job(source.scope, 'turn')
                first = source.bind_stint(path, {'model': 'qwen'}, job_turn='turn')
                first.on_accepted(SimpleNamespace(grant_id='grant', execution_id='first'))
                later = capture(history(later=True), api='api-3').bind_stint(
                    path, {'model': 'qwen'}, job_turn='turn')
                later.on_accepted(SimpleNamespace(grant_id='grant', execution_id='later'))
                self.assertEqual((first.job_key, later.job_key), (parent, parent))
                self.assertEqual(first.grant_key, later.grant_key)
                self.assertNotEqual(first.call_key, later.call_key)
                rows = store.db.execute('SELECT api_request_id,call_key FROM turn_api_call '
                                        'WHERE scope=? ORDER BY api_request_id', (source.scope,)).fetchall()
                self.assertEqual(rows, [('api-2', first.call_key), ('api-3', later.call_key)])
                self.assertEqual(store.observe_native_outcome(source.scope, 'turn',
                    completed=True, failed=False, interrupted=False), 1)
                self.assertEqual(store.pending_events()[0]['job_key'], parent)
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
