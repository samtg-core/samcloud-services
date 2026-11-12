"""Offline durability/ownership tests; no provider or registry calls."""
import concurrent.futures
import json
import sqlite3
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .inference_arbiter import Arbiter, Conflict, NotFound


class ArbitrationJournal(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'arbiter.sqlite'
        self.a = Arbiter(self.path)

    def tearDown(self):
        self.a.close()
        self.temp.cleanup()

    def submit(self, owner='creative', key='turn1', resource='slice'):
        return self.a.submit(owner, key, resource, 'qwen')

    def test_idempotency_and_conflicting_reuse(self):
        first = self.submit()
        self.assertEqual(first['id'], self.submit()['id'])
        with self.assertRaises(Conflict):
            self.a.submit('creative', 'turn1', 'wafer', 'qwen')
        self.assertNotEqual(first['id'], self.submit('assistant')['id'])

    def test_execution_grant_hands_off_only_at_confirmed_backend_end(self):
        first = self.a.submit('creative', 'call1', 'slice', 'qwen', lifetime='execution')
        second = self.submit('assistant')
        self.a.enqueue(first['id'], 'creative', 'one', {'prompt': 'A'})
        self.a.enqueue(second['id'], 'assistant', 'two', {'prompt': 'B'})
        active = self.a.dispatch_next('slice')
        self.assertEqual(active['grant']['id'], first['id'])
        self.a.uncertain(active['execution']['id'])
        self.assertIsNone(self.a.dispatch_next('slice'))
        self.a.finish(active['execution']['id'], {'answer': 'done'})
        self.assertEqual(self.a.get(first['id'], 'creative')['state'], 'released')
        self.assertEqual(self.a.dispatch_next('slice')['grant']['id'], second['id'])
        replay = self.a.enqueue(first['id'], 'creative', 'one', {'prompt': 'A'})
        self.assertEqual(replay['id'], active['execution']['id'])
        with self.assertRaises(Conflict):
            self.a.enqueue(first['id'], 'creative', 'new', {'prompt': 'A'})
        with self.assertRaises(Conflict):
            self.a.submit('creative', 'call1', 'slice', 'qwen')

    def test_queued_execution_grant_does_not_expire_and_failure_releases(self):
        first = self.a.submit('creative', 'call1', 'slice', 'qwen', lifetime='execution')
        self.a.enqueue(first['id'], 'creative', 'one', {})
        with patch('ollama.inference_arbiter.time.time', return_value=10**12):
            active = self.a.dispatch_next('slice')
        self.a.finish(active['execution']['id'], {'error': 'confirmed failure'}, failed=True)
        self.assertEqual(self.a.get(first['id'], 'creative')['state'], 'released')
        with self.assertRaises(ValueError):
            self.a.submit('creative', 'bad', 'slice', 'qwen', lifetime='timeout')

    def test_old_journal_migrates_without_changing_held_work_session(self):
        old_path = Path(self.temp.name) / 'old.sqlite'
        old_path.touch(mode=0o600)
        db = sqlite3.connect(old_path)
        try:
            db.executescript('''CREATE TABLE grants (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                owner TEXT NOT NULL, client_key TEXT NOT NULL, resource TEXT NOT NULL,
                model TEXT NOT NULL, state TEXT NOT NULL,
                release_requested INTEGER NOT NULL DEFAULT 0, active_execution TEXT,
                created REAL NOT NULL, updated REAL NOT NULL, UNIQUE(owner,client_key));
                INSERT INTO grants(id,owner,client_key,resource,model,state,created,updated)
                VALUES('old','creative','old-call','slice','qwen','granted',1,1);''')
        finally:
            db.close()
        migrated = Arbiter(old_path)
        try:
            self.assertEqual(migrated.get('old', 'creative')['lifetime'], 'work_session')
            other = migrated.submit('assistant', 'next', 'slice', 'qwen', lifetime='execution')
            migrated.enqueue(other['id'], 'assistant', 'one', {})
            self.assertIsNone(migrated.dispatch_next('slice'))
            migrated.request_release('old', 'creative')
            self.assertEqual(migrated.dispatch_next('slice')['grant']['id'], other['id'])
        finally:
            migrated.close()

    def test_pending_survives_abrupt_submitter_exit(self):
        code = """import os,sys
from ollama.inference_arbiter import Arbiter
a=Arbiter(sys.argv[1]); a.submit('child','request','slice','qwen'); os._exit(0)
"""
        subprocess.run([sys.executable, '-c', code, str(self.path)], check=True, timeout=5)
        grant = self.a.admit_next('slice')
        self.assertEqual(grant['owner'], 'child')
        self.assertEqual(grant['state'], 'granted')

    def test_fifo_and_exclusive_resource_across_connections(self):
        first = self.submit()
        second = self.submit('assistant')
        def admit(_):
            a = Arbiter(self.path)
            try:
                return a.admit_next('slice')
            finally:
                a.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            granted = [r for r in pool.map(admit, range(8)) if r is not None]
        self.assertEqual([r['id'] for r in granted], [first['id']])
        self.a.request_release(first['id'], 'creative')
        self.assertEqual(self.a.admit_next('slice')['id'], second['id'])

    def test_other_owner_cannot_read_execute_or_release(self):
        grant = self.submit()
        for action in (
            lambda: self.a.get(grant['id'], 'assistant'),
            lambda: self.a.start(grant['id'], 'assistant', 'x', {}),
            lambda: self.a.request_release(grant['id'], 'assistant'),
        ):
            with self.assertRaises(NotFound):
                action()

    def test_release_waits_for_actual_execution_end(self):
        first = self.submit()
        second = self.submit('assistant')
        self.a.admit_next('slice')
        execution, new = self.a.start(first['id'], 'creative', 'call1', {'prompt': 'A'})
        self.assertTrue(new)
        self.a.request_release(first['id'], 'creative')
        self.assertIsNone(self.a.admit_next('slice'))
        with self.assertRaises(Conflict):
            self.a.start(first['id'], 'creative', 'call2', {'prompt': 'B'})
        self.a.finish(execution['id'], {'answer': 'done'})
        self.assertEqual(self.a.admit_next('slice')['id'], second['id'])

    def test_reconnect_finds_execution_and_saved_result_without_redispatch(self):
        grant = self.submit()
        self.a.admit_next('slice')
        first, new = self.a.start(grant['id'], 'creative', 'call', {'prompt': 'A'})
        self.assertTrue(new)
        self.a.close()
        self.a = Arbiter(self.path)
        same, new = self.a.start(grant['id'], 'creative', 'call', {'prompt': 'A'})
        self.assertFalse(new)
        self.assertEqual(same['id'], first['id'])
        self.assertEqual(same['state'], 'running')
        self.a.finish(first['id'], {'answer': 'saved'})
        self.a.request_release(grant['id'], 'creative')
        same, new = self.a.start(grant['id'], 'creative', 'call', {'prompt': 'A'})
        self.assertFalse(new)
        self.assertEqual(json.loads(same['result']), {'answer': 'saved'})
        with self.assertRaises(Conflict):
            self.a.start(grant['id'], 'creative', 'call', {'prompt': 'different'})

    def test_multiple_calls_can_share_grant_but_never_overlap(self):
        grant = self.submit()
        self.a.admit_next('slice')
        first, _ = self.a.start(grant['id'], 'creative', 'first', {})
        with self.assertRaises(Conflict):
            self.a.start(grant['id'], 'creative', 'second', {})
        self.a.finish(first['id'], {})
        second, new = self.a.start(grant['id'], 'creative', 'second', {})
        self.assertTrue(new)
        self.assertNotEqual(first['id'], second['id'])

    def test_crash_uncertainty_blocks_resource_until_reconciled(self):
        grant = self.submit()
        self.submit('assistant')
        self.a.admit_next('slice')
        execution, _ = self.a.start(grant['id'], 'creative', 'call', {})
        self.a.quarantine_running()
        self.assertEqual(self.a.get(grant['id'], 'creative')['state'], 'uncertain')
        self.a.request_release(grant['id'], 'creative')
        self.assertIsNone(self.a.admit_next('slice'))
        same, new = self.a.start(grant['id'], 'creative', 'call', {})
        self.assertFalse(new)
        self.assertEqual(same['state'], 'uncertain')
        # A supervisor has now established the old backend execution ended.
        self.a.finish(execution['id'], {'error': 'backend stopped'}, failed=True)
        self.assertEqual(self.a.admit_next('slice')['owner'], 'assistant')

    def test_cancel_queued_does_not_take_resource(self):
        first = self.submit()
        second = self.submit('assistant')
        self.a.request_release(first['id'], 'creative')
        self.assertEqual(self.a.get(first['id'], 'creative')['state'], 'cancelled')
        self.assertEqual(self.a.admit_next('slice')['id'], second['id'])

    def test_five_hour_wait_changes_nothing_without_a_lifecycle_transition(self):
        with patch('ollama.inference_arbiter.time.time', return_value=1000):
            first = self.submit()
            second = self.submit('assistant')
            active = self.a.enqueue(first['id'], 'creative', 'call', {})
            waiting = self.a.enqueue(second['id'], 'assistant', 'call', {})
            self.a.dispatch_next('slice')
        with patch('ollama.inference_arbiter.time.time', return_value=1000+5*3600):
            self.assertEqual(self.a.execution(active['id'], 'creative')['state'], 'running')
            self.assertEqual(self.a.execution(waiting['id'], 'assistant')['state'], 'queued')
            self.assertIsNone(self.a.dispatch_next('slice'))
            self.a.request_release(first['id'], 'creative')
            self.assertIsNone(self.a.dispatch_next('slice'))
            self.a.finish(active['id'], {'answer':'done'})
            self.a.dispatch_next('slice')
            self.assertEqual(self.a.execution(waiting['id'], 'assistant')['state'], 'running')

    def test_terminal_result_is_immutable_and_finish_idempotent(self):
        grant = self.submit()
        self.a.admit_next('slice')
        execution, _ = self.a.start(grant['id'], 'creative', 'call', {})
        self.a.finish(execution['id'], {'value': 1})
        self.a.finish(execution['id'], {'value': 1})
        with self.assertRaises(Conflict):
            self.a.finish(execution['id'], {'value': 2})

    def test_journal_private_and_rejects_public_file(self):
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        other = Path(self.temp.name) / 'public.sqlite'
        other.touch(mode=0o644)
        other.chmod(0o644)
        with self.assertRaises(PermissionError):
            Arbiter(other)


if __name__ == '__main__':
    unittest.main()
