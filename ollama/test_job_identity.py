"""Durable local linkage, migration and recovery; no service or model."""
import hashlib
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from .inference_arbiter import Conflict, encoded
from .inference_identity import IdentityStore


class JobIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'identity.sqlite'
        self.store = IdentityStore(self.path)
        self.request = {'model': 'text', 'messages': ['synthetic']}

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_new_calls_across_work_types_share_job_but_other_turn_and_scope_do_not(self):
        main = self.store.bind('scope', 'main', 'main', self.request, job_turn='turn')
        compaction = self.store.bind_execution('scope', 'compression', 'summary', self.request, job_turn='turn')
        tool = self.store.bind_execution('scope', 'tool', 'vision', self.request, job_turn='turn')
        next_turn = self.store.bind('scope', 'main', 'next', self.request, job_turn='next')
        sibling = self.store.bind('sibling', 'main', 'main', self.request, job_turn='turn')
        self.assertEqual({main.job_key, compaction.job_key, tool.job_key}, {main.job_key})
        self.assertNotEqual(main.job_key, next_turn.job_key)
        self.assertNotEqual(main.job_key, sibling.job_key)
        self.assertEqual(main.headers()['X-Inference-Job-Key'], main.job_key)

    def test_retry_keeps_original_linkage_even_when_current_turn_changed(self):
        first = self.store.bind('scope', 'main', 'call', self.request, job_turn='original')
        self.store.close()
        self.store = IdentityStore(self.path)
        replay = self.store.bind('scope', 'main', 'call', self.request, job_turn='later')
        self.assertEqual(first, replay)
        self.assertEqual(self.store.db.execute('SELECT turn_id FROM native_job_binding').fetchall(), [('original',)])

    def test_legacy_allocated_call_is_never_promoted_on_retry(self):
        first = self.store.bind('scope', 'main', 'call', self.request)
        later = self.store.bind('scope', 'main', 'call', self.request, job_turn='new')
        self.assertEqual(first, later)
        self.assertIsNone(later.job_key)
        self.assertNotIn('X-Inference-Job-Key', later.headers())
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM native_job_binding').fetchone()[0], 0)

    def test_changed_payload_refuses_without_creating_another_job(self):
        self.store.bind('scope', 'main', 'call', self.request, job_turn='original')
        with self.assertRaises(Conflict):
            self.store.bind('scope', 'main', 'call', self.request | {'model': 'other'}, job_turn='later')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM native_job_binding').fetchone()[0], 1)

    def test_linked_child_does_not_emit_legacy_parent_yield(self):
        keys = self.store.bind_execution('scope', 'tool', 'child', self.request, job_turn='turn')
        frozen = self.store.bind_handoff(keys, ('old-parent',))
        self.assertEqual(frozen.job_key, keys.job_key)
        self.assertEqual(frozen.yield_grant_keys, ())
        self.assertEqual(self.store.bind_handoff(keys, ('different-parent',)), frozen)

    def test_acknowledged_compaction_model_switch_reads_same_job_link(self):
        keys = self.store.bind_compaction('scope', 'summary', self.request,
                                         stint_model='text', job_turn='original')
        self.store.record_turn_call('scope', 'original', keys,
            job=SimpleNamespace(grant_id='grant', execution_id='execution'))
        resumed = self.store.bind_compaction('scope', 'summary', self.request | {'model': 'other'},
                                            stint_model='other', job_turn='new')
        self.assertEqual(resumed.job_key, keys.job_key)
        self.assertEqual(resumed.resume_job, ('grant', 'execution'))

    def test_real_legacy_schema_migrates_with_null_frozen_linkage(self):
        self.store.close()
        self.path.unlink()
        connection = sqlite3.connect(self.path)
        self.path.chmod(0o600)
        connection.executescript('''
            CREATE TABLE work_identity(scope TEXT,boundary TEXT,key TEXT UNIQUE,PRIMARY KEY(scope,boundary));
            CREATE TABLE call_identity(work_key TEXT,boundary TEXT,key TEXT UNIQUE,digest TEXT,
                                       PRIMARY KEY(work_key,boundary));
        ''')
        connection.execute('INSERT INTO work_identity VALUES(?,?,?)', ('scope', 'main', 'grant'))
        connection.execute('INSERT INTO call_identity VALUES(?,?,?,?)',
            ('grant', 'call', 'execution', hashlib.sha256(encoded(self.request).encode()).hexdigest()))
        connection.commit()
        connection.close()
        self.store = IdentityStore(self.path)
        replay = self.store.bind('scope', 'main', 'call', self.request, job_turn='new')
        self.assertEqual((replay.grant_key, replay.call_key), ('grant', 'execution'))
        self.assertIsNone(replay.job_key)


if __name__ == '__main__':
    unittest.main()
