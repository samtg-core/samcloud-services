"""Committed event coverage, cursor replay and metadata privacy; no inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .inference_arbiter import Arbiter


class Telemetry(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'queue.sqlite'
        self.a=Arbiter(self.path)

    def tearDown(self):
        self.a.close()
        self.tmp.cleanup()

    def rows(self):
        return [dict(row) for row in self.a.db.execute('SELECT * FROM metadata_events ORDER BY seq')]

    def test_every_call_transition_is_durable_without_prompt_or_result(self):
        g=self.a.submit('owner','scope','gpu','model')
        c=self.a.enqueue(g['id'],'owner','call',{'prompt':'PRIVATE PROMPT'})
        self.a.dispatch_next('gpu')
        self.a.finish(c['id'],{'text':'PRIVATE RESULT'})
        rows=self.rows()
        self.assertEqual([r['state'] for r in rows if r['kind']=='execution'],
                         ['queued','running','complete'])
        self.assertNotIn('PRIVATE',json.dumps(rows))
        self.assertEqual([r['seq'] for r in rows],list(range(1,len(rows)+1)))
        self.assertEqual(self.path.with_suffix('.events-notify').stat().st_size,8)

    def test_rolled_back_transition_is_not_published(self):
        before=self.rows()
        with self.assertRaises(RuntimeError):
            with self.a.transaction():
                self.a.db.execute("INSERT INTO metadata_events(kind,at) VALUES('never',1)")
                raise RuntimeError('rollback')
        self.assertEqual(self.rows(),before)

    def test_notify_occurs_after_commit_and_can_fail_without_reclassifying_work(self):
        import sqlite3
        seen=[]
        def inspect(path,sequence):
            with sqlite3.connect(path) as other:
                seen.append(other.execute('SELECT MAX(seq) FROM metadata_events').fetchone()[0])
            self.assertEqual(seen[-1],sequence)
        with patch('ollama.inference_telemetry.notify',side_effect=inspect):
            self.a.capacity_event('gpu','model','unknown','observing')
        self.assertEqual(len(seen),1)
        with patch('ollama.inference_telemetry.os.open',side_effect=OSError('unavailable')):
            g=self.a.submit('owner','scope','gpu','model')
        self.assertEqual(self.a.get(g['id'],'owner')['state'],'queued')

    def test_reopen_does_not_synthesize_historical_events(self):
        self.a.submit('owner','scope','gpu','model')
        before=self.rows()
        self.a.close()
        self.a=Arbiter(self.path)
        self.assertEqual(self.rows(),before)


if __name__=='__main__':unittest.main()
