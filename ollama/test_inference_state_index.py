"""Control-path reads must not walk historical inference request bodies."""
import json
from pathlib import Path
import tempfile
import unittest

from .inference_arbiter import Arbiter


class JournalStateReads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'queue.sqlite'
        self.journal = Arbiter(self.path)

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def test_pending_demand_does_not_scan_terminal_request_history(self):
        grant = self.journal.submit('seat', 'history', 'gpu', 'qwen')
        request = json.dumps({'prompt': 'historical context ' * 4096})
        # A real on-disk overflow-page history, not a timing mock. No provider
        # runs: only the scheduler's metadata query is exercised below.
        with self.journal.transaction():
            self.journal.db.executemany('''INSERT INTO executions
                (id,grant_id,client_key,request,state,created,updated)
                VALUES (?,?,?,?,'complete',1,1)''',
                ((f'old-{i}', grant['id'], f'old-{i}', request) for i in range(512)))
        demand = self.journal.submit('next-seat', 'new', 'gpu', 'qwen')
        queued = self.journal.enqueue(demand['id'], 'next-seat', 'call', {'prompt': 'next'})

        # Bound SQLite work rather than wall time, which depends on page cache,
        # disk speed and host pressure. The old full scan exceeds this budget.
        steps = 0

        def budget():
            nonlocal steps
            steps += 1
            return int(steps > 800)

        self.journal.db.set_progress_handler(budget, 1)
        try:
            self.assertEqual(self.journal.pending_models('gpu'), ['qwen'])
        finally:
            self.journal.db.set_progress_handler(None, 0)
        self.assertEqual(self.journal.execution(queued['id'], 'next-seat')['state'], 'queued')

    def test_existing_journal_gets_covering_metadata_reads_without_changing_work(self):
        grant = self.journal.submit('seat', 'work', 'gpu', 'qwen')
        request = {'prompt': 'context ' * 8192}
        call = self.journal.enqueue(grant['id'], 'seat', 'call', request)
        self.journal.dispatch_next('gpu')
        self.journal.append_event(call['id'], 'saved partial output')
        self.journal.uncertain(call['id'])
        # Model the previous schema, which had only the id and idempotency
        # indexes. Reopening performs the ordinary additive schema upgrade.
        self.journal.db.execute('DROP INDEX executions_state_grant_created')
        before = {table: [tuple(row) for row in self.journal.db.execute('SELECT * FROM ' + table)]
                  for table in ('grants', 'executions', 'execution_events', 'execution_lifecycle')}
        self.journal.close()
        self.journal = Arbiter(self.path)
        after = {table: [tuple(row) for row in self.journal.db.execute('SELECT * FROM ' + table)]
                 for table in before}
        self.assertEqual(before, after)
        for sql in (
                'SELECT state,COUNT(*) FROM executions GROUP BY state',
                "SELECT 1 FROM executions WHERE state IN ('running','uncertain') LIMIT 1"):
            plan = ' '.join(str(tuple(row)) for row in
                            self.journal.db.execute('EXPLAIN QUERY PLAN ' + sql))
            self.assertIn('COVERING INDEX executions_state_grant_created', plan)
        self.assertEqual(self.journal.execution(call['id'], 'seat')['state'], 'uncertain')
        self.assertIsNone(self.journal.dispatch_next('gpu'))


if __name__ == '__main__':
    unittest.main()
