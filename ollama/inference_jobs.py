"""Requester-job continuity inside the existing inference journal.

This is scheduling ownership, never a physical allocator. The shared registry
still admits every physical claim. A job key is explicit, owner-scoped and
resource-scoped; sharing a username does not attach an independent request.
Execution/grant identities remain unchanged across reconnect and recovery.
"""
import time
import uuid


class JobScheduling:
    def _install_jobs(self):
        self.db.execute('''CREATE TABLE IF NOT EXISTS requester_job_ends (
            owner TEXT NOT NULL, client_key TEXT NOT NULL, at REAL NOT NULL,
            PRIMARY KEY(owner,client_key))''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS requester_jobs (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            client_key TEXT NOT NULL,
            resource TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('queued','owned','closed')),
            end_requested TEXT CHECK(end_requested IN ('finish','yield')),
            created REAL NOT NULL,
            updated REAL NOT NULL,
            UNIQUE(owner,client_key,resource))''')
        self.db.execute('''CREATE UNIQUE INDEX IF NOT EXISTS one_requester_job
            ON requester_jobs(resource) WHERE state='owned' ''')
        columns = {row['name'] for row in self.db.execute('PRAGMA table_info(executions)')}
        if 'job_id' not in columns:
            self.db.execute('ALTER TABLE executions ADD COLUMN job_id TEXT '
                            'REFERENCES requester_jobs(id)')

    def _find_job(self, owner, key, resource):
        return self.db.execute('''SELECT * FROM requester_jobs
            WHERE owner=? AND client_key=? AND resource=?''', (owner, key, resource)).fetchone()

    def _bind_job(self, owner, key, resource):
        """Inside the execution intake transaction; no admission before work exists."""
        if key is None:
            return None
        from .inference_arbiter import Conflict, required
        required(key)
        if len(key) > 512:
            raise ValueError('bounded job key required')
        if self.db.execute('SELECT 1 FROM requester_job_ends WHERE owner=? AND client_key=?',
                           (owner, key)).fetchone():
            raise Conflict('job is finishing or already finished')
        job = self._find_job(owner, key, resource)
        if job is not None:
            if job['state'] == 'closed' or job['end_requested']:
                raise Conflict('job is finishing or already finished')
            return job['id']
        ident, now = uuid.uuid4().hex, time.time()
        self.db.execute('INSERT INTO requester_jobs VALUES(?,?,?,?,\'queued\',NULL,?,?)',
                        (ident, owner, key, resource, now, now))
        return ident

    def job(self, owner, key, resource):
        from .inference_arbiter import NotFound
        row = self._find_job(owner, key, resource)
        if row is None:
            raise NotFound()
        return dict(row)

    def owns_job(self, resource):
        """True also between calls, during recovery, or while children drain."""
        return self.db.execute("SELECT 1 FROM requester_jobs WHERE resource=? AND state='owned'",
                               (resource,)).fetchone() is not None

    def jobs(self, owner, key):
        from .inference_arbiter import NotFound, required
        required(key)
        rows = self.db.execute('SELECT * FROM requester_jobs WHERE owner=? AND client_key=? '
                               'ORDER BY resource', (owner, key)).fetchall()
        if not rows:
            raise NotFound()
        return [dict(row) for row in rows]

    def end_job(self, owner, key, *, finish=True):
        """Explicit owner boundary, fenced by this exact job rather than a reused grant.

        Accepted children drain; unknown execution retains ownership. No call is
        cancelled, replayed, expired or presumed terminated by this operation.
        Finish seals the job against new calls. Yield can resume the same job
        with a later call after its accepted children have actually ended.
        """
        from .inference_arbiter import required
        required(key)
        with self.transaction():
            if finish:
                # End can arrive before a lost-ACK intake. Seal the exact
                # owner/key even if that request has not committed yet; no
                # late/new resource child can silently reopen the parent.
                self.db.execute('INSERT OR IGNORE INTO requester_job_ends VALUES(?,?,?)',
                                (owner, key, time.time()))
            rows = self.db.execute('SELECT * FROM requester_jobs WHERE owner=? AND client_key=?',
                                   (owner, key)).fetchall()
            for row in rows:
                if row['state'] == 'closed':
                    continue
                action = 'finish' if finish or row['end_requested'] == 'finish' else 'yield'
                self.db.execute('UPDATE requester_jobs SET end_requested=?,updated=? WHERE id=?',
                                (action, time.time(), row['id']))
                self._advance_job(row['id'])
            return [dict(row) for row in self.db.execute(
                'SELECT * FROM requester_jobs WHERE owner=? AND client_key=? ORDER BY resource',
                (owner, key))]

    def _advance_job(self, job_id):
        """Inside a write transaction, only after authoritative terminal transitions."""
        row = self.db.execute('SELECT * FROM requester_jobs WHERE id=?', (job_id,)).fetchone()
        if row is None or not row['end_requested']:
            return
        if self.db.execute("SELECT 1 FROM executions WHERE job_id=? "
                           "AND state IN ('queued','running','uncertain')", (job_id,)).fetchone():
            return
        now = time.time()
        if row['state'] == 'owned':
            grant = self.db.execute("SELECT * FROM grants WHERE resource=? AND state='granted'",
                                    (row['resource'],)).fetchone()
            if grant is not None:
                self._handoff(grant, 'job_' + row['end_requested'], now)
        self.db.execute('UPDATE requester_jobs SET state=?,end_requested=NULL,updated=? WHERE id=?',
                        ('closed' if row['end_requested'] == 'finish' else 'queued', now, job_id))

    def _dispatch_job(self, resource, permitted_model):
        """Return (owns scheduling, dispatched item). No wall-clock handoff policy."""
        job = self.db.execute("SELECT * FROM requester_jobs WHERE resource=? AND state='owned'",
                              (resource,)).fetchone()
        if job is None:
            return False, None
        incumbent = self.db.execute("SELECT * FROM grants WHERE resource=? "
                                    "AND state IN ('granted','uncertain')", (resource,)).fetchone()
        if incumbent is not None and (incumbent['state'] == 'uncertain' or
                                     incumbent['active_execution'] or incumbent['release_requested']):
            return True, None
        self._advance_job(job['id'])
        if not self.owns_job(resource):
            return False, None
        candidates = self.db.execute('''SELECT e.id AS execution_id,g.id AS grant_id
            FROM executions e JOIN grants g ON g.id=e.grant_id
            WHERE e.job_id=? AND e.state='queued' AND g.state IN ('queued','granted')
                AND g.release_requested=0 ORDER BY e.created,e.id''', (job['id'],)).fetchall()
        for candidate in candidates:
            grant = self.db.execute('SELECT * FROM grants WHERE id=?',
                                    (candidate['grant_id'],)).fetchone()
            if permitted_model is not None and permitted_model(resource, grant['model']) is not True:
                continue
            if incumbent is not None and incumbent['id'] != grant['id']:
                # Backend/grant changes inside one job do not return the job's
                # place. The unique resource owner still prevents overlap.
                self._handoff(incumbent, 'job_child', time.time())
            call = self.db.execute('SELECT * FROM executions WHERE id=?',
                                   (candidate['execution_id'],)).fetchone()
            return True, self._start_queued_call(grant, call, time.time())
        return True, None

    def _own_call_job(self, call):
        if call['job_id'] is not None:
            self.db.execute("UPDATE requester_jobs SET state='owned',updated=? WHERE id=?",
                            (time.time(), call['job_id']))

    def _advance_call_job(self, call):
        if call['job_id'] is not None:
            self._advance_job(call['job_id'])
