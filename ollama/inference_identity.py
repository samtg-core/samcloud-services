"""Caller-owned durable keys; no prompt matching or inferred turn boundaries.

The host supplies explicit stable scope/work/call boundaries BEFORE submission.
This store does not discover Hermes rows, resume turns, or declare backend work
finished. Compaction/retry intent mapping belongs to the host integration.
Only identifiers and a request-integrity digest are stored, never request bodies.
One connection per owning thread; independent processes/connections serialize
allocation with SQLite. Nothing here is installed in the live clients.
"""
from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
import sqlite3
import uuid

from .inference_arbiter import Conflict, encoded, required


class LocalSourceRefused(ValueError):
    """Known local ownership refusal, before an inference intent can be submitted."""


@dataclass(frozen=True)
class CallKeys:
    grant_key: str
    call_key: str
    yield_grant_keys: tuple[str, ...] = ()
    execution_only: bool = False
    work_session: bool = False
    on_accepted: object = field(default=None, compare=False, repr=False)
    resume_job: object = field(default=None, compare=False, repr=False)
    job_key: str | None = None

    def headers(self):
        result = {'X-Inference-Grant-Key': self.grant_key, 'Idempotency-Key': self.call_key}
        if self.job_key is not None:
            result['X-Inference-Job-Key'] = self.job_key
        return result


class IdentityStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(fd).st_mode & 0o077:
                raise PermissionError('call identity store must be private')
        finally:
            os.close(fd)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS work_identity (
                scope TEXT NOT NULL, boundary TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
                PRIMARY KEY(scope,boundary));
            CREATE TABLE IF NOT EXISTS native_foreground_turn (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id));
            CREATE TABLE IF NOT EXISTS native_job_binding (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL, job_key TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id));
            CREATE TABLE IF NOT EXISTS native_requester_job (
                scope TEXT PRIMARY KEY, job_key TEXT NOT NULL, turn_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('active','recoverable')));
            CREATE TABLE IF NOT EXISTS native_operator_finish (
                scope TEXT NOT NULL, revision TEXT NOT NULL, job_key TEXT NOT NULL,
                turn_id TEXT NOT NULL, event_id TEXT NOT NULL,
                PRIMARY KEY(scope,revision));
            CREATE TABLE IF NOT EXISTS tool_aux_intent (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL, api_request_id TEXT NOT NULL,
                tool_call_id TEXT NOT NULL, step INTEGER NOT NULL, operation TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL CHECK(state IN ('pending','complete')),
                PRIMARY KEY(scope,turn_id,api_request_id,tool_call_id,step));
            CREATE UNIQUE INDEX IF NOT EXISTS one_pending_tool_aux ON tool_aux_intent
                (scope,turn_id,api_request_id,tool_call_id) WHERE state='pending';
            CREATE TABLE IF NOT EXISTS call_identity (
                work_key TEXT NOT NULL REFERENCES work_identity(key),
                boundary TEXT NOT NULL, key TEXT NOT NULL UNIQUE, digest TEXT NOT NULL,
                PRIMARY KEY(work_key,boundary));
            CREATE TABLE IF NOT EXISTS compaction_binding (
                scope TEXT NOT NULL, source_boundary TEXT NOT NULL,
                work_boundary TEXT NOT NULL, call_boundary TEXT NOT NULL,
                lifetime TEXT NOT NULL CHECK(lifetime IN ('execution','work_session')),
                PRIMARY KEY(scope,source_boundary));
            CREATE TABLE IF NOT EXISTS native_turn_end (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id));
            CREATE TABLE IF NOT EXISTS accepted_native_call (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL,
                grant_key TEXT NOT NULL, call_key TEXT NOT NULL,
                grant_id TEXT NOT NULL, execution_id TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id,grant_key,call_key));
            CREATE TABLE IF NOT EXISTS turn_call (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL,
                grant_key TEXT NOT NULL, call_key TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id,grant_key));
            CREATE TABLE IF NOT EXISTS turn_api_call (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL, api_request_id TEXT NOT NULL,
                grant_key TEXT NOT NULL, call_key TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id,api_request_id,grant_key));
            CREATE TABLE IF NOT EXISTS native_tool_call (
                scope TEXT NOT NULL, turn_id TEXT NOT NULL, api_request_id TEXT NOT NULL,
                tool_call_id TEXT NOT NULL, grant_key TEXT NOT NULL, call_key TEXT NOT NULL,
                PRIMARY KEY(scope,turn_id,api_request_id,tool_call_id,grant_key));
            CREATE TABLE IF NOT EXISTS lifecycle_outbox (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, turn_id TEXT NOT NULL,
                grant_key TEXT NOT NULL, call_key TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','accepted','stale')),
                UNIQUE(scope,turn_id,grant_key));
        ''')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            columns = {row[1] for row in self.db.execute('PRAGMA table_info(call_identity)')}
            if 'yield_grant_keys' not in columns:
                self.db.execute('ALTER TABLE call_identity ADD COLUMN yield_grant_keys TEXT')
            if 'job_key' not in columns:
                # NULL is a frozen legacy binding, not an invitation to attach
                # an already allocated/accepted call to a newly inferred job.
                self.db.execute('ALTER TABLE call_identity ADD COLUMN job_key TEXT')
            outbox_columns = {row[1] for row in self.db.execute('PRAGMA table_info(lifecycle_outbox)')}
            if 'kind' not in outbox_columns:
                self.db.execute("ALTER TABLE lifecycle_outbox ADD COLUMN kind TEXT NOT NULL DEFAULT 'grant_yield'")
            if 'job_key' not in outbox_columns:
                self.db.execute('ALTER TABLE lifecycle_outbox ADD COLUMN job_key TEXT')
            compaction_columns = {row[1] for row in self.db.execute('PRAGMA table_info(compaction_binding)')}
            for column in ('model','shape_digest'):
                if column not in compaction_columns:
                    self.db.execute('ALTER TABLE compaction_binding ADD COLUMN '+column+' TEXT')
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _bind_in_transaction(self, scope, work_boundary, call_boundary, request, *, job_turn=''):
        if job_turn != '':
            required(job_turn)
        digest = hashlib.sha256(encoded(request).encode()).hexdigest()
        row = self.db.execute('SELECT key FROM work_identity WHERE scope=? AND boundary=?',
                              (scope, work_boundary)).fetchone()
        grant = row[0] if row else uuid.uuid4().hex
        if row is None:
            self.db.execute('INSERT INTO work_identity VALUES(?,?,?)', (scope, work_boundary, grant))
        row = self.db.execute('SELECT key,digest,job_key FROM call_identity WHERE work_key=? AND boundary=?',
                              (grant, call_boundary)).fetchone()
        if row and row[1] != digest:
            raise Conflict('persisted call boundary has different request content; reconcile before submitting')
        call = row[0] if row else uuid.uuid4().hex
        job_key = row[2] if row else None
        if row is None:
            if job_turn:
                required(job_turn)
                bound = self.db.execute('SELECT job_key FROM native_job_binding WHERE scope=? AND turn_id=?',
                                        (scope, job_turn)).fetchone()
                job_key = bound[0] if bound else 'job:' + hashlib.sha256(encoded([scope, job_turn]).encode()).hexdigest()
                self.db.execute('INSERT OR IGNORE INTO native_job_binding VALUES(?,?,?)',
                                (scope, job_turn, job_key))
            self.db.execute('INSERT INTO call_identity(work_key,boundary,key,digest,job_key) VALUES(?,?,?,?,?)',
                            (grant, call_boundary, call, digest, job_key))
        return CallKeys(grant, call, job_key=job_key)

    def bind(self, scope, work_boundary, call_boundary, request, *, job_turn=''):
        for value in (scope, work_boundary, call_boundary):
            required(value)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            keys = self._bind_in_transaction(scope, work_boundary, call_boundary, request, job_turn=job_turn)
            self.db.execute('COMMIT')
            return keys
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _saved_compaction(self, scope, work, call, *, required_receipt=True):
        rows = self.db.execute('SELECT DISTINCT w.key,c.key,a.grant_id,a.execution_id,c.job_key '
            'FROM work_identity w JOIN call_identity c ON c.work_key=w.key '
            'JOIN accepted_native_call a ON a.scope=w.scope AND a.grant_key=w.key AND a.call_key=c.key '
            'WHERE w.scope=? AND w.boundary=? AND c.boundary=?', (scope,work,call)).fetchall()
        if not rows and not required_receipt:
            return None
        if len(rows) != 1:
            raise Conflict('saved summary has no exact acknowledged intake; restore its original model and '
                           'request to reattach the same keys, never delete the mapping or create a new grant')
        grant, key, grant_id, execution_id, job_key = rows[0]
        return CallKeys(grant,key,work_session=True,resume_job=(grant_id,execution_id),job_key=job_key)

    def bind_compaction(self, scope, source_boundary, request, *, stint_model='', job_turn=''):
        """Freeze source policy. A model-only switch can READ its acknowledged job.

        Prompt/options changes cannot borrow an old summary. Unknown intake is
        recovered with the original model/request and keys, never a new grant.
        """
        required(scope); required(source_boundary)
        shape = hashlib.sha256(encoded({k:v for k,v in request.items() if k!='model'}).encode()).hexdigest()
        legacy = 'execution:' + hashlib.sha256(
            encoded(['compression', source_boundary]).encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT work_boundary,call_boundary,lifetime,model,shape_digest '
                'FROM compaction_binding WHERE scope=? AND source_boundary=?',
                (scope, source_boundary)).fetchone()
            if row is None:
                old = self.db.execute('SELECT 1 FROM work_identity WHERE scope=? AND boundary=?',
                                      (scope, legacy)).fetchone()
                if old is None and stint_model and stint_model == required(request.get('model')):
                    work = 'conversation-stint:' + hashlib.sha256(encoded(['main', stint_model]).encode()).hexdigest()
                    row = (work, 'compression:' + source_boundary, 'work_session',request.get('model'),shape)
                else:
                    row = (legacy, source_boundary, 'execution',request.get('model'),shape)
                self.db.execute('INSERT INTO compaction_binding '
                    '(scope,source_boundary,work_boundary,call_boundary,lifetime,model,shape_digest) '
                    'VALUES(?,?,?,?,?,?,?)',(scope,source_boundary,*row))
            work, call, lifetime, original_model, original_shape = row
            if lifetime == 'work_session':
                if not stint_model:
                    raise Conflict('compaction continuation requires an exact live foreground owner')
                expected = 'conversation-stint:' + hashlib.sha256(encoded(['main', stint_model]).encode()).hexdigest()
                if work != expected or request.get('model') != stint_model:
                    if not original_model or original_shape != shape:
                        raise Conflict('saved summary prompt/options differ; restore the original model and '
                                       'request before resuming this exact source')
                    keys = self._saved_compaction(scope,work,call)
                    self.db.execute('COMMIT')
                    return keys
            keys = self._bind_in_transaction(scope, work, call, request, job_turn=job_turn)
            if original_model is None or original_shape is None:
                # Backfill only after the original request digest has matched.
                self.db.execute('UPDATE compaction_binding SET model=?,shape_digest=? '
                    'WHERE scope=? AND source_boundary=?',(request.get('model'),shape,scope,source_boundary))
            saved = self._saved_compaction(scope,work,call,required_receipt=False) if lifetime == 'work_session' else None
            self.db.execute('COMMIT')
            return saved or replace(keys, work_session=lifetime == 'work_session')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def close(self):
        self.db.close()

    def foreground_turn(self,scope,turn_id,*,record=False):
        required(scope);required(turn_id)
        if record:
            self.db.execute('INSERT OR IGNORE INTO native_foreground_turn VALUES(?,?)',(scope,turn_id))
        return self.db.execute('SELECT 1 FROM native_foreground_turn WHERE scope=? AND turn_id=?',
                               (scope,turn_id)).fetchone() is not None

    def begin_native_job(self, scope, turn_id, *, is_current=None):
        """Called by the explicit foreground owner, never by username discovery.

        A new native attempt on this SAME conversation resumes its unfinished
        parent. Successful completion removes that parent; a sibling scope or
        a subsequent completed job gets a separate key.
        """
        required(scope); required(turn_id)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if is_current is not None and is_current() is not True:
                raise LocalSourceRefused('native foreground ended before its parent binding committed')
            self.foreground_turn(scope, turn_id, record=True)
            if self.db.execute('SELECT 1 FROM native_turn_end WHERE scope=? AND turn_id=?',
                               (scope, turn_id)).fetchone():
                self.db.execute('COMMIT')
                return None
            bound = self.db.execute('SELECT job_key FROM native_job_binding WHERE scope=? AND turn_id=?',
                                    (scope, turn_id)).fetchone()
            active = self.db.execute('SELECT job_key,turn_id FROM native_requester_job WHERE scope=?',
                                     (scope,)).fetchone()
            if bound and active and bound[0] != active[0]:
                raise Conflict('native turn belongs to another parent job')
            if bound and active and active[1] != turn_id:
                raise LocalSourceRefused('native attempt was superseded by its recovery owner')
            key = (bound[0] if bound else active[0] if active else
                   'job:' + hashlib.sha256(encoded([scope, turn_id]).encode()).hexdigest())
            self.db.execute('INSERT OR IGNORE INTO native_job_binding VALUES(?,?,?)', (scope, turn_id, key))
            self.db.execute("INSERT INTO native_requester_job VALUES(?,?,?,'active') "
                "ON CONFLICT(scope) DO UPDATE SET turn_id=excluded.turn_id,state='active'", (scope, key, turn_id))
            self.db.execute('COMMIT')
            return key
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def observe_native_outcome(self, scope, turn_id, *, completed, failed, interrupted):
        required(scope); required(turn_id)
        if any(type(value) is not bool for value in (completed, failed, interrupted)):
            raise LocalSourceRefused('native completion verdict is missing or invalid')
        if completed and (failed or interrupted):
            raise LocalSourceRefused('native completion verdict is contradictory')
        if completed:
            return self.finish_turn(scope, turn_id)
        # Late failure metadata cannot demote the subsequent live attempt.
        self.db.execute("UPDATE native_requester_job SET state='recoverable' WHERE scope=? AND turn_id=?",
                        (scope, turn_id))
        return 0

    def record_turn_call(self, scope, turn_id, keys, *, api_request_id='', job=None):
        for value in (scope, turn_id):
            required(value)
        if api_request_id:
            required(api_request_id)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            owned = self.db.execute('SELECT c.job_key FROM work_identity w JOIN call_identity c '
                'ON c.work_key=w.key WHERE w.scope=? AND w.key=? AND c.key=?',
                (scope, keys.grant_key, keys.call_key)).fetchone()
            if owned is None:
                raise Conflict('native turn does not own the durable call')
            if owned[0] != getattr(keys, 'job_key', None):
                raise Conflict('native call changed its frozen parent job')
            if job is not None:
                required(job.grant_id); required(job.execution_id)
                old = self.db.execute('SELECT grant_id,execution_id FROM accepted_native_call '
                    'WHERE scope=? AND grant_key=? AND call_key=?',
                    (scope,keys.grant_key,keys.call_key)).fetchone()
                if old and old != (job.grant_id,job.execution_id):
                    raise Conflict('intake receipt changed its exact service identity')
                self.db.execute('INSERT OR IGNORE INTO accepted_native_call VALUES(?,?,?,?,?,?)',
                    (scope,turn_id,keys.grant_key,keys.call_key,job.grant_id,job.execution_id))
            if api_request_id:
                # An API counter can survive retry/recovery. This is only a
                # candidate; successful intake binds the actual response tool IDs.
                self.db.execute('INSERT INTO turn_api_call '
                    '(scope,turn_id,api_request_id,grant_key,call_key) VALUES(?,?,?,?,?) '
                    'ON CONFLICT(scope,turn_id,api_request_id,grant_key) DO UPDATE SET call_key=excluded.call_key',
                    (scope,turn_id,api_request_id,keys.grant_key,keys.call_key))
            self.db.execute('INSERT INTO turn_call VALUES(?,?,?,?) '
                'ON CONFLICT(scope,turn_id,grant_key) DO UPDATE SET call_key=excluded.call_key',
                (scope, turn_id, keys.grant_key, keys.call_key))
            if self.db.execute('SELECT 1 FROM native_turn_end WHERE scope=? AND turn_id=?',
                               (scope,turn_id)).fetchone():
                # A late intake cannot rewrite an already relayed event. Give it
                # a new fenced event; a subsequent real call makes it stale safely.
                boundary = 'late-intake:'+turn_id+':'+keys.call_key
                if keys.job_key is not None:
                    active = self.db.execute(
                        'SELECT job_key,turn_id FROM native_requester_job WHERE scope=?',
                        (scope,)).fetchone()
                    if not (active and active[0] == keys.job_key and active[1] != turn_id):
                        self._queue_job_finishes(scope, boundary, (keys.job_key,))
                else:
                    self._queue_yields(scope,boundary,[(keys.grant_key,keys.call_key)])
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _queue_yields(self, scope, boundary, rows):
        # Caller owns the transaction. The legacy turn_id column carries the
        # event boundary: a native turn or a namespaced tool-dependency event.
        for grant_key, call_key in rows:
            old = self.db.execute('SELECT call_key FROM lifecycle_outbox '
                'WHERE scope=? AND turn_id=? AND grant_key=?', (scope,boundary,grant_key)).fetchone()
            if old and old[0] != call_key:
                raise Conflict('lifecycle event changed its call boundary')
            self.db.execute('INSERT OR IGNORE INTO lifecycle_outbox '
                '(id,scope,turn_id,grant_key,call_key,state) VALUES(?,?,?,?,?,?)',
                (uuid.uuid4().hex,scope,boundary,grant_key,call_key,'pending'))

    def _queue_job_finishes(self, scope, boundary, keys):
        for key in keys:
            self.db.execute('INSERT OR IGNORE INTO lifecycle_outbox '
                '(id,scope,turn_id,grant_key,call_key,state,kind,job_key) '
                "VALUES(?,?,?,?,'finish','pending','job_finish',?)",
                (uuid.uuid4().hex, scope, boundary, key, key))

    def complete_api(self, scope, turn_id, api_request_id, tool_call_ids):
        """Freeze actual returned tool IDs, not the reusable native API counter."""
        for value in (scope,turn_id,api_request_id):
            required(value)
        if not isinstance(tool_call_ids,tuple):
            raise ValueError('explicit returned tool identities required')
        for value in tool_call_ids:
            required(value)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rows = self.db.execute('SELECT grant_key,call_key FROM turn_api_call '
                'WHERE scope=? AND turn_id=? AND api_request_id=?',
                (scope,turn_id,api_request_id)).fetchall()
            for tool in tool_call_ids:
                for grant,call in rows:
                    identity = (scope,turn_id,api_request_id,tool,grant)
                    old = self.db.execute('SELECT call_key FROM native_tool_call '
                        'WHERE scope=? AND turn_id=? AND api_request_id=? AND tool_call_id=? AND grant_key=?',
                        identity).fetchone()
                    if old is not None and old[0] != call:
                        raise Conflict('native tool identity changed its durable call boundary')
                    self.db.execute('INSERT OR IGNORE INTO native_tool_call VALUES(?,?,?,?,?,?)',
                                    (*identity,call))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def dependency_wait(self, scope, turn_id, api_request_id, tool_call_id):
        """Explicit external wait, fenced to its originating API/tool call.

        Never selects the current/latest call in the turn. Delayed/replayed
        events keep their OLD expected-call fence; they cannot yield newer work.
        No tool command, prompt or result is retained in this identifier outbox.
        """
        for value in (scope,turn_id,api_request_id,tool_call_id):
            required(value)
        boundary = 'dependency:' + hashlib.sha256(encoded(
            [turn_id,api_request_id,tool_call_id]).encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rows = self.db.execute('SELECT grant_key,call_key FROM native_tool_call '
                'WHERE scope=? AND turn_id=? AND api_request_id=? AND tool_call_id=?',
                (scope,turn_id,api_request_id,tool_call_id)).fetchall()
            if any(self.db.execute('SELECT job_key FROM call_identity WHERE key=?', (call,)).fetchone()[0]
                   is not None for _, call in rows):
                raise LocalSourceRefused('linked job dependency requires qualified delegation or explicit job yield')
            self._queue_yields(scope,boundary,rows)
            self.db.execute('COMMIT')
            return len(rows)
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish_turn(self, scope, turn_id):
        """Persist identifiers only; the native callback performs no network I/O."""
        for value in (scope, turn_id):
            required(value)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT OR IGNORE INTO native_turn_end VALUES(?,?)',(scope,turn_id))
            rows = self.db.execute('SELECT t.grant_key,t.call_key,c.job_key FROM turn_call t '
                'JOIN call_identity c ON c.key=t.call_key WHERE t.scope=? AND t.turn_id=?',
                (scope, turn_id)).fetchall()
            jobs = {row[2] for row in rows if row[2] is not None}
            jobs.update(row[0] for row in self.db.execute('SELECT job_key FROM native_job_binding '
                                                       'WHERE scope=? AND turn_id=?', (scope, turn_id)))
            active = self.db.execute('SELECT job_key,turn_id FROM native_requester_job WHERE scope=?',
                                     (scope,)).fetchone()
            if active and active[1] != turn_id:
                jobs.discard(active[0])  # delayed old success cannot finish a resumed live parent
            self._queue_job_finishes(scope, turn_id, sorted(jobs))
            self.db.execute('DELETE FROM native_requester_job WHERE scope=? AND turn_id=?', (scope, turn_id))
            legacy = [(grant, call) for grant, call, job in rows if job is None]
            self._queue_yields(scope,turn_id,legacy)
            self.db.execute('COMMIT')
            return len(legacy) + len(jobs)
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def native_job_status(self, scope):
        """Exact owning scope only; revision fences a later resumed attempt."""
        required(scope)
        row = self.db.execute('SELECT job_key,turn_id,state FROM native_requester_job WHERE scope=?',
                              (scope,)).fetchone()
        if row is None:
            return None
        revision = hashlib.sha256(encoded([scope, *row]).encode()).hexdigest()
        return dict(job_key=row[0], turn_id=row[1], state=row[2], revision=revision)

    def request_native_job_finish(self, scope, revision):
        """Explicit operator intent, not inferred abandonment or cancellation.

        Seal every attempt bound to this parent before any late intake. The
        service still drains accepted/uncertain children; the exact finish is
        durable even when delivery fails or this process exits immediately.
        """
        required(scope); required(revision)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute('SELECT job_key,turn_id,event_id FROM native_operator_finish '
                                   'WHERE scope=? AND revision=?', (scope, revision)).fetchone()
            if prior is not None:
                result = dict(job_key=prior[0], turn_id=prior[1], event_id=prior[2], revision=revision)
            else:
                current = self.native_job_status(scope)
                if current is None or current['revision'] != revision:
                    raise Conflict('parent changed since status; inspect the exact owning session again')
                key = current['job_key']
                self.db.execute('INSERT OR IGNORE INTO native_turn_end SELECT scope,turn_id '
                    'FROM native_job_binding WHERE scope=? AND job_key=?', (scope,key))
                boundary = 'operator-finish:' + revision
                self._queue_job_finishes(scope, boundary, (key,))
                event = self.db.execute('SELECT id FROM lifecycle_outbox WHERE scope=? AND turn_id=? '
                                        'AND grant_key=?', (scope,boundary,key)).fetchone()[0]
                self.db.execute('INSERT INTO native_operator_finish VALUES(?,?,?,?,?)',
                                (scope,revision,key,current['turn_id'],event))
                self.db.execute('DELETE FROM native_requester_job WHERE scope=? AND job_key=?', (scope,key))
                result = dict(job_key=key,turn_id=current['turn_id'],event_id=event,revision=revision)
            self.db.execute('COMMIT')
            return {**result, 'state':'finish_requested'}
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def operator_finish_delivery(self, event_id):
        required(event_id)
        row = self.db.execute('SELECT rowid,id,grant_key,call_key,job_key,state FROM lifecycle_outbox '
                              "WHERE id=? AND kind='job_finish'", (event_id,)).fetchone()
        if row is None:
            raise Conflict('saved operator finish event is missing')
        return dict(cursor=row[0],id=row[1],grant_key=row[2],expected_call_key=row[3],
                    job_key=row[4],state=row[5],kind='job_finish')

    def pending_event_highwater(self):
        return self.db.execute("SELECT coalesce(max(rowid),0) FROM lifecycle_outbox WHERE state='pending'").fetchone()[0]

    def pending_events(self, limit=32, *, after=0):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('bounded event page required')
        if type(after) is not int or after < 0:
            raise ValueError('valid event cursor required')
        result = []
        for row in self.db.execute("SELECT rowid,id,grant_key,call_key,kind,job_key FROM lifecycle_outbox "
                                   "WHERE state='pending' AND rowid>? ORDER BY rowid LIMIT ?", (after, limit)):
            event = dict(zip(('cursor','id','grant_key','expected_call_key'), row[:4]))
            if row[4] != 'grant_yield':
                event.update(kind=row[4], job_key=row[5])
            result.append(event)
        return result

    def acknowledge_event(self, event, verdict):
        if verdict not in ('accepted', 'stale'):
            raise ValueError('explicit lifecycle acknowledgement required')
        self.db.execute("UPDATE lifecycle_outbox SET state=? WHERE id=? AND grant_key=? "
                        "AND call_key=? AND state='pending'", (verdict, event['id'],
                        event['grant_key'], event['expected_call_key']))

    def conversation_stints(self, scope):
        """Explicit owning conversation only, never a profile's latest work."""
        required(scope)
        return tuple(row[0] for row in self.db.execute(
            "SELECT key FROM work_identity WHERE scope=? AND boundary LIKE 'conversation-stint:%' ORDER BY key",
            (scope,)))

    def bind_handoff(self, keys, parents):
        """Freeze dependency intent with call identity BEFORE its first submit.

        A later-created parent cannot change a saved summary's retry metadata.
        An empty first parent set is meaningful and must be persisted too.
        """
        import json
        value = encoded([] if keys.job_key is not None else sorted(set(parents)))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT yield_grant_keys FROM call_identity WHERE key=? AND work_key=?',
                                  (keys.call_key, keys.grant_key)).fetchone()
            if row is None:
                raise Conflict('handoff requires an existing owning call identity')
            saved = row[0]
            if saved is None:
                self.db.execute('UPDATE call_identity SET yield_grant_keys=? WHERE key=?',
                                (value, keys.call_key))
                saved = value
            if keys.job_key is not None and json.loads(saved):
                raise Conflict('linked job cannot use legacy dependency yield')
            self.db.execute('COMMIT')
            return replace(keys, yield_grant_keys=tuple(json.loads(saved)))
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def bind_execution(self, scope, work_boundary, call_boundary, request, *, job_turn=''):
        """One durable grant per explicit host call, including within a tool loop.

        Hashes only the caller's declared boundary IDs to keep identifiers short;
        never hashes prompt equality to decide whether two intentions are one.
        A retry/restart of the same call recovers its released result too.
        """
        required(work_boundary)
        required(call_boundary)
        boundary = 'execution:' + hashlib.sha256(
            encoded([work_boundary, call_boundary]).encode()).hexdigest()
        return self.bind(scope, boundary, call_boundary, request, job_turn=job_turn)
