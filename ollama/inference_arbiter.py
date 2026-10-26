"""Durable single-resource grants, independent of transport and lease policy.

No elapsed-work deadline: a grant explicitly spans one execution or a work
session. The service releases execution grants only after confirmed backend end;
owners release work-session grants at safe boundaries. Disconnect changes nothing.
The service must enforce this journal on EVERY backend-dispatch path before it
can be used as an arbiter; this module alone is not a deployed queue.

Owners passed here must come from verified authentication, not request bodies.
Only the service execution supervisor calls finish/quarantine, never clients.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .inference_jobs import JobScheduling


class NotFound(Exception):
    """Same answer for nonexistent and other-owner records."""


class Conflict(Exception):
    pass


class LifecycleStale(Conflict):
    """The event names older work, not the current scheduling boundary."""


class CapacityDeferred(Exception):
    """Trusted pre-provider refusal, not a timeout or ambiguous backend result.

    The adapter must establish no provider operation started or remains running.
    Never translate a socket failure into this claim. A fresh authoritative
    capacity event, not another consumer request or elapsed time, permits retry.
    """
    def __init__(self, reason):
        if reason not in ('capacity_unavailable', 'work_mode'):
            raise ValueError('typed pre-provider wait reason required')
        self.reason = reason
        super().__init__(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def required(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("nonempty identifier of at most 512 characters required")
    return value


@dataclass(frozen=True)
class StintPolicy:
    """Service configuration, not consumer-selected execution deadlines.

    Event-driven by default: explicit dependency/tool yields drive handoff.
    Optional experimental clock bounds apply only between confirmed backend
    executions, never as compute deadlines. No policy is enabled implicitly.
    """
    idle_grace_seconds: float | None = None
    fairness_seconds: float | None = None

    def __post_init__(self):
        for value in (self.idle_grace_seconds, self.fairness_seconds):
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError('positive finite stint policy bounds required')


class Arbiter(JobScheduling):
    def __init__(self, path, *, stint_policy=None):
        if stint_policy is not None and not isinstance(stint_policy, StintPolicy):
            raise TypeError('stint_policy must be service-owned StintPolicy')
        self.stint_policy = stint_policy
        path = Path(path)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(fd).st_mode & 0o077:
                raise PermissionError("inference journal must be private (0600)")
        finally:
            os.close(fd)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS grants (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                owner TEXT NOT NULL,
                client_key TEXT NOT NULL,
                resource TEXT NOT NULL,
                model TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN
                    ('queued','granted','uncertain','released','cancelled')),
                release_requested INTEGER NOT NULL DEFAULT 0,
                active_execution TEXT,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                UNIQUE(owner,client_key)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_resource_owner
                ON grants(resource) WHERE state IN ('granted','uncertain');
            CREATE TABLE IF NOT EXISTS executions (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL REFERENCES grants(id),
                client_key TEXT NOT NULL,
                request TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN
                    ('queued','running','uncertain','complete','failed','cancelled')),
                result TEXT,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                UNIQUE(grant_id,client_key)
            );
            CREATE TABLE IF NOT EXISTS execution_events (
                execution_id TEXT NOT NULL REFERENCES executions(id),
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('delta','terminal')),
                data TEXT NOT NULL,
                PRIMARY KEY(execution_id,seq)
            );
            -- Metadata-only transitions, committed with the state they describe.
            -- No synthetic backfill: old journals lack historical timing.
            CREATE TABLE IF NOT EXISTS execution_lifecycle (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL REFERENCES executions(id),
                state TEXT NOT NULL,
                at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS resource_waits (
                resource TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                since REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS lifecycle_execution
                ON execution_lifecycle(execution_id,seq);
            CREATE TRIGGER IF NOT EXISTS lifecycle_insert AFTER INSERT ON executions
            BEGIN
                INSERT INTO execution_lifecycle(execution_id,state,at)
                VALUES(NEW.id,NEW.state,NEW.created);
            END;
            CREATE TRIGGER IF NOT EXISTS lifecycle_change AFTER UPDATE OF state ON executions
            WHEN NEW.state != OLD.state
            BEGIN
                INSERT INTO execution_lifecycle(execution_id,state,at)
                VALUES(NEW.id,NEW.state,NEW.updated);
            END;
        """)
        # Existing offline journals used work-session semantics. Serialize the
        # additive migration so independent callers cannot race ALTER TABLE.
        with self.transaction():
            columns = {row['name'] for row in self.db.execute('PRAGMA table_info(grants)')}
            if 'lifetime' not in columns:
                self.db.execute("ALTER TABLE grants ADD COLUMN lifetime TEXT NOT NULL "
                    "DEFAULT 'work_session' CHECK(lifetime IN ('work_session','execution'))")
            for name, declaration in (
                ('queue_order', 'INTEGER'), ('stint_started', 'REAL'),
                ('idle_since', 'REAL'), ('yield_requested', 'INTEGER NOT NULL DEFAULT 0'),
                ('handoff_reason', 'TEXT'),
            ):
                if name not in columns:
                    self.db.execute(f'ALTER TABLE grants ADD COLUMN {name} {declaration}')
            self.db.execute('UPDATE grants SET queue_order=seq WHERE queue_order IS NULL')
            self.db.execute("""UPDATE grants SET stint_started=COALESCE(stint_started,updated),
                idle_since=COALESCE(idle_since,updated)
                WHERE state='granted' AND active_execution IS NULL""")
            execution_columns = {row['name'] for row in self.db.execute('PRAGMA table_info(executions)')}
            if 'yield_grant_keys' not in execution_columns:
                self.db.execute("ALTER TABLE executions ADD COLUMN yield_grant_keys TEXT NOT NULL DEFAULT '[]'")
            if 'wait_reason' not in execution_columns:
                self.db.execute('ALTER TABLE executions ADD COLUMN wait_reason TEXT')
            self._install_jobs()
        if self.stint_policy is None and self.db.execute(
                "SELECT 1 FROM requester_jobs WHERE state!='closed'").fetchone():
            self.db.close()
            raise Conflict('unfinished requester jobs require their qualified scheduler')
        from .inference_telemetry import install, notify
        install(self.db)
        self._telemetry_seq = self.db.execute('SELECT COALESCE(MAX(seq),0) FROM metadata_events').fetchone()[0]
        notify(self.path,self._telemetry_seq)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if hasattr(self,'_telemetry_seq'):
            latest=self.db.execute('SELECT COALESCE(MAX(seq),0) FROM metadata_events').fetchone()[0]
            if latest != self._telemetry_seq:
                self._telemetry_seq=latest
                from .inference_telemetry import notify
                notify(self.path,latest)

    def capacity_event(self,resource,model,state,reason):
        """Trusted capacity source only; metadata committed before dispatch wake."""
        if state not in ('available','unavailable','unknown'):
            raise ValueError('explicit observation state required')
        with self.transaction():
            self.db.execute('''INSERT INTO metadata_events(kind,at,resource,model,state,reason)
                VALUES('capacity',?,?,?,?,?)''',(time.time(),resource,model,state,reason))

    def _owned(self, grant_id, owner):
        row = self.db.execute("SELECT * FROM grants WHERE id=? AND owner=?",
                              (grant_id, owner)).fetchone()
        if row is None:
            raise NotFound()
        return row

    def _next_order(self):
        # Called inside a write transaction, including across journal connections.
        return self.db.execute('SELECT COALESCE(MAX(queue_order),0)+1 FROM grants').fetchone()[0]

    def _ready_grant(self, resource, excluding='', permitted_model=None):
        rows = self.db.execute("""SELECT g.* FROM grants g
            WHERE g.resource=? AND g.state='queued' AND g.id!=? AND EXISTS
            (SELECT 1 FROM executions e WHERE e.grant_id=g.id AND e.state='queued')
            ORDER BY g.queue_order""", (resource, excluding)).fetchall()
        return next((row for row in rows if permitted_model is None or
                     permitted_model(resource,row['model']) is True),None)

    def pending_models(self, resource):
        """Actual saved inference demand, not an idle requester's affinity."""
        return [row['model'] for row in self.db.execute('''SELECT g.model, MIN(e.created) AS first
            FROM executions e JOIN grants g ON g.id=e.grant_id
            WHERE g.resource=? AND e.state='queued' AND g.state IN ('queued','granted')
                AND g.release_requested=0 GROUP BY g.model ORDER BY first,g.model''', (resource,))]

    def _handoff(self, grant, reason, now):
        """Yield scheduling preference, not residency, saved results or executions."""
        if grant['active_execution'] or grant['state'] != 'granted':
            raise Conflict('cannot hand off running or uncertain work')
        self.db.execute("""UPDATE grants SET state='queued',queue_order=?,
            stint_started=NULL,idle_since=NULL,yield_requested=0,handoff_reason=?,updated=?
            WHERE id=?""", (self._next_order(), reason, now, grant['id']))

    def next_handoff_at(self, resource):
        """Advisory scheduler wake-up; dispatch rechecks authority transactionally."""
        if self.owns_job(resource):
            return None
        policy = self.stint_policy
        if policy is None:
            return None
        row = self.db.execute("SELECT * FROM grants WHERE resource=? AND state='granted'",
                              (resource,)).fetchone()
        if (row is None or row['active_execution'] or row['release_requested'] or
                row['lifetime'] != 'work_session' or self._ready_grant(resource, row['id']) is None):
            return None
        bounds = [stamp + interval for stamp, interval in (
            (row['idle_since'], policy.idle_grace_seconds),
            (row['stint_started'], policy.fairness_seconds)) if stamp is not None and interval is not None]
        return min(bounds) if bounds else None

    def submit(self, owner, client_key, resource, model, lifetime='work_session'):
        """Retrying the same key finds the same grant; changing its ask conflicts."""
        for value in (owner, client_key, resource, model):
            required(value)
        if lifetime not in ('work_session', 'execution'):
            raise ValueError('invalid grant lifetime')
        with self.transaction():
            old = self.db.execute("SELECT * FROM grants WHERE owner=? AND client_key=?",
                                  (owner, client_key)).fetchone()
            if old:
                if (old['resource'], old['model'], old['lifetime']) != (resource, model, lifetime):
                    raise Conflict("idempotency key already names another request")
                return dict(old)
            now = time.time()
            grant_id = uuid.uuid4().hex
            self.db.execute("""INSERT INTO grants
                (id,owner,client_key,resource,model,state,created,updated,lifetime,queue_order)
                VALUES (?,?,?,?,?,'queued',?,?,?,?)""",
                (grant_id, owner, client_key, resource, model, now, now, lifetime, self._next_order()))
            return dict(self._owned(grant_id, owner))

    def get(self, grant_id, owner):
        return dict(self._owned(grant_id, owner))

    def execution(self, execution_id, owner):
        row = self.db.execute("""SELECT e.* FROM executions e
            JOIN grants g ON g.id=e.grant_id WHERE e.id=? AND g.owner=?""",
            (execution_id, owner)).fetchone()
        if row is None:
            raise NotFound()
        return dict(row)

    def append_event(self, execution_id, data, *, terminal=False):
        """Supervisor only: persist output before making it observable.

        Terminal frames remain hidden until finish commits complete. A provider
        can emit a finish frame and subsequently fail; partial output is not a
        successful result. No consumer route may append to this journal.
        """
        if not isinstance(data, str):
            raise ValueError('event must be text')
        with self.transaction():
            row = self.db.execute('SELECT state FROM executions WHERE id=?',
                                  (execution_id,)).fetchone()
            if row is None:
                raise NotFound()
            if row['state'] != 'running':
                raise Conflict('only running work can append output')
            if not terminal and self.db.execute("SELECT 1 FROM execution_events "
                    "WHERE execution_id=? AND kind='terminal' LIMIT 1", (execution_id,)).fetchone():
                raise Conflict('output after a terminal frame must remain provisional')
            seq = self.db.execute('SELECT COALESCE(MAX(seq),0)+1 FROM execution_events '
                                  'WHERE execution_id=?', (execution_id,)).fetchone()[0]
            self.db.execute('INSERT INTO execution_events VALUES (?,?,?,?)',
                            (execution_id, seq, 'terminal' if terminal else 'delta', data))
            return seq

    def events(self, execution_id, owner, *, after=0, limit=100):
        """Owner-scoped cursor read; polling/reconnecting never dispatches work.

        State and events are one snapshot. A terminal state may still have more
        pages: callers drain has_more before treating the stream as complete.
        """
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('invalid event cursor or page size')
        with self.transaction():
            call = self.execution(execution_id, owner)
            rows = self.db.execute('SELECT seq,kind,data FROM execution_events '
                "WHERE execution_id=? AND seq>? AND (kind='delta' OR ?) ORDER BY seq LIMIT ?",
                (execution_id, after, call['state'] == 'complete', limit + 1)).fetchall()
            page = [dict(row) for row in rows[:limit]]
            return {'execution_id': execution_id, 'state': call['state'], 'events': page,
                    'next_cursor': page[-1]['seq'] if page else after, 'has_more': len(rows) > limit}

    def existing_execution(self, grant_id, owner, client_key, request, *, yield_grant_keys=(), job_key=None):
        """Owner-scoped retry check before installing a durable input.

        Advisory only: enqueue repeats it inside its write transaction.
        """
        required(client_key)
        body = encoded(request)
        grant = self._owned(grant_id,owner)
        old = self.db.execute('SELECT * FROM executions WHERE grant_id=? AND client_key=?',
                              (grant_id,client_key)).fetchone()
        if old is not None:
            if old['request']!=body or old['yield_grant_keys'] != encoded(list(yield_grant_keys)):
                raise Conflict('execution key already names another request')
            linked = self._find_job(owner, job_key, grant['resource']) if job_key is not None else None
            if ((job_key is not None and linked is None) or
                    old['job_id'] != (linked['id'] if linked is not None else None)):
                raise Conflict('execution key already names another job')
            return dict(old)
        return None

    def enqueue(self, grant_id, owner, client_key, request, *, yield_grant_keys=(), job_key=None):
        """Persist a call while its grant waits; reconnect never submits twice.

        One pending/active call per grant. The grant can span multiple sequential
        calls; this is not a per-request lease policy. Queue wait has no deadline.
        """
        required(client_key)
        if not isinstance(yield_grant_keys, (tuple, list)) or len(yield_grant_keys) > 32:
            raise ValueError('bounded dependency grant keys required')
        for key in yield_grant_keys:
            required(key)
        yield_grant_keys = tuple(sorted(set(yield_grant_keys)))
        body = encoded(request)
        with self.transaction():
            grant = self._owned(grant_id, owner)
            old = self.existing_execution(grant_id,owner,client_key,request,
                                          yield_grant_keys=yield_grant_keys,job_key=job_key)
            if old is not None:
                return old
            parents = []
            if yield_grant_keys and self.stint_policy is None:
                raise Conflict('dependency handoff requires automatic stint policy')
            for key in yield_grant_keys:
                parent = self.db.execute('SELECT * FROM grants WHERE owner=? AND client_key=?',
                                         (owner, key)).fetchone()
                # Absent or other-owner keys are indistinguishable. A locally
                # allocated parent may never have been submitted before preflight.
                if parent is None:
                    continue
                if parent['id'] == grant_id or parent['lifetime'] != 'work_session':
                    raise Conflict('dependency parent must be another work session')
                if parent['resource'] == grant['resource']:
                    parents.append(parent)
            pending = self.db.execute("SELECT 1 FROM executions WHERE grant_id=? "
                                      "AND state='queued'", (grant_id,)).fetchone()
            if (grant['state'] not in ('queued', 'granted') or
                    grant['release_requested'] or grant['active_execution'] or pending):
                raise Conflict("grant cannot accept another execution")
            if job_key is not None and yield_grant_keys:
                raise Conflict('job continuity and legacy dependency yield are distinct contracts')
            if job_key is not None and self.stint_policy is None:
                raise Conflict('job continuity requires atomic dispatch policy')
            job_id = self._bind_job(owner, job_key, grant['resource'])
            execution_id, now = uuid.uuid4().hex, time.time()
            self.db.execute("""INSERT INTO executions
                (id,grant_id,client_key,request,state,created,updated,yield_grant_keys,job_id)
                VALUES (?,?,?,?,'queued',?,?,?,?)""",
                (execution_id, grant_id, client_key, body, now, now, encoded(list(yield_grant_keys)),job_id))
            if self.stint_policy is not None and grant['state'] == 'queued':
                # Empty grants don't bank a place while their tools run. A
                # returning conversation joins when it actually has ready work.
                self.db.execute('UPDATE grants SET queue_order=? WHERE id=?',
                                (self._next_order(), grant_id))
            # Commit dependency intent and parent yield together. Replaying a
            # saved call returns above and cannot yield a later foreground stint.
            for parent in parents:
                if parent['state'] == 'granted' and not parent['release_requested']:
                    self.db.execute('UPDATE grants SET yield_requested=1 WHERE id=?', (parent['id'],))
            return self.execution(execution_id, owner)

    def dispatch_next(self, resource, *, permitted_model=None):
        """Supervisor only: atomically own one persisted call before dispatch.

        Capacity/permission must be checked first. Work-session affinity survives
        short gaps; an optional service policy hands it off only between calls.
        A running or uncertain call always blocks handoff regardless of its age.
        Queued grants without a call do not reserve the runner prematurely.
        """
        with self.transaction():
            if self.db.execute('SELECT 1 FROM resource_waits WHERE resource=?', (resource,)).fetchone():
                return None
            owned, item = self._dispatch_job(resource, permitted_model)
            if owned:
                return item
            now = time.time()
            grant = self.db.execute("SELECT * FROM grants WHERE resource=? "
                                    "AND state IN ('granted','uncertain')", (resource,)).fetchone()
            if grant is not None and (grant['state'] == 'uncertain' or
                                      grant['active_execution'] or grant['release_requested']):
                return None
            if grant is not None:
                reason = 'explicit_yield' if grant['yield_requested'] else None
                model_permission = (True if permitted_model is None else
                                    permitted_model(resource,grant['model']))
                if model_permission is False:
                    # Running/unknown work returned above. Only an idle
                    # preference can yield to another model's available fit.
                    reason = 'capacity_changed'
                policy = self.stint_policy
                if (reason is None and policy is not None and
                        grant['lifetime'] == 'work_session' and
                        self._ready_grant(resource, grant['id'],permitted_model) is not None):
                    if (policy.idle_grace_seconds is not None and grant['idle_since'] is not None
                            and now - grant['idle_since'] >= policy.idle_grace_seconds):
                        reason = 'idle_grace'
                    elif (policy.fairness_seconds is not None and grant['stint_started'] is not None
                            and now - grant['stint_started'] >= policy.fairness_seconds):
                        reason = 'fair_share'
                if reason is not None:
                    self._handoff(grant, reason, now)
                    grant = None
                elif model_permission is not True:
                    # A partial/failed observation is not capacity loss. Keep
                    # affinity, but never dispatch the incumbent on UNKNOWN.
                    return None
            if grant is None:
                grant = self._ready_grant(resource,permitted_model=permitted_model)
            if grant is None:
                return None
            call = self.db.execute("SELECT * FROM executions WHERE grant_id=? AND state='queued'",
                                   (grant['id'],)).fetchone()
            if call is None:
                return None
            self._own_call_job(call)
            return self._start_queued_call(grant, call, now)

    def _start_queued_call(self, grant, call, now):
        started = grant['stint_started'] if grant['state'] == 'granted' else now
        self.db.execute("""UPDATE grants SET state='granted',active_execution=?,updated=?,
            stint_started=?,idle_since=NULL WHERE id=?""",
                        (call['id'], now, started if started is not None else now, grant['id']))
        self.db.execute("UPDATE executions SET state='running',wait_reason=NULL,updated=? WHERE id=?", (now, call['id']))
        return {'grant': dict(self._owned(grant['id'], grant['owner'])),
                'execution': self.execution(call['id'], grant['owner'])}

    def admit_next(self, resource):
        """FIFO admission only after the supervisor establishes capacity/permission.

        Does not load a model or infer availability from a missing status read.
        An uncertain existing execution keeps its resource occupied.
        """
        if self.stint_policy is not None:
            raise Conflict('stint scheduling requires atomic dispatch_next')
        with self.transaction():
            if self.db.execute('SELECT 1 FROM resource_waits WHERE resource=?', (resource,)).fetchone():
                return None
            held = self.db.execute("SELECT 1 FROM grants WHERE resource=? "
                                   "AND state IN ('granted','uncertain')", (resource,)).fetchone()
            if held:
                return None
            row = self.db.execute("SELECT * FROM grants WHERE resource=? AND state='queued' "
                                  "ORDER BY seq LIMIT 1", (resource,)).fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE grants SET state='granted',updated=? WHERE id=?",
                            (time.time(), row['id']))
            return dict(self.db.execute("SELECT * FROM grants WHERE id=?", (row['id'],)).fetchone())

    def start(self, grant_id, owner, client_key, request):
        """Persist execution intent BEFORE dispatch. Only new=True permits dispatch.

        A retry returns its existing execution/result, even after release. It must
        never cause another provider call. A crash after this commit is uncertain,
        not permission to replay an inference whose outcome is unknown.
        """
        if self.stint_policy is not None:
            raise Conflict('stint scheduling requires atomic dispatch_next')
        required(client_key)
        body = encoded(request)
        with self.transaction():
            grant = self._owned(grant_id, owner)
            old = self.db.execute("SELECT * FROM executions WHERE grant_id=? AND client_key=?",
                                  (grant_id, client_key)).fetchone()
            if old:
                if old['request'] != body:
                    raise Conflict("execution key already names another request")
                return dict(old), False
            pending = self.db.execute("SELECT 1 FROM executions WHERE grant_id=? AND state='queued'",
                                      (grant_id,)).fetchone()
            if grant['state'] != 'granted' or grant['release_requested'] or grant['active_execution'] or pending:
                raise Conflict("grant is not available for a new execution")
            execution_id, now = uuid.uuid4().hex, time.time()
            self.db.execute("""INSERT INTO executions
                (id,grant_id,client_key,request,state,created,updated)
                VALUES (?,?,?,?,'running',?,?)""",
                (execution_id, grant_id, client_key, body, now, now))
            self.db.execute("UPDATE grants SET active_execution=?,updated=? WHERE id=?",
                            (execution_id, now, grant_id))
            return dict(self.db.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()), True

    def request_release(self, grant_id, owner):
        """Release idle/queued grants; active work drains without admitting another."""
        with self.transaction():
            row = self._owned(grant_id, owner)
            if row['state'] == 'queued':
                state = 'cancelled'
            elif row['active_execution']:
                state = row['state']
            elif row['state'] in ('released', 'cancelled'):
                return dict(row)
            else:
                state = 'released'
            self.db.execute("UPDATE grants SET state=?,release_requested=1,updated=? WHERE id=?",
                            (state, time.time(), grant_id))
            jobs = self.db.execute("SELECT DISTINCT job_id FROM executions "
                "WHERE grant_id=? AND state='queued' AND job_id IS NOT NULL", (grant_id,)).fetchall()
            self.db.execute("UPDATE executions SET state='cancelled',updated=? "
                            "WHERE grant_id=? AND state='queued'", (time.time(), grant_id))
            for job in jobs:
                self._advance_job(job['job_id'])
            return dict(self._owned(grant_id, owner))

    def yield_by_key(self, owner, grant_key, expected_call_key):
        """A delayed lifecycle event cannot yield a later native call.

        Grant/call keys are durable intent IDs, not model finish reasons. This
        event changes scheduling preference only; it proves no backend ended.
        """
        required(grant_key)
        required(expected_call_key)
        with self.transaction():
            row = self.db.execute('SELECT * FROM grants WHERE owner=? AND client_key=?',
                                  (owner, grant_key)).fetchone()
            if row is None:
                raise NotFound()
            latest = self.db.execute('SELECT client_key FROM executions WHERE grant_id=? '
                                     'ORDER BY rowid DESC LIMIT 1', (row['id'],)).fetchone()
            if latest is None or latest['client_key'] != expected_call_key:
                raise LifecycleStale('lifecycle event does not name the current call boundary')
            return self._request_yield(row)

    def request_yield(self, grant_id, owner):
        """A long external tool may yield preference without canceling its job.

        An active/uncertain worker drains or remains quarantined. Repeating the
        signal is harmless; the same grant can resume with its next ready call.
        """
        with self.transaction():
            row = self._owned(grant_id, owner)
            return self._request_yield(row)

    def _request_yield(self, row):
        if row['lifetime'] != 'work_session':
            raise Conflict('only work-session grants can yield')
        if row['state'] in ('released', 'cancelled', 'queued') or row['release_requested']:
            return dict(row)
        if row['active_execution'] or row['state'] == 'uncertain':
            self.db.execute('UPDATE grants SET yield_requested=1 WHERE id=?', (row['id'],))
        else:
            self._handoff(row, 'explicit_yield', time.time())
        return dict(self._owned(row['id'], row['owner']))

    def finish(self, execution_id, result, *, failed=False):
        """Supervisor only: call after backend execution actually terminates."""
        body = encoded(result)
        state = 'failed' if failed else 'complete'
        with self.transaction():
            row = self.db.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
            if row is None:
                raise NotFound()
            if row['state'] in ('queued', 'cancelled'):
                raise Conflict("execution was not dispatched")
            if row['state'] in ('complete', 'failed'):
                if row['state'] != state or row['result'] != body:
                    raise Conflict("terminal execution cannot be overwritten")
                return
            grant = self.db.execute("SELECT * FROM grants WHERE id=?", (row['grant_id'],)).fetchone()
            if grant['active_execution'] != execution_id:
                raise Conflict("execution no longer owns the grant")
            now = time.time()
            self.db.execute("UPDATE executions SET state=?,result=?,updated=? WHERE id=?",
                            (state, body, now, execution_id))
            released = bool(grant['release_requested'] or grant['lifetime'] == 'execution')
            self.db.execute("""UPDATE grants SET state=?,active_execution=NULL,updated=?,
                idle_since=? WHERE id=?""",
                            ('released' if released else 'granted', now,
                             None if released else now, grant['id']))
            self._advance_call_job(row)

    def defer_capacity(self, execution_id, reason):
        """Put the SAME logical request back in queue, atomically blocking retry.

        Only before provider start, asserted by the trusted adapter. Saved output
        is an additional hard rejection: no partial answer may be replayed as a
        new generation. An uncertain execution can never use this escape hatch.
        """
        CapacityDeferred(reason)  # validate without persisting caller prose
        with self.transaction():
            row = self.db.execute('SELECT * FROM executions WHERE id=?',(execution_id,)).fetchone()
            if row is None:
                raise NotFound()
            grant = self.db.execute('SELECT * FROM grants WHERE id=?',(row['grant_id'],)).fetchone()
            if (row['state'] != 'running' or grant['active_execution'] != execution_id or
                    self.db.execute('SELECT 1 FROM execution_events WHERE execution_id=?',
                                    (execution_id,)).fetchone()):
                raise Conflict('only pre-provider, output-free owned work can wait for capacity')
            now = time.time()
            self.db.execute('INSERT INTO resource_waits VALUES(?,?,?) ON CONFLICT(resource) '
                'DO UPDATE SET reason=excluded.reason,since=excluded.since',
                (grant['resource'],reason,now))
            self.db.execute("UPDATE executions SET state='queued',wait_reason=?,updated=? WHERE id=?",
                            (reason,now,execution_id))
            self.db.execute("UPDATE grants SET state='queued',active_execution=NULL,updated=?,"
                'queue_order=?,stint_started=NULL,idle_since=NULL WHERE id=?',
                (now,self._next_order(),grant['id']))

    def capacity_available(self, resource):
        """Trusted fresh permission only; never exposed as an owner API write."""
        required(resource)
        with self.transaction():
            self.db.execute('DELETE FROM resource_waits WHERE resource=?',(resource,))
            self.db.execute('UPDATE executions SET wait_reason=NULL WHERE state=\'queued\' '
                'AND grant_id IN (SELECT id FROM grants WHERE resource=?)',(resource,))

    def uncertain(self, execution_id):
        """Transport loss is not proof that backend compute ended."""
        with self.transaction():
            row = self.db.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
            if row is None:
                raise NotFound()
            if row['state'] not in ('running', 'uncertain'):
                raise Conflict("only dispatched work can become uncertain")
            now = time.time()
            self.db.execute("UPDATE executions SET state='uncertain',updated=? WHERE id=?", (now, execution_id))
            self.db.execute("UPDATE grants SET state='uncertain',updated=? WHERE active_execution=?",
                            (now, execution_id))

    def quarantine_running(self):
        """Supervisor reconciliation after owner loss, NOT an automatic reset.

        Marks uncertain work, preserves queued work and prevents duplicate dispatch.
        Caller must establish that its previous scheduler is gone before using it.
        This records uncertainty; it does not cancel or kill backend execution.
        """
        with self.transaction():
            now = time.time()
            self.db.execute("UPDATE executions SET state='uncertain',updated=? WHERE state='running'", (now,))
            self.db.execute("UPDATE grants SET state='uncertain',updated=? "
                            "WHERE active_execution IS NOT NULL AND state='granted'", (now,))
