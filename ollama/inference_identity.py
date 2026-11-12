"""Caller-owned durable keys; no prompt matching or inferred turn boundaries.

The host supplies explicit stable scope/work/call boundaries BEFORE submission.
This store does not discover Hermes rows, resume turns, or declare backend work
finished. Compaction/retry intent mapping belongs to the host integration.
Only identifiers and a request-integrity digest are stored, never request bodies.
One connection per owning thread; independent processes/connections serialize
allocation with SQLite. Nothing here is installed in the live clients.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sqlite3
import uuid

from .inference_arbiter import Conflict, encoded, required


@dataclass(frozen=True)
class CallKeys:
    grant_key: str
    call_key: str
    yield_grant_keys: tuple[str, ...] = ()
    execution_only: bool = False

    def headers(self):
        return {'X-Inference-Grant-Key': self.grant_key, 'Idempotency-Key': self.call_key}


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
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def bind(self, scope, work_boundary, call_boundary, request):
        for value in (scope, work_boundary, call_boundary):
            required(value)
        # Integrity guard only. Equality of content never allocates identity.
        digest = hashlib.sha256(encoded(request).encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT key FROM work_identity WHERE scope=? AND boundary=?',
                                  (scope, work_boundary)).fetchone()
            grant = row[0] if row else uuid.uuid4().hex
            if row is None:
                self.db.execute('INSERT INTO work_identity VALUES(?,?,?)', (scope, work_boundary, grant))
            row = self.db.execute('SELECT key,digest FROM call_identity WHERE work_key=? AND boundary=?',
                                  (grant, call_boundary)).fetchone()
            if row and row[1] != digest:
                raise Conflict('persisted call boundary has different request content; reconcile before submitting')
            call = row[0] if row else uuid.uuid4().hex
            if row is None:
                self.db.execute('INSERT INTO call_identity(work_key,boundary,key,digest) VALUES(?,?,?,?)',
                                (grant, call_boundary, call, digest))
            self.db.execute('COMMIT')
            return CallKeys(grant, call)
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

    def record_turn_call(self, scope, turn_id, keys, *, api_request_id=''):
        for value in (scope, turn_id):
            required(value)
        if api_request_id:
            required(api_request_id)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            owned = self.db.execute('SELECT 1 FROM work_identity w JOIN call_identity c '
                'ON c.work_key=w.key WHERE w.scope=? AND w.key=? AND c.key=?',
                (scope, keys.grant_key, keys.call_key)).fetchone()
            if owned is None:
                raise Conflict('native turn does not own the durable call')
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
            rows = self.db.execute('SELECT grant_key,call_key FROM turn_call '
                                  'WHERE scope=? AND turn_id=?', (scope, turn_id)).fetchall()
            self._queue_yields(scope,turn_id,rows)
            self.db.execute('COMMIT')
            return len(rows)
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def pending_events(self, limit=32, *, after=0):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('bounded event page required')
        if type(after) is not int or after < 0:
            raise ValueError('valid event cursor required')
        return [dict(zip(('cursor','id','grant_key','expected_call_key'), row)) for row in self.db.execute(
            "SELECT rowid,id,grant_key,call_key FROM lifecycle_outbox WHERE state='pending' AND rowid>? ORDER BY rowid LIMIT ?",
            (after, limit))]

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
        value = encoded(sorted(set(parents)))
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
            self.db.execute('COMMIT')
            return CallKeys(keys.grant_key, keys.call_key, tuple(json.loads(saved)))
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def bind_execution(self, scope, work_boundary, call_boundary, request):
        """One durable grant per explicit host call, including within a tool loop.

        Hashes only the caller's declared boundary IDs to keep identifiers short;
        never hashes prompt equality to decide whether two intentions are one.
        A retry/restart of the same call recovers its released result too.
        """
        required(work_boundary)
        required(call_boundary)
        boundary = 'execution:' + hashlib.sha256(
            encoded([work_boundary, call_boundary]).encode()).hexdigest()
        return self.bind(scope, boundary, call_boundary, request)
