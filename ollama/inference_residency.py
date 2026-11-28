"""Private, service-owned residency handover intent; not enabled in the manager.

Only verified registry receipts may enter register/accept. Persisting a receipt
does not make it fresh after restart. No model loading, lease release, retry
clock or capacity permission is hidden here; callers bind those relationships.
Covered means the stated reservation has a validated receipt, not that its byte
count covers the model's actual working set/KV cache or authorizes more compute.
"""
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .inference_arbiter import Conflict, required


@dataclass(frozen=True)
class RolloverIntent:
    id: str
    allocation: str
    predecessor: str
    ttl_seconds: int


def _expiry(value):
    if not isinstance(value, str):
        raise ValueError('finite registry expiry required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone-aware registry expiry required')
    return parsed.timestamp()


class ResidencyStore:
    def __init__(self, path, *, registry_owner, service_id, clock=time.time):
        self.owner, self.service = required(registry_owner), required(service_id)
        self.clock, self.fresh = clock, {}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(descriptor).st_mode & 0o077:
                raise PermissionError('residency store must be private')
        finally:
            os.close(descriptor)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS residency_owner (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                registry_owner TEXT NOT NULL, service_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS allocations (
                id TEXT PRIMARY KEY, resource TEXT NOT NULL, model TEXT NOT NULL,
                lease_id TEXT NOT NULL UNIQUE, memory_mb INTEGER NOT NULL,
                expires_at TEXT NOT NULL, max_total_s INTEGER NOT NULL,
                state TEXT NOT NULL, pending_id TEXT,
                UNIQUE(resource,model));
            CREATE TABLE IF NOT EXISTS residency_rollovers (
                id TEXT PRIMARY KEY, allocation TEXT NOT NULL REFERENCES allocations(id),
                predecessor TEXT NOT NULL UNIQUE, ttl_seconds INTEGER NOT NULL,
                state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS residency_ends (
                allocation TEXT NOT NULL, lease_id TEXT NOT NULL UNIQUE,
                resource TEXT NOT NULL, model TEXT NOT NULL, memory_mb INTEGER NOT NULL,
                ended_status TEXT NOT NULL, observed_at REAL NOT NULL);
        ''')
        self.db.execute('INSERT OR IGNORE INTO residency_owner VALUES(1,?,?)',
                        (self.owner,self.service))
        owner = self.db.execute('SELECT registry_owner,service_id FROM residency_owner').fetchone()
        if tuple(owner) != (self.owner,self.service):
            self.db.close()
            raise PermissionError('residency store belongs to a different service identity')

    def close(self):
        self.db.close()

    @staticmethod
    def _marker(row):
        return tuple(row[key] for key in ('resource','model','lease_id','memory_mb','expires_at','max_total_s'))

    def snapshot(self, allocation):
        row = self.db.execute('SELECT * FROM allocations WHERE id=?', (allocation,)).fetchone()
        if row is None:
            raise Conflict('allocation is not registered')
        fresh = (row['state'] == 'covered' and row['pending_id'] is None and
                 self.fresh.get(allocation) == self._marker(row))
        return {**dict(row), 'fresh_receipt':fresh,
                'receipt_unexpired':_expiry(row['expires_at']) > self.clock()}

    def forget_confirmation(self, allocation):
        """A failed ownership observation withdraws freshness, not saved intent."""
        self.snapshot(allocation)
        self.fresh.pop(allocation,None)

    def register(self, allocation, *, resource, model, receipt):
        """Capture/confirm an exact owned allocation from a fresh trusted read.

        Never infer model ownership from equal memory sizes or identity-wide
        latest leases. The caller must already know the exact model/lease link.
        Pending handovers cannot be overwritten by a stale predecessor receipt.
        """
        for value in (allocation, resource, model):
            required(value)
        self.fresh.pop(allocation,None)
        if (receipt.get('agent_id') != self.owner or receipt.get('service_id') != self.service or
                receipt.get('resource_id') != resource or receipt.get('exclusive') != 0 or
                receipt.get('status') != 'active'):
            raise Conflict('receipt does not prove the configured owned shared allocation')
        lease_id = required(receipt.get('id'))
        memory, maximum = receipt.get('memory_mb'), receipt.get('max_total_s')
        if type(memory) is not int or memory <= 0 or type(maximum) is not int or maximum <= 0:
            raise ValueError('positive allocation size and exposure bound required')
        if _expiry(receipt.get('expires_at')) <= self.clock():
            raise Conflict('allocation receipt is expired')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM residency_ends WHERE lease_id=?',(lease_id,)).fetchone():
                raise Conflict('ended lease cannot establish a new residency')
            old = self.db.execute('SELECT * FROM allocations WHERE id=?', (allocation,)).fetchone()
            if old:
                if old['state']=='retired' and old['pending_id'] is None:
                    if (old['resource'],old['model'])!=(resource,model):
                        raise Conflict('retired allocation belongs to another model/resource')
                    self.db.execute('''UPDATE allocations SET lease_id=?,memory_mb=?,expires_at=?,
                        max_total_s=?,state='covered',pending_id=NULL WHERE id=?''',
                        (lease_id,memory,receipt['expires_at'],maximum,allocation))
                elif (old['resource'],old['model'],old['lease_id'],old['memory_mb']) != (
                        resource,model,lease_id,memory):
                    raise Conflict('allocation identity changed; reconcile before adoption')
                elif not old['pending_id']:
                    self.db.execute("UPDATE allocations SET expires_at=?,max_total_s=?,state='covered' WHERE id=?",
                        (receipt['expires_at'],maximum,allocation))
            else:
                if self.db.execute('SELECT 1 FROM residency_rollovers WHERE predecessor=?',
                                   (lease_id,)).fetchone():
                    raise Conflict('predecessor has already entered a handover')
                self.db.execute("INSERT INTO allocations VALUES(?,?,?,?,?,?,?,'covered',NULL)",
                    (allocation,resource,model,lease_id,memory,receipt['expires_at'],maximum))
            self.db.execute('COMMIT')
        except sqlite3.IntegrityError:
            self.db.execute('ROLLBACK')
            raise Conflict('model/resource or lease is already registered') from None
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        if not old or not old['pending_id']:
            self.fresh[allocation] = (resource,model,lease_id,memory,receipt['expires_at'],maximum)
        return self.snapshot(allocation)

    def retire(self,allocation,receipt):
        """End a known residency after caller-confirmed backend absence.

        A terminal exact registry receipt is required as well. Ordinary unload
        and reload may then reuse the logical model slot; pending rollovers and
        unknown/released-by-guess allocations cannot be overwritten.
        """
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.snapshot(allocation)
            from .inference_reservation import needs_settlement
            if (needs_settlement(receipt) or row['pending_id'] is not None or receipt.get('id')!=row['lease_id']
                    or receipt.get('agent_id')!=self.owner or receipt.get('service_id')!=self.service
                    or receipt.get('resource_id')!=row['resource'] or receipt.get('exclusive')!=0
                    or type(receipt.get('memory_mb')) is not int or receipt['memory_mb']!=row['memory_mb']
                    or receipt.get('status') not in ('released','expired')):
                raise Conflict('exact ended allocation required for residency retirement')
            self.db.execute('INSERT OR IGNORE INTO residency_ends VALUES(?,?,?,?,?,?,?)',
                (allocation,row['lease_id'],row['resource'],row['model'],row['memory_mb'],
                 receipt['status'],self.clock()))
            self.db.execute("UPDATE allocations SET state='retired' WHERE id=?",(allocation,))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK');raise
        self.fresh.pop(allocation,None)

    def prepare(self, allocation, *, ttl_seconds):
        """Commit intent BEFORE POST; reopen returns the same predecessor/key."""
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError('positive rollover TTL required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.snapshot(allocation)
            if row['pending_id']:
                intent = self.db.execute('SELECT * FROM residency_rollovers WHERE id=?',
                                         (row['pending_id'],)).fetchone()
                if intent['ttl_seconds'] != ttl_seconds:
                    raise Conflict('pending rollover TTL changed')
                operation = RolloverIntent(intent['id'],allocation,intent['predecessor'],ttl_seconds)
            else:
                if (not row['fresh_receipt'] or not row['receipt_unexpired'] or
                        row['state'] != 'covered' or ttl_seconds > row['max_total_s']):
                    raise Conflict('fresh covered allocation required for a new handover')
                operation = RolloverIntent(uuid.uuid4().hex,allocation,row['lease_id'],ttl_seconds)
                self.db.execute("INSERT INTO residency_rollovers VALUES(?,?,?,?,'pending')",
                    (operation.id,allocation,operation.predecessor,ttl_seconds))
                self.db.execute("UPDATE allocations SET pending_id=?,state='handover_pending' WHERE id=?",
                                 (operation.id,allocation))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        self.fresh.pop(allocation,None)
        return operation

    def accept(self, operation, response):
        """Apply only the exact linked response, never generic HTTP200 success.

        Unknown/malformed replies leave the pending intent intact. An ended
        replacement records uncovered accounting, NOT terminated backend work.
        A duplicate old response cannot overwrite a newer handover.
        """
        self.db.execute('BEGIN IMMEDIATE')
        try:
            intent = self.db.execute('SELECT * FROM residency_rollovers WHERE id=?',
                                     (operation.id,)).fetchone()
            if (intent is None or intent['allocation'] != operation.allocation or
                    intent['predecessor'] != operation.predecessor or intent['ttl_seconds'] != operation.ttl_seconds):
                raise Conflict('rollover intent does not match persisted operation')
            row = self.snapshot(operation.allocation)
            if intent['state'] == 'applied':
                self.db.execute('COMMIT')
                return {**row, 'changed':False}
            if row['pending_id'] != operation.id:
                raise Conflict('stale rollover response')
            new_id = required(response.get('lease_id'))
            if (response.get('replaces_lease_id') != operation.predecessor or
                    response.get('resource_id') != row['resource'] or
                    type(response.get('memory_mb')) is not int or response['memory_mb'] != row['memory_mb'] or
                    type(response.get('max_total_s')) is not int or
                    response['max_total_s'] != row['max_total_s'] or new_id == operation.predecessor or
                    response.get('status') not in ('active','expired','released')):
                raise Conflict('rollover response does not prove the linked allocation')
            expiry = _expiry(response.get('expires_at'))
            if expiry < _expiry(row['expires_at']):
                raise Conflict('rollover response shortened allocation coverage')
            covered = response['status'] == 'active' and expiry > self.clock()
            self.db.execute('UPDATE allocations SET lease_id=?,expires_at=?,state=?,pending_id=NULL WHERE id=?',
                (new_id,response['expires_at'],'covered' if covered else 'uncovered',operation.allocation))
            self.db.execute("UPDATE residency_rollovers SET state='applied' WHERE id=?", (operation.id,))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        if covered:
            self.fresh[operation.allocation] = self._marker({**row,
                'lease_id':new_id,'expires_at':response['expires_at']})
        else:
            self.fresh.pop(operation.allocation,None)
        return {**self.snapshot(operation.allocation), 'changed':True}
