"""Initial byte-reservation intent, independent of HTTP observers.

The registry's initial POST is not idempotent. Persist BEFORE sending; a lost
reply or process death is unknown, never a reason to POST again. A successful
receipt is handed to the model loader once. After that, model/residency recovery
must reconcile the exact saved lease; this component never infers provider end.
"""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .inference_arbiter import CapacityDeferred, Conflict, required


def needs_settlement(receipt):
    """Registry expiry is not physical settlement (0.12.99 hold policy).

    GET keeps status=expired after settlement; settled_at is the acknowledgment.
    Even on a free-policy resource, an absence-confirmed holder may explicitly
    settle its exact expired row. Never infer that a TTL released memory.
    """
    return (receipt.get('status') == 'expired' and
            receipt.get('expiry_reason') == 'ttl_expired' and not receipt.get('settled_at'))


class InitialReservations:
    """Synchronous manager-thread adapter; each call owns its SQLite connection.

    The file lock covers registry I/O too: two load threads/processes cannot
    send competing initial allocations through the same journal. It is not a
    replacement for the gateway's resource admission or working-set fit gate.
    """
    def __init__(self, path, client, *, registry_owner, service_id, models, budgets=None,
                 wait_s=None, wait=False, clock=time.time):
        self.path, self.client = Path(path), client
        self.owner, self.service = required(registry_owner), required(service_id)
        self.models = {required(model):required(resource) for model, resource in models.items()}
        self.budgets = dict(budgets or {})
        if wait_s is not None and (type(wait_s) is not int or wait_s < 0):
            raise ValueError('explicit non-negative registry queue wait required')
        if type(wait) is not bool:
            raise ValueError('wait must be a boolean')
        self.wait_s, self.wait = wait_s, wait
        if (set(self.budgets) - set(self.models) or
                any(type(value) is not int or value <= 0 for value in self.budgets.values())):
            raise ValueError('reservation budgets must name configured models with positive MiB')
        if not self.path.is_absolute() or not self.models:
            raise ValueError('absolute private journal and exact model/resource mapping required')
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._owned() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS reservation_owner (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    owner TEXT NOT NULL, service TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS initial_reservations (
                    model TEXT PRIMARY KEY, resource TEXT NOT NULL,
                    memory_mb INTEGER NOT NULL, ttl_seconds INTEGER NOT NULL,
                    intent TEXT NOT NULL, state TEXT NOT NULL,
                    lease_id TEXT, expires_at TEXT);
                CREATE TABLE IF NOT EXISTS sizing_reconciliations (
                    model TEXT NOT NULL, predecessor TEXT NOT NULL,
                    memory_mb INTEGER NOT NULL, ttl_seconds INTEGER NOT NULL,
                    state TEXT NOT NULL, successor TEXT, expires_at TEXT,
                    PRIMARY KEY(model,predecessor));
            ''')
            db.execute('INSERT OR IGNORE INTO reservation_owner VALUES(1,?,?)',
                       (self.owner, self.service))
            if tuple(db.execute('SELECT owner,service FROM reservation_owner').fetchone()) != (
                    self.owner, self.service):
                raise PermissionError('reservation journal belongs to another service identity')

    @contextmanager
    def _owned(self):
        descriptors = []
        db = None
        try:
            for path in (str(self.path)+'.owner', self.path):
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                descriptors.append(fd)
                if os.fstat(fd).st_mode & 0o077:
                    raise PermissionError('reservation files must be private')
            fcntl.flock(descriptors[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
            db = sqlite3.connect(self.path, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA synchronous=FULL')
            yield db
        finally:
            if db is not None:
                db.close()
            for fd in reversed(descriptors):
                os.close(fd)

    def snapshot(self, model):
        with self._owned() as db:
            row = db.execute('SELECT * FROM initial_reservations WHERE model=?', (model,)).fetchone()
            return dict(row) if row else None

    def required_memory(self, model, estimated_mb):
        """Existing service estimate, optionally raised by an explicit override.

        Reserving bytes does not enforce a provider allocation limit or prove
        this budget covers KV/cache growth. Deployment must qualify that bound.
        """
        if model not in self.models or type(estimated_mb) is not int or estimated_mb <= 0:
            raise ValueError('configured model and positive weight estimate required')
        return max(estimated_mb, self.budgets.get(model, 0))

    def resize(self, model, predecessor, memory_mb, ttl_seconds):
        """Journal the existing acquire-before-release sizing request.

        A received refusal keeps the predecessor. A lost POST reply remains
        unknown and is never blindly repeated, including after process reopen.
        This does not load/unload a model or release the predecessor; the manager
        changes its exact pointer only after this returns a confirmed grant.
        """
        if (model not in self.models or not isinstance(predecessor,str) or not predecessor
                or type(memory_mb) is not int or memory_mb <= 0
                or type(ttl_seconds) is not int or ttl_seconds <= 0):
            raise ValueError('exact sizing dimensions required')
        with self._owned() as db:
            old = db.execute('SELECT * FROM sizing_reconciliations WHERE model=? AND predecessor=?',
                             (model,predecessor)).fetchone()
            if old is not None:
                if (old['memory_mb'],old['ttl_seconds']) != (memory_mb,ttl_seconds):
                    raise Conflict('sizing dimensions changed; reconcile saved intent')
                if old['state']=='refused':
                    return None
                raise Conflict('sizing request already issued; reconcile saved intent')
            db.execute('INSERT INTO sizing_reconciliations VALUES(?,?,?,?,?,NULL,NULL)',
                       (model,predecessor,memory_mb,ttl_seconds,'unknown'))
            try:
                receipt = self.client.request_lease(resource_id=self.models[model],
                    service_id=self.service,memory_mb=memory_mb,ttl_seconds=ttl_seconds)
            except Exception:
                raise Conflict('sizing reply unknown; predecessor retained') from None
            if not isinstance(receipt,dict):
                raise Conflict('sizing reply invalid; predecessor retained')
            if receipt.get('status_code')==409:
                db.execute("UPDATE sizing_reconciliations SET state='refused' WHERE model=? AND predecessor=?",
                           (model,predecessor))
                return None
            successor = receipt.get('lease_id') or receipt.get('id')
            if not isinstance(successor,str) or not successor or successor==predecessor:
                raise Conflict('sizing successor identity unknown')
            db.execute('UPDATE sizing_reconciliations SET successor=? WHERE model=? AND predecessor=?',
                       (successor,model,predecessor))
            if receipt.get('status_code') not in (200,201,202):
                raise Conflict('sizing successor state unknown')
            if receipt.get('status')=='queued' or receipt.get('queue_position') is not None:
                # Same existing policy: do not retain an unused queued resize.
                # Persist its ID before release; a lost DELETE is not a refusal.
                try:
                    released = self.client.release_lease(successor)
                except Exception:
                    raise Conflict('queued sizing release unknown') from None
                if not isinstance(released,dict) or released.get('status') not in ('released', 'cancelled'):
                    raise Conflict('queued sizing release unconfirmed')
                db.execute("UPDATE sizing_reconciliations SET state='refused' WHERE model=? AND predecessor=?",
                           (model,predecessor))
                return None
            try:
                expiry = datetime.fromisoformat(receipt['expires_at'].replace('Z','+00:00'))
                valid = (receipt.get('status')=='active' and expiry.tzinfo is not None
                         and expiry.timestamp()>self.clock())
            except (KeyError,AttributeError,TypeError,ValueError,OverflowError):
                valid = False
            if not valid:
                raise Conflict('sizing successor is not a current finite grant')
            db.execute("UPDATE sizing_reconciliations SET state='handed_off',expires_at=? WHERE model=? AND predecessor=?",
                       (receipt['expires_at'],model,predecessor))
            return successor

    def _observe_waiting(self, db, row):
        """Reobserve the saved allocation, never buy a new queue position.

        The list route works before and after registry promotion ships. An
        event is only a wakeup hint: exact ownership, dimensions and a current
        finite grant must come from the ledger before provider handoff.
        """
        try:
            rows = self.client.list_leases(resource_id=row['resource'], service_id=self.service)
        except Exception:
            raise Conflict('saved reservation observation unknown') from None
        matches = [item for item in rows if isinstance(item, dict) and
                   item.get('id') == row['lease_id']] if isinstance(rows, list) else []
        if len(matches) != 1:
            raise Conflict('exact saved reservation is unconfirmed')
        receipt = matches[0]
        if (receipt.get('agent_id') != self.owner or receipt.get('service_id') != self.service
                or receipt.get('resource_id') != row['resource'] or receipt.get('exclusive') != 0
                or type(receipt.get('memory_mb')) is not int
                or receipt['memory_mb'] != row['memory_mb']):
            raise Conflict('saved reservation ownership or dimensions differ')
        if receipt.get('status') == 'queued':
            if row['state'] != 'queued':
                raise Conflict('granted reservation regressed to queued')
            if self.wait_s is not None:
                try:
                    deadline = datetime.fromisoformat(receipt['expires_at'].replace('Z', '+00:00'))
                    if deadline.tzinfo is None:
                        raise ValueError('queue deadline must be timezone-aware')
                except (KeyError, AttributeError, TypeError, ValueError):
                    raise Conflict('saved queue deadline unknown') from None
                if deadline.timestamp() <= self.clock() + self.wait_s / 2:
                    try:
                        self.client.renew_lease(row['lease_id'], self.wait_s)
                    except Exception:
                        raise Conflict('saved queue renewal unknown') from None
                    # Renewal can race promotion. Only a subsequent ledger read
                    # proves active ownership; a renewal reply never loads work.
            raise CapacityDeferred('capacity_unavailable')
        try:
            expiry = datetime.fromisoformat(receipt['expires_at'].replace('Z', '+00:00'))
            valid = (receipt.get('status') == 'active' and expiry.tzinfo is not None
                     and expiry.timestamp() > self.clock())
        except (KeyError, AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            # Expiry/cancellation is not a reason to silently POST again. Keep
            # this identity until explicit absence/release reconciliation.
            raise Conflict('saved reservation is not a current finite grant')
        db.execute("UPDATE initial_reservations SET state='reserved',expires_at=? WHERE model=?",
                   (receipt['expires_at'], row['model']))
        return row['lease_id']

    def stage(self, model, *, resource, service_id, memory_mb, ttl_seconds):
        """Request for actual queued demand, without authorizing provider load.

        The caller must establish demand; startup, affinity and model presence
        are not demand. This metadata intent is not another allocation ledger.
        A promoted reservation is consumed once by reserve(), after a fresh
        physical-fit check by the owning capacity controller.
        """
        return self._reserve(model, resource=resource, service_id=service_id,
                             memory_mb=memory_mb, ttl_seconds=ttl_seconds, consume=False)

    def reserve(self, model, *, resource, service_id, memory_mb, ttl_seconds):
        return self._reserve(model, resource=resource, service_id=service_id,
                             memory_mb=memory_mb, ttl_seconds=ttl_seconds, consume=True)

    def _reserve(self, model, *, resource, service_id, memory_mb, ttl_seconds, consume):
        memory_mb = self.required_memory(model, memory_mb)
        if (self.models.get(model) != resource or service_id != self.service or
                type(memory_mb) is not int or memory_mb <= 0 or
                type(ttl_seconds) is not int or ttl_seconds <= 0):
            raise ValueError('exact configured allocation and positive size/TTL required')
        with self._owned() as db:
            row = db.execute('SELECT * FROM initial_reservations WHERE model=?', (model,)).fetchone()
            # Refused or positively retired allocations already permit a new
            # intent. Their former dimensions must not pin later qualified
            # budgets forever; unfinished ownership still requires an exact
            # dimension match and its existing reconciliation path.
            if row is not None and row['state'] not in ('refused', 'retired'):
                if (row['resource'], row['memory_mb'], row['ttl_seconds']) != (
                        resource, memory_mb, ttl_seconds):
                    raise Conflict('reservation dimensions changed; reconcile saved allocation')
                if row['state'] in ('queued', 'reserved'):
                    lease = self._observe_waiting(db, row)
                    if consume:
                        db.execute("UPDATE initial_reservations SET state='handed_off' WHERE model=?", (model,))
                    return lease
                # Unknown and already handed-off receipts need recovery.
                # Never guess which identity-wide lease belonged to this call.
                raise Conflict('initial reservation requires exact lease reconciliation')
            intent = uuid.uuid4().hex
            db.execute('INSERT OR REPLACE INTO initial_reservations VALUES(?,?,?,?,?,?,NULL,NULL)',
                       (model, resource, memory_mb, ttl_seconds, intent, 'unknown'))
            try:
                queue_options = ({'wait_s':self.wait_s} if self.wait_s is not None
                                 else {'wait':True} if self.wait else {})
                receipt = self.client.request_lease(resource_id=resource,
                    service_id=service_id, memory_mb=memory_mb, ttl_seconds=ttl_seconds,
                    **queue_options)
            except Exception:
                raise Conflict('initial reservation reply unknown; do not repeat allocation') from None
            if not isinstance(receipt, dict):
                raise Conflict('initial reservation reply is invalid')
            code = receipt.get('status_code')
            if code == 409:
                db.execute("UPDATE initial_reservations SET state='refused' WHERE model=?", (model,))
                raise CapacityDeferred('capacity_unavailable')
            lease = receipt.get('lease_id') or receipt.get('id')
            if isinstance(lease, str) and lease.strip():
                # Preserve the exact identifier even if the rest is malformed.
                db.execute('UPDATE initial_reservations SET lease_id=? WHERE model=?', (lease, model))
            else:
                raise Conflict('initial reservation has no exact lease identifier')
            if code in (200, 201, 202) and (receipt.get('status') == 'queued' or
                                          receipt.get('queue_position') is not None):
                db.execute("UPDATE initial_reservations SET state='queued' WHERE model=?", (model,))
                raise CapacityDeferred('capacity_unavailable')
            if code not in (200, 201, 202) or receipt.get('status') != 'active':
                raise Conflict('initial reservation not active; reconcile saved lease')
            try:
                expires = datetime.fromisoformat(receipt['expires_at'].replace('Z', '+00:00'))
                valid = expires.tzinfo is not None and expires.timestamp() > self.clock()
            except (KeyError, AttributeError, TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise Conflict('initial reservation has no current finite expiry')
            # Commit before returning to ANY provider mutation. Reusing this
            # receipt after a failed load would assume that load never started.
            db.execute("UPDATE initial_reservations SET state=?,expires_at=? WHERE model=?",
                       ('handed_off' if consume else 'reserved', receipt['expires_at'], model))
            return lease

    def reconcile_absent(self, model, *, confirm_absent):
        """Retire ONLY the saved exact lease after affirmative provider absence.

        Trusted service code must hold resource admission against new compute
        throughout this synchronous operation. The callback is a fresh backend
        observation, not an HTTP disconnect, elapsed time, or absent manager row.
        An unknown initial POST without a returned ID cannot be resolved here.
        A lost DELETE reply retains the ID and state; next reconciliation reads
        that same lease, never posts another allocation or guesses a successor.
        """
        if model not in self.models or not callable(confirm_absent):
            raise ValueError('configured model and trusted provider observation required')
        with self._owned() as db:
            row = db.execute('SELECT * FROM initial_reservations WHERE model=?', (model,)).fetchone()
            if row is None or not row['lease_id']:
                raise Conflict('no exact initial lease identity to reconcile')
            try:
                absent = confirm_absent(model)
            except Exception:
                raise Conflict('provider absence is unknown') from None
            if absent is not True:
                raise Conflict('provider absence is not confirmed')
            try:
                leases = self.client.list_leases(resource_id=row['resource'], service_id=self.service)
            except Exception:
                raise Conflict('exact lease observation is unknown') from None
            if not isinstance(leases, list):
                raise Conflict('exact lease observation is invalid')
            # An ordinary unload may follow the existing sizing replacement.
            # Reconcile each exact known successor too; do not infer that a
            # lost resize POST created nothing merely because the model is gone.
            sizes = db.execute('SELECT * FROM sizing_reconciliations WHERE model=?', (model,)).fetchall()
            for sizing in sizes:
                if sizing['state'] in ('refused','retired'):
                    continue
                successor = sizing['successor']
                if not successor:
                    raise Conflict('sizing allocation identity unknown; cannot retire residency')
                found = [lease for lease in leases if isinstance(lease,dict) and lease.get('id')==successor]
                if len(found)!=1:
                    raise Conflict('exact sizing successor is unconfirmed')
                known = found[0]
                if (known.get('agent_id')!=self.owner or known.get('service_id')!=self.service
                        or known.get('resource_id')!=row['resource'] or known.get('exclusive')!=0
                        or type(known.get('memory_mb')) is not int or known['memory_mb']!=sizing['memory_mb']
                        or known.get('status') not in ('active','queued','released','expired','cancelled')):
                    raise Conflict('sizing successor ownership or state differs')
                if known['status'] in ('active','queued') or needs_settlement(known):
                    try:
                        if confirm_absent(model) is not True:
                            raise Conflict('provider absence changed before sizing release')
                        released = self.client.release_lease(successor)
                    except Exception:
                        raise Conflict('sizing release unconfirmed; reconcile same saved ID') from None
                    if not isinstance(released,dict) or released.get('status') not in ('released', 'cancelled', 'settled'):
                        raise Conflict('sizing release unconfirmed')
                db.execute("UPDATE sizing_reconciliations SET state='retired' WHERE model=? AND predecessor=?",
                           (model,sizing['predecessor']))
            matches = [lease for lease in leases if isinstance(lease, dict) and lease.get('id') == row['lease_id']]
            if len(matches) != 1:
                raise Conflict('registry does not prove the exact saved lease')
            lease = matches[0]
            if (lease.get('agent_id') != self.owner or lease.get('service_id') != self.service or
                    lease.get('resource_id') != row['resource'] or lease.get('exclusive') != 0 or
                    type(lease.get('memory_mb')) is not int or lease['memory_mb'] != row['memory_mb']):
                raise Conflict('exact lease ownership or dimensions differ')
            if lease.get('status') not in ('active', 'queued', 'released', 'expired', 'cancelled'):
                raise Conflict('exact lease state is unknown')
            if lease['status'] in ('active', 'queued') or needs_settlement(lease):
                # Recheck after network I/O; the admission owner must still
                # exclude new work. Silence/metadata failure never means absent.
                try:
                    absent = confirm_absent(model)
                except Exception:
                    raise Conflict('provider absence is unknown') from None
                if absent is not True:
                    raise Conflict('provider absence changed before release')
                db.execute("UPDATE initial_reservations SET state='retiring' WHERE model=?", (model,))
                try:
                    receipt = self.client.release_lease(row['lease_id'])
                except Exception:
                    raise Conflict('lease release outcome unknown; reconcile same saved ID') from None
                if not isinstance(receipt, dict) or receipt.get('status') not in ('released', 'cancelled', 'settled'):
                    raise Conflict('lease release is not confirmed')
            db.execute("UPDATE initial_reservations SET state='retired' WHERE model=?", (model,))
            return {'lease_id':row['lease_id'], 'state':'retired'}
