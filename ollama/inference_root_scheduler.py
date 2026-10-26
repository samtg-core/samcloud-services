"""Staged requester/physical-root boundary, not selected by shipping bootstrap.

Logical preparation precedes physical admission; it never starts an execution.
Only the registry owns capacity. A qualified controller supplies fresh exact
registry observations and performs backend cleanup outside this journal.
"""
from datetime import datetime
import json
import time
from uuid import uuid4

from .inference_arbiter import Arbiter,Conflict,StintPolicy,required
from .inference_root_client import _base


def _unexpired(lease):
    try:
        stamp=datetime.fromisoformat(lease['expires_at'].replace('Z','+00:00'))
        return stamp.tzinfo is not None and stamp.timestamp()>time.time()
    except (KeyError,TypeError,AttributeError,ValueError,OverflowError):
        return False


class RootPreparingArbiter(Arbiter):
    """Explicit alternate scheduler. No existing bootstrap selects this class.

    The preparation ID, NOT the reusable logical job/session, identifies one
    physical root intent in RootIntentStore. Explicit yield can later create a
    new preparation while retaining the same logical job and accepted call IDs.
    """
    supports_root_preparation=True

    def __init__(self,path,*,registry,registry_actor,service_id,stint_policy=None):
        required(registry_actor);required(service_id)
        self.registry=_base(registry)
        self.registry_actor,self.service_id=registry_actor,service_id
        self._fresh_roots={}
        super().__init__(path,stint_policy=StintPolicy() if stint_policy is None else stint_policy)
        try:
            with self.transaction():
                self.db.execute('''CREATE TABLE IF NOT EXISTS requester_root_identity (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),registry TEXT NOT NULL,actor TEXT NOT NULL,service TEXT NOT NULL)''')
                self.db.execute('INSERT OR IGNORE INTO requester_root_identity VALUES(1,?,?,?)',
                                (self.registry,registry_actor,service_id))
                if tuple(self.db.execute('SELECT registry,actor,service FROM requester_root_identity').fetchone())!=(self.registry,registry_actor,service_id):
                    raise PermissionError('Root journal belongs to another physical provider')
                self.db.execute('''CREATE TABLE IF NOT EXISTS requester_root_preparations (
                    id TEXT PRIMARY KEY,resource TEXT NOT NULL,job_id TEXT NOT NULL REFERENCES requester_jobs(id),
                    execution_id TEXT NOT NULL REFERENCES executions(id),
                    state TEXT NOT NULL CHECK(state IN ('prepared','admitted','ending','complete')),
                    root_id TEXT,initial_lease_id TEXT,current_lease_id TEXT,receipt TEXT,
                    end_action TEXT CHECK(end_action IN ('finish','yield')))''')
                self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_root_preparation ON requester_root_preparations(resource) WHERE state!='complete'")
        except BaseException:
            self.close()
            raise

    def preparation(self,preparation_id):
        row=self.db.execute('SELECT * FROM requester_root_preparations WHERE id=?',(preparation_id,)).fetchone()
        if row is None:raise Conflict('Root preparation not found')
        return dict(row)

    def _current_preparation(self,resource):
        return self.db.execute("SELECT * FROM requester_root_preparations WHERE resource=? AND state!='complete'",
                               (resource,)).fetchone()

    def prepare_root(self,resource):
        """Reserve the next eligible job's logical place, keeping its call queued.

        Existing grant FIFO decides the first job; do not skip earlier legacy
        work or adopt a parent that already started through another controller.
        Reconnect returns the same durable preparation, including end debt.
        """
        required(resource)
        with self.transaction():
            current=self._current_preparation(resource)
            if current is not None:
                self._advance_job(current['job_id'])
                return self.preparation(current['id'])
            if self.db.execute('SELECT 1 FROM resource_waits WHERE resource=?',(resource,)).fetchone():
                return None
            if self.owns_job(resource) or self.db.execute("SELECT 1 FROM grants WHERE resource=? AND state IN ('granted','uncertain')",(resource,)).fetchone():
                raise Conflict('Drain existing legacy scheduling before root preparation')
            grant=self._ready_grant(resource)
            if grant is None:return None
            call=self.db.execute("SELECT * FROM executions WHERE grant_id=? AND state='queued'",(grant['id'],)).fetchone()
            if call['job_id'] is None:
                raise Conflict('Earlier unlinked work must drain without rewriting its identity')
            job=self.db.execute('SELECT * FROM requester_jobs WHERE id=?',(call['job_id'],)).fetchone()
            if job['state']!='queued':raise Conflict('Job is not available for root preparation')
            identifier=uuid4().hex
            self.db.execute("INSERT INTO requester_root_preparations(id,resource,job_id,execution_id,state) VALUES(?,?,?,?,'prepared')",
                            (identifier,resource,job['id'],call['id']))
            # Logical scheduling ownership is not physical admission. No grant
            # or execution is marked running; dispatch stays blocked below.
            self.db.execute("UPDATE requester_jobs SET state='owned',updated=? WHERE id=?",(time.time(),job['id']))
            return self.preparation(identifier)

    def next_child(self,preparation_id):
        """Loop-local preparation target, never a dispatch or admission receipt.

        Match the owned-job candidate order. Running/uncertain work anywhere on
        the resource prevents a model switch; an observer cannot override it.
        """
        prep=self.preparation(preparation_id)
        if prep['state'] not in ('prepared','admitted') or self.db.execute(
                "SELECT 1 FROM executions e JOIN grants g ON g.id=e.grant_id WHERE g.resource=? AND e.state IN ('running','uncertain')",
                (prep['resource'],)).fetchone():
            return None
        row=self.db.execute("""SELECT e.id AS execution_id,g.model FROM executions e
            JOIN grants g ON g.id=e.grant_id WHERE e.job_id=? AND e.state='queued'
            AND g.state IN ('queued','granted') AND g.release_requested=0
            ORDER BY e.created,e.id LIMIT 1""",(prep['job_id'],)).fetchone()
        return dict(row) if row is not None else None

    def withdraw_root(self,preparation_id):
        """Synchronous admission withdrawal before any handoff/failed observation."""
        self._fresh_roots.pop(preparation_id,None)
        def withdraw():
            self.db.execute("UPDATE requester_root_preparations SET state='prepared' WHERE id=? AND state='admitted'",(preparation_id,))
        if self.db.in_transaction:
            withdraw()
        else:
            with self.transaction():withdraw()

    def confirm_root(self,preparation_id,receipt,resource_snapshot):
        """Fresh exact full-root observation, never a queued/cached permission.

        Caller supplies observations from the bound registry client; this is
        trusted controller code, not an HTTP request body or absence sensor.
        """
        self.withdraw_root(preparation_id)
        with self.transaction():
            prep=self.preparation(preparation_id)
            lease=receipt.get('lease')
            view=resource_snapshot
            if (prep['state'] not in ('prepared','admitted') or receipt.get('state')!='open'
                or receipt.get('lifetime')!='explicit_end' or receipt.get('resource_id')!=prep['resource']
                or not isinstance(lease,dict) or not isinstance(view,dict)
                or view.get('id')!=prep['resource'] or view.get('allocation_mode')!='shared'
                or view.get('expiry_policy')!='hold'):
                raise Conflict('Fresh current root admission required')
            for field in ('root_id','initial_lease_id','current_lease_id'):required(receipt.get(field))
            specs=view.get('specs')
            if isinstance(specs,str):
                try:specs=json.loads(specs)
                except ValueError:specs=None
            total=next((specs.get(key) for key in ('vram_mb','gpu_memory_mb','unified_memory_mb','ram_mb')
                        if specs.get(key)),0) if isinstance(specs,dict) else 0
            if (type(total) is not int or total<=0 or type(lease.get('memory_mb')) is not int
                or lease['memory_mb']!=total or lease.get('exclusive')!=1
                or lease.get('id')!=receipt['current_lease_id'] or lease.get('resource_id')!=prep['resource']
                or lease.get('agent_id')!=self.registry_actor or lease.get('service_id')!=self.service_id
                or lease.get('status')!='active' or not _unexpired(lease)):
                raise Conflict('Exact active provider envelope required')
            rows=view.get('leases');unsettled=view.get('unsettled')
            if (not isinstance(rows,list) or not isinstance(unsettled,list)
                or any(not isinstance(row,dict) or row.get('resource_id')!=prep['resource'] for row in rows+unsettled)):
                raise Conflict('Physical resource snapshot is unknown')
            matches=[row for row in rows if row.get('id')==lease['id']]
            fields=('id','resource_id','service_id','agent_id','memory_mb','exclusive','status','expires_at')
            if (len(matches)!=1 or any(matches[0].get(k)!=lease.get(k) for k in fields)
                or unsettled or any(row.get('status')=='active' and row.get('id')!=lease['id'] for row in rows)):
                raise Conflict('Physical root snapshot differs or has unsettled ownership')
            if prep['root_id'] is not None and (prep['root_id']!=receipt['root_id'] or prep['initial_lease_id']!=receipt['initial_lease_id']):
                raise Conflict('Saved physical root cannot be replaced')
            self.db.execute("UPDATE requester_root_preparations SET state='admitted',root_id=?,initial_lease_id=?,current_lease_id=?,receipt=? WHERE id=?",
                            (receipt['root_id'],receipt['initial_lease_id'],lease['id'],json.dumps(receipt),preparation_id))
        self._fresh_roots[preparation_id]=(receipt['root_id'],lease['id'],lease['expires_at'])
        return self.preparation(preparation_id)

    def dispatch_next(self,resource,*,permitted_model=None):
        prep=self._current_preparation(resource)
        if prep is None or prep['state']!='admitted':return None
        receipt=json.loads(prep['receipt'])
        marker=(prep['root_id'],prep['current_lease_id'],receipt['lease']['expires_at'])
        if self._fresh_roots.get(prep['id'])!=marker or not _unexpired(receipt['lease']):return None
        if permitted_model is None:
            raise Conflict('Root admission does not replace model/headroom permission')
        # Base dispatch remains the atomic execution/grant owner and enforces
        # running/uncertain exclusion plus the exact parent's accepted children.
        def exact_permission(target,model):
            current=self._current_preparation(resource)
            job=self.db.execute('SELECT state FROM requester_jobs WHERE id=?',(prep['job_id'],)).fetchone()
            if (current is None or current['id']!=prep['id'] or current['state']!='admitted'
                or current['current_lease_id']!=prep['current_lease_id']
                or self._fresh_roots.get(prep['id'])!=marker or not _unexpired(receipt['lease'])
                or job is None or job['state']!='owned'):
                return None
            return permitted_model(target,model)
        return super().dispatch_next(resource,permitted_model=exact_permission)

    def _advance_job(self,job_id):
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='requester_root_preparations'").fetchone():
            prep=self.db.execute("SELECT * FROM requester_root_preparations WHERE job_id=? AND state!='complete'",(job_id,)).fetchone()
            if prep is not None:
                job=self.db.execute('SELECT * FROM requester_jobs WHERE id=?',(job_id,)).fetchone()
                pending=self.db.execute("SELECT 1 FROM executions WHERE job_id=? AND state IN ('queued','running','uncertain')",(job_id,)).fetchone()
                if job['end_requested'] and not pending:
                    self.db.execute("UPDATE requester_root_preparations SET state='ending',end_action=? WHERE id=?",
                                    (job['end_requested'],prep['id']))
                    self.withdraw_root(prep['id'])
                return  # Explicit logical finish waits for physical end receipt.
        return super()._advance_job(job_id)

    def settle_preparation(self,preparation_id,receipt):
        """Exact registry terminal receipt completes logical handback atomically.

        No backend is presumed gone and no registry release occurs here. The
        qualified controller first drains/stops/observes/settles the same root.
        """
        self.withdraw_root(preparation_id)
        with self.transaction():
            prep=self.preparation(preparation_id)
            if prep['state']=='complete':return prep  # Old completion never targets a newer preparation.
            if prep['state']!='ending' or receipt.get('resource_id')!=prep['resource'] or receipt.get('state')!='complete' or receipt.get('lifetime')!='explicit_end':
                raise Conflict('Exact ended physical root receipt required')
            if self.db.execute("SELECT 1 FROM executions WHERE job_id=? AND state IN ('queued','running','uncertain')",(prep['job_id'],)).fetchone():
                raise Conflict('Accepted children still require reconciliation')
            if prep['root_id'] is not None and (receipt.get('root_id')!=prep['root_id'] or receipt.get('initial_lease_id')!=prep['initial_lease_id']):
                raise Conflict('Physical root identity differs')
            lease=receipt.get('lease')
            if receipt.get('never_admitted') is True:
                if (prep['root_id'] is not None or any(receipt.get(key) is not None for key in ('root_id','initial_lease_id','current_lease_id'))
                    or lease is not None or receipt.get('service_id')!=self.service_id):
                    raise Conflict('Invalid never-admitted root end')
            elif (not isinstance(lease,dict) or lease.get('id')!=receipt.get('current_lease_id')
                  or lease.get('resource_id')!=prep['resource']
                  or not (lease.get('status') in ('released','cancelled') or
                          lease.get('status')=='expired' and lease.get('settled_at'))):
                raise Conflict('Physical cleanup/release remains unresolved')
            else:
                for field in ('root_id','initial_lease_id','current_lease_id'):required(receipt.get(field))
            self.db.execute("UPDATE requester_root_preparations SET state='complete',root_id=?,initial_lease_id=?,current_lease_id=?,receipt=? WHERE id=?",
                            (receipt.get('root_id'),receipt.get('initial_lease_id'),receipt.get('current_lease_id'),json.dumps(receipt),preparation_id))
            # Once this exact physical boundary is acknowledged, reuse the
            # existing finish/yield semantics and FIFO requeue operation.
            super()._advance_job(prep['job_id'])
            return self.preparation(preparation_id)
