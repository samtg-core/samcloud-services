"""Staged whole-root manager binding. Explicit attachment only; no bootstrap.

Models consume one already-admitted root. Rows below are durable provider
intent/ownership, not another allocator. Unknown startup keeps debt; unloading a
model never ends/releases the root. Actual host/process qualification is separate.
"""
from copy import deepcopy
import asyncio
from datetime import datetime
import json
import hashlib
import sqlite3
import threading
import time
from uuid import uuid4

from . import capacity
from .inference_arbiter import CapacityDeferred,Conflict
from .inference_events import source_call
from .inference_root_client import _base
from .inference_root_scheduler import _unexpired


class RootAdmissionDeferred(CapacityDeferred):
    """Pre-provider refusal using the existing typed queue wait reason."""
    def __init__(self,cause):
        self.cause=cause
        super().__init__('capacity_unavailable')


def _physical_headroom():
    try:
        reading=capacity.collect()
        available=reading.get('memory_available_mb')
        device=reading.get('memory_device_inuse_mb')
    except Exception:
        raise RootAdmissionDeferred('host_headroom_unknown') from None
    if type(available) is not int or available<0 or type(device) is not int or device<0:
        raise RootAdmissionDeferred('host_headroom_unknown')
    return available,device


class _RootOnlyClient:
    def __init__(self,client):self._client=client
    def __getattr__(self,name):
        if name in ('request_lease','release_lease','renew_lease','rollover_lease'):
            raise RuntimeError('Root mode forbids ordinary per-model lease mutation')
        return getattr(self._client,name)


class RootModelBinding:
    def __init__(self,manager,scheduler,client,*,resource,models,confirm_cold):
        from . import manager as module
        from .manager import Backend
        store=client.store
        if (not manager.durable_inference or manager.models or getattr(manager,'root_binding',None) is not None
            or manager.initial_reservations is not None or manager.residency_renewal is not None
            or getattr(manager,'_unreleased',{}) or manager._background_started or not models
            or any(not isinstance(value,Backend) or value not in (Backend.OLLAMA,Backend.VLM) for value in models.values())):
            raise Conflict('Explicit cold durable manager without legacy accounting required')
        if (resource!=module.RESOURCE_ID or store.service!=module.SERVICE_ID
            or scheduler.registry!=store.registry or scheduler.registry_actor!=store.actor
            or scheduler.service_id!=store.service or _base(manager.sc.base)!=store.registry):
            raise Conflict('Root manager/provider/registry scope differs')
        if confirm_cold() is not True:
            raise Conflict('Fresh backend absence required before root-mode attachment')
        self.manager,self.scheduler,self.client,self.store=manager,scheduler,client,store
        self.resource,self.models=resource,dict(models)
        self.confirm_cold=confirm_cold
        self._lock=threading.Lock();self._permission=None;self._source_owner=None
        self._health_changed=asyncio.Event()
        self._health_marker=None;self._health_due=None;self._health_ttl=None
        self.on_health_observed=None
        self._health_stopped=False;self._boot_id=uuid4().hex
        with store._transaction() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS root_model_intents (
                id TEXT PRIMARY KEY,preparation_id TEXT NOT NULL,resource TEXT NOT NULL,
                model TEXT NOT NULL,backend TEXT NOT NULL,lease_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('loading','resident','unknown','gone')),
                expected_mb INTEGER NOT NULL,reported_mb INTEGER,observation TEXT)''')
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_root_model ON root_model_intents(resource,model) WHERE state!='gone'")
            conn.execute('''CREATE TABLE IF NOT EXISTS root_control_suspensions (
                resource TEXT PRIMARY KEY,boot_id TEXT NOT NULL,observation TEXT NOT NULL,
                scope_digest TEXT NOT NULL)''')
        self._proxy=_RootOnlyClient(manager.sc)
        manager.sc=self._proxy
        manager.root_binding=self

    def _mode(self,*,allow_stopped=False):
        if (self._health_stopped and not allow_stopped or not self.manager.durable_inference or self.manager.root_binding is not self or self.manager.sc is not self._proxy
            or self.manager.initial_reservations is not None or self.manager.residency_renewal is not None):
            raise Conflict('Root manager accounting configuration changed')
        if self._source_owner is not None and self.manager.inference_capacity_source is not self._source_owner:
            raise Conflict('Root manager source ownership changed')

    def publish(self,preparation_id,receipt,resource_snapshot):
        """Event-loop only: bind fresh registry observations to the scheduler."""
        self._permission=None
        self._mode()
        saved=self.store.prepare(preparation_id,self.resource)
        if saved['end_requested'] or saved['root_id']!=receipt.get('root_id') or saved['initial_lease_id']!=receipt.get('initial_lease_id'):
            raise Conflict('Root client receipt does not match saved physical intent')
        self._note_health(receipt)
        prep=self.scheduler.confirm_root(preparation_id,receipt,resource_snapshot)
        with self._lock:self._permission=(prep['id'],deepcopy(receipt))

    def _note_health(self,receipt):
        """Event-loop health scheduling, independent of compute/job lifetime."""
        lease=receipt.get('lease')
        if (receipt.get('state')=='complete' or not isinstance(lease,dict)
            or lease.get('status')!='active' or lease.get('agent_id')!=self.store.actor
            or lease.get('service_id')!=self.store.service):
            self._health_marker=None;self._health_due=None;self._health_changed.set()
            return
        ttl=receipt.get('health_ttl_s')
        if type(ttl) is not int or not 1<=ttl<=86400:
            raise Conflict('Finite root health contract is unknown')
        try:
            expiry=datetime.fromisoformat(lease['expires_at'].replace('Z','+00:00'))
            if expiry.tzinfo is None:raise ValueError('timezone missing')
            remaining=expiry.timestamp()-time.time()
        except (KeyError,TypeError,ValueError,AttributeError,OverflowError):
            raise Conflict('Root health observation is unknown') from None
        marker=(receipt['root_id'],lease['id'],lease['expires_at'],ttl)
        self._health_ttl=ttl
        if marker!=self._health_marker:
            self._health_marker=marker
            self._health_due=time.monotonic()+min(ttl/4,max(0,remaining-ttl*.75))
            self._health_changed.set()

    async def wait_for_renewal(self):
        """Wait for accepted health, then its due heartbeat; no idle busy poll."""
        while True:
            if self._health_stopped:raise asyncio.CancelledError()
            self._health_changed.clear()
            due=self._health_due
            if due is None:
                await self._health_changed.wait()
                continue
            try:
                await asyncio.wait_for(self._health_changed.wait(),max(0,due-time.monotonic()))
            except asyncio.TimeoutError:
                return

    def withdraw(self):
        """Event-loop only; provider threads lose permission synchronously."""
        with self._lock:
            current=self._permission;self._permission=None
        if current:self.scheduler.withdraw_root(current[0])

    def _admission(self):
        self._mode()
        if self._source_owner is not None and self._source_owner.paused:
            raise RootAdmissionDeferred('root_source_paused')
        with self._lock:current=deepcopy(self._permission)
        if current is None:raise RootAdmissionDeferred('root_admission_unknown')
        ident,receipt=current;lease=receipt['lease']
        # Provider operations run in worker threads. Never share the scheduler's
        # SQLite connection or infer permission from persisted historical data.
        conn=sqlite3.connect('file:'+str(self.scheduler.path)+'?mode=ro',uri=True)
        conn.row_factory=sqlite3.Row
        try:prep=conn.execute('SELECT * FROM requester_root_preparations WHERE id=?',(ident,)).fetchone()
        finally:conn.close()
        saved=self.store.prepare(ident,self.resource)
        historical=json.loads(saved['receipt']) if saved['receipt'] else {}
        canonical=historical.get('lease')
        if (prep is None or prep['state']!='admitted' or prep['root_id']!=receipt['root_id']
            or prep['current_lease_id']!=lease['id'] or saved['end_requested']
            or historical.get('state')!='open' or historical.get('current_lease_id')!=lease['id']
            or not isinstance(canonical,dict) or canonical.get('status')!='active' or not _unexpired(canonical)
            or any(canonical.get(key)!=lease.get(key) for key in ('id','resource_id','agent_id','service_id','memory_mb','exclusive'))
            or not _unexpired(lease)):
            raise RootAdmissionDeferred('root_admission_unknown')
        return ident,receipt

    def _rows(self):
        with self.store._transaction() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM root_model_intents WHERE resource=? AND state!='gone'",(self.resource,))]

    def _owned(self,mm):
        self._mode()
        matches=[row for row in self._rows() if row['id']==getattr(mm,'root_model_intent',None)]
        if (len(matches)!=1 or self.manager.models.get(mm.name) is not mm or not mm.managed or matches[0]['model']!=mm.name
            or matches[0]['lease_id']!=mm.lease_id or matches[0]['backend']!=mm.backend.value):
            raise Conflict('Exact model/root intent ownership requires reconciliation')
        return matches[0]

    def validate_resident(self,mm):
        ident,receipt=self._admission();row=self._owned(mm)
        if row['state']!='resident' or row['preparation_id']!=ident or mm.lease_id!=receipt['current_lease_id']:
            raise Conflict('Model readiness requires reconciliation')

    def prepare_space(self,model,backend,need_mb):
        ident,receipt=self._admission()
        if self.models.get(model)!=backend or type(need_mb) is not int or need_mb<=0:
            raise Conflict('Unsupported or unknown root model profile')
        rows=self._rows()
        if any(row['state']!='resident' or row['preparation_id']!=ident for row in rows):
            raise Conflict('Unresolved model startup/recovery debt')
        for row in rows:
            mm=self.manager.models.get(row['model'])
            if mm is None or self._owned(mm)['id']!=row['id']:
                raise Conflict('Resident model ownership is unknown')
        if set(self.manager.models)!={row['model'] for row in rows}:
            raise Conflict('Untracked manager residency prevents root load')
        limit=receipt['lease']['memory_mb']
        if need_mb>limit:raise RootAdmissionDeferred('model_exceeds_root_envelope')
        def fits():
            available,device=_physical_headroom()
            used=sum(row['reported_mb'] for row in self._rows() if row['state']=='resident')
            # A cached model size excludes later KV/context growth. Do not
            # infer spare root bytes from that smaller historical number.
            return used+need_mb<=limit and device+need_mb<=limit and capacity.fits(need_mb,available)
        if not fits():
            for row in rows:
                # Strict manager unload proves owned backend absence. Its root
                # hook records model gone without releasing the parent claim.
                result=self.manager.unload(row['model'],force=True,require_confirmed=True)
                if result.get('status')!='unloaded':raise Conflict('Root model switch remains unresolved')
                self._admission()
                if fits():break
        if not fits():raise RootAdmissionDeferred('host_headroom_unavailable')

    def start_load(self,model,memory_mb,*,before_load):
        ident,receipt=self._admission()
        if not before_load or model not in self.models or type(memory_mb) is not int or memory_mb<=0:
            raise Conflict('Explicit supported root load required')
        with self.store._transaction() as conn:
            rows=list(conn.execute("SELECT * FROM root_model_intents WHERE resource=? AND state!='gone'",(self.resource,)))
            if (any(row['state']!='resident' or row['preparation_id']!=ident or row['model']==model for row in rows)
                or sum(row['reported_mb'] for row in rows)+memory_mb>receipt['lease']['memory_mb']):
                raise Conflict('Unresolved or insufficient root model ownership')
            conn.execute("INSERT INTO root_model_intents VALUES(?,?,?,?,?,?,'loading',?,NULL,NULL)",
                (uuid4().hex,ident,self.resource,model,self.models[model].value,receipt['current_lease_id'],memory_mb))
        return receipt['current_lease_id']

    def bind_handle(self,mm):
        """Save the MM link before post-load sizing/readiness can fail."""
        matches=[row for row in self._rows() if row['model']==mm.name and row['lease_id']==mm.lease_id and row['state']=='loading']
        if len(matches)!=1 or matches[0]['backend']!=mm.backend.value:
            raise Conflict('Exact startup intent is unknown')
        mm.root_model_intent=matches[0]['id']

    def loaded(self,mm):
        self.bind_handle(mm)
        row=self._owned(mm)
        try:ident,receipt=self._retained_owner(row)
        except Conflict:
            with self.store._transaction() as conn:
                conn.execute("UPDATE root_model_intents SET state='unknown' WHERE id=?",(row['id'],))
            # Provider mutation has happened. Never describe this as a safe
            # pre-provider refusal or let the supervisor silently repeat it.
            raise Conflict('Post-load root admission requires reconciliation') from None
        try:_,device=_physical_headroom()
        except CapacityDeferred:
            # The load already started under positive admission. An observer
            # gap does not undo known provider completion or admit another
            # call; the source still needs fresh physical data for dispatch.
            device=None
            with self._lock:self._permission=None
        with self.store._transaction() as conn:
            others=list(conn.execute("SELECT * FROM root_model_intents WHERE resource=? AND state!='gone' AND id!=?",(self.resource,row['id'])))
            valid=(row['preparation_id']==ident and row['lease_id']==receipt['current_lease_id']
                and type(mm.memory_mb) is int and mm.memory_mb>0
                and (device is None or device<=receipt['lease']['memory_mb'])
                and all(r['state']=='resident' for r in others)
                and sum(r['reported_mb'] for r in others)+mm.memory_mb<=receipt['lease']['memory_mb'])
            conn.execute('UPDATE root_model_intents SET state=?,reported_mb=? WHERE id=?',
                         ('resident' if valid else 'unknown',mm.memory_mb,row['id']))
        if not valid:raise Conflict('Loaded model exceeds or differs from confirmed root')
        return mm.lease_id

    def _retained_owner(self,row):
        """Already-started load may finish during admission withdrawal/health loss.

        This never permits a new backend start. The exact physical root/member
        must still be held, including expired unsettled health under the frozen
        hold policy. Released/changed/settled ownership is reconciliation.
        """
        saved=self.store.prepare(row['preparation_id'],self.resource)
        receipt=json.loads(saved['receipt']) if saved['receipt'] else {}
        lease=receipt.get('lease')
        conn=sqlite3.connect('file:'+str(self.scheduler.path)+'?mode=ro',uri=True)
        conn.row_factory=sqlite3.Row
        try:prep=conn.execute('SELECT * FROM requester_root_preparations WHERE id=?',(row['preparation_id'],)).fetchone()
        finally:conn.close()
        if (prep is None or prep['state'] not in ('prepared','admitted','ending')
            or prep['root_id']!=saved['root_id'] or prep['initial_lease_id']!=saved['initial_lease_id']
            or prep['current_lease_id']!=row['lease_id'] or receipt.get('root_id')!=saved['root_id']
            or receipt.get('state') not in ('open','ending') or receipt.get('resource_id')!=self.resource
            or receipt.get('current_lease_id')!=row['lease_id'] or not isinstance(lease,dict)
            or lease.get('id')!=row['lease_id'] or lease.get('resource_id')!=self.resource
            or lease.get('agent_id')!=self.store.actor or lease.get('service_id')!=self.store.service
            or lease.get('exclusive')!=1 or type(lease.get('memory_mb')) is not int or lease['memory_mb']<=0
            or not (lease.get('status')=='active' or lease.get('status')=='expired'
                    and lease.get('expiry_reason')=='ttl_expired' and not lease.get('settled_at'))):
            raise Conflict('Exact started-load root ownership requires reconciliation')
        return row['preparation_id'],receipt

    def model_gone(self,mm):
        """Called only after strict manager unload verified actual absence."""
        row=self._owned(mm)
        with self.store._transaction() as conn:
            conn.execute("UPDATE root_model_intents SET state='gone',observation='strict-manager-unload' WHERE id=?",(row['id'],))

    def retain_claim(self,lease_id):
        """A startup/release error cannot release a full parent envelope."""
        rows=[row for row in self._rows() if row['lease_id']==lease_id]
        if not rows:raise Conflict('Unknown root claim requires reconciliation')
        with self._lock:self._permission=None
        with self.store._transaction() as conn:
            conn.execute("UPDATE root_model_intents SET state='unknown' WHERE lease_id=? AND state!='gone'",(lease_id,))

    def reconcile_gone(self,intent_id,*,observation,confirm_absent,confirm_no_work):
        """Explicit fresh provider+execution evidence; no physical release here."""
        if not isinstance(observation,str) or not observation.strip():raise ValueError('Absence observation required')
        with self.store._transaction() as conn:
            row=conn.execute('SELECT * FROM root_model_intents WHERE id=? AND resource=?',(intent_id,self.resource)).fetchone()
        if row is None:raise Conflict('Model intent not found')
        if row['model'] in self.manager.models or confirm_no_work() is not True or confirm_absent(row['model']) is not True:
            raise Conflict('Model/process execution remains unresolved')
        with self.store._transaction() as conn:
            conn.execute("UPDATE root_model_intents SET state='gone',observation=? WHERE id=?",(observation,intent_id))

    def _suspension_digest(self):
        # Only stable identity/model history, not a lease's changing health,
        # current holder or a caller prompt. New work/model intent invalidates
        # the witness; ordinary heartbeat cannot do so accidentally.
        with self.store._transaction() as conn:
            intents=[tuple(row) for row in conn.execute(
                'SELECT job_id,submit_key,root_id,initial_lease_id FROM root_client_intents WHERE resource=? ORDER BY job_id',(self.resource,))]
            models=[tuple(row) for row in conn.execute(
                'SELECT id,preparation_id,lease_id,state FROM root_model_intents WHERE resource=? ORDER BY id',(self.resource,))]
        conn=sqlite3.connect('file:'+str(self.scheduler.path)+'?mode=ro',uri=True)
        try:preps=conn.execute("SELECT id,job_id FROM requester_root_preparations WHERE resource=? AND state!='complete' ORDER BY id",(self.resource,)).fetchall()
        finally:conn.close()
        scope=[self.resource,self.store.registry,self.store.actor,self.store.service,
               sorted((name,kind.value) for name,kind in self.models.items()),intents,models,preps]
        return hashlib.sha256(json.dumps(scope,separators=(',',':')).encode()).hexdigest()

    def _has_suspension(self):
        with self.store._transaction() as conn:
            row=conn.execute('SELECT * FROM root_control_suspensions WHERE resource=?',(self.resource,)).fetchone()
        return bool(row and row['boot_id']==self._boot_id and row['scope_digest']==self._suspension_digest())

    async def suspend_for_restart(self,*,observation):
        """Explicit qualified backend drain while retaining job/root ownership.

        Unlike end, this never sends a registry mutation or fabricates logical
        completion. Caller pauses/drains the service first. A fresh cold sensor
        plus no unknown compute permits stopping control while exact charged /
        queued / unacknowledged physical intents remain durable for reconnect.
        """
        self._mode()
        if not isinstance(observation,str) or not observation.strip() or len(observation)>512:
            raise ValueError('Fresh suspension observation required')
        source=self.manager.inference_capacity_source
        if source is None or source is not self._source_owner or not source.paused:
            raise Conflict('Paused bound root source required for suspension')
        def drained():
            current=asyncio.current_task()
            if any(not task.done() and task is not current for task in
                    (*source.supervisor.workers.values(),*source.supervisor.preparations)):
                raise Conflict('Known service work must drain before suspension')
            if self.scheduler.db.execute("SELECT 1 FROM executions e JOIN grants g ON g.id=e.grant_id WHERE g.resource=? AND e.state IN ('running','uncertain')",(self.resource,)).fetchone():
                raise Conflict('Running or uncertain execution requires reconciliation before suspension')
        drained()
        self.withdraw()
        for name,mm in list(self.manager.models.items()):
            self._owned(mm)
            result=await source_call(lambda:self.manager.unload(name,force=True,require_confirmed=True))
            if result.get('status')!='unloaded':raise Conflict('Owned backend cleanup remains unconfirmed')
        if self._rows():
            raise Conflict('Unknown model startup requires explicit reconciliation before suspension')
        if await source_call(self.confirm_cold) is not True:
            raise Conflict('Fresh backend/cache absence required for suspension')
        drained()
        if not source.paused or self.manager.models or self._rows():
            raise Conflict('Suspension source/backend ownership changed')
        digest=self._suspension_digest()
        with self.store._transaction() as conn:
            conn.execute('INSERT INTO root_control_suspensions VALUES(?,?,?,?) ON CONFLICT(resource) DO UPDATE SET boot_id=excluded.boot_id,observation=excluded.observation,scope_digest=excluded.scope_digest',
                         (self.resource,self._boot_id,observation,digest))
        return {'state':'suspended','resource':self.resource,'ownership_retained':True,
                'backend_absence_observation':observation,'scope_digest':digest}

    def assert_shutdown_ready(self):
        """No mutation or HTTP: audit terminal or explicitly suspended ownership.

        Stop new source work and drain tracked tasks before this check. Queued
        calls may remain durable. A preparation with no client intent is known
        never submitted (SDK persists before HTTP); it is suspended, not ended.
        Unknown/lost-ACK intent requires a terminal receipt or a current-boot
        suspension witness; neither invents completion or releases ownership.
        """
        self._mode(allow_stopped=True)
        source=self.manager.inference_capacity_source
        if source is not None:
            if getattr(source,'binding',None) is not self or not getattr(source,'paused',False):
                raise Conflict('Root source must be paused before manager shutdown')
            supervisor=source.supervisor
            if any(not task.done() for task in (*supervisor.workers.values(),*supervisor.preparations)):
                raise Conflict('Root control/backend work remains; shutdown refused')
        if self.manager.models or self._rows():
            raise Conflict('Root model ownership requires explicit cleanup before shutdown')
        with self.store._transaction() as conn:
            intents={row['job_id']:dict(row) for row in conn.execute(
                'SELECT * FROM root_client_intents WHERE resource=?',(self.resource,))}
        receipts={};suspended=self._has_suspension()
        for key,row in intents.items():
            try:receipt=json.loads(row['receipt']) if row['receipt'] else None
            except (TypeError,ValueError):receipt=None
            if not isinstance(receipt,dict) or receipt.get('state')!='complete':
                if not suspended:raise Conflict('Root intent lacks exact terminal acknowledgement or qualified suspension; shutdown refused')
            else:receipts[key]=receipt
        conn=sqlite3.connect('file:'+str(self.scheduler.path)+'?mode=ro',uri=True)
        conn.row_factory=sqlite3.Row
        try:
            if conn.execute("SELECT 1 FROM executions e JOIN grants g ON g.id=e.grant_id WHERE g.resource=? AND e.state IN ('running','uncertain')",(self.resource,)).fetchone():
                raise Conflict('Root execution ownership remains unresolved; shutdown refused')
            preps=[dict(row) for row in conn.execute(
                "SELECT * FROM requester_root_preparations WHERE resource=? AND state!='complete'",(self.resource,))]
            owned={row['id'] for row in conn.execute(
                "SELECT id FROM requester_jobs WHERE resource=? AND state='owned'",(self.resource,))}
        finally:conn.close()
        if owned-{row['job_id'] for row in preps}:
            raise Conflict('Unqualified logical owner requires reconciliation before shutdown')
        for prep in preps:
            receipt=receipts.get(prep['id'])
            if receipt is None:
                if not suspended and any(prep[key] is not None for key in ('root_id','initial_lease_id','current_lease_id')):
                    raise Conflict('Root preparation lost its physical receipt; shutdown refused')
                continue  # Unsubmitted or explicitly suspended: keep queued identity.
            if prep['root_id'] is not None and any(prep[key]!=receipt.get(key) for key in ('root_id','initial_lease_id')):
                raise Conflict('Terminal root acknowledgement differs from preparation')
        # A terminal physical receipt may precede logical settlement. Preserve
        # that metadata for exact recovery; do not fabricate logical completion.

    def begin_shutdown(self):
        """One-way observation fence after positive physical drain, no release."""
        self.assert_shutdown_ready()
        source=self.manager.inference_capacity_source
        if source is not None:source._stop_observations()
        self.withdraw()
        self.on_health_observed=None
        self._health_stopped=True;self._health_due=None;self._health_marker=None
        self._health_changed.set()

    async def renew_models(self):
        """One exact root heartbeat even with no resident models; no swap."""
        try:
            prep=self.scheduler._current_preparation(self.resource)
            if prep is None:
                self._health_due=None;self._health_marker=None
                return []
            # Failed observations retain intent/charge, but cannot make this a
            # zero-delay retry loop after a missed heartbeat.
            if self._health_ttl is not None:self._health_due=time.monotonic()+self._health_ttl/4
            self.withdraw()
            current=await source_call(self.client.observe,prep['id'],self.resource)
            self._note_health(current)
            if self._health_due is None:
                raise Conflict('Current root health requires exact-holder reconciliation')
            renewed=await source_call(lambda:self.client.renew(prep['id'],self.resource,lease_id=current['current_lease_id']))
            self._note_health(renewed)
            if prep['state']!='ending' and renewed['state']=='open':
                view=await source_call(self.manager.sc.get_resource,self.resource)
                self.publish(prep['id'],renewed,view)
            return [renewed]
        finally:
            # A completed health observation (including a failed read) is a
            # real event, not an admission receipt. The optional source owns
            # its coalesced task; it cannot mask the maintenance result.
            if self.on_health_observed is not None:
                try:self.on_health_observed()
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning('Root health source notification incomplete')

    async def refresh(self):
        """Withdraw before reading; failed observation never reuses old permission."""
        prep=self.scheduler._current_preparation(self.resource)
        self.withdraw()
        if prep is None:return None
        receipt=await source_call(self.client.observe,prep['id'],self.resource)
        self._note_health(receipt)
        if prep['state']=='ending' or receipt['state']!='open':return receipt
        view=await source_call(self.manager.sc.get_resource,self.resource)
        self.publish(prep['id'],receipt,view)
        return receipt

    async def prepare_root(self,preparation_id):
        """Explicit saved begin; arm health even when model admission is unknown."""
        self._mode()  # Refuse stopped/drifted ownership BEFORE any root submit.
        prep=self.scheduler.preparation(preparation_id)
        if prep['state'] not in ('prepared','admitted'):
            raise Conflict('Root preparation is ending; flush its saved end instead')
        self.withdraw()
        receipt=await source_call(self.client.begin,preparation_id,self.resource)
        self._note_health(receipt)
        if self.scheduler.preparation(preparation_id)['state']=='ending':return receipt
        if receipt.get('state')=='open' and receipt.get('lease',{}).get('status')=='active':
            view=await source_call(self.manager.sc.get_resource,self.resource)
            self.publish(preparation_id,receipt,view)
        return receipt

    async def end_drained_root(self,preparation_id,*,observation):
        """Explicit controller boundary: stop/observe/settle, then logical end.

        The supplied cold observer must qualify process/cache absence on the
        actual shipping host. A clock, socket loss or registry DELETE is not
        that evidence. This method creates no replacement physical claim.
        """
        self._mode()
        if not isinstance(observation,str) or not observation.strip():
            raise ValueError('Fresh absence observation required')
        prep=self.scheduler.preparation(preparation_id)
        if prep['state']=='complete':return prep
        if prep['state']!='ending' or self.scheduler.db.execute(
                "SELECT 1 FROM executions WHERE job_id=? AND state IN ('queued','running','uncertain')",
                (prep['job_id'],)).fetchone():
            raise Conflict('Accepted execution ownership has not drained')
        self.withdraw()
        ended=await source_call(self.client.end,preparation_id,self.resource)
        self._note_health(ended)
        if ended['state']=='complete':
            if self.manager.models or self._rows():
                raise Conflict('Terminal root still has local model debt; reconcile without stopping another holder')
            return self.scheduler.settle_preparation(preparation_id,ended)
        lease=ended.get('lease')
        if (not isinstance(lease,dict) or lease.get('agent_id')!=self.store.actor
            or lease.get('service_id')!=self.store.service or lease.get('resource_id')!=self.resource
            or lease.get('status') not in ('active','expired') or lease.get('exclusive')!=1
            or type(lease.get('memory_mb')) is not int or lease['memory_mb']<=0
            or lease['status']=='expired' and (lease.get('expiry_reason')!='ttl_expired' or lease.get('settled_at'))):
            raise Conflict('Exact current root holder must reconcile its backend')
        for name,mm in list(self.manager.models.items()):
            if self._owned(mm)['preparation_id']!=preparation_id:
                raise Conflict('Another physical intent still has model ownership')
            result=await source_call(lambda:self.manager.unload(name,force=True,require_confirmed=True))
            if result.get('status')!='unloaded':raise Conflict('Backend cleanup remains unresolved')
        if await source_call(self.confirm_cold) is not True:
            raise Conflict('Backend/process/cache absence remains unconfirmed')
        rows=self._rows()
        if any(row['preparation_id']!=preparation_id for row in rows):
            raise Conflict('Unrelated model debt requires reconciliation')
        # After accepted work drained AND fresh global provider absence, a
        # pre-handle crash/start intent is positively gone. Never infer that
        # from empty inventory alone or use it to silently repeat generation.
        with self.store._transaction() as conn:
            conn.execute("UPDATE root_model_intents SET state='gone',observation=? WHERE preparation_id=? AND resource=? AND state!='gone'",
                         (observation,preparation_id,self.resource))
        final=await source_call(lambda:self.client.settle(preparation_id,self.resource,
            lease_id=lease['id'],observation=observation))
        self._note_health(final)
        return self.scheduler.settle_preparation(preparation_id,final)
