"""Service-owned physical handoff, separate from requester/tool affinity.

Not automatically installed. A fresh registry snapshot/event reconciliation may
ask this actor to yield; the registry remains the only allocation authority.
The local journal stores recovery identities, never another capacity budget.
"""
import asyncio
import json
import time
import subprocess
from datetime import datetime

from .inference_arbiter import Conflict
from .inference_events import source_call
from .inference_reservation import needs_settlement


def cold_vision_absent():
    """Cold recovery only: process inventory AND refused backend connection.

    This never terminates a process or qualifies live VLM unload. A failed or
    unreadable inventory is unknown; a closed port alone is not absence.
    """
    from .config import VLM_HOST, VLM_PORT
    from .inference_bootstrap import vision_backend_absent
    try:
        result = subprocess.run(['/bin/ps','-axo','command='], capture_output=True,
                                text=True, timeout=5)
        if result.returncode or not result.stdout.strip():
            return False
        for line in result.stdout.splitlines():
            # ps renders command text, not a shell-quoted round-trip. Quoting
            # in an unrelated process must not invalidate the whole inventory.
            args = line.split()
            if (any(args[i:i+2] == ['-m','mlx_vlm.server'] for i in range(len(args)))
                    or any(part.rsplit('/',1)[-1] == 'mlx_vlm.server' for part in args)):
                return False
        return vision_backend_absent(VLM_HOST, VLM_PORT) is True
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


class ResourceHandoff:
    def __init__(self, runtime):
        self.runtime = runtime
        self.initial = runtime.manager.initial_reservations
        resources = set(runtime.settings.models.values())
        if len(resources) != 1:
            raise ValueError('one physical resource per manager handoff required')
        self.resource = resources.pop()
        self.db = runtime.queue.supervisor.journal.db
        self.db.execute('''CREATE TABLE IF NOT EXISTS service_handoffs (
            resource TEXT PRIMARY KEY, phase TEXT NOT NULL, contender TEXT,
            allocations TEXT NOT NULL, updated REAL NOT NULL)''')
        columns = {row['name'] for row in self.db.execute('PRAGMA table_info(service_handoffs)')}
        if 'recovering' not in columns:
            self.db.execute("ALTER TABLE service_handoffs ADD COLUMN recovering TEXT NOT NULL DEFAULT '[]'")
        self.task = None
        if self.snapshot()['phase'] != 'idle':
            self._pause()  # restart cannot resurrect stale dispatch permission

    def snapshot(self):
        row = self.db.execute('SELECT * FROM service_handoffs WHERE resource=?',
                              (self.resource,)).fetchone()
        return ({**dict(row), 'allocations':json.loads(row['allocations']),
                 'recovering':json.loads(row['recovering'])} if row else
                {'resource':self.resource,'phase':'idle','contender':None,'allocations':[], 'recovering':[]})

    def _save(self, phase, *, contender=None, allocations=None, recovering=None):
        old = self.snapshot()
        selected = contender if contender is not None else old['contender']
        pointers = old['allocations'] if allocations is None else allocations
        recovery = old['recovering'] if recovering is None else recovering
        now = time.time()
        with self.runtime.queue.supervisor.journal.transaction():
            self.db.execute('''INSERT OR REPLACE INTO service_handoffs
                (resource,phase,contender,allocations,updated,recovering) VALUES(?,?,?,?,?,?)''',
                (self.resource, phase, selected, json.dumps(pointers), now, json.dumps(recovery)))
            if (phase, selected, pointers) != (old['phase'], old['contender'], old['allocations']):
                self.db.execute('''INSERT INTO metadata_events(kind,at,resource,state,reason)
                    VALUES('service_handoff',?,?,?,?)''', (now,self.resource,phase,selected))

    def _pause(self):
        self.runtime.capacity.pause()
        if self.runtime.capacity_source is not None:
            self.runtime.capacity_source.pause()

    def begin(self):
        """Internal wakeup hint, not permission to unload or release.

        Withdraw dispatch synchronously before an await. The service owns the
        returned task; an HTTP/event observer must shield it, not cancel it.
        Repeated hints coalesce, including hints while a handoff is uncertain.
        """
        if self.task is not None and not self.task.done():
            return self.task
        if getattr(self.runtime, '_external_quiesce', False):
            raise RuntimeError('Physical handoff is paused by service quiescence')
        self._pause()
        self.task = asyncio.create_task(self._run())
        return self.task

    def _foreign(self, view):
        if (not isinstance(view, dict) or view.get('id') != self.resource or
                view.get('expiry_policy') != 'hold' or not isinstance(view.get('leases'), list)):
            raise Conflict('shared resource observation unknown')
        for row in view['leases']:
            if not isinstance(row, dict) or row.get('resource_id') != self.resource:
                raise Conflict('shared lease observation unknown')
        return next((row for row in view['leases'] if row.get('status') == 'queued' and
                     row.get('service_id') != self.initial.service), None)

    def _exact(self, rows, allocation):
        matches = [row for row in rows if isinstance(row, dict) and
                   row.get('id') == allocation['lease_id']] if isinstance(rows, list) else []
        if len(matches) != 1:
            raise Conflict('exact physical allocation observation unknown')
        row = matches[0]
        if (row.get('agent_id') != self.initial.owner or
                row.get('service_id') != self.initial.service or
                row.get('resource_id') != self.resource or row.get('exclusive') != 0 or
                type(row.get('memory_mb')) is not int or row['memory_mb'] != allocation['memory_mb']):
            raise Conflict('physical allocation ownership differs')
        return row

    def _absent(self, model):
        from .manager import VLM_MODELS
        if model in self.runtime.manager.models:
            return False
        if any(info['default'] == model for info in VLM_MODELS.values()):
            # Only cold recovery can reach this branch: live VLM unload still
            # refuses durable accounting and admission remains fail-closed.
            return cold_vision_absent()
        running = self.runtime.manager.ollama.list_running()
        return (isinstance(running, list) and all(isinstance(row, dict) and
            isinstance(row.get('name'), str) and row['name'].removesuffix(':latest') !=
            model.removesuffix(':latest') for row in running))

    def _confirm_retained_vision(self, model):
        """Fresh exact admission for recovery, never a new registry request."""
        if self.runtime.manager.models.get(model.name) is not model:
            raise Conflict('retained vision pointer changed')
        rows = self.runtime.manager.sc.list_leases(
            resource_id=self.resource, service_id=self.initial.service)
        matches = [row for row in rows if isinstance(row, dict) and
                   row.get('id') == model.lease_id] if isinstance(rows, list) else []
        if len(matches) != 1 or type(matches[0].get('memory_mb')) is not int:
            raise Conflict('retained vision claim is unknown')
        row = self._exact(rows, {'lease_id':model.lease_id,
                                'memory_mb':matches[0]['memory_mb']})
        try:
            expiry = datetime.fromisoformat(row['expires_at'].replace('Z', '+00:00'))
            valid = expiry.tzinfo is not None and expiry.timestamp() > time.time()
        except (KeyError, AttributeError, ValueError, TypeError, OverflowError):
            valid = False
        if (row.get('status') != 'active' or needs_settlement(row) or not valid
                or type(model.memory_mb) is not int or row['memory_mb'] < model.memory_mb):
            raise Conflict('retained vision claim is not current admission')

    async def _run(self):
        manager = self.runtime.manager
        try:
            # This task outlives its observers. Unknown native work prevents
            # unloading even if there is no remaining Python worker task.
            self._save('draining')
            if (await self.runtime.quiesce(_resource_handoff=True)).get('drained') is not True:
                self._save('uncertain')
                return self.snapshot()
            saved = self.snapshot()
            view = await source_call(manager.sc.get_resource, self.resource)
            foreign = self._foreign(view)
            own_debt = any(row.get('service_id') == self.initial.service for row in
                           view.get('unsettled', []) if isinstance(row, dict))
            allocations = saved['allocations']
            if (not allocations and not own_debt and
                    self.runtime.queue.supervisor.journal.owns_job(self.resource)):
                # A foreign waiter is demand, not authority to end a requester
                # job. Recheck after draining: the current call may have ended
                # but its parent is still doing a tool or preparing its child.
                # Saved/uncertain teardown and physical accounting debt still
                # require reconciliation; never undo an unload already begun.
                from .manager import VLM_MODELS
                from .vision_lifecycle import recover_owned_vision
                vision = {info['default'] for info in VLM_MODELS.values()}
                if set(saved['recovering']) - set(manager.models):
                    raise Conflict('saved vision recovery has no owned process handle')
                for name, mm in list(manager.models.items()):
                    if (name in vision and getattr(mm, 'managed', False)
                            and getattr(mm, 'vlm_process', None) is not None
                            and (mm.vlm_process.poll() is not None or name in saved['recovering'])):
                        self._save('recovering', recovering=[name])
                        binding = getattr(manager, 'residency_renewal', None)
                        if binding is not None:
                            binding.resume()  # startup wait must keep renewing its exact charge
                        await source_call(lambda:recover_owned_vision(mm,
                            confirm_admission=lambda:self._confirm_retained_vision(mm),
                            resume=name in saved['recovering']))
                        self._save('draining', recovering=[])
                # Ownership retains resident work, not a stale allocation for
                # a backend affirmatively proved absent. The capacity source
                # cannot admit this parent's next child until that exact saved
                # initial intent is reconciled. Unknown identity/absence/release
                # keeps this handoff uncertain and dispatch paused.
                for name in self.runtime.settings.models:
                    if name in manager.models:
                        continue
                    intent = self.initial.snapshot(name)
                    if intent is not None and intent['state'] in ('unknown', 'handed_off', 'retiring'):
                        await source_call(lambda:self.initial.reconcile_absent(
                            name, confirm_absent=self._absent))
                return self._finish()
            if not allocations:
                from .manager import VLM_MODELS
                vision = {info['default'] for info in VLM_MODELS.values()}
                ended_child = any(model in vision and getattr(mm,'managed',False)
                    and getattr(mm,'vlm_process',None) is not None
                    and mm.vlm_process.poll() is not None
                    for model,mm in manager.models.items())
                if (foreign is None and manager.models and not own_debt and not ended_child
                        and not saved['recovering']):
                    return self._finish()
                # A parent can end after a replacement spawned but before its
                # readiness was proved. Its saved recovery is still teardown
                # debt: reconcile the exact child/claim below, even without a
                # foreign waiter. Never clear that intent by declaring idle.
                rows = await source_call(lambda:manager.sc.list_leases(
                    resource_id=self.resource, service_id=self.initial.service))
                for model, mm in list(manager.models.items()):
                    if self.runtime.settings.models.get(model) != self.resource:
                        raise Conflict('unconfigured residency prevents handoff')
                    candidates = [row for row in rows if isinstance(row, dict) and
                                  row.get('id') == mm.lease_id] if isinstance(rows, list) else []
                    if len(candidates) != 1 or type(candidates[0].get('memory_mb')) is not int:
                        raise Conflict('resident allocation unknown')
                    allocation = {'model':model,'lease_id':mm.lease_id,
                                  'memory_mb':candidates[0]['memory_mb']}
                    self._exact(rows, allocation)
                    allocations.append(allocation)
                # An expired owner can outlive manager inventory or a restart.
                # Recover ONLY exact pointers already in this service's journals,
                # never a latest identity-wide lease or a similar byte count.
                for model in self.runtime.settings.models:
                    if model in manager.models:
                        continue
                    known_ids = set()
                    intent = self.initial.snapshot(model)
                    if intent is not None and intent['state'] == 'unknown' and not intent['lease_id']:
                        raise Conflict('initial allocation identity is still unknown')
                    if intent is not None and intent['state'] != 'retired' and intent['lease_id']:
                        known_ids.add(intent['lease_id'])
                    actor = getattr(self.runtime, 'residency_actor', None)
                    if actor is not None:
                        try:
                            prior = actor.store.snapshot(model)
                        except Conflict:
                            prior = None
                        if prior is not None and prior['state'] != 'retired':
                            known_ids.add(prior['lease_id'])
                    for lease_id in sorted(known_ids):
                        found = [row for row in rows if isinstance(row, dict) and row.get('id') == lease_id]
                        if len(found) != 1 or type(found[0].get('memory_mb')) is not int:
                            raise Conflict('journaled physical allocation unknown')
                        allocation = {'model':model,'lease_id':lease_id,'memory_mb':found[0]['memory_mb']}
                        self._exact(rows, allocation)
                        allocations.append(allocation)
                # Save exact IDs BEFORE unloading can erase manager inventory.
                self._save('unloading', contender=foreign['id'] if foreign else None, allocations=allocations)
            for allocation in allocations:
                model = allocation['model']
                mm = manager.models.get(model)
                if mm is not None:
                    if mm.lease_id != allocation['lease_id']:
                        raise Conflict('residency pointer changed during handoff')
                    await source_call(lambda:manager.unload(model, require_confirmed=True))
                if await source_call(self._absent, model) is not True:
                    raise Conflict('provider absence unconfirmed')
                # Retire initial/sizing intents too, including their expired
                # hold-policy charges. This operation reconfirms backend absence.
                if self.initial.snapshot(model) is not None:
                    await source_call(lambda:self.initial.reconcile_absent(
                        model, confirm_absent=self._absent))
                rows = await source_call(lambda:manager.sc.list_leases(
                    resource_id=self.resource, service_id=self.initial.service))
                row = self._exact(rows, allocation)
                if row.get('status') == 'active' or needs_settlement(row):
                    if await source_call(self._absent, model) is not True:
                        raise Conflict('provider absence changed before settlement')
                    await source_call(manager.sc.release_lease, allocation['lease_id'])
                    rows = await source_call(lambda:manager.sc.list_leases(
                        resource_id=self.resource, service_id=self.initial.service))
                    row = self._exact(rows, allocation)
                if row.get('status') not in ('released', 'expired', 'cancelled') or needs_settlement(row):
                    raise Conflict('physical release unconfirmed')
            return self._finish()
        except asyncio.CancelledError:
            self._save('uncertain')
            raise  # cancellation never resumes or declares physical settlement
        except Exception:
            self._save('uncertain')
            # A failed unload may still own residency. Keep metadata renewal
            # available while dispatch remains paused and reconciliation waits.
            binding = getattr(manager, 'residency_renewal', None)
            if manager.models and binding is not None:
                binding.resume()
            return self.snapshot()

    def _finish(self):
        self._save('idle', allocations=[], recovering=[])
        if not getattr(self.runtime, '_external_quiesce', False):
            self.runtime.resume()  # requires fresh capacity; never directly pumps
        return self.snapshot()
