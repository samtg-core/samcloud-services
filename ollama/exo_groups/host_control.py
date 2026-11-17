"""Opt-in, fixed-configuration host metadata and rank control on model-service.

The existing authentication middleware must supply the exact configured group
owner. A request cannot select a model, path, command, credential or deployment.
Preparation accepts only the admitted member of the fixed deployment; no caller
selects runtime configuration. Operations belong to the supervisor and journal.
"""

import asyncio
import base64
from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path
import re
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .deployment import HostDeployment
from .cleanup import CleanupExpectation, NoStartExpectation, validate_cleanup
from .http import decode_json
from .member import ExpectedMember
from .plan import LaunchBinding, MemberPlan
from .prepare import canonical, digest
from .receipt_files import collect_attempt
from .token_receipts import collect_tokens
from .rank_process import Listener, ProcessIdentity, RunBinding


_SOURCE_NAMES = ('host_control.py', 'deployment.py', 'rank_runtime.py', 'receipt_files.py', 'token_receipts.py', 'pf_transport.py')
_SOURCE = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
           for name in _SOURCE_NAMES}
SOURCE_SHA256 = digest(_SOURCE)


def run_identity(binding):
    return {name: getattr(binding, name) for name in (
        'group_id', 'generation', 'lease_id', 'resource_id',
        'rank_manifest_sha256', 'runtime_sha256')}


def configuration_sha256(deployment):
    return digest({'plan': asdict(deployment.plan), 'files': deployment.files_json,
        'runtime_sha256': deployment.runtime_sha256, 'boot_id': deployment.boot_id,
        'home': str(deployment.home), 'state_base': str(deployment.state_base),
        'peer_ips': deployment.peer_ips, 'ports': deployment.ports,
        'cleanup_tolerance_bytes': deployment.cleanup_tolerance_bytes,
        'qualification_token_limit': deployment.qualification_token_limit,
        'transport_sha256': deployment.transport_sha256})


class HostControl:
    def __init__(self, runtime, deployment, composition):
        if (not isinstance(deployment, HostDeployment) or composition.runtime is not runtime
                or getattr(runtime, 'exo_host_control', None) is not None
                or set(runtime.settings.models.values()) != {deployment.files.resource_id}
                or any(getattr(route, 'path', '').startswith('/v1/exo/host')
                       for route in runtime.queue.service.app.routes)):
            raise ValueError('one explicit host deployment on its own queue runtime required')
        self.runtime, self.deployment, self.composition = runtime, deployment, composition
        self.configuration_sha256 = configuration_sha256(deployment)
        self.accepting = True
        self.tasks = {}
        journal = runtime.queue.supervisor.journal
        with journal.transaction():
            journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_host_controls (
                resource_id TEXT NOT NULL, lease_id TEXT NOT NULL, action TEXT NOT NULL,
                request_id TEXT NOT NULL UNIQUE, identity_json TEXT NOT NULL,
                state TEXT NOT NULL, error_type TEXT, created REAL NOT NULL,
                PRIMARY KEY(resource_id,lease_id,action))''')
        runtime.queue.service.app.include_router(self.router())
        runtime.exo_host_control = self

    def _authorized(self, request):
        caller = getattr(request.state, 'caller', None)
        if not isinstance(caller, dict) or caller.get('valid') is not True:
            raise HTTPException(401, 'Verified coordinator required')
        if caller.get('owner_id') != self.deployment.plan.owner_id:
            raise HTTPException(403, 'Configured coordinator required')
        if (configuration_sha256(self.deployment) != self.configuration_sha256
                or any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != pin
                       for name, pin in _SOURCE.items())):
            raise HTTPException(503, 'Host control source or configuration changed')

    async def _body(self, request, fields):
        self._authorized(request)  # Do not read a foreign caller's body.
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 128 * 1024:
                raise HTTPException(413, 'Host control metadata exceeds bound')
            raw.extend(chunk)
        try:
            body = decode_json(raw)
            if (set(body) != fields | {'request_id'} or not isinstance(body['request_id'], str)
                    or not re.fullmatch('[a-f0-9]{64}', body['request_id'])):
                raise ValueError('shape')
            return body
        except (ValueError, TypeError, KeyError):
            raise HTTPException(422, 'Invalid host control metadata') from None

    def _response(self, request_id, report):
        return JSONResponse({'request_id': request_id, 'source_sha256': SOURCE_SHA256,
            'configuration_sha256': self.configuration_sha256, 'report': report},
            headers={'Cache-Control': 'no-store'})

    def _owner(self, identity):
        owner = getattr(self.runtime, 'host_rank', None)
        if owner is None or self.composition.roles.get('host') is not owner or owner.closed:
            raise HTTPException(409, 'No registered live host attempt')
        binding = owner.actor.spec.binding
        ExpectedMember.from_launch(self.deployment.plan, binding.expected.binding)
        compiled = self.deployment.compile(binding.expected.binding)
        if (canonical(identity) != canonical(run_identity(binding))
                or compiled.spec != owner.actor.spec
                or canonical(compiled.environment) != canonical(owner._environment)):
            raise HTTPException(409, 'Host attempt identity differs')
        return owner

    def _status(self, owner):
        owner.inventory_changed()  # Explicit physical reconciliation, not a new timer.
        row = owner._attempt()
        child = owner.actor.process
        return {'identity': run_identity(owner.actor.spec.binding), 'phase': row['phase'],
            'runtime_phase': owner.actor.phase, 'launch_attempted': owner.actor.launch_attempted,
            'child': asdict(child.identity) if child is not None else None,
            'cleanup_observed': owner.actor.receipt is not None,
            'events_reconciled': owner._events_ready,
            'host_reader_owner': owner.adapter.reader.owner_id,
            'authority_from_this_status': False,
            'operations': [dict(v) for v in self.runtime.queue.supervisor.journal.db.execute(
                'SELECT action,request_id,state,error_type FROM exo_host_controls '
                'WHERE resource_id=? AND lease_id=? ORDER BY action',
                (row['resource_id'], row['lease_id']))]}

    def _launch(self, value):
        try:
            if (not isinstance(value, dict) or set(value) != set(LaunchBinding.__dataclass_fields__)
                    or type(value['generation']) is not int or type(value['pipeline_rank']) is not int
                    or not isinstance(value['layers'], list) or len(value['layers']) != 2
                    or any(type(v) is not int for v in value['layers'])
                    or any(not isinstance(value[k], str) for k in value if k not in (
                        'generation', 'pipeline_rank', 'layers'))
                    or datetime.fromisoformat(value['expires_at']).utcoffset() is None):
                raise ValueError('shape')
            binding = LaunchBinding(**(value | {'layers': tuple(value['layers'])}))
            self.deployment.compile(binding)
            return binding
        except (KeyError, TypeError, ValueError):
            raise HTTPException(422, 'Exact fixed-deployment launch binding required') from None

    def _preparation_status(self, binding):
        row = self.runtime.queue.supervisor.journal.db.execute('SELECT * FROM exo_host_controls '
            'WHERE resource_id=? AND lease_id=? AND action=?',
            (binding.resource_id, binding.lease_id, 'prepare')).fetchone()
        if row is None or canonical(asdict(binding)).decode() != row['identity_json']:
            raise HTTPException(409, 'No exact preparation record')
        owner = getattr(self.runtime, 'host_rank', None)
        if owner is not None and self.composition.roles.get('host') is owner:
            return self._status(self._owner(run_identity(self.deployment.compile(binding).spec.binding)))
        return {'identity': run_identity(self.deployment.compile(binding).spec.binding),
            'phase': 'preparation_unconfirmed', 'authority_from_this_status': False,
            'operations': [{k: row[k] for k in ('action', 'request_id', 'state', 'error_type')}]}

    def _observed_status(self, identity):
        if getattr(self.runtime, 'host_rank', None) is not None:
            return self._status(self._owner(identity))
        if not isinstance(identity, dict) or not isinstance(identity.get('lease_id'), str):
            raise HTTPException(409, 'No exact registered host attempt')
        row = self.runtime.queue.supervisor.journal.db.execute('SELECT * FROM exo_host_controls '
            'WHERE resource_id=? AND lease_id=? AND action=?',
            (self.deployment.files.resource_id, identity.get('lease_id'), 'prepare')).fetchone()
        if row is None:
            raise HTTPException(409, 'No exact registered host attempt')
        binding = self._launch(decode_json(row['identity_json']))
        if canonical(identity) != canonical(run_identity(self.deployment.compile(binding).spec.binding)):
            raise HTTPException(409, 'Host preparation identity differs')
        return self._preparation_status(binding)

    async def _prepare(self, body):
        if body['plan_sha256'] != digest(asdict(self.deployment.plan)):
            raise HTTPException(409, 'Host deployment plan differs')
        binding = self._launch(body['launch'])
        key = binding.resource_id, binding.lease_id, 'prepare'
        journal = self.runtime.queue.supervisor.journal
        identity_json = canonical(body['launch']).decode()
        history = self._preparation_history(binding.lease_id)
        existing = journal.db.execute('SELECT 1 FROM exo_host_controls '
            'WHERE resource_id=? AND lease_id=? AND action=?', key).fetchone()
        if history and existing is None:
            await self._completed_preparations(history)
        with journal.transaction():
            old = journal.db.execute('SELECT * FROM exo_host_controls '
                'WHERE resource_id=? AND lease_id=? AND action=?', key).fetchone()
            if old is not None:
                if old['request_id'] != body['request_id'] or old['identity_json'] != identity_json:
                    raise HTTPException(409, 'Preparation already owned; inspect the same attempt')
            else:
                if (not self.accepting or not self.composition.started or self.composition.draining
                        or self.composition.closed or getattr(self.runtime, 'host_rank', None) is not None
                        or self._preparation_history(binding.lease_id) != history):
                    raise HTTPException(409, 'Fresh configured host preparation required')
                try:
                    journal.db.execute('INSERT INTO exo_host_controls VALUES(?,?,?,?,?,?,NULL,?)',
                        (*key, body['request_id'], identity_json, 'owned', time.time()))
                except sqlite3.IntegrityError:
                    raise HTTPException(409, 'Preparation request identity already used') from None
        task = self.tasks.get(key)
        if old is None:
            operation = self._bind_prepared(binding, key)
            try:
                task = self.tasks[key] = self.runtime.queue.supervisor.prepare(operation)
            except BaseException:
                operation.close()
                raise
        if task is not None:
            await asyncio.shield(task)
        # A failed/unknown preparation with no registered owner remains owned.
        # This report permits no retry, launch, absence inference or settlement.
        return self._preparation_status(binding)

    def _preparation_history(self, lease_id):
        return [dict(row) for row in self.runtime.queue.supervisor.journal.db.execute(
            "SELECT * FROM exo_host_controls WHERE action='prepare' AND lease_id<>? "
            'ORDER BY resource_id,lease_id', (lease_id,))]

    async def _completed_preparations(self, history):
        """Allow a fresh fixed session only after exact prior cleanup/settlement.

        Read the old host-owned journal/immutable receipts and a fresh projection
        with this service's own credential. No history is removed or reset.
        A started attempt requires its independent child birth and launch
        receipt, plus cleanup validated against that exact child. Incomplete,
        rebooted or unknown attempts stay refused. Old work is never resumed.
        """
        journal = self.runtime.queue.supervisor.journal
        reader = self.deployment.host_reader(self.runtime)
        try:
            for prior in history:
                row = journal.db.execute('SELECT * FROM exo_host_rank_attempts '
                    'WHERE resource_id=? AND lease_id=?',
                    (prior['resource_id'], prior['lease_id'])).fetchone()
                if (prior['state'] != 'returned' or row is None or row['phase'] not in ('launch_fenced', 'spawning')
                        or row['boot_id'] != self.deployment.boot_id
                        or row['resource_id'] != self.deployment.files.resource_id
                        or not Path(row['state_path']).is_relative_to(self.deployment.state_base)):
                    raise ValueError('prior attempt is not a fixed completed session')
                spec = {key: row[key] for key in ('resource_id', 'group_id', 'generation',
                    'lease_id', 'runtime_sha256', 'launch_sha256', 'state_path')}
                spec.update(boot_id=row['boot_id'], uid=self.deployment.files.uid,
                            journal_path=str(self.runtime.settings.journal))
                report = await asyncio.to_thread(collect_attempt, spec)
                prepared = report['prepared']
                data = prepared['binding']['expected']
                launch = dict(data['binding']); launch['layers'] = tuple(launch['layers'])
                member = dict(data['member']); member['layers'] = tuple(member['layers'])
                expected = ExpectedMember(LaunchBinding(**launch), MemberPlan(**member))
                run = RunBinding.from_expected(expected, row['runtime_sha256'])
                pin = NoStartExpectation(run, prepared['host_boot'], prepared['launch_sha256'],
                    prepared['attempt_sha256'], tuple(Listener(**v) for v in prepared['listeners']),
                    prepared['baseline_in_use_bytes'], prepared['tolerance_bytes'], prepared['prepared_at'])
                if (canonical(asdict(pin)) != canonical(prepared)
                        or canonical(asdict(expected.binding)) != canonical(decode_json(prior['identity_json']))
                        or expected.binding.rank != self.deployment.files.rank
                        or expected.member.rank != expected.binding.rank
                        or expected.member.resource_id != row['resource_id']
                        or any(getattr(expected.binding, key) != row[key]
                               for key in ('resource_id', 'lease_id', 'group_id', 'generation'))):
                    raise ValueError('prior preparation identity changed')
                started, launch = report['started'], report['launch_intent']
                if started is None:
                    if report['phase'] != 'launch_fenced' or launch is not None:
                        raise ValueError('prior no-start fence is unconfirmed')
                else:
                    if (report['phase'] != 'spawning' or not isinstance(started, dict)
                            or started.get('type') != 'rank_started' or not isinstance(launch, dict)
                            or canonical({k: started.get(k) for k in ('group_id', 'generation', 'lease_id', 'resource_id')})
                                != canonical({k: getattr(expected.binding, k) for k in ('group_id', 'generation', 'lease_id', 'resource_id')})
                            or started.get('host_boot') != row['boot_id'] or launch.get('host_boot') != row['boot_id']
                            or canonical(launch.get('binding')) != canonical(asdict(run))):
                        raise ValueError('prior child launch differs from preparation')
                    child = started.get('child')
                    fields = ProcessIdentity.__dataclass_fields__
                    if (not isinstance(child, dict) or set(child) != set(fields)
                            or any(type(child[k]) is not int or child[k] < 0 for k in fields)
                            or any(child[k] == 0 for k in ('pid', 'group_id', 'started_seconds'))
                            or child['started_microseconds'] >= 1000000
                            or child['uid'] != self.deployment.files.uid):
                        raise ValueError('prior exact child birth identity unavailable')
                    pin = CleanupExpectation(run, ProcessIdentity(**child), pin.listeners,
                        launch.get('baseline_in_use_bytes'), pin.tolerance_bytes)
                cleanup = report['cleanup']
                raw = base64.b64decode(cleanup['base64'], validate=True)
                if (len(raw) != cleanup['bytes']
                        or hashlib.sha256(raw).hexdigest() != cleanup['sha256']):
                    raise ValueError('prior cleanup bytes changed')
                validate_cleanup(raw, pin)
                projection = await reader.projection(expected)
                observed = projection.get('member', {})
                lease = observed.get('lease', {})
                if (projection.get('group_id') != run.group_id
                        or projection.get('owner_id') != expected.binding.owner_id
                        or projection.get('generation') != run.generation
                        or projection.get('phase') != 'released'
                        or projection.get('release_requested') is not True
                        or observed.get('resource_id') != run.resource_id
                        or observed.get('lease_id') != run.lease_id
                        or observed.get('cleanup_attested') is not True
                        or observed.get('settlement_recorded') is not True
                        or lease.get('id') != run.lease_id or lease.get('resource_id') != run.resource_id
                        or lease.get('agent_id') != expected.binding.owner_id
                        or lease.get('exclusive') is not True
                        or lease.get('memory_mb') != expected.member.memory_mb
                        or lease.get('status') not in ('released', 'cancelled', 'expired')
                        or lease.get('status') == 'expired' and not lease.get('settled_at')):
                    raise ValueError('prior exact member settlement is unconfirmed')
        except Exception:
            raise HTTPException(409, 'Prior host preparation needs exact cleanup and settlement') from None
        finally:
            await reader.close()

    async def _bind_prepared(self, binding, key):
        state, error_type = 'returned', None
        try:
            await self.deployment.admit_preparation(self.runtime, binding)
            if not self.accepting or self.composition.draining:
                raise RuntimeError('service draining before preparation')

            def emit(event):
                # The exact host lifecycle retains immutable local event files;
                # metadata notification contains no bearer, prompt or stdout.
                self.runtime.queue.supervisor.journal.capacity_event(
                    binding.resource_id, None, 'unknown', 'exo_rank_' + event['type'])

            try:
                owner = await self.deployment.bind(self.runtime, binding, emit=emit)
            finally:
                # Partial private setup may already have journaled a host owner.
                # Preserve its subscription and permanent no-start cleanup path.
                owner = getattr(self.runtime, 'host_rank', None)
                if (owner is not None and not owner.closed
                        and owner.actor.spec.binding == self.deployment.compile(binding).spec.binding):
                    self.composition.attach_host(owner)
            await owner.wait_connected(self.composition.tasks['host'])
        except (Exception, asyncio.CancelledError) as error:
            state, error_type = 'unconfirmed', type(error).__name__
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            with self.runtime.queue.supervisor.journal.transaction():
                self.runtime.queue.supervisor.journal.db.execute(
                    'UPDATE exo_host_controls SET state=?,error_type=? '
                    'WHERE resource_id=? AND lease_id=? AND action=?', (state, error_type, *key))

    async def _operate(self, action, body):
        owner = self._owner(body['identity'])
        if not self.accepting and action == 'start':
            raise HTTPException(409, 'Host control is draining')
        journal = self.runtime.queue.supervisor.journal
        binding = owner.actor.spec.binding
        key = binding.resource_id, binding.lease_id, action
        identity_json = canonical(body['identity']).decode()
        with journal.transaction():
            old = journal.db.execute('SELECT * FROM exo_host_controls '
                'WHERE resource_id=? AND lease_id=? AND action=?', key).fetchone()
            if old is not None:
                if old['request_id'] != body['request_id'] or old['identity_json'] != identity_json:
                    raise HTTPException(409, 'Host operation already owned; inspect it')
            else:
                if action == 'start' and not owner._events_ready:
                    raise HTTPException(409, 'Reconciled host feed required before accepting start')
                try:
                    journal.db.execute('INSERT INTO exo_host_controls VALUES(?,?,?,?,?,?,NULL,?)',
                        (*key, body['request_id'], identity_json, 'owned', time.time()))
                except sqlite3.IntegrityError:
                    raise HTTPException(409, 'Host operation identity already used') from None
        task = self.tasks.get(key)
        if old is None:
            # Commit the one operation before scheduling it. A lost HTTP reply
            # or service restart cannot turn this into another start decision.
            operation = self._perform(owner, key)
            try:
                task = self.tasks[key] = self.runtime.queue.supervisor.prepare(operation)
            except BaseException:
                operation.close()
                raise
        if task is not None:
            await asyncio.shield(task)
        return self._status(owner)

    async def _perform(self, owner, key):
        state, error_type = 'returned', None
        try:
            if key[2] == 'start':
                await owner.start()
            elif key[2] == 'stop':
                owner.stop('configured coordinator requested group drain')
            else:
                await owner.attest_never_started()
        except (Exception, asyncio.CancelledError) as error:
            state, error_type = 'unconfirmed', type(error).__name__
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            with self.runtime.queue.supervisor.journal.transaction():
                self.runtime.queue.supervisor.journal.db.execute(
                    'UPDATE exo_host_controls SET state=?,error_type=? '
                    'WHERE resource_id=? AND lease_id=? AND action=?', (state, error_type, *key))

    def pause(self):
        self.accepting = False

    async def drain(self):
        tasks = tuple(task for task in self.tasks.values() if not task.done())
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    def router(self):
        router = APIRouter(prefix='/v1/exo/host')

        @router.post('/files')
        async def files(request: Request):
            body = await self._body(request, {'policy_sha256'})
            if body['policy_sha256'] != digest(asdict(self.deployment.files)):
                raise HTTPException(409, 'Host file policy differs')
            member = next(m for m in self.deployment.plan.members if m.rank == self.deployment.files.rank)
            # Structural preflight only, before any group exists. The identity
            # is never submitted/read and cannot be used by the start route.
            binding = LaunchBinding('group_' + '0' * 24, 1, self.deployment.plan.owner_id,
                member.rank, member.pipeline_rank, member.resource_id, 'lease_' + '0' * 16,
                member.layers, '2100-01-01T00:00:00+00:00', self.deployment.plan.rank_manifest_sha256)
            try:
                report = await self.deployment.observe(self.deployment.compile(binding))
            except Exception:
                raise HTTPException(503, 'Host files or live transport unconfirmed') from None
            return self._response(body['request_id'], report)

        @router.post('/status')
        async def status(request: Request):
            body = await self._body(request, {'identity'})
            return self._response(body['request_id'], self._observed_status(body['identity']))

        @router.post('/prepare')
        async def prepare(request: Request):
            body = await self._body(request, {'launch', 'plan_sha256'})
            return self._response(body['request_id'], await self._prepare(body))

        @router.post('/receipts')
        async def receipts(request: Request):
            body = await self._body(request, {'identity'})
            owner = self._owner(body['identity'])
            binding = owner.actor.spec.binding
            spec = {**run_identity(binding), 'launch_sha256': owner.launch_sha256,
                'boot_id': owner.boot_id, 'uid': self.deployment.files.uid,
                'state_path': str(owner.actor.state), 'journal_path': str(self.runtime.settings.journal)}
            spec.pop('rank_manifest_sha256')  # Collector derives it from the actual prepared row.
            try:
                report = await asyncio.to_thread(collect_attempt, spec)
            except Exception:
                raise HTTPException(503, 'Exact host receipt unconfirmed') from None
            return self._response(body['request_id'], report)

        @router.post('/tokens')
        async def tokens(request: Request):
            body = await self._body(request, {'identity', 'command_id', 'task_id'})
            owner = self._owner(body['identity'])
            if self.deployment.qualification_token_limit != 16:
                raise HTTPException(409, 'Raw token qualification is not configured')
            binding = owner.actor.spec.binding
            compiled = self.deployment.compile(binding.expected.binding)
            expected = {'group_id': binding.group_id, 'generation': binding.generation,
                'rank': self.deployment.files.rank, 'device_rank': binding.expected.binding.pipeline_rank,
                'rank_manifest_sha256': binding.rank_manifest_sha256,
                'command_id': body['command_id'], 'task_id': body['task_id'],
                'model': self.deployment.plan.model_id}
            try:
                report = await asyncio.to_thread(collect_tokens, str(compiled.native_home),
                    self.deployment.files.uid, owner.boot_id, expected)
            except Exception:
                raise HTTPException(503, 'Exact raw token receipt unconfirmed') from None
            return self._response(body['request_id'], {'identity': run_identity(binding), **report})

        def add_operation(action):
            async def operation(request: Request):
                body = await self._body(request, {'identity'})
                return self._response(body['request_id'], await self._operate(action, body))
            router.add_api_route('/' + action, operation, methods=['POST'], name='exo_host_' + action)

        for action in ('start', 'stop', 'fence-unstarted'):
            add_operation(action)
        return router
