"""Trusted fixed fleet startup and an explicit attended coordinator session.

The already-authenticated owner supplies its own header once, in memory. Host
readers retain their manager credentials. Requests cannot select configuration,
models, commands, paths, audiences or a different principal. No default selector.
"""

import asyncio
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..inference_arbiter import Conflict
from .acquire import GroupAcquisition
from .authority import GroupAuthorityReader
from .compiler import ContractCompiler
from .deployment import HostDeployment
from .factory import AssemblyFactory, HostFleet
from .host_control import configuration_sha256
from .http import credential_headers, decode_json
from .intents import GroupIntents
from .plane import CoordinatorPlane
from .prepare import digest
from .qualification import ChatReference, TokenComparison
from .renew import GroupRenewals
from .service import EXOService


_PINS = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
         for name in ('configured.py', 'qualification.py', 'calls.py', 'factory.py', 'host_control.py')}


class AttendedCredential:
    def __init__(self):
        self._value = None

    def header(self):
        if self._value is None:
            raise RuntimeError('attended coordinator is not activated')
        return self._value

    def bind(self, header):
        value = credential_headers(lambda: header)['Authorization']
        if self._value is not None and self._value != value:
            raise Conflict('this attended coordinator already has its credential')
        self._value = value

    def clear(self):
        self._value = None


class AttendedService(EXOService):
    def __init__(self, runtime, deployment, credential):
        self.credential = credential
        super().__init__(runtime, deployment=deployment, defer_coordinator=True)

    async def close(self):
        self.draining = True
        session = getattr(self, 'session', None)
        if session is not None:
            await session.drain()
        await super().close()  # A cleanup/settlement refusal retains the credential.
        self.credential.clear()


class FleetConfiguration:
    """Pass this exact configure callback to EXOQueueStartup on Slice only.

    Other hosts use host_configuration(their_fixed_deployment). Every path,
    runtime, endpoint and audience is trusted install-time data. No JSON name
    imports, environment-selected profiles or credential-file fallback exist.
    """
    def __init__(self, hosts, endpoints, *, manifest, parent, journals, group_scope,
                 renew_before_s, raw_reference=None, reference_sha256=None, transports=None,
                 plane_transport=None):
        if (not isinstance(hosts, tuple) or len(hosts) != 3
                or any(not isinstance(host, HostDeployment) for host in hosts)
                or tuple(host.files.rank for host in hosts) != ('slice', 'wafer', 'obelisk')
                or not isinstance(group_scope, str) or not re.fullmatch('[a-zA-Z0-9_-]{1,100}', group_scope)
                or raw_reference is None and reference_sha256 is not None
                or raw_reference is not None and any(host.qualification_token_limit != 16 for host in hosts)):
            raise ValueError('fixed pipeline fleet and optional pinned raw qualification required')
        self.hosts, self.endpoints = hosts, endpoints
        self.manifest, self.parent, self.journals = manifest, parent, dict(journals)
        self.group_scope, self.renew_before_s = group_scope, renew_before_s
        self.raw_reference, self.reference_sha256 = raw_reference, reference_sha256
        self.transports, self.plane_transport = transports, plane_transport
        if raw_reference is not None:
            ChatReference(raw_reference, reference_sha256, hosts[0].plan)
        if renew_before_s is None and any(m.ttl_seconds != m.max_total_s for m in hosts[0].plan.members):
            raise ValueError('explicit renewal controls required for a renewable session')
        if renew_before_s is not None and (
                not isinstance(renew_before_s, dict)
                or set(renew_before_s) != {m.rank for m in hosts[0].plan.members}
                or any(type(renew_before_s[m.rank]) not in (float, int)
                       or not math.isfinite(renew_before_s[m.rank])
                       or not 0 < renew_before_s[m.rank] < m.ttl_seconds for m in hosts[0].plan.members)):
            raise ValueError('explicit finite per-member renewal controls required')
        self._configured = False

    async def __call__(self, runtime):
        if self._configured:
            raise ValueError('fixed configure callback is already consumed')
        local = self.hosts[0]
        if (str(runtime.settings.journal) != self.journals['slice']
                or runtime.settings.models.get(local.plan.model_id) != local.files.resource_id
                or set(runtime.settings.models.values()) != {local.files.resource_id}):
            raise ValueError('existing exact Slice journal and model catalogue required')
        self._configured = True
        credential = AttendedCredential()
        plane = CoordinatorPlane(local.plan.owner_id, credential.header, transport=self.plane_transport)
        try:
            fleet = HostFleet(self.hosts, self.endpoints, plane, manifest=self.manifest,
                parent=self.parent, journals=self.journals, transports=self.transports)
        except Exception:
            await plane.close()
            raise
        intents = GroupIntents(runtime.queue.supervisor.journal)

        def authority(plan, launches):
            return GroupAuthorityReader(plan, launches, credential.header)

        renewals = None
        if self.renew_before_s is not None:
            renewals = GroupRenewals(local.plan, plane, renew_before_s=self.renew_before_s)
        elif any(member.ttl_seconds != member.max_total_s for member in local.plan.members):
            raise ValueError('explicit renewal controls required for a renewable session')
        actor = GroupAcquisition(intents, local.plan, self.manifest, plane, authority,
                                 verify_staging=fleet.verify, renewals=renewals)
        installation = runtime.bind_exo({local.plan.model_id: actor})
        service = AttendedService(runtime, local, credential)
        compiler = ContractCompiler(local, runtime.queue.supervisor)
        AssemblyFactory(fleet, service, compiler, group_scope=self.group_scope).bind(
            installation.providers[local.plan.model_id])
        comparison = None
        if self.raw_reference is not None:
            comparison = TokenComparison(installation.providers[local.plan.model_id], fleet,
                                         self.raw_reference, self.reference_sha256)
        service.session = SessionControl(service, fleet, comparison)
        return service


def host_configuration(deployment):
    if not isinstance(deployment, HostDeployment) or deployment.files.rank == 'slice':
        raise ValueError('exact non-coordinator host deployment required')

    async def configure(runtime):
        return EXOService(runtime, deployment=deployment)
    return configure


class SessionControl:
    def __init__(self, service, fleet, comparison):
        self.service, self.fleet, self.comparison = service, fleet, comparison
        self.provider = service.roles['coordinator'].providers[fleet.plan.model_id]
        self.journal = service.runtime.queue.supervisor.journal
        self.plan = fleet.plan
        self.tasks = {}
        self.pin = self.configuration()
        with self.journal.transaction():
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_session_controls (
                owner TEXT NOT NULL,request_key TEXT NOT NULL,action TEXT NOT NULL,
                request_id TEXT NOT NULL UNIQUE,body_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,error_type TEXT,created REAL NOT NULL,
                PRIMARY KEY(owner,request_key,action))''')
        service.runtime.queue.service.app.include_router(self.router())

    def configuration(self):
        return digest({'hosts': [configuration_sha256(host) for host in self.fleet.hosts],
            'endpoints': [asdict(endpoint) for endpoint in self.fleet.endpoints],
            'journals': self.fleet.journals,
            'reference_sha256': None if self.comparison is None else self.comparison.reference.sha256})

    def authorize(self, request):
        caller = getattr(request.state, 'caller', None)
        if not isinstance(caller, dict) or caller.get('valid') is not True:
            raise HTTPException(401, 'Verified coordinator required')
        if caller.get('owner_id') != self.plan.owner_id:
            raise HTTPException(403, 'Configured coordinator required')
        if (self.configuration() != self.pin
                or any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha
                       for name, sha in _PINS.items())):
            raise HTTPException(503, 'Fixed coordinator source or configuration changed')

    async def body(self, request, fields):
        self.authorize(request)
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 4096:
                raise HTTPException(413, 'Session metadata exceeds bound')
            raw.extend(chunk)
        try:
            value = decode_json(raw)
            if (set(value) != fields | {'request_id'} or not isinstance(value['request_id'], str)
                    or not re.fullmatch('[a-f0-9]{64}', value['request_id'])):
                raise ValueError('shape')
            return value
        except (KeyError, TypeError, ValueError):
            raise HTTPException(422, 'Invalid fixed session metadata') from None

    def status(self):
        saved = self.provider._saved()
        task = self.service.tasks.get('coordinator')
        return {'group_id': None if saved is None else saved['group_id'],
            'generation': None if saved is None else saved['generation'],
            'phase': self.provider.actor.report.phase, 'reason': self.provider.actor.report.reason,
            'group_phase': self.provider.actor.group_snapshot.get('phase')
                if isinstance(self.provider.actor.group_snapshot, dict) else None,
            'credential_bound': self.service.credential._value is not None,
            'subscription': 'not_started' if task is None else 'ended' if task.done() else 'running',
            'authority_from_status': False}

    def response(self, request_id=None, **values):
        return JSONResponse({'request_id': request_id, 'configuration_sha256': self.pin,
            'source_sha256': digest(_PINS), **values}, headers={'Cache-Control': 'no-store'})

    async def operate(self, action, body, *, header=None):
        if not self.service.started or self.service.closed or self.service.draining:
            raise HTTPException(409, 'Session control is not accepting operations')
        if action in ('begin-release', 'settle'):
            saved = self.provider._saved()
            if (self.provider.receipts is None or saved is None or saved['group_id'] is None
                    or body['generation'] != saved['generation']
                    or action == 'settle' and body['lease_id'] not in self.provider.receipts.routes):
                raise HTTPException(409, 'Original group, receipt routes and generation required')
        key_action = action if action != 'settle' else 'settle:' + str(body.get('lease_id'))
        key = self.plan.owner_id, self.plan.request_key, key_action
        if action != 'activate':
            # Explicit metadata reconciliation can use a new request id.
            # The underlying group/settlement owner re-reads its same saved
            # identity; no acquisition, rank or inference is resubmitted.
            key = (*key[:2], key_action + ':' + body['request_id'])
        with self.journal.transaction():
            old = self.journal.db.execute('SELECT * FROM exo_session_controls '
                'WHERE owner=? AND request_key=? AND action=?', key).fetchone()
            if old is not None:
                if old['request_id'] != body['request_id'] or old['body_sha256'] != digest(body):
                    raise HTTPException(409, 'Session operation already owned; inspect it')
            else:
                try:
                    self.journal.db.execute('INSERT INTO exo_session_controls VALUES(?,?,?,?,?,?,NULL,?)',
                        (*key, body['request_id'], digest(body), 'owned', time.time()))
                except sqlite3.IntegrityError:
                    raise HTTPException(409, 'Session request id already belongs to another operation') from None
        task = self.tasks.get(key)
        # Activation's durable row records the owner decision, never its
        # credential. After a process restart the same authenticated request
        # must explicitly restore that in-memory credential and subscription.
        # Other operations retain their once-owned task/reconciliation rules.
        restore_activation = (action == 'activate' and task is None
            and self.service.credential._value is None)
        if old is None or restore_activation:
            operation = self.perform(action, body, key, header)
            try:
                task = self.tasks[key] = self.service.runtime.queue.supervisor.prepare(operation)
            except BaseException:
                operation.close()
                raise
        if task is not None:
            await asyncio.shield(task)
        row = self.journal.db.execute('SELECT state,error_type FROM exo_session_controls '
            'WHERE owner=? AND request_key=? AND action=?', key).fetchone()
        return self.response(body['request_id'], operation=dict(row), status=self.status())

    async def drain(self):
        if self.tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in self.tasks.values()))

    async def perform(self, action, body, key, header):
        state, error = 'returned', None
        try:
            if action == 'activate':
                self.service.credential.bind(header)
                await self.fleet.plane.verify_owner()
                self.service.connect_coordinator()
            elif action == 'reconnect':
                self.service.reconnect('coordinator')
            elif action == 'begin-release':
                if self.provider.receipts is None:
                    raise Conflict('original host receipts are not configured')
                await self.provider.receipts.begin_release(generation=body['generation'])
            else:
                if self.provider.receipts is None:
                    raise Conflict('original host receipts are not configured')
                await self.provider.receipts.settle_host(body['lease_id'], generation=body['generation'])
        except (Exception, asyncio.CancelledError) as exception:
            state, error = 'unconfirmed', type(exception).__name__
            if isinstance(exception, asyncio.CancelledError):
                raise
        finally:
            with self.journal.transaction():
                self.journal.db.execute('UPDATE exo_session_controls SET state=?,error_type=? '
                    'WHERE owner=? AND request_key=? AND action=?', (state, error, *key))

    def router(self):
        router = APIRouter(prefix='/v1/exo/session')

        @router.get('/status')
        async def status(request: Request):
            self.authorize(request)
            return self.response(status=self.status())

        @router.post('/activate')
        async def activate(request: Request):
            body = await self.body(request, set())
            return await self.operate('activate', body, header=request.headers.get('authorization'))

        @router.post('/reconnect')
        async def reconnect(request: Request):
            return await self.operate('reconnect', await self.body(request, set()))

        @router.post('/begin-release')
        async def release(request: Request):
            body = await self.body(request, {'generation'})
            if type(body['generation']) is not int or body['generation'] < 1:
                raise HTTPException(422, 'Explicit generation required')
            return await self.operate('begin-release', body)

        @router.post('/settle')
        async def settle(request: Request):
            body = await self.body(request, {'generation', 'lease_id'})
            if (type(body['generation']) is not int or body['generation'] < 1
                    or not isinstance(body['lease_id'], str)
                    or not re.fullmatch('lease_[a-f0-9]{16,64}', body['lease_id'])):
                raise HTTPException(422, 'Exact member and generation required')
            return await self.operate('settle', body)

        @router.post('/compare')
        async def compare(request: Request):
            body = await self.body(request, {'execution_id'})
            if (self.comparison is None or not isinstance(body['execution_id'], str)
                    or not re.fullmatch('[a-f0-9]{32}', body['execution_id'])):
                raise HTTPException(409, 'Fixed raw-token qualification is not configured')
            try:
                report = await self.comparison.observe(body['execution_id'])
            except Exception:
                raise HTTPException(503, 'Exact raw-token comparison unconfirmed') from None
            return self.response(body['request_id'], comparison=report)

        return router
