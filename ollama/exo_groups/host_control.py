"""Opt-in, fixed-configuration host metadata and rank control on model-service.

The existing authentication middleware must supply the exact configured group
owner. A request cannot select a model, path, command, credential or deployment.
Registration remains trusted launcher code; there is deliberately no prepare
endpoint. Stateful operations belong to the existing supervisor and journal.
"""

import asyncio
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .deployment import HostDeployment
from .http import decode_json
from .member import ExpectedMember
from .plan import LaunchBinding
from .prepare import canonical, digest
from .receipt_files import collect_attempt


_SOURCE_NAMES = ('host_control.py', 'deployment.py', 'rank_runtime.py', 'receipt_files.py')
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
        'cleanup_tolerance_bytes': deployment.cleanup_tolerance_bytes})


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
            'host_reader_owner': owner.adapter.reader.owner_id,
            'authority_from_this_status': False,
            'operations': [dict(v) for v in self.runtime.queue.supervisor.journal.db.execute(
                'SELECT action,request_id,state,error_type FROM exo_host_controls '
                'WHERE resource_id=? AND lease_id=? ORDER BY action',
                (row['resource_id'], row['lease_id']))]}

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
            return self._response(body['request_id'], self._status(self._owner(body['identity'])))

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

        def add_operation(action):
            async def operation(request: Request):
                body = await self._body(request, {'identity'})
                return self._response(body['request_id'], await self._operate(action, body))
            router.add_api_route('/' + action, operation, methods=['POST'], name='exo_host_' + action)

        for action in ('start', 'stop', 'fence-unstarted'):
            add_operation(action)
        return router
