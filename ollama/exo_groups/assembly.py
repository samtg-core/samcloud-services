"""Admitted group -> exact hosts -> private instance -> attributed readiness.

Runs inside the existing provider's finite supervisor task. Persisted POST
ownership permits observations, never another submission after an unknown reply.
There is no assembly timer, retry loop, inference call or second queue here.
"""

from dataclasses import asdict
import time

from ..inference_arbiter import Conflict, NotFound
from .host_control import run_identity
from .host_http import HostCommands
from .prepare import canonical, digest


class GroupAssembly:
    def __init__(self, provider, commands):
        backend, readiness, receipts = provider.backend, provider.readiness, provider.receipts
        if (backend is None or readiness is None or receipts is None
                or readiness.provider is not provider or not isinstance(commands, tuple) or len(commands) != len(provider.actor.plan.pipeline_ranks)
                or any(not isinstance(command, HostCommands) or command.actor is not provider.actor
                       or command.journal is not provider.queue.supervisor.journal for command in commands)
                or tuple(command.binding.expected.binding for command in commands) != backend.launches
                or any(command.binding.runtime_sha256 != backend.contract.runtime_sha256 for command in commands)
                or len({id(command.service) for command in commands}) != 1
                or set(receipts.routes) != {launch.lease_id for launch in backend.launches}
                or any(receipts.routes[launch.lease_id].binding != launch
                       or receipts.routes[launch.lease_id].reader.route.uid != command.client.endpoint.uid
                       for launch, command in zip(backend.launches, commands))):
            raise ValueError('one exact installed provider, three host controls, receipts and readiness required')
        self.provider, self.commands = provider, commands
        self.actor, self.plan = provider.actor, provider.actor.plan
        self.journal, self.contract = provider.queue.supervisor.journal, backend.contract
        self.service = commands[0].service
        with self.journal.transaction():
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_group_assembly (
                owner TEXT NOT NULL,request_key TEXT NOT NULL,group_id TEXT NOT NULL,
                generation INTEGER NOT NULL,contract_sha256 TEXT NOT NULL,
                phase TEXT NOT NULL,error_type TEXT,updated REAL NOT NULL,
                PRIMARY KEY(owner,request_key))''')
            old = self.saved()
            if old is None:
                self.journal.db.execute('INSERT INTO exo_group_assembly VALUES(?,?,?,?,?,?,NULL,?)',
                    (self.plan.owner_id, self.plan.request_key, self.contract.group_id,
                     self.contract.generation, self.contract.sha256, 'configured', time.time()))
            elif (old['group_id'], old['generation'], old['contract_sha256']) != (
                    self.contract.group_id, self.contract.generation, self.contract.sha256):
                raise Conflict('assembly differs from the original private instance')

    def saved(self):
        row = self.journal.db.execute('SELECT * FROM exo_group_assembly WHERE owner=? AND request_key=?',
            (self.plan.owner_id, self.plan.request_key)).fetchone()
        return dict(row) if row is not None else None

    def _record(self, phase, error_type=None):
        with self.journal.transaction():
            self.journal.db.execute('UPDATE exo_group_assembly SET phase=?,error_type=?,updated=? '
                'WHERE owner=? AND request_key=?',
                (phase, error_type, time.time(), self.plan.owner_id, self.plan.request_key))

    def _current(self):
        saved = self.saved()
        if (saved is None or (saved['group_id'], saved['generation'], saved['contract_sha256']) != (
                self.contract.group_id, self.contract.generation, self.contract.sha256)
                or not self.service.started or self.service.draining or self.service.closed
                or self.provider._closed or self.provider._busy() or not self.actor.may_start_ranks()
                or self.actor.intents.closing(self.plan.owner_id, self.plan.request_key)):
            raise Conflict('fresh admitted assembly authority required')

    def _submission(self, command, action):
        binding = command.binding
        row = self.journal.db.execute('SELECT * FROM exo_host_submissions '
            'WHERE group_id=? AND generation=? AND lease_id=? AND action=?',
            (binding.group_id, binding.generation, binding.lease_id, action)).fetchone()
        if row is not None and (row['identity_json'] != canonical(run_identity(binding)).decode()
                or row['endpoint_sha256'] != digest(asdict(command.client.endpoint))):
            raise Conflict('saved host submission identity changed')
        return row

    async def _command(self, command, action):
        self._current()
        row = self._submission(command, action)
        if row is None:
            report = await command.submit(action, verify_runtime=self.provider.backend.verify_runtime)
            row = self._submission(command, action)
        else:
            # Reconcile the same durable operation. The host may resume its
            # explicitly saved read-only prefix, then commit its FIRST bind or
            # launch decision. An unknown effect is never submitted again.
            report = await command.client.status(command.binding)
        if (canonical(report.get('identity')) != canonical(run_identity(command.binding))
                or not isinstance(report.get('operations'), list)
                or sum(isinstance(v, dict) and v.get('request_id') == row['request_id']
                       and v.get('action') == action for v in report['operations']) != 1
                or report.get('cleanup_observed') is not False
                or report.get('events_reconciled') is not True
                or report.get('phase') not in ('prepared', 'spawning')):
            raise Conflict('exact current host operation is not confirmed')
        return report

    async def _receipt(self, command):
        route = self.provider.receipts.routes[command.binding.lease_id]
        report = self.provider.receipts._observation(route, await route.read(self.plan))
        if report.cleanup_bytes is not None:
            raise Conflict('ended host cannot enter assembly')
        await report.pin(self.provider.completion, generation=route.binding.generation)
        return report

    async def advance(self):
        """One event-driven pass; False needs another external wake, not a loop."""
        phase = 'preparing_hosts'
        try:
            self._current()
            ready = self.provider.readiness.saved()
            if ready is not None and ready['confirmed'] == 1:
                self._record('ready')
                return True
            self._record(phase)
            for command in self.commands:
                await self._command(command, 'prepare')
                report = await self._receipt(command)
                if report.started is not None and self._submission(command, 'start') is None:
                    raise Conflict('host started without this coordinator submission')
            # Every independent preparation pin precedes the first start.
            phase = 'starting_hosts'
            self._record(phase)
            for command in self.commands:
                report = await self._command(command, 'start')
                observed = await self._receipt(command)
                if (report.get('phase') != 'spawning' or report.get('runtime_phase') != 'running'
                        or observed.started is None or observed.phase != 'spawning'):
                    raise Conflict('exact live host start is not confirmed')
            self._current()
            phase = 'starting_instance'
            self._record(phase)
            startup = self.provider.readiness.startup
            try:
                submitted = startup.starts.snapshot(self.plan.owner_id, self.plan.request_key)
            except NotFound:
                await startup.submit()
            else:
                if submitted['contract_sha256'] != self.contract.sha256:
                    raise Conflict('owned private instance differs')
                # Even a lost acceptance is never re-POSTed. Readiness uses
                # the same exact contract, real hosts and live private state.
            phase = 'confirming_readiness'
            self._record(phase)
            await self.provider.readiness.attest()
            self._record('ready')
            return True
        except Exception as error:
            self._record(phase, type(error).__name__)
            self.journal.capacity_event(self.provider.resource, self.provider.model,
                                        'unknown', 'exo_assembly_unconfirmed')
            return False
