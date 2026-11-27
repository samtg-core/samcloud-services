"""Fixed fleet preflight and the trusted queued provider assembly factory.

No profile discovery or public configuration API. Each host's own gateway
verifies its complete files and live transport before any group is acquired.
The existing provider later calls the bound factory on its admitted members.
"""

from dataclasses import asdict
from pathlib import PurePosixPath
import re

from ..inference_arbiter import Conflict
from .assembly import GroupAssembly
from .backend import EXOBackend
from .calls import ExecutionBindings
from .compiler import ContractCompiler
from .deployment import HostDeployment
from .host_control import SOURCE_SHA256, configuration_sha256
from .host_http import HostCommands, HostControlClient, HostEndpoint, HTTPReceiptReader
from .plane import CoordinatorPlane
from .plan import bind_admitted_group
from .prepare import canonical
from .private import PrivateEXO
from .rank_runtime import spec_launch_sha256
from .receipts import ReceiptRoute
from .staging import StagingVerifier
from .startup import InstanceStarts, InstanceStartup


class HostFleet:
    def __init__(self, hosts, endpoints, plane, *, manifest, parent, journals, transports=None):
        if (not isinstance(hosts, tuple) or len(hosts) not in (2, 3)
                or any(not isinstance(host, HostDeployment) for host in hosts)
                or tuple(host.files.rank for host in hosts) != hosts[0].plan.pipeline_ranks
                or not isinstance(endpoints, tuple) or len(endpoints) != len(hosts)
                or any(not isinstance(endpoint, HostEndpoint) for endpoint in endpoints)
                or not isinstance(plane, CoordinatorPlane) or plane.owner_id != hosts[0].plan.owner_id
                or any(canonical(asdict(host.plan)) != canonical(asdict(hosts[0].plan))
                       or host.runtime_sha256 != hosts[0].runtime_sha256
                       or host.peer_ips != hosts[0].peer_ips or host.ports != hosts[0].ports
                       or host.qualification_token_limit != hosts[0].qualification_token_limit
                       or host.prefill_step_size != hosts[0].prefill_step_size
                       or host.served_context_length != hosts[0].served_context_length
                       for host in hosts)):
            raise ValueError('fixed pipeline-ordered fleet, endpoints and own coordinator required')
        for host, endpoint in zip(hosts, endpoints):
            if ((endpoint.rank, endpoint.resource_id, endpoint.uid) != (
                    host.files.rank, host.files.resource_id, host.files.uid)
                    or endpoint.source_sha256 != SOURCE_SHA256
                    or endpoint.configuration_sha256 != configuration_sha256(host)):
                raise ValueError('host endpoint differs from its qualified fixed deployment')
        if transports is not None and set(transports) != set(host.files.rank for host in hosts):
            raise ValueError('exact isolated test transports or production HTTPS required')
        if (not isinstance(journals, dict) or set(journals) != set(host.files.rank for host in hosts)
                or any(not isinstance(path, str) or not path.startswith('/') or '\x00' in path
                       or str(PurePosixPath(path)) != path or '..' in PurePosixPath(path).parts
                       for path in journals.values())):
            raise ValueError('exact qualified existing journal path for every host required')
        self.hosts, self.endpoints, self.plane = hosts, endpoints, plane
        self.journals = dict(journals)
        self.plan, self.local = hosts[0].plan, hosts[0]
        self.manifest, self.parent = manifest, parent
        runtime_files = {host.files.rank: {'files': host.files.runtime['files'],
            'directories': [d for d in host.files.runtime['views'] if d != '.']} for host in hosts}
        settings = {host.files.rank: {'model_root': str(host.model_root),
            'runtime_root': str(host.runtime_root), 'uid': host.files.uid,
            'scratch_bytes': host.files.scratch_bytes} for host in hosts}
        self.staging = StagingVerifier(self.plan, manifest, parent, self.local.runtime_sha256,
            runtime_files, settings, self._read_files)
        if any(canonical(asdict(given.files)) != canonical(asdict(rebuilt))
               for given, rebuilt in zip(hosts, self.staging.policies)):
            raise ValueError('reconstructed fleet policy differs from exact host policies')
        self.clients = {endpoint.rank: HostControlClient(endpoint, plane.header,
            transport=None if transports is None else transports[endpoint.rank]) for endpoint in endpoints}

    async def _read_files(self, host, previous):
        return await self.clients[host.rank].files(host, previous)

    async def verify(self, plan, manifest):
        return await self.staging.verify(plan, manifest)

    async def close(self):
        for client in self.clients.values():
            await client.close()


class AssemblyFactory:
    def __init__(self, fleet, service, compiler, *, group_scope):
        if (not isinstance(fleet, HostFleet) or not isinstance(compiler, ContractCompiler)
                or compiler.deployment is not fleet.local
                or compiler.supervisor is not service.runtime.queue.supervisor
                or getattr(service, 'control', None) is None or service.control.deployment is not fleet.local
                or service.roles.get('coordinator') is not getattr(service.runtime, 'exo', None)
                or fleet.journals['slice'] != str(service.runtime.settings.journal)
                or service.closed or service.draining or not isinstance(group_scope, str)
                or not re.fullmatch('[a-zA-Z0-9_-]{1,100}', group_scope)):
            raise ValueError('one exact local host, compiler and coordinator service required')
        self.fleet, self.service, self.compiler, self.group_scope = fleet, service, compiler, group_scope
        for client in fleet.clients.values():
            service.own_host_client(client)

    def bind(self, provider):
        if (provider.installation is not self.service.roles['coordinator']
                or provider.actor.plane is not self.fleet.plane
                or provider.actor.plan != self.fleet.plan
                or provider.actor.verify_staging != self.fleet.verify):
            raise ValueError('provider must use this exact own-plane fleet preflight')
        provider.bind_assembly_factory(self)

    async def __call__(self, provider):
        actor, plan = provider.actor, self.fleet.plan
        if (provider.installation is not self.service.roles['coordinator']
                or actor.plane is not self.fleet.plane or actor.plan != plan
                or actor.verify_staging != self.fleet.verify or not self.service.started
                or self.service.draining or not actor.may_start_ranks()):
            raise Conflict('exact admitted queued provider required')
        saved = actor.intents.snapshot(plan.owner_id, plan.request_key)
        group = await actor.plane.group(saved['group_id'])
        leases = {member['lease_id']: await actor.plane.lease(member['lease_id']) for member in group['members']}
        group = await actor.plane.group(saved['group_id'])
        launches = bind_admitted_group(plan, group, leases,
            expected_generation=saved['generation'], now=actor.wall_clock())
        actor.intents.bind_launches(plan.owner_id, plan.request_key, launches)
        contract = await self.compiler.compile(launches)
        contract._check_binding(plan, launches, runtime_sha256=self.fleet.local.runtime_sha256)
        compiled = tuple(host.compile(launch) for host, launch in zip(self.fleet.hosts, launches))
        instance = contract._instance
        wanted_hosts = [{'ip': address, 'port': self.fleet.local.ports[3]} for address in self.fleet.local.peer_ips]
        if (set(instance['shardAssignments']['nodeToRunner']) != {value.node_id for value in compiled}
                or canonical(instance['hostsByNode']) != canonical(
                    {value.node_id: wanted_hosts for value in compiled})):
            raise ValueError('compiled contract differs from exact host node and ring endpoints')
        if self.service.draining:
            raise Conflict('service draining; configuration retained without rank launch')

        async def verify_runtime():
            proof = await self.fleet.verify(plan, self.fleet.manifest)
            actor.intents.record_staging(plan.owner_id, plan.request_key, proof)
            return proof.runtime_sha256

        bindings = ExecutionBindings(actor.intents)
        bindings.bind_instance(plan.owner_id, plan.request_key, contract)
        private = PrivateEXO(self.fleet.local.ports[0], compiled[0].native_home/'private-api.key')
        backend = EXOBackend(actor, private, bindings, contract, launches, verify_runtime)
        provider.bind_backend(backend)
        provider.bind_publication(group_scope=self.group_scope)
        commands, routes = [], []
        for host, launch, value in zip(self.fleet.hosts, launches, compiled):
            client = self.fleet.clients[host.files.rank]
            commands.append(HostCommands(actor, client, launch, contract.runtime_sha256, service=self.service))
            routes.append(ReceiptRoute(HTTPReceiptReader(client), launch,
                self.fleet.journals[host.files.rank],
                str(value.session/'attempt'), host.boot_id,
                spec_launch_sha256(value.spec, value.environment), contract.runtime_sha256))
        provider.bind_receipts(tuple(routes))
        startup = InstanceStartup(actor, private, InstanceStarts(bindings), contract, launches, verify_runtime)
        provider.bind_readiness(startup)
        return GroupAssembly(provider, tuple(commands))
