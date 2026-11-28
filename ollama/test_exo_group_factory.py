"""Fixed fleet preflight and factory with sealed files, real queue and fake I/O.

No model/host process, physical lease or native import. The compiler substitute
retargets the existing qualified wire fixture; real compilation is checked in
the frozen interpreter separately.
"""

import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import httpx

from . import test_exo_group_deployment as deployed
from . import test_exo_group_instance as native_fixture
from . import test_exo_group_plan as core_fixture
from .exo_groups.acquire import GroupAcquisition
from .exo_groups.authority import GroupAuthorityReader
from .exo_groups.compiler import ContractCompiler
from .exo_groups.deployment import HostDeployment
from .exo_groups.factory import AssemblyFactory, HostFleet
from .exo_groups.host_control import SOURCE_SHA256, configuration_sha256
from .exo_groups.host_http import HostEndpoint
from .exo_groups.instance import InstanceContract
from .exo_groups.intents import GroupIntents
from .exo_groups.plane import CoordinatorPlane
from .exo_groups.prepare import canonical, digest
from .exo_groups.service import EXOService
from .exo_groups.staging import HostFiles, runtime_file_policy


class FixedFactoryControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = deployed.DeploymentControls()
        self.context.include_group_model = True
        self.context.addAsyncCleanup = self.addAsyncCleanup
        await self.context.asyncSetUp()
        c = self.context
        self.runtime, self.plan = c.runtime, c.plan
        self.runtime.manager._collect_stats = lambda: {'memory_available_mb': 20000, 'memory_device_inuse_mb': 0}
        self.runtime.capacity_source.pause()
        self.layouts = [c.config]
        for rank in ('wafer', 'obelisk'):
            old = next(host for host in c.stage.verifier.policies if host.rank == rank)
            directory = Path(old.runtime['root'])
            directory.chmod(0o700)
            pins = list(c.stage.runtime_files[rank])
            for name in ('run_text.py', 'python/bin/python3.13', 'source/src/exo/main.py'):
                path = directory/name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b'fixture must never execute\n'
                path.write_bytes(raw)
                pins.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'link': None})
            for path in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if not path.is_symlink():
                    path.chmod(0o500 if path.is_dir() else 0o400)
            directory.chmod(0o500)
            base = c.root/('attempts-'+rank)
            base.mkdir(mode=0o700)
            files = HostFiles(rank, old.resource_id, old.uid, old.scratch_bytes, old.model,
                              runtime_file_policy(str(directory), pins))
            self.layouts.append(HostDeployment(self.plan, files, **(c.options | {'state_base': str(base)})))
        self.layouts = tuple(self.layouts)
        self.endpoints = tuple(HostEndpoint(host.files.rank, host.files.resource_id, host.files.uid,
            'https://'+host.files.rank+'.test', SOURCE_SHA256, configuration_sha256(host)) for host in self.layouts)
        self.journals = {host.files.rank: str(c.root/(host.files.rank+'.sqlite')) for host in self.layouts}
        self.journals['slice'] = str(self.runtime.settings.journal)
        self.core_requests, self.host_operations, self.errors = [], [], []
        self.creates, self.network = 0, True
        core = core_fixture.GroupBindingControls()
        core.setUp()
        self.now = core.now
        self.group = copy.deepcopy(core.group)
        self.group.update(id='group_'+'a'*24, intent_sha256=self.plan.intent_sha256)
        self.leases = {}
        for i, member in enumerate(self.group['members']):
            lease = copy.deepcopy(core.leases[member['lease_id']])
            lease['id'] = 'lease_'+f'{i+1:016x}'
            member['lease_id'] = lease['id']
            self.leases[lease['id']] = lease

        async def core_transport(request):
            self.core_requests.append(request)
            if request.url.path == '/api/v1/auth/verify':
                value = {'valid': True, 'owner_id': self.plan.owner_id}
            elif request.method == 'POST':
                self.creates += 1
                self.assertEqual(request.url.path, '/api/v1/lease-groups')
                self.assertEqual(json.loads(request.content), self.plan.request())
                value = self.group
            elif request.url.path.startswith('/api/v1/leases/'):
                value = self.leases[request.url.path.rsplit('/', 1)[-1]]
            else:
                self.assertEqual(request.url.path, '/api/v1/lease-groups/'+self.group['id'])
                value = self.group
            return httpx.Response(200, json=copy.deepcopy(value))

        self.core_transport_callback = core_transport
        self.plane = CoordinatorPlane(self.plan.owner_id, lambda: 'Bearer synthetic-coordinator',
                                      transport=httpx.MockTransport(core_transport))
        transports = {}
        for layout, endpoint in zip(self.layouts, self.endpoints):
            async def host_transport(request, *, host=layout, route=endpoint):
                try:
                    self.host_operations.append(request.url.path.rsplit('/', 1)[-1])
                    self.assertTrue(request.url.path.endswith('/files'))
                    if not self.network:
                        return httpx.Response(503)
                    body = json.loads(request.content)
                    self.assertEqual(body['policy_sha256'], digest(asdict(host.files)))
                    binding = replace(c.binding, rank=host.files.rank,
                        resource_id=host.files.resource_id,
                        pipeline_rank=next(m.pipeline_rank for m in self.plan.members if m.rank==host.files.rank),
                        layers=next(m.layers for m in self.plan.members if m.rank==host.files.rank))
                    report = await host.observe(host.compile(binding))
                    return httpx.Response(200, headers={'cache-control': 'no-store'}, json={
                        'request_id': body['request_id'], 'source_sha256': route.source_sha256,
                        'configuration_sha256': route.configuration_sha256, 'report': report})
                except AssertionError as error:
                    self.errors.append(str(error))
                    raise
            transports[layout.files.rank] = httpx.MockTransport(host_transport)
        self.fleet_args = dict(manifest=c.stage.manifest, parent=canonical(c.stage.parent),
                               journals=self.journals, transports=transports)
        self.fleet = HostFleet(self.layouts, self.endpoints, self.plane, **self.fleet_args)
        if getattr(self, 'configuration_only', False):
            self.addAsyncCleanup(self.fleet.close)
            self.addAsyncCleanup(self.plane.close)
            return
        self.intents = GroupIntents(self.runtime.queue.supervisor.journal)

        def authority(plan, launches):
            async def transport(request):
                if request.url.path.endswith('/auth/verify'):
                    value = {'valid': True, 'owner_id': plan.owner_id}
                else:
                    binding = next(b for b in launches if b.lease_id == request.url.path.split('/')[-2])
                    value = {'group_id': self.group['id'], 'owner_id': plan.owner_id,
                        'generation': self.group['generation'], 'phase': self.group['phase'],
                        'authoritative': True, 'release_requested': False,
                        'observed_at': self.now.isoformat(), 'authority_expires_at': '2026-11-14T10:10:00+00:00',
                        'member': {'resource_id': binding.resource_id, 'lease_id': binding.lease_id,
                            'cleanup_attested': False, 'settlement_recorded': False,
                            'lease': self.leases[binding.lease_id]}}
                return httpx.Response(200, json=value)
            return GroupAuthorityReader(plan, launches, self.plane.header,
                transport=httpx.MockTransport(transport), clock=lambda: 100)

        self.actor = GroupAcquisition(self.intents, self.plan, c.stage.manifest, self.plane,
            authority, verify_staging=self.fleet.verify, wall_clock=lambda: self.now)
        self.installation = self.runtime.bind_exo({self.plan.model_id: self.actor})
        self.provider = self.installation.providers[self.plan.model_id]
        self.service = EXOService(self.runtime, deployment=c.config)

        async def feed():
            await asyncio.Event().wait()

        self.installation.consume_events_once = feed
        self.service.start()
        test = self

        class FixtureCompiler(ContractCompiler):
            async def compile(self, launches):
                reference = native_fixture.PrivateInstanceControls()
                reference.setUp()
                original = reference.contract
                value = copy.deepcopy(original._instance)
                compiled = tuple(host.compile(launch) for host, launch in zip(test.layouts, launches))
                mapping = {original.instance_id: digest([launches[0].group_id, 1, test.plan.rank_manifest_sha256, 'instance']),
                           reference.plan.model_id: test.plan.model_id}
                for oldnode, oldrunner in value['shardAssignments']['nodeToRunner'].items():
                    rank = value['shardAssignments']['runnerToShard'][oldrunner]['PipelineShardMetadata']['deviceRank']
                    mapping[oldnode] = compiled[rank].node_id
                    mapping[oldrunner] = digest([launches[0].group_id, 1, test.plan.rank_manifest_sha256,
                                                'runner:'+launches[rank].rank])
                def remap(item):
                    if isinstance(item, str):
                        return mapping.get(item, item)
                    if isinstance(item, list):
                        return [remap(v) for v in item]
                    if isinstance(item, dict):
                        return {mapping.get(k, k): remap(v) for k,v in item.items()}
                    return item
                state = remap(json.loads(original.state_instance_json))
                request = remap(original.create_request())
                hosts = {item.node_id: [{'ip': ip, 'port': test.layouts[0].ports[3]}
                    for ip in test.layouts[0].peer_ips] for item in compiled}
                state['MlxRingInstance']['hostsByNode'] = hosts
                request['instance']['MlxRingInstance']['hosts_by_node'] = hosts
                return InstanceContract(digest(asdict(test.plan)), launches[0].group_id, 1,
                    c.config.runtime_sha256, canonical(request).decode(), canonical(state).decode())

        self.compiler = FixtureCompiler(c.config, self.runtime.queue.supervisor)
        self.factory = AssemblyFactory(self.fleet, self.service, self.compiler, group_scope='projects')
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        # Synthetic plane ownership has no physical resource to settle. Drain
        # and close these test transports; this is not production cleanup proof.
        self.runtime.capacity_source.pause()
        self.provider._closed = True
        await self.actor.close()
        tasks = tuple(self.runtime.queue.supervisor.preparations)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in self.service.tasks.values():
            task.cancel()
        await asyncio.gather(*self.service.tasks.values(), return_exceptions=True)
        for client in self.service.host_clients:
            await client.close()
        await self.fleet.close()
        if self.provider.backend is not None:
            await self.provider.backend.private.close()
        self.installation.closed = self.service.closed = True
        if self.errors:
            raise AssertionError('masked fake transport assertion: '+repr(self.errors))

    async def admit(self):
        journal = self.runtime.queue.supervisor.journal
        grant = journal.submit('caller', 'one', self.layouts[0].files.resource_id, self.plan.model_id)
        call = journal.enqueue(grant['id'], 'caller', 'one', {'operation': 'chat'})
        self.actor.note_demand(call['id'])
        self.actor.events_reconciled()
        return await self.actor.wait()

    async def test_real_sealed_preflight_happens_without_any_group_request(self):
        proof = await self.fleet.verify(self.plan, self.context.stage.manifest)
        proof.check(self.plan)
        self.assertEqual(len(proof.boots), 3)
        self.assertFalse(self.core_requests)
        self.assertEqual(self.host_operations, ['files']*3)

    async def test_missing_host_transport_blocks_before_acquiring_any_member(self):
        self.network = False
        result = await self.admit()
        self.assertEqual(result.phase, 'unknown')
        self.assertEqual(self.creates, 0)
        self.assertFalse(any(list(host.state_base.iterdir()) for host in self.layouts))

    async def test_admitted_factory_wires_same_queue_clients_paths_and_readiness_without_start(self):
        self.assertEqual((await self.admit()).phase, 'admitted')
        assembly = await self.factory(self.provider)
        self.assertIs(assembly.provider, self.provider)
        self.assertIs(self.provider.backend.bindings.journal, self.runtime.queue.supervisor.journal)
        self.assertEqual(self.creates, 1)
        for host, command in zip(self.layouts, assembly.commands):
            route = self.provider.receipts.routes[command.binding.lease_id]
            self.assertEqual(route.journal_path, self.journals[host.files.rank])
            self.assertEqual(route.state_path, str(host.compile(command.binding.expected.binding).session/'attempt'))
            self.assertIs(route.reader.client, self.fleet.clients[host.files.rank])
        self.assertFalse(self.provider.backend.private.key_file.exists())
        self.assertTrue(all(op=='files' for op in self.host_operations))
        self.assertFalse(any(list(host.state_base.iterdir()) for host in self.layouts))

    async def test_endpoint_or_journal_scope_changes_refuse_before_network_or_group(self):
        bad = (replace(self.endpoints[0], configuration_sha256='0'*64), *self.endpoints[1:])
        with self.assertRaises(ValueError):
            HostFleet(self.layouts, bad, self.plane, **self.fleet_args)
        for changes in ({'slice': '/wrong/local.sqlite'}, {'slice': '/root/../alias'}):
            if '..' in changes['slice']:
                with self.assertRaises(ValueError):
                    HostFleet(self.layouts, self.endpoints, self.plane,
                        **(self.fleet_args | {'journals': self.journals | changes}))
            else:
                fleet = HostFleet(self.layouts, self.endpoints, self.plane,
                    **(self.fleet_args | {'journals': self.journals | changes}))
                try:
                    with self.assertRaises(ValueError):
                        AssemblyFactory(fleet, self.service, self.compiler, group_scope='projects')
                finally:
                    await fleet.close()
        self.assertEqual(self.creates, 0)
        self.assertFalse(self.host_operations)

    async def test_mismatched_prefill_steps_refuse_before_any_host_or_group_request(self):
        changed = copy.copy(self.layouts[1])
        changed.prefill_step_size = 256
        hosts = (self.layouts[0], changed, self.layouts[2])
        with self.assertRaises(ValueError):
            HostFleet(hosts, self.endpoints, self.plane, **self.fleet_args)
        self.assertEqual(self.creates, 0)
        self.assertFalse(self.host_operations)

    async def test_mismatched_served_contexts_refuse_before_any_host_or_group_request(self):
        changed = copy.copy(self.layouts[1])
        changed.served_context_length = 1024
        with self.assertRaises(ValueError):
            HostFleet((self.layouts[0], changed, self.layouts[2]), self.endpoints,
                      self.plane, **self.fleet_args)
        self.assertEqual(self.creates, 0)
        self.assertFalse(self.host_operations)

    async def test_factory_refuses_a_provider_using_a_different_staging_verifier(self):
        self.actor.verify_staging = core_fixture.staged_fixture
        with self.assertRaises(ValueError):
            self.factory.bind(self.provider)
        self.assertIsNone(self.provider.assembly_factory)
        self.assertFalse(self.core_requests)

    async def test_unused_host_clients_close_with_service_without_a_group(self):
        await self.service.close()
        self.assertTrue(all(client._client.is_closed for client in self.fleet.clients.values()))
        self.assertEqual(self.creates, 0)
