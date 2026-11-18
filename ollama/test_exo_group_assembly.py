"""Provider-owned complete startup, fake HTTPS/native I/O, real queue journal.

Independent host receipt objects reuse the full preparation/child fixtures. No
physical lease, file upload, native process or model is created by these tests.
"""

import asyncio
import copy
from dataclasses import asdict, replace
import json
from unittest import IsolatedAsyncioTestCase

import httpx

from . import test_exo_group_readiness as controls
from .exo_groups.assembly import GroupAssembly
from .exo_groups.cleanup import NoStartExpectation
from .exo_groups.host_control import run_identity
from .exo_groups.host_http import HostCommands, HostControlClient, HostEndpoint
from .exo_groups.prepare import canonical, digest
from .exo_groups.service import EXOService


class AssemblyControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = controls.ReadinessControls()
        self.context.addAsyncCleanup = self.addAsyncCleanup
        await self.context.asyncSetUp()
        self.native = self.context.fixture
        self.fixture, self.parts = self.native.control, self.context.parts
        self.provider, self.actor = self.context.provider, self.native.actor
        self.journal = self.fixture.journal
        with self.journal.transaction():
            self.journal.db.execute('DELETE FROM exo_instance_starts')
        self.native.posts = 0
        self.native_ready = copy.deepcopy(self.native.initial)
        empty = copy.deepcopy(self.parts.state)
        for name in ('instances', 'runners', 'tasks', 'downloads', 'prefillServerPorts', 'instanceLinks'):
            empty[name] = {}
        empty['lastEventAppliedIdx'] = 12
        empty['topology']['nodes'] = list(self.parts.contract._instance['shardAssignments']['nodeToRunner'])
        self.native.initial = empty
        self.started_reports = copy.deepcopy(self.context.reports)
        for key, report in self.context.reports.items():
            self.context.reports[key] = replace(report, started=None, phase='prepared')
        self.effects, self.operations, self.errors = [], {}, []
        self.mode, self.lost = 'ok', False
        self.low_memory = set()
        self.capacity_reads = []
        self.capacity_fault = None
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.service = EXOService(self.fixture.runtime)
        connected = asyncio.Event()

        async def feed():
            connected.set()
            await asyncio.Event().wait()

        self.fixture.installation.consume_events_once = feed
        self.service.start()
        await connected.wait()
        self.commands = []
        for launch in self.parts.launches:
            endpoint = HostEndpoint(launch.rank, launch.resource_id, 501,
                'https://' + launch.rank + '.test', 'a' * 64, 'b' * 64)

            async def transport(request, *, route=endpoint, member=launch):
                try:
                    return await self.host(request, route, member)
                except AssertionError as error:
                    self.errors.append(str(error))
                    raise

            client = HostControlClient(endpoint, lambda: 'Bearer synthetic-coordinator',
                transport=httpx.MockTransport(transport))
            self.commands.append(HostCommands(self.actor, client, launch,
                self.parts.runtime, service=self.service))
        self.assembly = GroupAssembly(self.provider, tuple(self.commands))
        original = self.native.transport

        async def native(request):
            if self.mode == 'private-unavailable':
                raise httpx.ConnectError('private API has not opened')
            result = await original(request)
            if request.method == 'POST':
                self.effects.append(('instance',))
                for launch in self.parts.launches:
                    self.assertNotIsInstance(self.provider.completion._pin(launch.lease_id), NoStartExpectation)
                self.native.initial = copy.deepcopy(self.native_ready)
                if self.mode == 'lost-instance' and not self.lost:
                    self.lost = True
                    raise httpx.ReadError('accepted instance, reply lost')
            return result

        await self.parts.private._client.aclose()
        self.parts.private._client = httpx.AsyncClient(base_url='http://127.0.0.1:52515',
            transport=httpx.MockTransport(native), timeout=None)
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        self.release.set()
        for command in self.commands:
            await command.drain()
            await command.client.close()
        for task in self.service.tasks.values():
            task.cancel()
        await asyncio.gather(*self.service.tasks.values(), return_exceptions=True)
        if self.errors:
            raise AssertionError('masked fake transport assertion: ' + repr(self.errors))

    async def host(self, request, endpoint, launch):
        value = json.loads(request.content)
        action = request.url.path.rsplit('/', 1)[-1]
        command = next(c for c in self.commands if c.binding.lease_id == launch.lease_id)
        if action == 'prepare':
            self.assertEqual(canonical(value['launch']), canonical(asdict(launch)))
            self.assertEqual(value['plan_sha256'], digest(asdict(self.actor.plan)))
        else:
            self.assertEqual(value['identity'], run_identity(command.binding))
        if action == 'capacity':
            self.capacity_reads.append(launch.rank)
            needed = command.binding.expected.member.memory_mb * 1024 * 1024
            report = {
                'identity': run_identity(command.binding),
                'uid': endpoint.uid, 'boot_id': self.provider.receipts.routes[launch.lease_id].boot_id,
                'manager_empty': True, 'total_bytes': needed * 2,
                'available_bytes': 0,
                'wired_limit_bytes': needed - 1 if launch.rank in self.low_memory else needed,
                'kernel_wired_bytes': needed // 4, 'device_inuse_bytes': needed // 8}
            if launch.rank == 'wafer' and self.capacity_fault:
                if self.capacity_fault == 'contact':
                    return httpx.Response(503)
                if self.capacity_fault == 'occupied':
                    report['kernel_wired_bytes'] = needed + 1
                else:
                    report.update(self.capacity_fault)
            return httpx.Response(200, headers={'cache-control': 'no-store'}, json={
                'request_id': value['request_id'], 'source_sha256': endpoint.source_sha256,
                'configuration_sha256': endpoint.configuration_sha256, 'report': report})
        if action in ('prepare', 'start'):
            # Coordinator journal ownership precedes every network effect.
            row = self.assembly._submission(command, action)
            self.assertEqual(value['request_id'], row['request_id'])
            self.effects.append((action, launch.rank))
            if action == 'start':
                self.entered.set()
                await self.release.wait()
                if self.mode == 'dropped-start' and not self.lost:
                    self.lost = True
                    raise httpx.ReadError('start never reached the server')
                if launch.pipeline_rank == 0:
                    for member in self.parts.launches:
                        self.assertIsInstance(self.provider.completion._pin(member.lease_id), NoStartExpectation)
                self.context.reports[launch.lease_id] = self.started_reports[launch.lease_id]
                if self.mode == 'memory-lost-during-start':
                    self.low_memory.add('wafer')
            self.operations[launch.lease_id, action] = {
                'request_id': value['request_id'], 'action': action, 'state': 'returned'}
            if action == 'start' and self.mode == 'lost-start' and not self.lost:
                self.lost = True
                raise httpx.ReadError('start accepted, reply lost')
        else:
            self.assertEqual(action, 'status')
        started = self.context.reports[launch.lease_id].started is not None
        return httpx.Response(200, headers={'cache-control': 'no-store'}, json={
            'request_id': value['request_id'], 'source_sha256': endpoint.source_sha256,
            'configuration_sha256': endpoint.configuration_sha256, 'report': {
                'identity': run_identity(command.binding), 'phase': 'spawning' if started else 'prepared',
                'runtime_phase': 'running' if started else 'prepared',
                'cleanup_observed': False, 'events_reconciled': True,
                'operations': [v for (lease, _), v in self.operations.items() if lease == launch.lease_id]}})

    async def advance(self):
        task = self.fixture.queue.supervisor.prepare(self.assembly.advance())
        return await asyncio.shield(task)

    async def test_all_preparations_precede_starts_then_one_instance_and_exact_serving(self):
        self.assertTrue(await self.advance(), self.assembly.saved())
        self.assertEqual(self.effects, [
            ('prepare', 'slice'), ('prepare', 'wafer'), ('prepare', 'obelisk'),
            ('start', 'slice'), ('start', 'wafer'), ('start', 'obelisk'), ('instance',)])
        self.assertEqual(self.assembly.saved()['phase'], 'ready')
        self.assertEqual(self.provider.readiness.saved()['confirmed'], 1)
        self.assertEqual(len(self.context.writes), 1)
        self.assertEqual(self.fixture.state(self.native.call), 'queued')
        self.assertEqual(self.parts.posts, 0)  # No inference in assembly.
        self.assertTrue(await self.advance())

    async def test_rank_wired_policy_wait_keeps_claims_and_original_call_before_any_native_start(self):
        original_group = copy.deepcopy(self.parts.group)
        self.low_memory = {'wafer'}
        for _ in range(2):
            self.assertFalse(await self.advance())
            self.assertEqual(self.assembly.saved()['phase'], 'waiting_capacity')
            self.assertEqual(self.fixture.state(self.native.call), 'queued')
            self.assertEqual(self.parts.group, original_group)
            self.assertEqual(self.native.posts, 0)
            self.assertFalse(any(x[0] == 'start' for x in self.effects))
        self.assertEqual(self.effects, [('prepare', 'slice'), ('prepare', 'wafer'), ('prepare', 'obelisk')])
        self.assertEqual(set(self.capacity_reads), {'slice', 'wafer', 'obelisk'})
        self.provider.assembly = self.assembly
        self.provider.publish()
        self.assertEqual(self.provider.source.last_model[self.provider.resource, self.provider.model]['reason'],
                         'exo_rank_memory')
        # A fresh external capacity observation can resume the same demand.
        self.low_memory.clear()
        self.assertTrue(await self.advance(), self.assembly.saved())
        self.assertEqual(self.native.posts, 1)
        for rank in ('slice', 'wafer', 'obelisk'):
            self.assertEqual(self.effects.count(('prepare', rank)), 1)
            self.assertEqual(self.effects.count(('start', rank)), 1)

    async def test_ram_is_rechecked_after_host_launches_before_instance_post(self):
        self.mode = 'memory-lost-during-start'
        self.assertFalse(await self.advance())
        self.assertEqual(self.assembly.saved()['phase'], 'waiting_capacity')
        self.assertEqual(self.native.posts, 0)
        self.assertEqual(self.fixture.state(self.native.call), 'queued')
        for rank in ('slice', 'wafer', 'obelisk'):
            self.assertEqual(self.effects.count(('start', rank)), 1)
        self.mode = 'ok'
        self.low_memory.clear()
        self.assertTrue(await self.advance(), self.assembly.saved())
        self.assertEqual(self.native.posts, 1)
        for rank in ('slice', 'wafer', 'obelisk'):
            self.assertEqual(self.effects.count(('start', rank)), 1)

    async def test_started_instance_is_not_denied_by_its_own_resident_memory(self):
        self.assertTrue(await self.advance())
        reads = len(self.capacity_reads)
        self.low_memory = {'slice', 'wafer', 'obelisk'}
        self.assertTrue(await self.advance())
        self.assertEqual(len(self.capacity_reads), reads)
        self.assertEqual(self.native.posts, 1)

    async def test_bad_capacity_observation_is_unknown_and_never_starts_or_frees_claims(self):
        original = copy.deepcopy(self.parts.group)
        for fault in ('contact', {'boot_id': 'foreign'}, {'uid': True},
                      {'identity': {}}, {'manager_empty': False},
                      {'available_bytes': True}, {'available_bytes': -1},
                      {'available_bytes': None}, {'available_bytes': 10**30},
                      {'available_bytes': 0, 'total_bytes': 1},
                      {'wired_limit_bytes': None}, {'wired_limit_bytes': True},
                      {'wired_limit_bytes': 0}, {'kernel_wired_bytes': -1},
                      {'device_inuse_bytes': None}):
            self.capacity_fault = fault
            with self.subTest(fault=fault):
                self.assertFalse(await self.advance())
                self.assertIsNotNone(self.assembly.saved()['error_type'])
                self.assertEqual(self.native.posts, 0)
                self.assertFalse(any(x[0] == 'start' for x in self.effects))
                self.assertEqual(self.fixture.state(self.native.call), 'queued')
                self.assertEqual(self.parts.group, original)

    async def test_zero_free_ram_does_not_reject_a_queue_admitted_metal_envelope(self):
        # The fixture advertises zero currently available RAM on every host.
        # The explicit wired envelopes and non-pageable headroom still fit.
        self.assertTrue(await self.advance(), self.assembly.saved())
        self.assertEqual(self.native.posts, 1)

    async def test_foreign_locked_memory_holds_same_claims_until_capacity_returns(self):
        original = copy.deepcopy(self.parts.group)
        self.capacity_fault = 'occupied'
        self.assertFalse(await self.advance())
        self.assertEqual(self.assembly.saved()['phase'], 'waiting_capacity')
        self.assertEqual(self.native.posts, 0)
        self.assertEqual(self.parts.group, original)
        self.capacity_fault = None
        self.assertTrue(await self.advance(), self.assembly.saved())

    async def test_capacity_wait_has_no_hot_retry_and_resumes_on_existing_external_wake(self):
        self.low_memory = {'wafer'}
        async def factory(provider):
            self.assertIs(provider, self.provider)
            return self.assembly
        self.fixture.source.paused = False
        self.provider.bind_assembly_factory(factory)
        async with asyncio.timeout(2):
            await asyncio.shield(self.provider._task)
        reads = len(self.capacity_reads)
        self.assertEqual(self.assembly.saved()['phase'], 'waiting_capacity')
        self.assertIs(self.provider.permitted(self.provider.resource), False)
        await asyncio.sleep(0)
        self.assertEqual(len(self.capacity_reads), reads)
        self.assertEqual(self.native.posts, 0)
        self.low_memory.clear()
        self.provider.wake()  # Existing host/resource event, not a new loop.
        async with asyncio.timeout(2):
            await asyncio.shield(self.provider._task)
        self.assertEqual(self.assembly.saved()['phase'], 'ready')
        self.assertEqual(self.native.posts, 1)
        self.assertEqual(len(self.effects), 7)

    async def test_lost_start_reply_recovers_actual_receipt_without_another_post(self):
        self.mode = 'lost-start'
        self.assertFalse(await self.advance())
        self.assertEqual(self.native.posts, 0)
        self.assertTrue(await self.advance(), self.assembly.saved())
        self.assertEqual(self.effects.count(('start', 'slice')), 1)
        self.assertEqual(self.native.posts, 1)

    async def test_unknown_start_without_actual_receipt_never_reposts_or_creates_instance(self):
        self.mode = 'dropped-start'
        self.assertFalse(await self.advance())
        self.assertFalse(await self.advance())
        self.assertEqual(self.effects.count(('start', 'slice')), 1)
        self.assertNotIn(('start', 'wafer'), self.effects)
        self.assertEqual(self.native.posts, 0)
        self.assertFalse(self.context.writes)

    async def test_lost_instance_reply_reconciles_same_ready_contract_without_repost(self):
        self.mode = 'lost-instance'
        self.assertFalse(await self.advance())
        self.assertEqual(self.native.posts, 1)
        self.assertTrue(await self.advance(), self.assembly.saved())
        self.assertEqual(self.native.posts, 1)
        self.assertEqual(len(self.context.writes), 1)

    async def test_observer_cancel_keeps_existing_supervisor_assembly_owned(self):
        self.release.clear()
        owned = self.fixture.queue.supervisor.prepare(self.assembly.advance())

        async def observe():
            return await asyncio.shield(owned)

        observer = asyncio.create_task(observe())
        await self.entered.wait()
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertIn(owned, self.fixture.queue.supervisor.preparations)
        self.assertFalse(owned.done())
        self.release.set()
        self.assertTrue(await owned, self.assembly.saved())
        self.assertEqual(self.native.posts, 1)

    async def test_provider_unavailable_private_api_waits_for_external_wake_without_hot_retry(self):
        self.mode = 'private-unavailable'
        calls = []

        async def factory(provider):
            self.assertIs(provider, self.provider)
            calls.append('factory')
            return self.assembly

        self.fixture.source.paused = False
        self.provider.bind_assembly_factory(factory)
        # Explicit bounded test observation, not a product deadline.
        async with asyncio.timeout(2):
            await asyncio.shield(self.provider._task)
        self.assertEqual(calls, ['factory'])
        self.assertEqual(self.native.posts, 0)
        self.assertFalse(self.provider._ready)
        self.assertEqual(len(self.effects), 6)
        await asyncio.sleep(0)
        self.assertEqual(len(self.effects), 6)
        self.provider.wake()  # Existing registry/owner event, not elapsed time.
        async with asyncio.timeout(2):
            await asyncio.shield(self.provider._task)
        self.assertEqual(calls, ['factory'])
        self.assertEqual(len(self.effects), 6)

    async def test_unconfirmed_group_or_draining_service_cannot_prepare_hosts(self):
        self.actor.events_lost()
        self.assertFalse(await self.advance())
        self.assertFalse(self.effects)
        self.actor.events_reconciled()
        await self.actor.wait()
        self.service.draining = True
        self.assertFalse(await self.advance())
        self.assertFalse(self.effects)

    async def test_failed_configuration_factory_is_never_automatically_called_again(self):
        calls = []

        async def factory(_provider):
            calls.append('factory')
            raise OSError('private configuration detail')

        self.fixture.source.paused = False
        self.provider.bind_assembly_factory(factory)
        async with asyncio.timeout(2):
            await asyncio.shield(self.provider._task)
        self.provider.wake()
        async with asyncio.timeout(2):
            await asyncio.shield(self.provider._task)
        self.assertEqual(calls, ['factory'])
        self.assertFalse(self.effects)
        self.assertFalse(self.provider._ready)
        row = self.journal.db.execute('SELECT * FROM exo_assembly_factory').fetchone()
        self.assertEqual(row['state'], 'owned')
        self.assertNotIn('private configuration', str(dict(row)))

    async def test_service_shutdown_drains_provider_control_before_refusing_unsettled_group(self):
        async def factory(_provider):
            return self.assembly

        self.release.clear()
        self.fixture.source.paused = False
        self.provider.bind_assembly_factory(factory)
        await self.entered.wait()
        closing = asyncio.create_task(self.service.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertTrue(self.service.draining)
        self.release.set()
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            async with asyncio.timeout(2):
                await closing
        self.assertFalse(self.service.closed)
        self.assertFalse(self.service.tasks['coordinator'].done())
        self.assertNotIn(('start', 'wafer'), self.effects)
        self.assertEqual(self.native.posts, 0)
