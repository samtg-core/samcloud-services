"""Host service ownership, real kernel child-exit event and fake plane metadata."""

import asyncio
import copy
from dataclasses import replace
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from uuid import UUID

import httpx

from . import test_exo_group_host_authority as fixtures
from . import test_exo_group_member as member_fixtures
from .exo_groups.host_authority import HostAuthorityAdapter, HostPlaneReader
from .exo_groups.rank_lifecycle import RankLifecycle, RankSpec, darwin_boot_id
from .exo_groups.rank_process import Listener, ObservationUnknown, RunBinding
from .exo_groups.rank_runtime import HostRankRuntime, RuntimeObservation, launch_sha256


class FakeWatch:
    def __init__(self, child, notify, failed):
        self.child, self.notify, self.failed = child, notify, failed
        self.closed = False

    def close(self):
        self.closed = True


class RankRuntimeControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = fixtures.HostAuthorityControls()
        await self.context.asyncSetUp()
        self.fixture, self.adapter = self.context.fixture, self.context.adapter
        self.actor = self.fixture.actor
        self.fixture.boot = str(UUID(int=123))
        self.environment = {'PATH': '/usr/bin:/bin'}
        self.verifications, self.watches = [], []
        self.observation = RuntimeObservation('b' * 64,
            launch_sha256(self.actor, self.environment), self.fixture.boot, os.getuid())
        async def verify():
            self.verifications.append(1)
            return self.observation
        def watch(*args):
            result = FakeWatch(*args)
            self.watches.append(result)
            return result
        self.runtime = HostRankRuntime(self.actor, self.adapter, verify_runtime=verify,
            environment=self.environment, boot_id=self.fixture.boot, watcher_factory=watch)
        await self.runtime.on_connect()  # Existing feed established, then fresh snapshot.

    async def asyncTearDown(self):
        if self.actor.process is not None:
            self.fixture.child.code = 0
            self.actor.process_exited()
        await self.runtime.close_after_cleanup()
        self.fixture.tearDown()

    async def test_simultaneous_starters_own_one_launch_and_one_exit_event(self):
        results = await asyncio.gather(self.runtime.start(), self.runtime.start(), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, ObservationUnknown) for result in results), 1)
        self.assertEqual(self.verifications, [1])
        self.assertEqual(self.fixture.spawn_count, 1)
        self.assertEqual(len(self.watches), 1)
        self.assertFalse(await self.runtime.may_admit())
        self.context.projection['phase'] = 'serving'
        self.assertTrue(await self.runtime.may_admit())

    async def test_unconfirmed_runtime_uid_boot_or_launch_spec_never_starts(self):
        original = self.observation
        for changed in (replace(original, runtime_sha256='f' * 64),
                        replace(original, boot_id=str(UUID(int=124))),
                        replace(original, uid=os.getuid() + 1),
                        replace(original, launch_sha256='a' * 64)):
            self.observation = changed
            with self.subTest(changed=changed), self.assertRaises(ObservationUnknown):
                await self.runtime.start()
        self.assertEqual(self.fixture.spawn_count, 0)
        self.assertFalse((self.actor.state / 'launch-intent.json').exists())

    async def test_configuration_change_during_fresh_authority_read_never_launches(self):
        async def change():
            self.actor.spec = replace(self.actor.spec, argv=(sys.executable, '-c', 'raise SystemExit(7)'))
        self.context.before_read = change
        with self.assertRaises(ObservationUnknown):
            await self.runtime.start()
        self.assertEqual(self.fixture.spawn_count, 0)

    async def test_failed_plane_read_does_not_turn_runtime_observation_into_authority(self):
        self.context.status = 503
        with self.assertRaises(ObservationUnknown):
            await self.runtime.start()
        self.assertEqual(self.fixture.spawn_count, 0)

    async def test_fence_stop_and_child_event_require_physical_cleanup_before_close(self):
        await self.runtime.start()
        self.context.projection['generation'] += 1
        await self.runtime.on_event(self.context.hint())
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertIsNone(self.actor.receipt)
        with self.assertRaises(ObservationUnknown):
            await self.runtime.close_after_cleanup()
        self.fixture.memory = 1000
        self.fixture.child.code = 0
        self.watches[0].notify()
        self.assertIsNone(self.actor.receipt)
        self.fixture.memory = 100
        receipt = self.runtime.inventory_changed()
        self.assertEqual(receipt['lease_id'], self.actor.spec.binding.lease_id)
        self.assertFalse(receipt['lease_released'])
        await self.runtime.close_after_cleanup()
        self.assertTrue(self.watches[0].closed)

    async def test_observer_failure_stops_same_attempt_and_never_relaunches(self):
        await self.runtime.start()
        self.watches[0].failed()
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertEqual(self.runtime.observation_error, 'child_exit_unconfirmed')
        self.assertIsNone(self.actor.receipt)
        with self.assertRaises(ObservationUnknown):
            await self.runtime.start()
        self.assertEqual(self.fixture.spawn_count, 1)

    async def test_disconnect_revokes_admission_without_replacement_or_immediate_termination(self):
        await self.runtime.start()
        self.context.projection['phase'] = 'serving'
        self.assertTrue(await self.runtime.may_admit())
        self.runtime.on_disconnect()
        self.assertFalse(self.actor.may_admit())
        self.assertFalse(await self.runtime.may_admit())
        self.assertEqual(self.fixture.child.stop_count, 0)
        self.fixture.now = self.actor.authority_deadline
        self.adapter.alarm.notify()
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertEqual(self.fixture.spawn_count, 1)

    async def test_disconnected_feed_cannot_launch_from_a_successful_snapshot_alone(self):
        self.runtime.on_disconnect()
        await self.adapter.refresh()  # Snapshot is valid; the event cursor is not.
        with self.assertRaises(ObservationUnknown):
            await self.runtime.start()
        self.assertEqual(self.verifications, [])
        self.assertEqual(self.fixture.spawn_count, 0)

    async def test_disconnect_during_launch_preparation_does_not_start(self):
        original = self.runtime.verify_runtime
        async def verify():
            result = await original()
            self.runtime.on_disconnect()
            return result
        self.runtime.verify_runtime = verify
        with self.assertRaises(ObservationUnknown):
            await self.runtime.start()
        self.assertEqual(self.fixture.spawn_count, 0)

    def test_no_plane_bearer_in_rank_environment_and_no_mutable_argv_spec(self):
        with self.assertRaises(ValueError):
            HostRankRuntime(self.actor, self.adapter, verify_runtime=self.runtime.verify_runtime,
                environment={'SAMCLOUD_AUTH_HDR': '/never/forward'}, boot_id=self.fixture.boot)
        with self.assertRaises(ValueError):
            replace(self.actor.spec, argv=[sys.executable, '-c', 'pass'])


@unittest.skipUnless(sys.platform == 'darwin', 'Darwin native child-exit event')
class ActualRankRuntimeControls(unittest.IsolatedAsyncioTestCase):
    async def test_child_already_reaped_before_watch_is_reported_once(self):
        from . import test_exo_group_rank_process as process_fixtures
        from .exo_groups.rank_process import OwnedRankProcess
        from .exo_groups.rank_runtime import ChildExitWatch
        with tempfile.TemporaryDirectory() as directory:
            child = OwnedRankProcess.spawn((sys.executable, '-I', '-S', '-c', 'pass'),
                binding=process_fixtures.binding(), authority=process_fixtures.authorized(),
                cwd=Path(directory), environment={'PATH': os.defpath},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child.process.wait(timeout=5)  # Force the exit-before-registration race.
            done, errors, calls = asyncio.Event(), [], []
            def notify():
                calls.append(1)
                done.set()
            watcher = ChildExitWatch(child, notify, lambda: errors.append(1))
            try:
                await asyncio.wait_for(done.wait(), 5)
                self.assertEqual(calls, [1])
                self.assertEqual(errors, [])
                self.assertEqual(child.exit_code(), 0)
                self.assertFalse(child.group_members())
            finally:
                watcher.close()

    async def test_native_child_exit_delivers_cleanup_on_existing_loop_without_polling(self):
        fixture = member_fixtures.MemberAuthorityControls()
        fixture.setUp()
        projection = copy.deepcopy(fixture.projection)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            code = (f'import socket,pathlib,signal;s=socket.socket();s.bind(("127.0.0.1",{port}));'
                    's.listen();pathlib.Path("ready").write_text("ready");signal.pause()')
            spec = RankSpec(RunBinding.from_expected(fixture.expected, 'b' * 64),
                (sys.executable, '-I', '-S', '-B', '-c', code), root,
                (Listener('127.0.0.1', port),), 0)
            cleaned = asyncio.Event()
            def emit(event):
                if 'evidence_sha256' in event:
                    cleaned.set()
            actor = RankLifecycle(fixture.expected, spec, root / 'run',
                                  memory_reader=lambda: 100, emit=emit)
            def handler(request):
                if request.url.path == '/api/v1/auth/verify':
                    return httpx.Response(200, json={'valid': True, 'owner_id': 'user:9999'})
                return httpx.Response(200, json=projection)
            reader = HostPlaneReader(lambda: 'Bearer synthetic-host', owner_id='user:9999',
                                     transport=httpx.MockTransport(handler))
            adapter = HostAuthorityAdapter(actor, reader)
            environment = {'PATH': os.defpath}
            boot = darwin_boot_id()
            async def verify():
                return RuntimeObservation('b' * 64, launch_sha256(actor, environment), boot, os.getuid())
            runtime = HostRankRuntime(actor, adapter, verify_runtime=verify,
                                      environment=environment, boot_id=boot)
            try:
                await runtime.on_connect()
                await runtime.start()
                async def ready():
                    while not (root / 'ready').exists():
                        await asyncio.sleep(.01)
                await asyncio.wait_for(ready(), 5)  # CPU test harness bound only.
                runtime.stop('CPU fixture finished')
                # No refresh_cleanup poll: the actual kernel exit event drives this.
                await asyncio.wait_for(cleaned.wait(), 5)
                self.assertEqual(actor.phase, 'awaiting_settlement')
                self.assertFalse(actor.receipt['lease_released'])
                self.assertFalse(actor.process.group_members())
                self.assertIsNotNone(actor.process.exit_code())
                await runtime.close_after_cleanup()
            finally:
                if actor.process is not None and actor.process.exit_code() is None:
                    actor.process.stop()
                    actor.process.process.wait(timeout=5)
                    actor.process_exited()
                if actor.phase == 'awaiting_settlement' or not actor.launch_attempted:
                    await runtime.close_after_cleanup()
