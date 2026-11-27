"""Real service/host runtime/feed lifetime; isolated HTTP and physical fixtures.

No fleet, model, native compute or lease operation. Callback assertions are
retained separately because the production transport sanitizes exceptions.
"""

import asyncio
import unittest

import httpx

from . import server
from . import test_exo_group_events as events
from . import test_exo_group_host_install as host_install
from . import test_exo_group_install as installed
from . import test_exo_group_rank_runtime as host_runtime
from . import test_queue_startup as startup
from .exo_groups.service import EXOQueueStartup, EXOService, ServiceFeedUnavailable
from .exo_groups.plane import CoordinatorPlane
from .exo_groups.rank_process import ObservationUnknown


class HostServiceControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = host_install.HostInstallControls()
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.runtime, self.host = self.context.runtime, self.context.owner
        self.runtime.bind_exo_rank(self.host)
        self.source = self.context.parts.context
        self.errors = []
        events.capture_assertions(self.source, self.errors)
        self.entered, self.release = asyncio.Event(), asyncio.Event()

        async def frames():
            self.entered.set()
            await self.release.wait()
            yield ': observed\n\n'

        self.source.frames = frames
        self.owner = EXOService(self.runtime)
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        if self.host.actor.process is not None:
            self.context.parts.fixture.child.code = 0
            self.host.actor.process_exited()
        elif self.host.actor.receipt is None:
            await self.host.attest_never_started()
        if self.owner.started:
            await self.owner.close()
        if self.errors:
            raise AssertionError('masked transport assertion: ' + repr(self.errors))

    async def start(self):
        self.owner.start()
        await self.entered.wait()

    async def test_feed_is_service_owned_and_not_a_permanent_supervisor_preparation(self):
        await self.start()
        self.assertIs(self.runtime.exo_service, self.owner)
        self.assertFalse(self.runtime.queue.supervisor.preparations)
        self.assertTrue(self.owner.snapshot()['subscriptions']['host']['observing'])
        self.assertTrue(self.host._events_ready)
        self.assertEqual(self.context.parts.fixture.spawn_count, 0)
        with self.assertRaises(RuntimeError):
            self.owner.start()
        with self.assertRaises(ValueError):
            EXOService(self.runtime)
        with self.assertRaises(RuntimeError):
            self.owner.reconnect('host')

    async def test_eof_is_reported_durably_and_does_not_reconnect_or_launch(self):
        await self.start()
        self.release.set()
        with self.assertLogs('ollama.exo_groups.service', level='WARNING'):
            with self.assertRaises(ServiceFeedUnavailable):
                await self.owner.wait('host')
        self.assertFalse(self.host._events_ready)
        count = sum(r.url.path == '/api/v1/events' for r in self.source.requests)
        self.assertEqual(count, 1)
        rows = self.runtime.queue.supervisor.journal.db.execute(
            "SELECT reason FROM metadata_events WHERE reason='exo_host_feed_unavailable'").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.context.parts.fixture.spawn_count, 0)
        self.assertFalse(self.runtime.capacity_source.permitted_model(self.context.resource, 'local'))

    async def test_observer_cancellation_does_not_cancel_the_owned_feed(self):
        await self.start()
        observer = asyncio.create_task(self.owner.wait('host'))
        await asyncio.sleep(0)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(self.owner.tasks['host'].done())
        self.assertTrue(self.host._events_ready)

    async def test_cancelled_connection_wait_keeps_the_original_feed_owned(self):
        await self.start()
        self.assertTrue(self.host._connected.is_set())
        await self.host.wait_connected(self.owner.tasks['host'])
        self.host.on_disconnect()
        self.assertFalse(self.host._connected.is_set())
        observer = asyncio.create_task(self.host.wait_connected(self.owner.tasks['host']))
        await asyncio.sleep(0)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(self.owner.tasks['host'].done())

    async def test_connection_wait_refuses_when_same_subscription_has_ended(self):
        await self.start()
        self.release.set()
        with self.assertLogs('ollama.exo_groups.service', level='WARNING'):
            with self.assertRaises(ServiceFeedUnavailable):
                await self.owner.wait('host')
        self.assertFalse(self.host._connected.is_set())
        with self.assertRaises(ObservationUnknown):
            await self.host.wait_connected(self.owner.tasks['host'])

    async def test_explicit_reconnect_uses_same_runtime_attempt_and_feed(self):
        await self.start()
        self.release.set()
        with self.assertRaises(ServiceFeedUnavailable):
            await self.owner.wait('host')
        feed = self.host._event_feed
        child_attempt = self.host._attempt()
        self.entered.clear()
        self.release.clear()
        self.owner.reconnect('host')
        await self.entered.wait()
        self.assertIs(self.host._event_feed, feed)
        self.assertEqual(self.host._attempt(), child_attempt)
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.source.requests), 2)
        self.assertEqual(self.context.parts.fixture.spawn_count, 0)

    async def test_shutdown_refuses_unobserved_cleanup_without_losing_feed_or_child(self):
        await self.start()
        await self.host.start()
        deadline = self.host.actor.authority_deadline
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            await self.owner.close()
        self.assertFalse(self.owner.tasks['host'].done())
        self.assertFalse(self.host.closed)
        self.assertEqual(self.host.actor.authority_deadline, deadline)
        self.assertEqual(self.context.parts.fixture.spawn_count, 1)
        self.assertEqual(self.context.parts.fixture.child.stop_count, 0)

    async def test_cleanup_then_shutdown_drains_feed_before_closing_journal_owner(self):
        await self.start()
        await self.host.attest_never_started()
        with self.assertRaisesRegex(RuntimeError, 'subscriptions'):
            self.runtime.close()
        await self.owner.close()
        self.assertTrue(self.owner.tasks['host'].done())
        self.assertTrue(self.host.closed)
        self.assertTrue(self.owner.closed)
        self.assertTrue((await self.runtime.quiesce())['drained'])
        with self.assertRaises(RuntimeError):
            self.owner.reconnect('host')
        self.assertEqual(self.context.parts.fixture.spawn_count, 0)


class StartupServiceControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = startup.Startup()
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.host_context = host_runtime.RankRuntimeControls()
        self.host_context.register_journal = False
        await self.host_context.asyncSetUp()
        self.host = self.host_context.runtime
        self.errors = []
        events.capture_assertions(self.host_context.context, self.errors)
        self.context.registry.verify_token.return_value['owner_id'] = self.host.adapter.reader.owner_id
        self.entered = asyncio.Event()

        async def frames():
            self.entered.set()
            await asyncio.Event().wait()
            yield ': unused\n\n'

        self.host_context.context.frames = frames
        self.owner = None
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        if self.host._journal is not None and self.host.actor.receipt is None:
            await self.host.attest_never_started()
        if self.owner is not None:
            await self.owner.close()
        await self.host_context.asyncTearDown()
        if self.errors:
            raise AssertionError('masked transport assertion: ' + repr(self.errors))

    async def test_actual_lifespan_composes_before_maintenance_and_closes_after_proof(self):
        async def configure(runtime):
            runtime.bind_exo_rank(self.host)
            self.owner = EXOService(runtime)
            return self.owner

        def maintenance():
            self.assertIs(server.inference_runtime.exo_service, self.owner)
            self.assertTrue(self.owner.started)

        self.context.manager.start_background_tasks.side_effect = maintenance
        server.inference_startup = EXOQueueStartup(self.context.settings, configure=configure)
        await self.context.boot()
        await self.entered.wait()
        self.assertFalse(self.context.manager.discover.called)
        self.assertFalse(self.context.manager.claim_leases.called)
        await self.host.attest_never_started()
        await self.context.context.__aexit__(None, None, None)
        self.context.context = None
        self.assertTrue(self.owner.closed)
        self.context.manager.shutdown.assert_called_once()

    async def test_missing_factory_refuses_before_install_or_manager_maintenance(self):
        server.mgr = self.context.manager
        with self.assertRaises(ValueError):
            await EXOQueueStartup(self.context.settings).start(server)
        self.assertIsNone(server.inference_runtime)
        self.assertFalse(self.context.settings.journal.exists())
        self.context.manager.start_background_tasks.assert_not_called()

    async def test_factory_failure_does_not_fall_back_to_legacy_adoption(self):
        async def configure(runtime):
            raise ValueError('synthetic binding unavailable')

        server.inference_startup = EXOQueueStartup(self.context.settings, configure=configure)
        with self.assertRaises(ValueError):
            await self.context.boot()
        self.context.manager.discover.assert_not_called()
        self.context.manager.claim_leases.assert_not_called()
        self.context.manager.start_background_tasks.assert_not_called()
        self.assertIsNotNone(server.inference_runtime)
        server.inference_runtime.close()

    async def test_actual_shutdown_does_not_release_manager_on_unobserved_host(self):
        async def configure(runtime):
            runtime.bind_exo_rank(self.host)
            self.owner = EXOService(runtime)
            return self.owner

        server.inference_startup = EXOQueueStartup(self.context.settings, configure=configure)
        await self.context.boot()
        await self.entered.wait()
        await self.host.start()
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            await self.context.context.__aexit__(None, None, None)
        self.context.context = None
        self.context.manager.shutdown.assert_not_called()
        self.assertFalse(self.owner.tasks['host'].done())
        self.assertEqual(self.host_context.fixture.child.stop_count, 0)
        # The failed lifespan does not pretend it cleaned up. Test teardown
        # independently observes child exit, then closes this same owner.
        self.host_context.fixture.child.code = 0
        self.host.actor.process_exited()
        await self.owner.close()
        runtime = server.inference_runtime
        self.assertTrue((await runtime.quiesce())['drained'])
        runtime.close()


class CoHostedServiceControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = installed.InstalledControls()
        self.context.registry_owner = 'user:9999'  # Own host-service principal.
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.host_context = host_runtime.RankRuntimeControls()
        self.host_context.register_journal = False
        await self.host_context.asyncSetUp()
        self.addAsyncCleanup(self.host_context.asyncTearDown)
        self.runtime, self.host = self.context.runtime, self.host_context.runtime
        self.runtime.bind_exo_rank(self.host)
        self.errors = []
        events.capture_assertions(self, self.errors)
        events.capture_assertions(self.host_context.context, self.errors)
        self.host_entered, self.coordinator_entered = asyncio.Event(), asyncio.Event()
        self.coordinator_release = asyncio.Event()
        self.requests = []
        self.old_plane = self.context.actor.plane

        async def host_frames():
            self.host_entered.set()
            await asyncio.Event().wait()
            yield ': unreachable\n\n'

        async def coordinator_frames():
            self.coordinator_entered.set()
            await self.coordinator_release.wait()
            yield ': observed\n\n'

        async def handle(request):
            self.requests.append(request)
            self.assertEqual(request.headers['Authorization'], 'Bearer synthetic-coordinator-only')
            if request.url.path == '/api/v1/auth/verify':
                return httpx.Response(200, json={'valid': True, 'owner_id': self.context.actor.plan.owner_id})
            if request.url.path == '/api/v1/events':
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                    stream=events.BytesStream(coordinator_frames))
            if request.method == 'GET' and request.url.path.startswith('/api/v1/lease-groups/'):
                return httpx.Response(200, json=self.context.parts.group)
            raise AssertionError('unexpected metadata operation')

        self.host_context.context.frames = host_frames
        self.plane = CoordinatorPlane(self.context.actor.plan.owner_id,
            lambda: 'Bearer synthetic-coordinator-only', transport=httpx.MockTransport(handle))
        self.context.actor.plane = self.plane
        self.owner = EXOService(self.runtime)
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        if self.host.actor.receipt is None:
            await self.host.attest_never_started()
        await self.owner.close()
        if self.errors:
            raise AssertionError('masked callback assertion: ' + repr(self.errors))

    async def test_two_principals_have_separate_feeds_and_one_lifecycle_owner(self):
        self.owner.start()
        await self.host_entered.wait()
        await self.coordinator_entered.wait()
        self.assertEqual(set(self.owner.tasks), {'host', 'coordinator'})
        self.assertEqual(self.runtime.manager.initial_reservations.owner, 'user:9999')
        self.assertNotEqual(self.host.adapter.reader.owner_id, self.plane.owner_id)
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.requests), 1)
        self.assertEqual(sum(r.url.path == '/api/v1/events'
            for r in self.host_context.context.requests), 1)
        self.assertFalse(self.runtime.queue.supervisor.workers)
        self.assertEqual(self.host_context.fixture.spawn_count, 0)

    async def test_coordinator_loss_does_not_reconnect_or_drop_the_host_owner(self):
        self.owner.start()
        await self.host_entered.wait()
        await self.coordinator_entered.wait()
        self.coordinator_release.set()
        with self.assertRaises(ServiceFeedUnavailable):
            await self.owner.wait('coordinator')
        self.assertFalse(self.owner.tasks['host'].done())
        self.assertTrue(self.host._events_ready)
        self.assertFalse(self.context.actor._events_ready)
        self.assertFalse(self.host.closed)
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.requests), 1)
        self.assertEqual(self.host_context.fixture.spawn_count, 0)

    async def test_unsettled_group_refuses_close_before_stopping_either_feed(self):
        # Prepare real durable group ownership through the existing acquisition
        # fixture, with dispatch paused. No native/private request is run.
        self.context.actor.plane = self.old_plane
        self.context.source.pause()
        await self.context.enqueue()
        self.context.actor.events_reconciled()
        await self.context.actor.wait()
        self.assertEqual(self.context.created, 1)
        self.context.actor.plane = self.plane
        self.owner.start()
        await self.host_entered.wait()
        await self.coordinator_entered.wait()
        await self.context.actor.wait()
        await self.host.attest_never_started()
        self.assertTrue(self.context.installation.owned())
        with self.assertRaisesRegex(RuntimeError, 'settlement'):
            await self.owner.close()
        self.assertFalse(self.owner.tasks['host'].done())
        self.assertFalse(self.owner.tasks['coordinator'].done())
        self.assertFalse(self.context.installation.closed)
        self.assertFalse(self.host.closed)
        self.assertFalse(self.context.queue.supervisor.workers)
        # Explicit isolated terminal receipt for teardown; not fleet evidence.
        self.context.parts.group['phase'] = 'released'
        self.context.actor.group_snapshot = dict(self.context.parts.group)
        self.context.actor.ownership_uncertain = False


if __name__ == '__main__':
    unittest.main()
