"""Authenticated request shapes and event races, without a production request."""

import asyncio
import copy
import json
import sys
import unittest

import httpx

from .exo_groups.alarm import ContinuousAuthorityAlarm
from .exo_groups.clock import authority_clock
from .exo_groups.host_authority import HostAuthorityAdapter, HostPlaneReader, ReadUnavailable
from .exo_groups.rank_process import ObservationUnknown
from . import test_exo_group_rank_lifecycle as fixtures


class FakeAlarm:
    def __init__(self, _loop, notify, failed):
        self.notify, self.failed = notify, failed
        self.deadline = None
        self.fail_next = False
        self.closed = False

    def arm(self, deadline):
        if self.fail_next:
            self.fail_next = False
            raise OSError("test registration failure")
        self.deadline = deadline

    def close(self):
        self.closed = True


class FrameStream(httpx.AsyncByteStream):
    def __init__(self, frames):
        self.frames = frames

    async def __aiter__(self):
        async for value in self.frames():
            yield value.encode()


class HostAuthorityControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.RankLifecycleControls()
        self.fixture.setUp()
        self.actor = self.fixture.actor
        self.projection = copy.deepcopy(self.fixture.projection)
        self.requests = []
        self.before_read = None
        self.status = 200
        self.location = None
        self.frames = None
        self.header = "Bearer synthetic-host-only"
        self.auth_owner = "user:9999"
        self.reader = HostPlaneReader(
            lambda: self.header,
            owner_id="user:9999",
            transport=httpx.MockTransport(self.handle),
        )
        self.adapter = HostAuthorityAdapter(
            self.actor, self.reader, alarm_factory=FakeAlarm
        )

    async def asyncTearDown(self):
        if self.actor.process is not None:
            self.fixture.child.code = 0
            self.actor.process_exited()
        await self.adapter.close_after_cleanup()
        self.fixture.tearDown()

    async def handle(self, request):
        self.requests.append(request)
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.scheme, "https")
        self.assertEqual(request.url.host, "cloud.samtg.xyz")
        self.assertEqual(request.headers["Authorization"], self.header)
        self.assertTrue(
            all(value is None for value in request.extensions["timeout"].values())
        )
        if request.url.path == "/api/v1/auth/verify":
            return httpx.Response(
                200, json={"valid": True, "owner_id": self.auth_owner}
            )
        if request.url.path == "/api/v1/events":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FrameStream(self.frames),
            )
        binding = self.actor.expected.binding
        self.assertEqual(
            request.url.path,
            f"/api/v1/lease-groups/{binding.group_id}/members/{binding.lease_id}/authority",
        )
        value = copy.deepcopy(self.projection)
        if self.before_read:
            await self.before_read()
        if self.location:
            return httpx.Response(302, headers={"Location": self.location})
        return httpx.Response(self.status, json=value)

    async def start(self):
        await self.adapter.before_start(environment={"PATH": "/usr/bin:/bin"})

    def hint(self):
        b = self.actor.expected.binding
        return {
            "type": "lease.group_changed",
            "target": b.lease_id,
            "data": {
                "resource_id": b.resource_id,
                "lease_id": b.lease_id,
                "group_id": b.group_id,
                "authoritative": True,
            },
        }

    async def test_launch_and_each_admission_require_actual_get_and_serving(self):
        await self.start()
        self.assertEqual(self.fixture.spawn_count, 1)
        self.assertFalse(await self.adapter.before_admit())
        self.projection["phase"] = "serving"
        self.assertTrue(await self.adapter.before_admit())
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(self.adapter.alarm.deadline, self.actor.authority_deadline)
        self.assertNotIn("synthetic-host-only", json.dumps(self.fixture.events))

    async def test_http_error_stops_admission_preserves_same_child_and_expiry(self):
        await self.start()
        deadline = self.actor.authority_deadline
        self.status = 503
        self.assertFalse(await self.adapter.before_admit())
        self.assertEqual(self.fixture.child.stop_count, 0)
        self.assertEqual(self.actor.authority_deadline, deadline)
        self.fixture.now = deadline
        self.adapter.alarm.notify()
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertEqual(self.fixture.spawn_count, 1)

    async def test_failed_renewal_alarm_keeps_old_authority_and_old_alarm(self):
        await self.start()
        deadline = self.actor.authority_deadline
        self.projection["authority_expires_at"] = "2026-11-14T10:10:30+00:00"
        self.projection["member"]["lease"]["expires_at"] = "2026-11-14T10:10:30+00:00"
        self.adapter.alarm.fail_next = True
        decision = await self.adapter.refresh()
        self.assertEqual(decision.state, "unknown")
        self.assertEqual(self.actor.authority_deadline, deadline)
        self.assertEqual(self.adapter.alarm.deadline, deadline)
        self.assertEqual(self.fixture.child.stop_count, 0)

    async def test_fence_hint_invalidates_older_inflight_response(self):
        await self.start()
        entered, release = asyncio.Event(), asyncio.Event()

        async def pause():
            entered.set()
            await release.wait()

        self.before_read = pause
        old = asyncio.create_task(self.adapter.refresh("older read"))
        await entered.wait()
        self.projection["generation"] = 2
        newer = asyncio.create_task(self.adapter.on_event(self.hint()))
        await asyncio.sleep(0)  # Allow immediate invalidation, not a job clock.
        self.assertFalse(self.actor.may_admit())
        self.before_read = None
        release.set()
        self.assertIsNone(await old)
        self.assertEqual((await newer).state, "stop_required")
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertEqual(self.actor.expected.binding.generation, 1)

    async def test_disconnect_discards_inflight_success_without_replay(self):
        await self.start()
        entered, release = asyncio.Event(), asyncio.Event()

        async def pause():
            entered.set()
            await release.wait()

        self.before_read = pause
        old = asyncio.create_task(self.adapter.refresh())
        await entered.wait()
        self.adapter.on_disconnect()
        release.set()
        self.assertIsNone(await old)
        self.assertIsNone(self.actor.authority)
        self.assertEqual(self.fixture.child.stop_count, 0)
        self.before_read = None
        self.assertEqual((await self.adapter.on_connect()).state, "authorized")
        self.assertEqual(self.fixture.spawn_count, 1)

    async def test_cancelled_read_retains_alarm_and_same_attempt(self):
        await self.start()
        entered = asyncio.Event()

        async def pause():
            entered.set()
            await asyncio.Event().wait()

        self.before_read = pause
        reading = asyncio.create_task(self.adapter.refresh())
        await entered.wait()
        reading.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reading
        self.assertIsNone(self.actor.authority)
        self.assertEqual(self.fixture.child.stop_count, 0)
        self.assertEqual(self.adapter.alarm.deadline, self.actor.authority_deadline)

    async def test_redirect_never_sends_header_to_another_origin(self):
        self.location = "https://untrusted.invalid/authority"
        self.assertEqual((await self.adapter.refresh()).state, "unknown")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0].url.host, "cloud.samtg.xyz")

    async def test_missing_header_never_sends_request(self):
        self.header = ""
        self.assertEqual((await self.adapter.refresh()).state, "unknown")
        self.assertEqual(self.requests, [])

    async def test_coordinator_header_cannot_masquerade_as_host_principal(self):
        self.auth_owner = self.actor.expected.binding.owner_id
        self.assertEqual((await self.adapter.refresh()).state, "unknown")
        self.assertEqual([r.url.path for r in self.requests], ["/api/v1/auth/verify"])
        self.assertEqual(self.actor.authority.state, "unknown")
        self.assertFalse(self.actor.authority.allows_start(self.fixture.now))

    async def test_coordinator_identity_is_refused_before_any_connection(self):
        reader = HostPlaneReader(
            lambda: "Bearer synthetic-coordinator",
            owner_id=self.actor.expected.binding.owner_id,
            transport=httpx.MockTransport(self.handle),
        )
        try:
            with self.assertRaises(ValueError):
                HostAuthorityAdapter(self.actor, reader, alarm_factory=FakeAlarm)
        finally:
            await reader.close()
        self.assertEqual(self.requests, [])

    async def test_rotated_header_is_verified_again_without_replaying_rank(self):
        await self.start()
        self.header = "Bearer different-host-header"
        self.auth_owner = self.actor.expected.binding.owner_id
        self.assertFalse(await self.adapter.before_admit())
        self.assertEqual(
            sum(r.url.path == "/api/v1/auth/verify" for r in self.requests), 2
        )
        self.assertEqual(self.fixture.spawn_count, 1)
        self.assertEqual(self.fixture.child.stop_count, 0)

    async def test_unrelated_callback_does_not_read_or_authorize(self):
        event = self.hint()
        event["target"] = "another-lease"
        self.assertIsNone(await self.adapter.on_event(event))
        self.assertEqual(self.requests, [])
        self.assertIsNone(self.actor.authority)

    async def test_malformed_event_revokes_admission_without_replaying_child(self):
        await self.start()
        self.assertIsNone(await self.adapter.on_event(None))
        self.assertIsNone(self.actor.authority)
        self.assertEqual(self.fixture.child.stop_count, 0)
        self.assertEqual(self.fixture.spawn_count, 1)

    async def test_alarm_rearm_failure_cannot_abandon_live_rank(self):
        await self.start()
        self.adapter.alarm.fail_next = True
        self.adapter.alarm.notify()  # Simulated early event, before exact expiry.
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertTrue(self.actor.draining)

    async def test_sse_connect_heartbeat_fence_and_eof_require_fresh_reads(self):
        await self.start()
        self.projection["phase"] = "serving"

        async def frames():
            yield ": keepalive\n\n"
            self.projection["generation"] = 2
            yield (
                "id: 23\nevent: lease.group_changed\ndata: "
                + json.dumps(self.hint())
                + "\n\n"
            )

        self.frames = frames
        with self.assertRaises(ReadUnavailable):
            await self.adapter.consume_once()
        self.assertEqual(self.adapter.cursor, "23")
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertIsNone(self.actor.authority)
        self.assertEqual(
            len(self.requests), 6
        )  # launch, stream, connect, heartbeat, fence

    async def test_reconnect_cursor_never_substitutes_for_snapshot(self):
        await self.start()
        self.adapter.cursor = "23"
        self.projection["generation"] = 2

        async def frames():
            yield ": keepalive\n\n"

        self.frames = frames
        with self.assertRaises(ReadUnavailable):
            await self.adapter.consume_once()
        stream = [r for r in self.requests if r.url.path == "/api/v1/events"][0]
        self.assertEqual(stream.headers["Last-Event-ID"], "23")
        self.assertEqual(self.fixture.child.stop_count, 1)

    async def test_malformed_sse_stops_admission_not_same_running_child(self):
        await self.start()

        async def frames():
            yield "id: 4\ndata: not-json\n\n"

        self.frames = frames
        with self.assertRaises(ValueError):
            await self.adapter.consume_once()
        self.assertIsNone(self.actor.authority)
        self.assertEqual(self.fixture.child.stop_count, 0)
        self.assertEqual(self.adapter.alarm.deadline, self.actor.authority_deadline)

    async def test_close_refuses_to_abandon_owned_attempt(self):
        await self.start()
        with self.assertRaises(ObservationUnknown):
            await self.adapter.close_after_cleanup()
        self.assertFalse(self.adapter.alarm.closed)
        self.fixture.child.code = 0
        self.actor.process_exited()
        await self.adapter.close_after_cleanup()
        self.assertTrue(self.adapter.alarm.closed)

    async def test_native_alarm_unavailable_stops_and_retains_cleanup(self):
        await self.start()
        self.adapter.alarm.failed()
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertTrue(self.actor.draining)
        self.assertIsNone(self.actor.receipt)

    @unittest.skipUnless(sys.platform == "darwin", "native continuous expiry event")
    async def test_real_alarm_stops_on_explicit_expiry_without_more_stream_bytes(self):
        self.actor.monotonic = authority_clock
        self.actor.spawner = lambda *args, **kwargs: self.fixture.child
        self.projection["authority_expires_at"] = "2026-11-14T10:05:00.050000+00:00"
        self.adapter.alarm.close()
        notified = asyncio.Event()

        def expired():
            self.adapter._expired()
            notified.set()

        self.adapter.alarm = ContinuousAuthorityAlarm(
            asyncio.get_running_loop(), expired, self.adapter._alarm_failed
        )
        await self.start()
        await asyncio.wait_for(notified.wait(), 3)
        self.assertEqual(self.fixture.child.stop_count, 1)
        self.assertTrue(self.actor.draining)


if __name__ == "__main__":
    unittest.main()
