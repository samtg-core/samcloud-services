"""Real authority/lifecycle/acquisition callbacks on a single injected SSE feed.

No fleet connection, native model or GPU lease. The host fixture observes fake
physical counters; the coordinator uses the real credential/HTTP adapter and
inference journal, with an isolated metadata transport.
"""

import asyncio
import copy
from dataclasses import replace
import json
import unittest

import httpx

from . import test_exo_group_acquire as acquisitions
from . import test_exo_group_install as installations
from . import test_exo_group_rank_runtime as hosts
from .exo_groups.acquire import GroupAcquisition
from .exo_groups.events import EventStreamUnavailable, RegistryEventFeed
from .exo_groups.plane import CoordinatorPlane
from .exo_groups.rank_process import ObservationUnknown


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, produce):
        self.produce = produce

    async def __aiter__(self):
        async for chunk in self.produce():
            yield chunk


def capture_assertions(case, errors):
    """A production transport masks arbitrary errors; tests must retain theirs."""
    class ObservedAssertion(AssertionError):
        def __init__(self, *args):
            errors.append(args)
            super().__init__(*args)
    case.failureException = ObservedAssertion


class HostFeedControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.callback_assertions = []
        capture_assertions(self, self.callback_assertions)
        self.context = hosts.RankRuntimeControls()
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.owner = self.context.runtime
        self.adapter = self.owner.adapter
        self.source = self.context.context
        capture_assertions(self.source, self.callback_assertions)
        self.feed = RegistryEventFeed.host(self.owner)

    async def asyncTearDown(self):
        if self.callback_assertions:
            raise AssertionError('masked callback assertions: ' + repr(self.callback_assertions))

    async def consume(self, chunks):
        async def frames():
            for chunk in chunks:
                yield chunk.decode('utf-8')
        self.source.frames = frames
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()

    async def test_runtime_gate_is_connected_before_start_then_eof_withdraws(self):
        async def frames():
            self.assertTrue(self.owner._events_ready)
            await self.owner.start()
            self.assertEqual(self.source.fixture.spawn_count, 1)
            yield ': keepalive\n\n'
        self.source.frames = frames
        self.owner._event_feed = self.feed
        with self.assertRaises(EventStreamUnavailable):
            await self.owner.consume_events_once()
        self.assertEqual(self.source.fixture.spawn_count, 1)
        self.assertFalse(self.owner._events_ready)
        self.assertFalse(await self.owner.may_admit())
        self.assertEqual(self.source.fixture.child.stop_count, 0)

    async def test_fence_from_stream_stops_same_child_but_does_not_release(self):
        async def frames():
            await self.owner.start()
            self.source.projection['generation'] += 1
            yield 'id: 17\nevent: lease.group_changed\ndata: ' + json.dumps(self.source.hint()) + '\n\n'
            self.assertEqual(self.source.fixture.child.stop_count, 1)
            self.assertEqual(self.source.fixture.spawn_count, 1)
            self.assertIsNone(self.owner.actor.receipt)
            with self.assertRaises(ObservationUnknown):
                await self.owner.start()
        self.source.frames = frames
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()
        self.assertEqual(self.feed.cursor, '17')
        self.assertFalse(self.adapter._stream_running)

    async def test_unrelated_activity_still_reads_authority(self):
        async def frames():
            before = len(self.source.requests)
            yield 'id: 18\ndata: {"type":"ticket.updated","data":{}}\n\n'
            self.assertGreater(len(self.source.requests), before)
            self.assertTrue(self.owner._events_ready)
        self.source.frames = frames
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()
        self.assertEqual(self.feed.cursor, '18')

    async def test_duplicate_consumers_and_adapter_bypass_refuse(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def frames():
            entered.set()
            await release.wait()
            yield ': end\n\n'
        self.source.frames = frames
        task = asyncio.create_task(self.feed.consume_once())
        try:
            await entered.wait()
            for call in (self.feed.consume_once, self.adapter.consume_once):
                with self.assertRaises((EventStreamUnavailable, ObservationUnknown)):
                    await call()
            with self.assertRaises(ValueError):
                RegistryEventFeed.host(self.owner)
            with self.assertRaises(EventStreamUnavailable):
                self.feed.close()
        finally:
            release.set()
            with self.assertRaises(EventStreamUnavailable):
                await task
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.source.requests), 1)

    async def test_cancellation_keeps_same_child_and_explicit_expiry(self):
        entered = asyncio.Event()
        async def frames():
            await self.owner.start()
            entered.set()
            await asyncio.Event().wait()
            yield ': unreachable\n\n'
        self.source.frames = frames
        task = asyncio.create_task(self.feed.consume_once())
        await entered.wait()
        deadline = self.owner.actor.authority_deadline
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.owner._events_ready)
        self.assertFalse(self.adapter._stream_running)
        self.assertFalse(self.feed.running)
        self.assertEqual(self.owner.actor.authority_deadline, deadline)
        self.assertEqual(self.source.fixture.child.stop_count, 0)
        self.source.fixture.now = deadline
        self.adapter.alarm.notify()
        self.assertEqual(self.source.fixture.child.stop_count, 1)
        self.assertEqual(self.source.fixture.spawn_count, 1)

    async def test_explicit_reconnect_retains_cursor_without_replaying_compute(self):
        await self.consume([b'id: 21\ndata: {"type":"ticket.updated"}\n\n'])
        await self.consume([b': keepalive\n\n'])
        requests = [r for r in self.source.requests if r.url.path == '/api/v1/events']
        self.assertNotIn('Last-Event-ID', requests[0].headers)
        self.assertEqual(requests[1].headers['Last-Event-ID'], '21')
        self.assertEqual(self.source.fixture.spawn_count, 0)

    async def test_runtime_entrypoint_creates_and_reuses_its_own_registered_feed(self):
        self.source.reader._event_feed_owner = None  # Discard only the unstarted test fixture feed.
        async def frames():
            self.assertTrue(self.owner._events_ready)
            self.assertIs(self.owner._event_feed, self.source.reader._event_feed_owner)
            yield 'id: 31\ndata: {"type":"ticket.updated"}\n\n'
        self.source.frames = frames
        with self.assertRaises(EventStreamUnavailable):
            await self.owner.consume_events_once()
        feed = self.owner._event_feed
        with self.assertRaises(EventStreamUnavailable):
            await self.owner.consume_events_once()
        self.assertIs(self.owner._event_feed, feed)
        self.assertEqual(feed.cursor, '31')
        self.assertEqual(self.source.fixture.spawn_count, 0)


class CoordinatorFeedControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.callback_assertions = []
        capture_assertions(self, self.callback_assertions)
        self.context = acquisitions.AcquisitionControls()
        await self.context.asyncSetUp()
        self.addAsyncCleanup(self.context.asyncTearDown)
        self.actor = self.context.actor
        # The old in-process acquisition fixture uses readable lease names;
        # the real HTTP client correctly accepts only production-shaped IDs.
        for number, member in enumerate(self.context.group['members']):
            old = member['lease_id']
            new = 'lease_' + format(number + 1, '016x')
            member['lease_id'] = new
            lease = self.context.leases.pop(old)
            lease['id'] = new
            self.context.leases[new] = lease
        self.old_plane = self.actor.plane
        self.requests = []
        self.produce = None
        self.status = 200
        self.content_type = 'text/event-stream'
        self.header = 'Bearer synthetic-coordinator-only'
        self.auth_owner = self.actor.plan.owner_id
        self.plane = CoordinatorPlane(self.actor.plan.owner_id,
            lambda: self.header, transport=httpx.MockTransport(self.handle))
        self.actor.plane = self.plane
        self.feed = RegistryEventFeed.coordinator(self.plane, (self.actor,))

    async def asyncTearDown(self):
        if self.callback_assertions:
            raise AssertionError('masked callback assertions: ' + repr(self.callback_assertions))

    async def handle(self, request):
        self.requests.append(request)
        self.assertEqual(str(request.url).split('/api/')[0], 'https://cloud.samtg.xyz')
        self.assertEqual(request.headers['Authorization'], self.header)
        self.assertTrue(all(v is None for v in request.extensions['timeout'].values()))
        if request.url.path == '/api/v1/auth/verify':
            return httpx.Response(200, json={'valid': True, 'owner_id': self.auth_owner})
        if request.url.path == '/api/v1/events':
            return httpx.Response(self.status, headers={'content-type': self.content_type},
                                  stream=BytesStream(self.produce))
        if request.method == 'POST' and request.url.path == '/api/v1/lease-groups':
            return httpx.Response(200, json=await self.old_plane.create(json.loads(request.content)))
        if request.method == 'POST' and request.url.path.endswith('/advance'):
            return httpx.Response(200, json=await self.old_plane.advance(self.context.group['id'], 1))
        if request.url.path.startswith('/api/v1/lease-groups/'):
            return httpx.Response(200, json=copy.deepcopy(self.context.group))
        if request.url.path.startswith('/api/v1/leases/'):
            return httpx.Response(200, json=copy.deepcopy(self.context.leases[request.url.path.rsplit('/', 1)[1]]))
        raise AssertionError('unexpected registry effect')

    async def chunks(self, chunks):
        async def produce():
            for chunk in chunks:
                yield chunk
        self.produce = produce
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()

    async def test_connect_drives_real_queued_demand_then_loss_revokes_authority(self):
        self.actor.note_demand(self.context.call['id'])
        await self.actor.wait()
        self.assertFalse(self.context.server_created)
        async def produce():
            await self.actor.wait()
            self.assertTrue(self.context.server_created)
            self.assertTrue(self.actor.may_start_ranks())
            yield b'id: 7\ndata: {"type":"resource.changed","data":{"resource_id":"wafer-services/gpu-metal"}}\n\n'
            await self.actor.wait()
        self.produce = produce
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()
        await self.actor.wait()
        self.assertFalse(self.actor.may_start_ranks())
        self.assertFalse(self.actor._events_ready)
        self.assertEqual(self.feed.cursor, '7')
        self.assertEqual(sum(r.method == 'POST' and r.url.path == '/api/v1/lease-groups' for r in self.requests), 1)
        self.assertEqual(self.context.journal.execution(self.context.call['id'], 'caller')['state'], 'queued')

    async def test_many_models_share_one_exact_principal_feed(self):
        self.plane._event_feed_owner = None  # Replace only this unstarted fixture feed.
        second = GroupAcquisition(self.actor.intents,
            replace(self.actor.plan, request_key=self.actor.plan.request_key + '-second'),
            self.actor.manifest, self.plane, self.actor.authority_factory,
            verify_staging=self.actor.verify_staging)
        self.addAsyncCleanup(second.close)
        self.feed = RegistryEventFeed.coordinator(self.plane, (self.actor, second))
        async def produce():
            self.assertTrue(self.actor._events_ready)
            self.assertTrue(second._events_ready)
            yield b': keepalive\n\n'
        self.produce = produce
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()
        self.assertFalse(self.actor._events_ready)
        self.assertFalse(second._events_ready)
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.requests), 1)

    async def test_wrong_owner_or_client_and_second_feed_refuse(self):
        with self.assertRaises(ValueError):
            RegistryEventFeed.coordinator(self.plane, (self.actor,))
        other = CoordinatorPlane('user:other', lambda: 'Bearer other')
        self.addAsyncCleanup(other.close)
        with self.assertRaises(ValueError):
            RegistryEventFeed.coordinator(other, (self.actor,))
        with self.assertRaises(ValueError):
            RegistryEventFeed.coordinator(self.plane, (self.actor, self.actor))

    async def test_redirect_status_and_content_type_errors_are_not_retried(self):
        for status, content in [(302, 'text/event-stream'), (403, 'application/json'), (200, 'text/plain')]:
            self.status, self.content_type = status, content
            before = len(self.requests)
            await self.chunks([])
            self.assertFalse(self.actor._events_ready)
            self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.requests[before:]), 1)
        self.assertFalse(self.context.server_created)

    async def test_bad_frames_never_advance_cursor_or_expose_details(self):
        for raw in [
            b'id: 1\ndata: {"type":"x","type":"y"}\n\n',
            b'id: 1\ndata: {"type":"x","value":NaN}\n\n',
            b'id: 1\ndata: {"type":"x","value":1e999}\n\n',
            b'id: 1\ndata: []\n\n',
            b'id: 1\ndata: {}\n\n',
            b'id: 1\ndata: {"type":null}\n\n',
            b'id: 1\nevent: wrong\ndata: {"type":"x"}\n\n',
            b'id: -1\ndata: {"type":"x"}\n\n',
            b'id: 1\nid: 2\ndata: {"type":"x"}\n\n',
            b'event: x\nevent: x\ndata: {"type":"x"}\n\n',
            b'data: \xff\n\n',
            b'data: {"type":"x"}',
        ]:
            with self.subTest(raw=raw):
                await self.chunks([raw])
                self.assertIsNone(self.feed.cursor)
                self.assertFalse(self.actor._events_ready)

    async def test_split_lines_multiline_json_crlf_and_activity_frames(self):
        await self.chunks([b'id: 1\r', b'\nevent: x\r\ndata: {"type":\r\n', b'data: "x"}\r\n\r\n'])
        self.assertEqual(self.feed.cursor, '1')

    async def test_frame_byte_bound_includes_utf8_and_unfinished_lines(self):
        await self.chunks([b'data: ' + b'x' * (128 * 1024)])
        await self.chunks([('data: ' + '\u00e9' * (70 * 1024) + '\n\n').encode()])
        await self.chunks([b'data: ' + b'x' * 65536, b'x' * 65536])
        self.assertIsNone(self.feed.cursor)

    async def test_one_large_chunk_of_bounded_frames_is_not_one_large_frame(self):
        chunk = b''.join(('id: '+str(i)+'\ndata: {"type":"ticket.updated","padding":"'+'x'*100+'"}\n\n').encode()
                         for i in range(1000))
        self.assertGreater(len(chunk), 128 * 1024)
        await self.chunks([chunk])
        self.assertEqual(self.feed.cursor, '999')

    async def test_explicit_cursor_reconnect_and_close_are_owned_not_automatic(self):
        await self.chunks([b'id: 25\ndata: {"type":"x"}\n\n'])
        await self.chunks([b': keepalive\n\n'])
        requests = [r for r in self.requests if r.url.path == '/api/v1/events']
        self.assertNotIn('Last-Event-ID', requests[0].headers)
        self.assertEqual(requests[1].headers['Last-Event-ID'], '25')
        self.feed.close()
        with self.assertRaises(EventStreamUnavailable):
            await self.feed.consume_once()
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.requests), 2)

    async def test_cursor_rewind_refuses_and_failed_callback_keeps_previous_ack(self):
        await self.chunks([b'id: 10\ndata: {"type":"x"}\n\nid: 9\ndata: {"type":"x"}\n\n'])
        self.assertEqual(self.feed.cursor, '10')
        original = self.actor.on_event
        def failed(_event):
            raise ValueError('synthetic private callback detail')
        self.actor.on_event = failed
        try:
            async def produce():
                yield b'id: 11\ndata: {"type":"x"}\n\n'
            self.produce = produce
            with self.assertRaises(EventStreamUnavailable) as error:
                await self.feed.consume_once()
            self.assertNotIn('private callback detail', str(error.exception))
            self.assertEqual(self.feed.cursor, '10')
            self.assertFalse(self.actor._events_ready)
        finally:
            self.actor.on_event = original

    async def test_registry_restart_allows_new_cursor_epoch_after_reconciliation(self):
        await self.chunks([b'id: 25\ndata: {"type":"x"}\n\n'])
        await self.chunks([b'id: 2\ndata: {"type":"x"}\n\n'])
        self.assertEqual(self.feed.cursor, '2')
        requests = [r for r in self.requests if r.url.path == '/api/v1/events']
        self.assertEqual(requests[1].headers['Last-Event-ID'], '25')
        self.assertFalse(self.actor._events_ready)

    async def test_all_participants_invalidate_even_when_one_callback_fails(self):
        self.plane._event_feed_owner = None
        second = GroupAcquisition(self.actor.intents,
            replace(self.actor.plan, request_key=self.actor.plan.request_key + '-second'),
            self.actor.manifest, self.plane, self.actor.authority_factory,
            verify_staging=self.actor.verify_staging)
        self.addAsyncCleanup(second.close)
        self.feed = RegistryEventFeed.coordinator(self.plane, (self.actor, second))
        second.events_reconciled()
        original = self.actor.events_lost
        def failed():
            raise OSError('synthetic audit failure')
        self.actor.events_lost = failed
        try:
            with self.assertRaises(EventStreamUnavailable):
                await self.feed.consume_once()
            self.assertFalse(second._events_ready)
            self.assertFalse(self.feed.running)
            self.assertEqual(self.requests, [])
        finally:
            self.actor.events_lost = original

    async def test_gateway_installation_entrypoint_drives_its_real_actor(self):
        self.plane._event_feed_owner = None  # Discard only this unstarted test feed.
        context = installations.InstalledControls()
        await context.asyncSetUp()
        self.addAsyncCleanup(context.asyncTearDown)
        context.source.pause()
        context.actor.plane = self.plane
        async def produce():
            self.assertTrue(context.actor._events_ready)
            self.assertFalse(self.actor._events_ready)
            yield b': keepalive\n\n'
        self.produce = produce
        with self.assertRaises(EventStreamUnavailable):
            await context.installation.consume_events_once()
        self.assertFalse(context.actor._events_ready)
        self.assertEqual(context.created, 0)
        self.assertEqual(context.parts.posts, 0)
        context.manager.sc.request_lease.assert_not_called()
        self.assertEqual(sum(r.url.path == '/api/v1/events' for r in self.requests), 1)

    async def test_wrong_verified_principal_never_opens_an_event_stream(self):
        self.auth_owner = 'user:other'
        await self.chunks([])
        self.assertEqual([(r.method, r.url.path) for r in self.requests], [('GET', '/api/v1/auth/verify')])
        self.assertFalse(self.actor._events_ready)

    async def test_credential_rotation_reverifies_same_owner_and_keeps_cursor(self):
        await self.chunks([b'id: 23\ndata: {"type":"x"}\n\n'])
        self.header = 'Bearer synthetic-rotated-coordinator'
        await self.chunks([b': keepalive\n\n'])
        self.assertEqual(sum(r.url.path == '/api/v1/auth/verify' for r in self.requests), 2)
        requests = [r for r in self.requests if r.url.path == '/api/v1/events']
        self.assertEqual(requests[1].headers['Last-Event-ID'], '23')
        self.assertEqual(requests[1].headers['Authorization'], self.header)


if __name__ == '__main__':
    unittest.main()
