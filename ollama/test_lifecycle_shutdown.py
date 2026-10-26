"""One-shot SDK shutdown delivers saved lifecycle intent, never infers an end."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from .hermes_queue_identity import HostBindings
from .hermes_queue_source import owner_scope
from .inference_events import relay_events
from .inference_identity import IdentityStore
from .inference_sdk import QueuedClient


class ShutdownLifecycle(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, mode, *, count=1, failure=None, completed=True, late_event=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.sqlite'
            host = HostBindings(path, directory, job_continuity=True)
            idle = threading.Event()
            pending = host.pending_events
            def read(after=0):
                events = pending(after)
                if not events:
                    idle.set()
                return events
            host.pending_events = read
            writes = []
            def handle(request):
                if request.url.path.endswith('/capabilities'):
                    return httpx.Response(200, json={'jobs':{'version':1, 'explicit_finish':True}})
                self.assertEqual(request.url.path, '/v1/inference/jobs/end')
                self.assertEqual(request.method, 'POST')
                writes.append(json.loads(request.content))
                if failure == 'first-page' and writes[-1]['key'] in keys[:32]:
                    return httpx.Response(503)
                if failure == 'network':
                    raise httpx.ReadError('observer unavailable')
                return httpx.Response(202, json=({'status':'wrong'} if failure == 'wrong-receipt' else
                    {**writes[-1], 'jobs':[]}))
            async def slow_poll(consumer, source):
                await relay_events(consumer, source, interval=3600)
            with patch('ollama.inference_events.relay_events', side_effect=slow_poll):
                client = QueuedClient(api_key='synthetic', base_url='https://synthetic.invalid/v1',
                    http_factory=lambda:httpx.AsyncClient(base_url='https://synthetic.invalid',
                        transport=httpx.MockTransport(handle)), event_source=host)
                self.assertTrue(await asyncio.to_thread(idle.wait, 3))
                keys = []
                store = IdentityStore(path)
                try:
                    for index in range(count):
                        scope = owner_scope(directory, 'session-' + str(index))
                        keys.append(store.begin_native_job(scope, 'turn'))
                        store.observe_native_outcome(scope, 'turn', completed=completed,
                            failed=not completed, interrupted=False)
                finally:
                    store.close()
                late = []
                if late_event:
                    highwater = host.pending_event_highwater
                    def append_after_snapshot():
                        through = highwater()
                        store = IdentityStore(path)
                        try:
                            scope = owner_scope(directory, 'later-session')
                            late.append(store.begin_native_job(scope, 'later-turn'))
                            store.finish_turn(scope, 'later-turn')
                        finally:
                            store.close()
                        return through
                    host.pending_event_highwater = append_after_snapshot
                if mode == 'sync':
                    await asyncio.to_thread(client.close)
                else:
                    await client.aclose()
            reopened = HostBindings(path, directory, job_continuity=True)
            events = []
            after = 0
            while page := reopened.pending_events(after):
                events.extend(page); after = page[-1]['cursor']
            if not completed:
                self.assertEqual(writes, [])
                store = IdentityStore(path)
                try:
                    self.assertEqual(store.native_job_status(owner_scope(directory, 'session-0'))['job_key'], keys[0])
                finally:
                    store.close()
            elif failure == 'first-page':
                self.assertEqual({event['job_key'] for event in events}, set(keys[:32]))
                self.assertEqual(len(writes), count)
                self.assertEqual({row['key'] for row in writes}, set(keys))
            elif failure:
                self.assertEqual({event['job_key'] for event in events}, set(keys))
            else:
                self.assertEqual({event['job_key'] for event in events}, set(late))
                self.assertEqual({row['key'] for row in writes}, set(keys))
                self.assertTrue(all(row['action'] == 'finish' for row in writes))

    async def test_sync_and_async_close_deliver_finish_created_during_poll_sleep(self):
        for mode in ('sync', 'async'):
            with self.subTest(mode=mode):
                await self.exercise(mode)

    async def test_shutdown_covers_pending_pages_without_repeating_failed_first_page(self):
        await self.exercise('sync', count=35, failure='first-page')

    async def test_shutdown_does_not_chase_events_created_after_its_snapshot(self):
        await self.exercise('async', late_event=True)

    async def test_shutdown_preserves_failed_observation_on_disk(self):
        for failure in ('network', 'wrong-receipt'):
            with self.subTest(failure=failure):
                await self.exercise('async', failure=failure)

    async def test_closing_client_does_not_finish_recoverable_native_parent(self):
        await self.exercise('async', completed=False)
