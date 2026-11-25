"""Existing queue coordinator owns host POSTs; fake HTTPS, no native/lease I/O."""

import asyncio
from dataclasses import replace
import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from . import test_exo_group_install as installed
from .exo_groups.host_control import run_identity
from .exo_groups.host_http import HostCommands, HostControlClient, HostControlUnavailable, HostEndpoint
from .exo_groups.prepare import digest
from .exo_groups.service import EXOService
from .inference_arbiter import Conflict


class HostCommandControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = installed.InstalledControls()
        await self.fixture.asyncSetUp()
        self.addAsyncCleanup(self.fixture.asyncTearDown)
        f = self.fixture
        self.actor = f.actor
        self.call = await f.enqueue()
        await f.bind_ready()
        # This existing fixture completes its synthetic inference call; the
        # acquired group session remains owned for subsequent host controls.
        self.assertEqual(f.state(self.call), 'complete')
        self.service = EXOService(f.runtime)
        entered = asyncio.Event()

        async def isolated_feed():
            entered.set()
            await asyncio.Event().wait()

        f.installation.consume_events_once = isolated_feed
        self.service.start()
        await entered.wait()
        self.endpoint = HostEndpoint('slice', f.resource, 1, 'https://host.test', 'a' * 64, 'b' * 64)
        self.requests = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.mode = 'ok'

        async def transport(request):
            self.requests.append(request)
            value = json.loads(request.content)
            self.entered.set()
            await self.release.wait()
            if self.mode == 'lost':
                raise httpx.ReadError('private host response and credential')
            report = {'identity': value.get('identity', run_identity(self.commands.binding)), 'operations': [{
                'action': request.url.path.split('/')[-1], 'request_id': value['request_id'], 'state': 'returned'}]}
            if self.mode == 'wrong-attempt':
                report['identity'] = report['identity'] | {'lease_id': 'foreign'}
            if self.mode == 'wrong-operation':
                report['operations'][0]['request_id'] = '0' * 64
            return httpx.Response(200, headers={'cache-control': 'no-store'}, json={
                'request_id': value['request_id'], 'source_sha256': self.endpoint.source_sha256,
                'configuration_sha256': self.endpoint.configuration_sha256, 'report': report})

        self.client = HostControlClient(self.endpoint, lambda: 'Bearer synthetic-coordinator',
            transport=httpx.MockTransport(transport))
        self.commands = HostCommands(self.actor, self.client, f.parts.launches[0],
            f.parts.contract.runtime_sha256, service=self.service)
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        self.release.set()
        await self.commands.drain()
        await self.client.close()
        # The test fixture does not attest real physical settlement. Cancel
        # only its synthetic feed; production close refuses the unended group.
        for task in self.service.tasks.values():
            task.cancel()
        await asyncio.gather(*self.service.tasks.values(), return_exceptions=True)

    async def verify(self):
        return self.commands.binding.runtime_sha256

    def row(self):
        return self.actor.intents.journal.db.execute('SELECT * FROM exo_host_submissions').fetchone()

    async def test_existing_supervisor_owns_post_across_observer_cancel(self):
        self.release.clear()
        observer = asyncio.create_task(self.commands.submit('start', verify_runtime=self.verify))
        await self.entered.wait()
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertEqual(self.row()['state'], 'owned')
        self.assertIn(self.commands.tasks['start'], self.fixture.queue.supervisor.preparations)
        self.assertFalse(self.commands.tasks['start'].done())
        self.release.set()
        result = await self.commands.wait('start')
        self.assertEqual(result['identity'], run_identity(self.commands.binding))
        self.assertEqual(self.row()['state'], 'returned')
        with self.assertRaises(Conflict):
            await self.commands.submit('start', verify_runtime=self.verify)
        self.assertEqual(len(self.requests), 1)

    async def test_prepare_carries_only_full_saved_binding_and_fixed_plan_before_start(self):
        result = await self.commands.submit('prepare', verify_runtime=self.verify)
        payload = json.loads(self.requests[0].content)
        self.assertEqual(set(payload), {'request_id', 'launch', 'plan_sha256'})
        self.assertEqual(payload['plan_sha256'], digest(self.fixture.actor.intents.snapshot(
            self.actor.plan.owner_id, self.actor.plan.request_key)['plan']))
        self.assertEqual(payload['launch']['lease_id'], self.commands.binding.lease_id)
        self.assertEqual(payload['launch']['rank_manifest_sha256'], self.actor.plan.rank_manifest_sha256)
        self.assertEqual(result['operations'][0]['action'], 'prepare')
        with self.assertRaises(Conflict):
            await self.commands.submit('prepare', verify_runtime=self.verify)
        await self.commands.submit('start', verify_runtime=self.verify)
        self.assertEqual([r.url.path.rsplit('/', 1)[-1] for r in self.requests], ['prepare', 'start'])

    async def test_lost_reply_retains_once_only_row_even_with_a_new_command_owner(self):
        self.mode = 'lost'
        with self.assertRaises(HostControlUnavailable) as caught:
            await self.commands.submit('start', verify_runtime=self.verify)
        self.assertNotIn('private', str(caught.exception))
        self.assertEqual(self.row()['state'], 'owned')
        next_owner = HostCommands(self.actor, self.client, self.fixture.parts.launches[0],
            self.fixture.parts.contract.runtime_sha256, service=self.service)
        with self.assertRaises(Conflict):
            await next_owner.submit('start', verify_runtime=self.verify)
        self.assertEqual(len(self.requests), 1)

    async def test_changed_runtime_or_events_during_verification_never_posts(self):
        async def wrong():
            return '0' * 64

        with self.assertRaises(Conflict):
            await self.commands.submit('start', verify_runtime=wrong)

        async def lost():
            self.actor.events_lost()
            return await self.verify()

        with self.assertRaises(Conflict):
            await self.commands.submit('start', verify_runtime=lost)
        self.assertFalse(self.requests)
        self.assertIsNone(self.row())

    async def test_same_plan_but_unbound_lease_cannot_stop_another_attempt(self):
        launch = replace(self.fixture.parts.launches[0], lease_id='foreign')
        commands = HostCommands(self.actor, self.client, launch,
            self.fixture.parts.contract.runtime_sha256, service=self.service)
        with self.assertRaises(Conflict):
            await commands.submit('stop', verify_runtime=self.verify)
        self.assertFalse(self.requests)
        self.assertIsNone(self.row())

    async def test_drain_during_runtime_read_prevents_late_submission(self):
        async def draining():
            self.service.draining = True
            return await self.verify()

        with self.assertRaises(Conflict):
            await self.commands.submit('start', verify_runtime=draining)
        self.assertFalse(self.requests)
        self.assertIsNone(self.row())

    async def test_acknowledgement_identity_is_checked_before_recording_returned(self):
        for mode, action in (('wrong-attempt', 'start'), ('wrong-operation', 'stop')):
            self.mode = mode
            with self.assertRaises(HostControlUnavailable):
                await self.commands.submit(action, verify_runtime=self.verify)
        self.assertEqual(len(self.requests), 2)
        states = self.actor.intents.journal.db.execute('SELECT state FROM exo_host_submissions').fetchall()
        self.assertEqual([r['state'] for r in states], ['owned', 'owned'])

    async def test_service_close_drains_owned_control_before_ownership_refusal(self):
        self.release.clear()
        observer = asyncio.create_task(self.commands.submit('start', verify_runtime=self.verify))
        await self.entered.wait()
        closing = asyncio.create_task(self.service.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertFalse(self.commands.accepting)
        self.release.set()
        await observer
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            await closing
        self.assertFalse(self.service.closed)
        with self.assertRaises(Conflict):
            await self.commands.submit('start', verify_runtime=self.verify)
        self.assertFalse(self.service.tasks['coordinator'].done())

    async def test_commands_cannot_use_a_foreign_service_or_unowned_task_factory(self):
        with patch.object(self.actor, 'task_factory', asyncio.create_task):
            with self.assertRaises(ValueError):
                HostCommands(self.actor, self.client, self.fixture.parts.launches[0],
                    self.fixture.parts.contract.runtime_sha256, service=self.service)
        with patch.object(self.service, 'closed', True):
            with self.assertRaises(ValueError):
                HostCommands(self.actor, self.client, self.fixture.parts.launches[0],
                    self.fixture.parts.contract.runtime_sha256, service=self.service)
        self.assertFalse(self.requests)
