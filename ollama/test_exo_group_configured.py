"""Trusted startup/attended activation; real queues and ASGI, isolated registry.

No physical group, native rank or service install. No caller credential is
written to a file, journal, host reader or compiled native environment.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import httpx

from . import test_exo_group_factory as fixture
from . import test_exo_group_host_authority as stream_fixture
from .exo_groups.configured import FleetConfiguration, host_configuration


class ConfiguredFleetControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = fixture.FixedFactoryControls()
        self.context.configuration_only = True
        self.context.addAsyncCleanup = self.addAsyncCleanup
        await self.context.asyncSetUp()
        c = self.context
        self.runtime = c.runtime
        self.connected, self.verify_entered, self.verify_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.verify_release.set()
        self.calls = []

        async def frames():
            self.connected.set()
            yield ': current\n\n'
            await asyncio.Event().wait()

        async def transport(request):
            self.calls.append(request)
            if request.url.path == '/api/v1/events':
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                    stream=stream_fixture.FrameStream(frames))
            if request.url.path == '/api/v1/auth/verify':
                self.verify_entered.set()
                await self.verify_release.wait()
            return await c.core_transport_callback(request)

        self.configuration = FleetConfiguration(c.layouts, c.endpoints, manifest=c.fleet_args['manifest'],
            parent=c.fleet_args['parent'], journals=c.journals, group_scope='projects',
            renew_before_s={member.rank: 10 for member in c.plan.members},
            transports=c.fleet_args['transports'], plane_transport=httpx.MockTransport(transport))
        self.service = await self.configuration(self.runtime)
        self.service.start()
        self.caller = {'valid': True, 'owner_id': c.plan.owner_id}

        @self.runtime.queue.service.app.middleware('http')
        async def own_caller(request, call_next):
            request.state.caller = self.caller
            return await call_next(request)

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.runtime.queue.service.app),
            base_url='https://slice.test', headers={'authorization': 'Bearer attended-operator-only'})
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        self.verify_release.set()
        await self.service.close()
        await self.client.aclose()

    async def post(self, action, request_id='a'*64, **fields):
        return await self.client.post('/v1/exo/session/'+action, json={'request_id': request_id, **fields})

    async def test_startup_has_one_fixed_provider_and_no_coordinator_credential_or_subscription(self):
        await asyncio.sleep(0)
        self.assertFalse(self.calls)
        self.assertNotIn('coordinator', self.service.tasks)
        self.assertIs(self.runtime.exo, self.service.roles['coordinator'])
        response = await self.client.get('/v1/exo/session/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status']['subscription'], 'not_started')
        self.assertFalse(response.json()['status']['credential_bound'])
        self.assertEqual(self.context.creates, 0)
        with self.assertRaises(RuntimeError):
            self.service.credential.header()

    async def test_verified_activation_starts_one_feed_and_keeps_header_out_of_journal_and_native_env(self):
        response = await self.post('activate')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['operation']['state'], 'returned')
        await self.connected.wait()
        original = self.service.tasks['coordinator']
        self.assertEqual(self.service.credential.header(), 'Bearer attended-operator-only')
        self.assertTrue(all(request.headers['authorization']=='Bearer attended-operator-only' for request in self.calls))
        repeated = await self.post('activate')
        self.assertEqual(repeated.status_code, 200)
        self.assertIs(self.service.tasks['coordinator'], original)
        self.assertEqual((await self.post('activate', request_id='b'*64)).status_code, 409)
        self.assertEqual(self.context.creates, 0)
        journal = '\n'.join(self.runtime.queue.supervisor.journal.db.iterdump())
        self.assertNotIn('attended-operator-only', journal)
        self.assertNotEqual(self.runtime.manager.sc.token, 'attended-operator-only')
        for host in self.context.layouts:
            member = next(m for m in self.context.plan.members if m.rank==host.files.rank)
            launch = replace(self.context.context.binding, rank=member.rank, resource_id=member.resource_id,
                             pipeline_rank=member.pipeline_rank, layers=member.layers)
            self.assertNotIn('attended-operator-only', repr(host.compile(launch).environment))

    async def test_wrong_owner_or_configuration_cannot_activate_or_supply_paths(self):
        self.caller = {'valid': True, 'owner_id': 'foreign'}
        self.assertEqual((await self.post('activate')).status_code, 403)
        self.assertFalse(self.calls)
        self.caller = {'valid': True, 'owner_id': self.context.plan.owner_id}
        self.assertEqual((await self.post('activate', path='/other')).status_code, 422)
        self.assertFalse(self.calls)
        self.service.session.fleet.journals['slice'] = '/other.sqlite'
        self.assertEqual((await self.post('activate')).status_code, 503)
        self.service.session.fleet.journals['slice'] = str(self.runtime.settings.journal)

    async def test_observer_disconnect_retains_the_same_owned_activation(self):
        self.verify_release.clear()
        observer = asyncio.create_task(self.post('activate'))
        await self.verify_entered.wait()
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        owned = next(iter(self.service.session.tasks.values()))
        self.assertIn(owned, self.runtime.queue.supervisor.preparations)
        self.assertFalse(owned.done())
        self.verify_release.set()
        await asyncio.shield(owned)
        await self.connected.wait()
        self.assertEqual((await self.post('activate')).status_code, 200)
        self.assertEqual(sum(request.url.path.endswith('/auth/verify') for request in self.calls), 1)

    async def test_shutdown_drains_activation_and_never_starts_a_feed_after_draining(self):
        self.verify_release.clear()
        observer = asyncio.create_task(self.post('activate'))
        await self.verify_entered.wait()
        closing = asyncio.create_task(self.service.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.verify_release.set()
        response = await observer
        await closing
        self.assertEqual(response.json()['operation']['state'], 'unconfirmed')
        self.assertNotIn('coordinator', self.service.tasks)
        self.assertIsNone(self.service.credential._value)

    async def test_credential_is_retained_on_unsettled_shutdown_then_cleared_after_success(self):
        await self.post('activate')
        await self.connected.wait()
        with patch.object(self.runtime, 'close_exo', side_effect=RuntimeError('unsettled')):
            with self.assertRaises(RuntimeError):
                await self.service.close()
        self.assertEqual(self.service.credential.header(), 'Bearer attended-operator-only')
        await self.service.close()
        self.assertIsNone(self.service.credential._value)

    async def test_source_configuration_is_once_only_and_slice_is_not_an_implicit_host_only_role(self):
        with self.assertRaises(ValueError):
            await self.configuration(self.runtime)
        with self.assertRaises(ValueError):
            host_configuration(self.context.layouts[0])

    async def test_release_and_settlement_require_original_group_then_use_existing_receipt_owner(self):
        self.assertEqual((await self.post('begin-release', generation=1)).status_code, 409)
        self.assertEqual((await self.post('begin-release', generation=True)).status_code, 422)
        rows = self.runtime.queue.supervisor.journal.db.execute('SELECT COUNT(*) FROM exo_session_controls').fetchone()[0]
        self.assertEqual(rows, 0)
        receipts = SimpleNamespace(routes={'lease_'+'1'*16: object()},
            begin_release=AsyncMock(return_value={'state': 'host_cleanup_required'}),
            settle_host=AsyncMock(return_value={'state': 'member_settlement_confirmed'}))
        provider = self.service.session.provider
        with patch.object(provider, 'receipts', receipts), patch.object(provider, '_saved',
                return_value={'group_id': 'group_'+'a'*24, 'generation': 1}):
            response = await self.post('begin-release', generation=1)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['operation']['state'], 'returned')
            self.assertEqual((await self.post('begin-release', generation=1)).status_code, 200)
            self.assertEqual(receipts.begin_release.await_count, 1)
            self.assertEqual((await self.post('begin-release', request_id='b'*64, generation=1)).status_code, 200)
            self.assertEqual(receipts.begin_release.await_count, 2)
            self.assertEqual((await self.post('settle', request_id='c'*64, generation=1,
                lease_id='lease_'+'2'*16)).status_code, 409)
            self.assertEqual((await self.post('settle', request_id='c'*64, generation=1,
                lease_id='lease_'+'1'*16)).status_code, 200)
            receipts.settle_host.assert_awaited_once_with('lease_'+'1'*16, generation=1)
        self.assertEqual(self.context.creates, 0)
