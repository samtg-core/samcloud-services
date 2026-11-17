"""Fixed host HTTPS control with real sealed files/journal and no native spawn.

ASGI transports and explicit event/physical fixtures are local test inputs,
not network confinement or distributed model qualification.
"""

import asyncio
import copy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import httpx

from . import test_exo_group_deployment as deployed
from . import test_exo_group_host_authority as authority_fixtures
from .exo_groups import deployment, host_control, receipt_files
from .exo_groups.host_authority import HostPlaneReader
from .exo_groups.host_control import configuration_sha256, run_identity
from .exo_groups.host_http import HostControlClient, HostEndpoint, HTTPReceiptReader
from .exo_groups.prepare import digest
from .exo_groups.service import EXOService


class HostControlControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = deployed.DeploymentControls()
        self.fixture.addAsyncCleanup = self.addAsyncCleanup
        await self.fixture.asyncSetUp()
        self.runtime, self.config = self.fixture.runtime, self.fixture.config
        self.caller = {'valid': True, 'owner_id': self.config.plan.owner_id}
        app = self.runtime.queue.service.app

        @app.middleware('http')
        async def authenticated_fixture(request, call_next):
            request.state.caller = self.caller
            return await call_next(request)

        self.composition = EXOService(self.runtime, deployment=self.config).start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://host.test')
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        self.release.set()
        self.fixture.owner = getattr(self.runtime, 'host_rank', None)
        if self.fixture.owner is not None and self.fixture.owner.actor.receipt is None:
            await self.fixture.owner.attest_never_started()
        await self.composition.close()
        await self.client.aclose()

    async def bind(self):
        owner = await self.fixture.bind()

        async def isolated_feed():
            owner._events_ready = True
            owner._connected.set()
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                owner._events_ready = False
                owner._connected.clear()

        owner.consume_events_once = isolated_feed
        self.composition.attach_host(owner)
        await self.entered.wait()
        return owner

    def payload(self, **values):
        return {'request_id': 'a' * 64, **values}

    async def post(self, action, **values):
        return await self.client.post('/v1/exo/host/' + action, json=self.payload(**values))

    def isolated_member_plane(self, *, changes=None):
        """Real own-credential reader/feed against isolated admitted metadata."""
        self.fixture.binding = replace(self.fixture.binding, expires_at='2026-11-16T00:01:00+00:00')
        binding = self.fixture.binding
        member = self.config.plan.members[0]
        lease = {'id': binding.lease_id, 'agent_id': binding.owner_id,
            'resource_id': binding.resource_id, 'status': 'active', 'service_id': None,
            'exclusive': True, 'priority': 'normal', 'memory_mb': member.memory_mb,
            'ttl_s': member.ttl_seconds, 'max_total_s': member.max_total_s,
            'granted_at': '2026-11-16T00:00:00+00:00', 'expires_at': '2026-11-16T00:01:00+00:00',
            'released_at': None, 'settled_at': None}
        projection = {'group_id': binding.group_id, 'owner_id': binding.owner_id,
            'generation': binding.generation, 'phase': 'admitted', 'release_requested': False,
            'authoritative': True, 'observed_at': '2026-11-16T00:00:00+00:00',
            'authority_expires_at': lease['expires_at'], 'member': {'resource_id': binding.resource_id,
                'lease_id': binding.lease_id, 'cleanup_attested': False, 'settlement_recorded': False,
                'lease': lease}, **(changes or {})}
        self.plane_requests, self.readers = [], []
        self.admit_entered, self.admit_release = asyncio.Event(), asyncio.Event()
        self.admit_release.set()

        async def frames():
            yield ': current\n\n'
            await asyncio.Event().wait()

        async def transport(request):
            self.plane_requests.append(request)
            self.assertEqual(request.method, 'GET')
            self.assertEqual(request.headers['authorization'], 'Bearer synthetic-host-only')
            if request.url.path == '/api/v1/auth/verify':
                return httpx.Response(200, json={'valid': True, 'owner_id': 'user:9999'})
            if request.url.path == '/api/v1/events':
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                    stream=authority_fixtures.FrameStream(frames))
            self.assertEqual(request.url.path,
                f'/api/v1/lease-groups/{binding.group_id}/members/{binding.lease_id}/authority')
            self.admit_entered.set()
            await self.admit_release.wait()
            return httpx.Response(200, json=projection)

        def reader(header, *, owner_id):
            result = HostPlaneReader(header, owner_id=owner_id, transport=httpx.MockTransport(transport))
            self.readers.append(result)
            return result

        self.enterContext(patch.object(deployment, 'HostPlaneReader', side_effect=reader))
        original = deployment.HostAuthorityAdapter
        self.enterContext(patch.object(deployment, 'HostAuthorityAdapter', side_effect=lambda actor, r:
            original(actor, r, alarm_factory=authority_fixtures.FakeAlarm)))
        return projection

    async def prepare(self, **changes):
        launch = json.loads(json.dumps(asdict(self.fixture.binding)))
        launch.update(changes)
        return await self.post('prepare', launch=launch, plan_sha256=digest(asdict(self.config.plan)))

    async def test_prepare_reads_own_admitted_member_and_attaches_existing_feed_without_start(self):
        self.isolated_member_plane()
        response = await self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()['report']
        self.assertEqual(report['phase'], 'prepared')
        self.assertTrue(report['events_reconciled'])
        self.assertEqual(report['operations'][0]['action'], 'prepare')
        self.assertEqual(report['operations'][0]['state'], 'returned')
        self.assertEqual(report['host_reader_owner'], 'user:9999')
        self.assertFalse(report['launch_attempted'])
        self.assertIs(self.runtime.host_rank, self.composition.roles['host'])
        self.assertTrue(self.readers[0]._client.is_closed)
        self.assertFalse(self.readers[1]._client.is_closed)
        before = self.runtime.host_rank
        replay = await self.prepare()
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertIs(self.runtime.host_rank, before)
        self.assertEqual(len(self.readers), 2)
        other = await self.client.post('/v1/exo/host/prepare', json=self.payload(
            request_id='b' * 64, launch=asdict(self.fixture.binding), plan_sha256=digest(asdict(self.config.plan))))
        self.assertEqual(other.status_code, 409)

    async def settled_history(self):
        projection = self.isolated_member_plane()
        response = await self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        owner = self.runtime.host_rank
        fenced = await self.client.post('/v1/exo/host/fence-unstarted', json=self.payload(
            request_id='b'*64, identity=run_identity(owner.actor.spec.binding)))
        self.assertTrue(fenced.json()['report']['cleanup_observed'])
        projection.update(phase='released', release_requested=True, authoritative=False,
                          authority_expires_at=None)
        projection['member'].update(cleanup_attested=True, settlement_recorded=True)
        projection['member']['lease'].update(status='released', released_at='2026-11-16T00:01:00+00:00')
        history = self.runtime.exo_host_control._preparation_history('lease_'+'9'*16)
        reader = SimpleNamespace(projection=AsyncMock(return_value=projection), close=AsyncMock())
        self.enterContext(patch.object(self.config, 'host_reader', return_value=reader))
        collector = host_control.collect_attempt
        self.enterContext(patch.object(host_control, 'collect_attempt', side_effect=lambda spec:
            collector(spec, boot_reader=lambda: self.config.boot_id)))
        return projection, history, reader

    async def test_historical_no_start_cleanup_and_exact_settlement_allow_new_session_preflight(self):
        _, history, reader = await self.settled_history()
        before = list(self.runtime.queue.supervisor.journal.db.execute(
            'SELECT * FROM exo_host_controls ORDER BY action'))
        await self.runtime.exo_host_control._completed_preparations(history)
        reader.projection.assert_awaited_once()
        reader.close.assert_awaited_once()
        after = list(self.runtime.queue.supervisor.journal.db.execute(
            'SELECT * FROM exo_host_controls ORDER BY action'))
        self.assertEqual(before, after)
        self.fixture.spawner.assert_not_called()

    async def test_historical_cleanup_without_exact_terminal_receipt_stays_refused(self):
        projection, history, reader = await self.settled_history()
        original = copy.deepcopy(projection)
        variants = [original | {'phase': 'releasing'}, original | {'group_id': 'foreign'},
            original | {'generation': 2},
            original | {'member': original['member'] | {'cleanup_attested': False}},
            original | {'member': original['member'] | {'settlement_recorded': False}},
            original | {'member': original['member'] | {'lease_id': 'foreign'}},
            original | {'member': original['member'] | {'lease': original['member']['lease'] | {'status': 'active'}}},
            original | {'member': original['member'] | {'lease': original['member']['lease'] | {'memory_mb': 999}}}]
        for value in variants:
            reader.projection.return_value = value
            with self.subTest(value=value), self.assertRaises(host_control.HTTPException) as caught:
                await self.runtime.exo_host_control._completed_preparations(history)
            self.assertEqual(caught.exception.status_code, 409)
        reader.projection.side_effect = OSError('unavailable')
        with self.assertRaises(host_control.HTTPException):
            await self.runtime.exo_host_control._completed_preparations(history)

    async def test_prepare_disconnect_keeps_same_supervisor_task_and_member(self):
        self.isolated_member_plane()
        self.admit_release.clear()
        observer = asyncio.create_task(self.prepare())
        await self.admit_entered.wait()
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(list(self.fixture.state_base.iterdir()))
        self.assertEqual(len(self.runtime.queue.supervisor.preparations), 1)
        self.admit_release.set()
        response = await self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['report']['phase'], 'prepared')
        self.assertEqual(len(self.readers), 2)

    async def test_unadmitted_or_wrong_generation_preparation_keeps_owned_failure_without_files(self):
        self.isolated_member_plane(changes={'generation': 2})
        response = await self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['report']['phase'], 'preparation_unconfirmed')
        self.assertEqual(response.json()['report']['operations'][0]['state'], 'unconfirmed')
        self.assertIsNone(getattr(self.runtime, 'host_rank', None))
        self.assertFalse(list(self.fixture.state_base.iterdir()))
        identity = run_identity(self.config.compile(self.fixture.binding).spec.binding)
        status = await self.post('status', identity=identity)
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()['report']['phase'], 'preparation_unconfirmed')
        await self.prepare()
        self.assertEqual(len(self.readers), 1)

    async def test_prepare_schema_cannot_choose_runtime_or_change_fixed_rank(self):
        for changes in ({'generation': True}, {'pipeline_rank': False}, {'layers': [0, True]},
                        {'argv': ['arbitrary']}, {'rank': 'wafer'}, {'expires_at': '2040-01-01'},
                        {'group_id': '../wrong'}):
            self.assertEqual((await self.prepare(**changes)).status_code, 422)
        self.assertFalse(self.runtime.queue.supervisor.journal.db.execute('SELECT 1 FROM exo_host_controls').fetchall())

    async def test_prepare_cannot_invent_a_future_lease_expiry(self):
        self.isolated_member_plane()
        response = await self.prepare(expires_at='2040-01-01T00:00:00+00:00')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['report']['phase'], 'preparation_unconfirmed')
        self.assertIsNone(getattr(self.runtime, 'host_rank', None))
        self.assertFalse(list(self.fixture.state_base.iterdir()))

    async def test_partial_private_setup_failure_still_attaches_negative_fence_cleanup(self):
        self.isolated_member_plane()
        with patch.object(deployment.secrets, 'token_hex', side_effect=OSError('fixture private setup')):
            response = await self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['report']['operations'][0]['state'], 'unconfirmed')
        self.assertIs(self.runtime.host_rank, self.composition.roles['host'])
        identity = run_identity(self.runtime.host_rank.actor.spec.binding)
        fenced = await self.client.post('/v1/exo/host/fence-unstarted',
            json=self.payload(request_id='b' * 64, identity=identity))
        self.assertEqual(fenced.status_code, 200, fenced.text)
        self.assertTrue(fenced.json()['report']['cleanup_observed'])

    async def test_standby_files_preflight_checks_live_transport_without_host_or_lease(self):
        response = await self.post('files', policy_sha256=digest(asdict(self.config.files)))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(response.json()['report']['boot_id'], self.config.boot_id)
        self.assertIsNone(getattr(self.runtime, 'host_rank', None))
        self.assertEqual(list(self.fixture.state_base.iterdir()), [])
        self.fixture.network = False
        self.assertEqual((await self.post('files', policy_sha256=digest(asdict(self.config.files)))).status_code, 503)
        self.assertEqual((await self.post('files', policy_sha256='0' * 64)).status_code, 409)
        self.assertFalse(self.runtime.queue.supervisor.journal.db.execute('SELECT 1 FROM grants').fetchall())

    async def test_authentication_is_exact_and_precedes_body_decode(self):
        for caller, status in ((None, 401), ({'owner_id': self.config.plan.owner_id}, 401),
                               ({'valid': True, 'owner_id': 'user:other'}, 403),
                               ({'valid': 1, 'owner_id': self.config.plan.owner_id}, 401)):
            self.caller = caller
            response = await self.client.post('/v1/exo/host/start', content=b'not json')
            self.assertEqual(response.status_code, status)
        self.assertIsNone(getattr(self.runtime, 'host_rank', None))

    async def test_malformed_duplicate_nonfinite_or_extra_fields_refuse(self):
        for body in ('[]', '{"request_id":"x","request_id":"x"}', '{"request_id":NaN}',
                     json.dumps(self.payload(identity={}, argv=['unsafe'])),
                     json.dumps({'request_id': True, 'identity': {}})):
            self.assertEqual((await self.client.post('/v1/exo/host/start', content=body)).status_code, 422)
        self.assertEqual((await self.client.post('/v1/exo/host/start', content=b' ' * (128 * 1024 + 1))).status_code, 413)

    async def test_source_or_fixed_configuration_change_refuses(self):
        with patch.dict(host_control._SOURCE, {'deployment.py': '0' * 64}):
            self.assertEqual((await self.post('status', identity={})).status_code, 503)
        with patch.object(self.config, 'boot_id', '00000000-0000-0000-0000-000000000001'):
            self.assertEqual((await self.post('status', identity={})).status_code, 503)

    async def test_status_requires_registered_exact_identity_and_is_not_authority(self):
        self.assertEqual((await self.post('status', identity={})).status_code, 409)
        owner = await self.bind()
        identity = run_identity(owner.actor.spec.binding)
        for changes in ({'lease_id': 'lease_' + '2' * 16}, {'generation': True},
                        {'runtime_sha256': '0' * 64}, {'state_path': '/tmp/arbitrary'}):
            self.assertEqual((await self.post('status', identity=identity | changes)).status_code, 409)
        report = (await self.post('status', identity=identity)).json()['report']
        self.assertEqual(report['phase'], 'prepared')
        self.assertFalse(report['authority_from_this_status'])
        self.assertIsNone(report['child'])
        self.assertFalse(report['operations'])

    async def test_disconnect_does_not_cancel_start_and_same_request_never_restarts(self):
        owner = await self.bind()
        started = asyncio.Event()

        async def start():
            started.set()
            await self.release.wait()

        with patch.object(owner, 'start', new=AsyncMock(side_effect=start)) as effect:
            identity = run_identity(owner.actor.spec.binding)
            observer = asyncio.create_task(self.post('start', identity=identity))
            await started.wait()
            observer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await observer
            self.assertEqual(effect.await_count, 1)
            self.assertEqual(len(self.runtime.queue.supervisor.preparations), 1)
            other = await self.client.post('/v1/exo/host/start',
                json={'request_id': 'b' * 64, 'identity': identity})
            self.assertEqual(other.status_code, 409)
            self.release.set()
            replay = await self.post('start', identity=identity)
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(effect.await_count, 1)
            self.assertEqual(replay.json()['report']['operations'][0]['state'], 'returned')

    async def test_old_owned_row_is_observed_without_launch_after_lost_task(self):
        owner = await self.bind()
        identity = run_identity(owner.actor.spec.binding)
        control = self.composition.control
        with patch.object(owner, 'start', new=AsyncMock(side_effect=ValueError('private response text'))) as effect:
            first = await self.post('start', identity=identity)
            control.tasks.clear()  # Same durable journal, no in-memory operation.
            again = await self.post('start', identity=identity)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(again.status_code, 200, again.text)
            self.assertEqual(effect.await_count, 1)
            self.assertNotIn('private response text', again.text)
            self.assertEqual(again.json()['report']['operations'][0]['state'], 'unconfirmed')

    async def test_missing_host_feed_refuses_before_committing_start(self):
        owner = await self.bind()
        owner._events_ready = False
        self.assertEqual((await self.post('start', identity=run_identity(owner.actor.spec.binding))).status_code, 409)
        self.assertFalse(self.runtime.queue.supervisor.journal.db.execute('SELECT 1 FROM exo_host_controls').fetchall())

    async def test_failed_task_registration_preserves_owned_row_without_retry(self):
        owner = await self.bind()
        identity = run_identity(owner.actor.spec.binding)
        with patch.object(self.runtime.queue.supervisor, 'prepare', side_effect=RuntimeError('schedule unavailable')):
            # Direct operation call observes scheduling failure after commit.
            with self.assertRaises(RuntimeError):
                await self.composition.control._operate('start', self.payload(identity=identity))
        with patch.object(owner, 'start', new=AsyncMock()) as effect:
            response = await self.post('start', identity=identity)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['report']['operations'][0]['state'], 'owned')
            effect.assert_not_called()

    async def test_negative_fence_receipt_is_same_exact_validated_cleanup_contract(self):
        owner = await self.bind()
        identity = run_identity(owner.actor.spec.binding)
        response = await self.post('fence-unstarted', identity=identity)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['report']['cleanup_observed'])
        self.assertEqual(owner._attempt()['phase'], 'launch_fenced')
        endpoint = HostEndpoint(self.config.files.rank, self.config.files.resource_id, self.config.files.uid,
            'https://host.test', host_control.SOURCE_SHA256, configuration_sha256(self.config))
        client = HostControlClient(endpoint, lambda: 'Bearer synthetic-coordinator',
            transport=httpx.ASGITransport(app=self.runtime.queue.service.app))
        try:
            def collect(spec):
                return receipt_files.collect_attempt(spec, boot_reader=lambda: self.config.boot_id)
            with patch.object(host_control, 'collect_attempt', side_effect=collect):
                proof = await HTTPReceiptReader(client).read(self.config.plan, binding=self.fixture.binding,
                    journal_path=str(self.runtime.settings.journal), state_path=str(owner.actor.state),
                    boot_id=self.config.boot_id, launch_sha256=owner.launch_sha256,
                    runtime_sha256=self.config.runtime_sha256)
                self.assertEqual(proof.phase, 'launch_fenced')
                self.assertIsNone(proof.started)
                self.assertIsNotNone(proof.cleanup_bytes)
                with self.assertRaises(ValueError):
                    await HTTPReceiptReader(client).read(self.config.plan, binding=self.fixture.binding,
                        journal_path=str(self.runtime.settings.journal), state_path='/tmp/wrong',
                        boot_id=self.config.boot_id, launch_sha256=owner.launch_sha256,
                        runtime_sha256=self.config.runtime_sha256)
        finally:
            await client.close()

    async def test_pause_blocks_new_start_but_allows_cleanup_and_drains_owned_task(self):
        owner = await self.bind()
        identity = run_identity(owner.actor.spec.binding)
        self.composition.control.pause()
        self.assertEqual((await self.post('start', identity=identity)).status_code, 409)
        self.assertEqual((await self.post('fence-unstarted', identity=identity)).status_code, 200)
        await self.composition.close()
        self.assertTrue(self.composition.closed)

    async def test_client_checks_nonce_source_configuration_and_never_redirects(self):
        endpoint = HostEndpoint('slice', self.config.files.resource_id, self.config.files.uid,
            'https://host.test', host_control.SOURCE_SHA256, configuration_sha256(self.config))
        for mutation in ('redirect', 'nonce', 'source', 'configuration', 'cache', 'duplicate'):
            requests = []

            def transport(request):
                requests.append(request)
                value = {'request_id': json.loads(request.content)['request_id'],
                    'source_sha256': endpoint.source_sha256,
                    'configuration_sha256': endpoint.configuration_sha256, 'report': {}}
                headers = {'cache-control': 'no-store'}
                if mutation == 'redirect':
                    return httpx.Response(307, headers={'location': 'https://foreign.test'})
                if mutation in ('nonce', 'source', 'configuration'):
                    value[{'nonce': 'request_id', 'source': 'source_sha256',
                           'configuration': 'configuration_sha256'}[mutation]] = '0' * 64
                if mutation == 'cache':
                    headers.clear()
                if mutation == 'duplicate':
                    return httpx.Response(200, content=b'{"report":{},"report":{}}',
                        headers=headers | {'content-type': 'application/json'})
                return httpx.Response(200, json=value, headers=headers)

            client = HostControlClient(endpoint, lambda: 'Bearer synthetic-coordinator',
                transport=httpx.MockTransport(transport))
            try:
                with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                    await client.status(self.config.compile(self.fixture.binding).spec.binding)
                self.assertEqual(len(requests), 1)
                self.assertTrue(all(v is None for v in requests[0].extensions['timeout'].values()))
            finally:
                await client.close()

    async def test_client_has_no_payload_route_or_credential_escape(self):
        endpoint = HostEndpoint('slice', self.config.files.resource_id, self.config.files.uid,
            'https://host.test', host_control.SOURCE_SHA256, configuration_sha256(self.config))
        client = HostControlClient(endpoint, lambda: 'Bearer synthetic-coordinator')
        try:
            for action, payload in (('arbitrary', {}), ('status', {'request_id': 'override'})):
                with self.assertRaises(ValueError):
                    await client._request(action, payload)
            with self.assertRaises(ValueError):
                await client.files(replace(self.config.files, resource_id='foreign/gpu'))
        finally:
            await client.close()
