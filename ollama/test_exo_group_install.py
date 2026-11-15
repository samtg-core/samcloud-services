"""Real GatewayQueue/routes/capacity/journal; injected plane, ranks and device.

No model, host process or physical lease request. Runtime tests live separately.
"""

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fastapi import FastAPI
import httpx

from . import server
from .exo_groups.acquire import GroupAcquisition
from .exo_groups.backend import EXOBackend
from .exo_groups.calls import ExecutionBindings
from .exo_groups.intents import GroupIntents
from .exo_groups.plan import HOST_RESOURCES
from .inference_bootstrap import QueueRuntime, QueueSettings
from . import test_exo_group_backend as backend_controls
from . import test_exo_group_plan as plan_controls
from datetime import datetime, timezone


class InstalledControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reuse the complete pinned native-state fixture and private transport.
        self.parts = backend_controls.PrivateBackendControls()
        await self.parts.asyncSetUp()
        p = self.parts
        p.supervisor.close()
        self.resource = HOST_RESOURCES['slice']
        self.manager = SimpleNamespace(models={}, residency_renewal=None, sc=SimpleNamespace(
            get_resource=lambda _: {'id': self.resource, 'allocation_mode': 'shared',
                'available_memory_mb': 10000, 'leases': []},
            request_lease=Mock(side_effect=AssertionError('no extra physical lease'))),
            ollama=SimpleNamespace(memory_estimate_mb=Mock(side_effect=AssertionError('no EXO estimate'))),
            _collect_stats=lambda: {'memory_available_mb': 20000, 'memory_device_inuse_mb': 0})
        service = SimpleNamespace(app=FastAPI(), mgr=self.manager, ChatRequest=server.ChatRequest,
            config=SimpleNamespace(WHISPER_MAX_UPLOAD_MB=1, SC_RESOURCE_ID=self.resource,
                                   SC_SERVICE_ID='fixture-service'))
        self.runtime = QueueRuntime.install(service, QueueSettings(
            Path(p.temp.name) / 'queue.sqlite', {p.plan.model_id: self.resource, 'local': self.resource},
            reservation_mb={p.plan.model_id: 1000, 'local': 100}), registry_owner='user:58')
        self.runtime.bind_capacity_source(work_permitted=True)
        self.queue = self.runtime.queue
        self.journal = self.queue.supervisor.journal
        p.supervisor, p.journal = self.queue.supervisor, self.journal
        p.intents = GroupIntents(self.journal)
        p.bindings = ExecutionBindings(p.intents)
        self.created = 0
        self.create_entered, self.create_release = asyncio.Event(), asyncio.Event()
        self.create_release.set()
        self.fail_create = False
        controls = self
        p.group['phase'] = 'serving'

        class Plane:
            owner_id = p.plan.owner_id
            async def verify_owner(self):
                pass
            async def create(self, _body):
                controls.created += 1
                controls.create_entered.set()
                await controls.create_release.wait()
                if controls.fail_create:
                    raise httpx.ReadError('unknown create reply')
                return copy.deepcopy(p.group)
            async def group(self, _id):
                return copy.deepcopy(p.group)
            async def lease(self, lease_id):
                f = plan_controls.GroupBindingControls()
                f.setUp()
                return f.leases[lease_id]
            async def close(self):
                pass

        async def nothing():
            pass
        self.actor = GroupAcquisition(p.intents, p.plan, p.manifest, Plane(),
            lambda *_: SimpleNamespace(refresh=nothing, close=nothing, invalidate=lambda: None,
                may_start_ranks=lambda: True, may_serve_requests=lambda: True),
            wall_clock=lambda: datetime(2026,11,14,10,0,tzinfo=timezone.utc))
        p.actor = self.actor
        p.backend = EXOBackend(self.actor, p.private, p.bindings, p.contract,
                               p.launches, p.verify_runtime)
        self.installation = self.runtime.bind_exo({p.plan.model_id: self.actor})
        self.provider = self.installation.providers[p.plan.model_id]
        self.source = self.runtime.capacity_source
        @service.app.middleware('http')
        async def caller(request, call_next):
            request.state.caller = {'username': 'caller'}
            return await call_next(request)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url='http://fixture')

    async def asyncTearDown(self):
        self.create_release.set()
        self.parts.end_release.set()
        self.source.pause()
        await self.drain()
        self.provider._closed = True
        await self.actor.close()  # Test teardown does not claim physical release.
        await self.http.aclose()
        self.runtime.residency_actor.close()
        await self.parts.asyncTearDown()

    async def drain(self):
        async with asyncio.timeout(3):
            while True:
                tasks = [task for task in (*self.queue.supervisor.workers.values(),
                    *self.queue.supervisor.preparations) if not task.done()]
                if not tasks:
                    return
                await asyncio.gather(*tasks)

    async def enqueue(self, key='one', *, stream=True):
        grant = await self.http.post('/v1/inference/grants', json={
            'key': key, 'model': self.parts.plan.model_id, 'lifetime': 'execution'})
        self.assertEqual(grant.status_code, 202, grant.text)
        response = await self.http.post('/v1/inference/grants/' + grant.json()['id'] + '/executions', json={
            'key': key, 'request': {'model': self.parts.plan.model_id,
                'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream}})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def state(self, execution):
        return self.journal.execution(execution, 'caller')['state']

    async def bind_ready(self):
        self.actor.events_reconciled()
        await self.drain()
        self.assertEqual(self.actor.report.phase, 'lease_serving')
        self.parts.bindings.bind_instance(self.parts.plan.owner_id,
            self.parts.plan.request_key, self.parts.contract)
        self.provider.bind_backend(self.parts.backend)
        await self.drain()

    async def test_local_capacity_cannot_authorize_group_and_no_local_lease_is_taken(self):
        await self.source.refresh(self.manager._collect_stats())
        self.assertTrue(self.source.permitted_model(self.resource, 'local'))
        call = await self.enqueue()
        self.assertEqual(self.state(call), 'queued')
        self.assertEqual(self.created, 0)
        self.assertIsNone(self.source.permitted_model(self.resource, self.parts.plan.model_id))
        await self.bind_ready()
        self.assertEqual(self.state(call), 'complete')
        self.assertEqual(self.parts.posts, 1)
        self.manager.sc.request_lease.assert_not_called()
        self.manager.ollama.memory_estimate_mb.assert_not_called()
        self.assertEqual(self.manager.models, {})
        self.assertIsNone(self.manager.initial_reservations.snapshot(self.parts.plan.model_id))

    async def test_terminal_failure_refreshes_readiness_for_next_queued_call(self):
        self.parts.mode = 'error'
        first, second = await self.enqueue(), await self.enqueue('two', stream=False)
        await self.bind_ready()
        self.assertEqual((self.state(first), self.state(second)), ('failed', 'complete'))
        self.assertEqual(self.parts.posts, 2)
        self.assertTrue(self.provider.owns_group())  # Logical result is not physical release.

    async def test_transport_loss_keeps_unknown_execution_and_next_call_waiting(self):
        self.parts.mode = 'reset'
        first, second = await self.enqueue(), await self.enqueue('two')
        await self.bind_ready()
        self.assertEqual((self.state(first), self.state(second)), ('uncertain', 'queued'))
        self.actor.events_reconciled()
        await self.drain()
        self.assertIsNone(self.queue.supervisor.pump(self.resource))
        self.assertEqual(self.parts.posts, 1)
        report = await self.runtime.quiesce()
        self.assertFalse(report['drained'])
        self.assertEqual((report['uncertain'], report['groups_owned']), (1, 1))
        with self.assertRaisesRegex(RuntimeError, 'EXO provider'):
            self.runtime.close()

    async def test_lost_create_blocks_local_until_current_group_read_confirms_slice_untouched(self):
        await self.enqueue()
        self.fail_create = True
        self.actor.events_reconciled()
        await self.drain()
        self.assertEqual(self.created, 1)
        self.assertIsNone(self.source.permitted_model(self.resource, 'local'))
        self.assertTrue(self.provider.owns_group())
        self.fail_create = False
        self.parts.group['phase'] = 'acquiring'
        for member in self.parts.group['members']:
            member.update(status='not_requested', lease_id=None)
        self.parts.group['members'][0].update(status='queued', lease_id='waiting-obelisk')
        async def advance(*_):
            return copy.deepcopy(self.parts.group)
        self.actor.plane.advance = advance
        self.actor.events_reconciled()
        await self.drain()
        self.assertIs(self.provider.local_permission(), True)
        self.assertTrue(self.source.permitted_model(self.resource, 'local'))
        self.parts.group['members'][1].update(status='queued', lease_id='waiting-slice')
        self.actor.wake()
        await self.drain()
        self.assertIs(self.source.permitted_model(self.resource, 'local'), False)
        self.actor.events_lost()
        self.assertIsNone(self.source.permitted_model(self.resource, 'local'))

    async def test_quiesce_does_not_claim_idle_owned_group_is_drained_or_release_it(self):
        call = await self.enqueue()
        await self.bind_ready()
        self.assertEqual(self.state(call), 'complete')
        report = await self.runtime.quiesce()
        self.assertFalse(report['drained'])
        self.assertEqual(report['groups_owned'], 1)
        with self.assertRaisesRegex(RuntimeError, 'observed cleanup'):
            await self.runtime.close_exo()
        self.assertFalse(self.actor._closed)
        self.assertEqual(self.actor.group_snapshot['phase'], 'serving')

    async def test_no_demand_installation_can_drain_without_creating_group(self):
        self.assertTrue((await self.runtime.quiesce())['drained'])
        await self.runtime.close_exo()
        self.assertTrue(self.installation.closed)
        self.assertEqual(self.created, 0)
        self.assertEqual(self.parts.posts, 0)

    async def test_event_loss_withdraws_cached_ready_and_does_not_fall_back(self):
        first = await self.enqueue()
        await self.bind_ready()
        self.assertEqual(self.state(first), 'complete')
        self.assertTrue(self.source.permitted_model(self.resource, self.parts.plan.model_id))
        self.actor.events_lost()
        self.assertIsNone(self.source.permitted_model(self.resource, self.parts.plan.model_id))
        second = await self.enqueue('after-loss')
        await self.drain()
        self.assertEqual(self.state(second), 'queued')
        self.assertEqual(self.parts.posts, 1)
        self.manager.sc.request_lease.assert_not_called()

    async def test_close_drains_started_create_then_refuses_over_returned_ownership(self):
        await self.enqueue()
        self.create_release.clear()
        self.actor.events_reconciled()
        await asyncio.wait_for(self.create_entered.wait(), 2)
        closer = asyncio.create_task(self.runtime.close_exo())
        await asyncio.sleep(0)
        self.assertFalse(closer.done())
        self.create_release.set()
        with self.assertRaisesRegex(RuntimeError, 'observed cleanup'):
            await closer
        self.assertTrue(self.provider.owns_group())
        self.assertFalse(self.installation.closed)
        self.assertFalse(self.actor._closed)

    async def test_local_inventory_must_drain_before_native_dispatch_and_its_change_wakes_once(self):
        call = await self.enqueue()
        self.manager.models['local'] = SimpleNamespace(in_flight=0)
        await self.bind_ready()
        self.assertEqual(self.state(call), 'queued')
        self.assertEqual(self.parts.posts, 0)
        self.manager.models.clear()
        await self.source.refresh(self.manager._collect_stats())
        await self.drain()
        self.assertEqual(self.state(call), 'complete')
        self.assertEqual(self.parts.posts, 1)
        epoch = self.provider._epoch
        await self.source.refresh(self.manager._collect_stats())
        await self.drain()
        self.assertEqual(self.provider._epoch, epoch)

    async def test_observed_released_group_can_close_without_becoming_owned_again(self):
        call = await self.enqueue()
        await self.bind_ready()
        self.assertEqual(self.state(call), 'complete')
        # Inject the core's fully released-group result, not local cleanup proof.
        self.parts.group.update(phase='released', release_requested=True)
        self.actor.wake()
        await self.drain()
        self.assertEqual(self.installation.owned(), 0)
        await self.runtime.close_exo()
        self.assertTrue(self.actor._closed)
        self.assertEqual(self.installation.owned(), 0)
        self.assertTrue((await self.runtime.quiesce())['drained'])


if __name__ == '__main__':
    unittest.main()
