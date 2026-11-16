"""Service-owned acquisition controls; metadata backend injected, no compute."""

import asyncio
import copy
from pathlib import Path
import tempfile
import unittest

import httpx

from . import test_exo_group_plan as fixtures
from .exo_groups.acquire import GroupAcquisition
from .exo_groups.authority import GroupAuthorityReader
from .exo_groups.intents import GroupIntents
from .inference_arbiter import Arbiter


class AcquisitionControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.journal = Arbiter(Path(self.temp.name) / 'queue.sqlite')
        self.intents = GroupIntents(self.journal)
        self.fixture = fixtures.GroupBindingControls()
        self.fixture.setUp()
        self.plan = self.fixture.plan
        self.group = dict(copy.deepcopy(self.fixture.group), id='group_' + 'a' * 24)
        self.leases = copy.deepcopy(self.fixture.leases)
        self.calls, self.readers = [], []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.lose_reply = False
        self.server_created = False
        self.close_count = 0
        controls = self

        class Plane:
            owner_id = controls.plan.owner_id
            async def verify_owner(self):
                controls.calls.append('verify_owner')
            async def create(self, body):
                controls.calls.append(('create', copy.deepcopy(body)))
                assert controls.intents.snapshot(controls.plan.owner_id,
                    controls.plan.request_key)['create_started'] == 1
                controls.entered.set()
                await controls.release.wait()
                controls.server_created = True
                if controls.lose_reply:
                    controls.lose_reply = False
                    raise httpx.ReadError('private token and response detail')
                return copy.deepcopy(controls.group)
            async def group(self, group_id):
                controls.calls.append(('group', group_id))
                assert group_id == controls.group['id']
                return copy.deepcopy(controls.group)
            async def lease(self, lease_id):
                controls.calls.append(('lease', lease_id))
                return copy.deepcopy(controls.leases[lease_id])
            async def advance(self, group_id, generation):
                controls.calls.append(('advance', group_id, generation))
                return copy.deepcopy(controls.group)
            async def close(self):
                controls.close_count += 1

        def factory(plan, launches):
            async def transport(request):
                if request.url.path == '/api/v1/auth/verify':
                    return httpx.Response(200, json={'valid': True, 'owner_id': plan.owner_id})
                binding = next(b for b in launches if b.lease_id == request.url.path.split('/')[-2])
                return httpx.Response(200, json={
                    'group_id': self.group['id'], 'owner_id': plan.owner_id,
                    'generation': self.group['generation'], 'phase': self.group['phase'],
                    'release_requested': self.group['release_requested'], 'authoritative': True,
                    'observed_at': self.fixture.now.isoformat(),
                    'authority_expires_at': '2026-11-14T10:10:00+00:00',
                    'member': {'resource_id': binding.resource_id, 'lease_id': binding.lease_id,
                        'cleanup_attested': False, 'settlement_recorded': False,
                        'lease': self.leases[binding.lease_id]},
                })
            reader = GroupAuthorityReader(plan, launches, lambda: 'Bearer test-owner',
                clock=lambda: 100, transport=httpx.MockTransport(transport))
            self.readers.append(reader)
            return reader

        self.actor = GroupAcquisition(self.intents, self.plan, self.fixture.manifest,
            Plane(), factory, wall_clock=lambda: self.fixture.now, verify_staging=fixtures.staged_fixture)
        self.grant = self.journal.submit('caller', 'one', 'claude-services-slice/gpu-0',
            self.plan.model_id, lifetime='execution')
        self.call = self.journal.enqueue(self.grant['id'], 'caller', 'one', {'operation': 'chat'})

    async def asyncTearDown(self):
        self.release.set()
        await self.actor.close()
        self.journal.close()
        self.temp.cleanup()

    async def start(self):
        self.actor.note_demand(self.call['id'])
        self.actor.events_reconciled()
        return await self.actor.wait()

    def saved(self):
        return self.intents.snapshot(self.plan.owner_id, self.plan.request_key)

    async def test_unconfirmed_event_source_does_not_start_acquisition(self):
        self.actor.note_demand(self.call['id'])
        await self.actor.wait()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.saved()['create_started'])
        report = await self.start()
        self.assertEqual(report.phase, 'admitted')
        self.assertTrue(self.actor.authority.may_start_ranks())
        self.assertFalse(self.actor.authority.may_serve_requests())
        self.assertEqual(len(self.saved()['launches']), 3)

    async def test_unbound_or_failed_staging_never_submits_a_physical_group(self):
        self.actor.verify_staging = None
        self.assertEqual((await self.start()).phase, 'unknown')
        self.assertFalse(self.saved()['create_started'])
        self.assertFalse(self.server_created)
        async def missing(*_):
            raise OSError('pinned files absent')
        self.actor.verify_staging = missing
        self.actor.wake()
        await self.actor.wait()
        self.assertFalse(self.saved()['create_started'])
        self.assertFalse(self.server_created)
        self.assertIsNone(self.intents.staging(self.plan.owner_id, self.plan.request_key))

    async def test_verified_proof_commits_before_create_and_policy_cannot_change(self):
        await self.start()
        saved = self.intents.staging(self.plan.owner_id, self.plan.request_key)
        self.assertEqual(saved['runtime_sha256'], fixtures.staging_fixture(self.plan).runtime_sha256)
        from dataclasses import replace
        async def changed(*_):
            return replace(fixtures.staging_fixture(self.plan), policy_sha256='f' * 64)
        self.actor.verify_staging = changed
        self.actor.wake()
        self.assertEqual((await self.actor.wait()).phase, 'unknown')
        self.assertFalse(self.actor.may_start_ranks())
        self.assertEqual(self.intents.staging(self.plan.owner_id, self.plan.request_key), saved)

    async def test_request_permission_requires_serving_current_cursor_and_authority(self):
        self.group['phase'] = 'serving'
        await self.start()
        self.assertTrue(self.actor.may_serve_requests())
        self.actor.events_lost()
        self.assertFalse(self.actor.may_serve_requests())
        self.actor.events_reconciled()
        await self.actor.wait()
        self.assertTrue(self.actor.may_serve_requests())
        self.actor.authority.invalidate()
        self.assertFalse(self.actor.may_serve_requests())

    async def test_observer_cancellation_leaves_metadata_task_owned(self):
        self.release.clear()
        self.actor.note_demand(self.call['id'])
        self.actor.events_reconciled()
        await asyncio.wait_for(self.entered.wait(), 2)
        observer = asyncio.create_task(self.actor.wait())
        await asyncio.sleep(0)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(self.actor._task.done())
        self.release.set()
        self.assertEqual((await self.actor.wait()).phase, 'admitted')
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == 'create' for c in self.calls), 1)

    async def test_lost_create_reply_is_not_retried_until_an_explicit_wakeup(self):
        self.lose_reply = True
        self.assertEqual((await self.start()).phase, 'unknown')
        self.assertTrue(self.server_created)
        self.assertIsNone(self.saved()['group_id'])
        await asyncio.sleep(0)
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == 'create' for c in self.calls), 1)
        self.journal.request_release(self.grant['id'], 'caller')
        self.actor.wake()  # Reconcile the owned metadata operation, not inference.
        self.assertEqual((await self.actor.wait()).phase, 'admitted')
        creates = [c[1] for c in self.calls if isinstance(c, tuple) and c[0] == 'create']
        self.assertEqual(creates, [self.plan.request(), self.plan.request()])
        self.assertEqual(self.saved()['group_id'], self.group['id'])
        self.assertEqual(self.journal.execution(self.call['id'], 'caller')['state'], 'cancelled')

    async def test_cancelled_unsubmitted_demand_does_not_acquire(self):
        self.actor.note_demand(self.call['id'])
        self.journal.request_release(self.grant['id'], 'caller')
        self.actor.events_reconciled()
        self.assertEqual((await self.actor.wait()).phase, 'idle')
        self.assertFalse(self.server_created)
        self.assertFalse(self.saved()['create_started'])

    async def test_event_contact_loss_during_create_records_identity_but_no_permission(self):
        self.release.clear()
        self.actor.note_demand(self.call['id'])
        self.actor.events_reconciled()
        await asyncio.wait_for(self.entered.wait(), 2)
        self.actor.events_lost()
        self.release.set()
        self.assertEqual((await self.actor.wait()).phase, 'unknown')
        self.assertEqual(self.saved()['group_id'], self.group['id'])
        self.assertIsNone(self.actor.authority)
        self.actor.events_reconciled()
        self.assertEqual((await self.actor.wait()).phase, 'admitted')
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == 'create' for c in self.calls), 1)

    async def test_fence_event_withdraws_before_io_and_never_rebinds_generation(self):
        await self.start()
        self.group.update(generation=2, phase='releasing')
        self.assertTrue(self.actor.on_event({'type': 'lease.group_changed',
            'data': {'resource_id': self.plan.members[0].resource_id}}))
        self.assertFalse(self.actor.authority.may_start_ranks())
        self.assertEqual((await self.actor.wait()).phase, 'fenced')
        self.assertEqual(self.saved()['generation'], 1)
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == 'create' for c in self.calls), 1)

    async def test_liveness_hints_during_slow_staging_do_not_starve_create(self):
        from .exo_groups.events import RegistryEventFeed
        feed = RegistryEventFeed(self.actor.plane, (self.actor,), None)
        entered, release = asyncio.Event(), asyncio.Event()
        reads = 0
        async def slow_stage(*args):
            nonlocal reads
            reads += 1
            if reads == 1:
                entered.set()
                await release.wait()
            return await fixtures.staged_fixture(*args)
        self.actor.verify_staging = slow_stage
        self.actor.note_demand(self.call['id'])
        self.actor.events_reconciled()
        await asyncio.wait_for(entered.wait(), 2)
        epoch = self.actor._epoch
        for _ in range(5):
            await feed._heartbeat()
            await feed._event({'type': 'resource.changed',
                'data': {'resource_id': 'unrelated/gpu-0'}})
        try:
            self.assertEqual(self.actor._epoch, epoch)
            self.assertFalse(self.saved()['create_started'])
        finally:
            release.set()
        self.assertEqual((await self.actor.wait()).phase, 'admitted')
        self.assertTrue(self.actor.may_start_ranks())
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == 'create' for c in self.calls), 1)
        self.assertEqual(reads, 2)  # Before create, then before admitted authority.

    async def test_liveness_cannot_mask_relevant_event_during_authority_refresh(self):
        await self.start()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.actor.authority.refresh
        async def slow_refresh():
            entered.set()
            await release.wait()
            return await original()
        self.actor.authority.refresh = slow_refresh
        self.actor.refresh()
        await asyncio.wait_for(entered.wait(), 2)
        self.actor.refresh()
        self.group.update(generation=2, phase='releasing')
        self.actor.on_event({'type': 'lease.group_changed',
            'data': {'resource_id': self.plan.members[0].resource_id}})
        self.assertFalse(self.actor.may_start_ranks())
        release.set()
        self.assertEqual((await self.actor.wait()).phase, 'fenced')
        self.assertFalse(self.actor.may_start_ranks())
        self.assertEqual(self.saved()['generation'], 1)

    async def test_contact_loss_during_slow_staging_still_prevents_create(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_stage(*args):
            entered.set()
            await release.wait()
            return await fixtures.staged_fixture(*args)
        self.actor.verify_staging = slow_stage
        self.actor.note_demand(self.call['id'])
        self.actor.events_reconciled()
        await asyncio.wait_for(entered.wait(), 2)
        self.actor.refresh()
        self.actor.events_lost()
        release.set()
        self.assertEqual((await self.actor.wait()).phase, 'unknown')
        self.assertFalse(self.saved()['create_started'])
        self.assertFalse(self.server_created)
        self.assertFalse(self.actor.may_start_ranks())

    async def test_close_drains_pending_metadata_and_keeps_ownership_record(self):
        self.release.clear()
        self.actor.note_demand(self.call['id'])
        self.actor.events_reconciled()
        await asyncio.wait_for(self.entered.wait(), 2)
        closer = asyncio.create_task(self.actor.close())
        await asyncio.sleep(0)
        self.assertFalse(closer.done())
        self.release.set()
        await closer
        await self.actor.close()
        self.assertEqual(self.saved()['group_id'], self.group['id'])
        self.assertIsNone(self.actor.authority)
        self.assertEqual(self.close_count, 1)
        with self.assertRaises(RuntimeError):
            self.actor.note_demand(self.call['id'])


if __name__ == '__main__':
    unittest.main()
