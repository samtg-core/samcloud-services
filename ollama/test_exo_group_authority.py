"""Provider authority against HTTP transport controls; no writes or inference."""

import asyncio
import copy
from dataclasses import replace
import unittest

import httpx

from . import test_exo_group_plan as fixtures
from .exo_groups.authority import AuthorityReadUnavailable, GroupAuthorityReader
from .exo_groups.plan import BindingRefused


class ProviderGroupAuthority(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fixture = fixtures.GroupBindingControls()
        fixture.setUp()
        self.plan = fixture.plan
        self.launches = fixture.bind()
        self.now = 100.0
        self.header = 'Bearer synthetic-coordinator-secret'
        self.auth_owner = self.plan.owner_id
        self.requests = []
        self.hook = None
        self.override = None
        self.projections = {
            binding.lease_id: {
                'group_id': binding.group_id, 'owner_id': binding.owner_id,
                'generation': binding.generation, 'phase': 'serving',
                'release_requested': False, 'authoritative': True,
                'observed_at': '2026-11-14T10:05:00+00:00',
                'authority_expires_at': '2026-11-14T10:10:00+00:00',
                'member': {'resource_id': binding.resource_id, 'lease_id': binding.lease_id,
                    'cleanup_attested': False, 'settlement_recorded': False,
                    'lease': copy.deepcopy(fixture.leases[binding.lease_id])},
            }
            for binding in self.launches
        }
        self.reader = GroupAuthorityReader(self.plan, self.launches, lambda: self.header,
            clock=lambda: self.now, transport=httpx.MockTransport(self.handle))

    async def asyncTearDown(self):
        await self.reader.close()

    async def handle(self, request):
        self.requests.append(request)
        self.assertEqual(request.method, 'GET')
        self.assertEqual(request.url.scheme, 'https')
        self.assertEqual(request.url.host, 'cloud.samtg.xyz')
        self.assertEqual(request.headers['Authorization'], self.header)
        self.assertEqual(request.headers['Cache-Control'], 'no-cache')
        self.assertTrue(all(value is None for value in request.extensions['timeout'].values()))
        if request.url.path == '/api/v1/auth/verify':
            return httpx.Response(200, json={'valid': True, 'owner_id': self.auth_owner})
        self.assertTrue(request.url.path.startswith('/api/v1/lease-groups/group-one/members/'))
        lease_id = request.url.path.split('/')[-2]
        self.assertIn(lease_id, self.projections)
        self.assertTrue(request.url.path.endswith('/authority'))
        if self.hook is not None:
            await self.hook(lease_id)
        if self.override is not None:
            return self.override
        return httpx.Response(200, json=self.projections[lease_id])

    async def test_exact_all_member_reads_without_lease_or_provider_writes(self):
        report = await self.reader.refresh()
        self.assertTrue(report.permits_requests(self.now))
        self.assertTrue(self.reader.may_start_ranks())
        self.assertTrue(self.reader.may_serve_requests())
        self.assertEqual(len(self.requests), 4)
        self.assertEqual([request.url.path for request in self.requests[1:]], [
            f'/api/v1/lease-groups/group-one/members/{binding.lease_id}/authority'
            for binding in self.launches])
        await self.reader.refresh()
        self.assertEqual(len(self.requests), 7)  # Same credential, three fresh projections.

    async def test_admitted_group_can_start_but_cannot_serve(self):
        for projection in self.projections.values():
            projection['phase'] = 'admitted'
        await self.reader.refresh()
        self.assertTrue(self.reader.may_start_ranks())
        self.assertFalse(self.reader.may_serve_requests())

    async def test_any_member_phase_fence_cleanup_or_lease_mismatch_withdraws(self):
        changed = self.projections[self.launches[-1].lease_id]
        original = copy.deepcopy(changed)
        variations = [dict(original, generation=2), dict(original, authoritative=False),
            dict(original, phase='acquiring'), dict(original, release_requested=True),
            dict(original, member=original['member'] | {'cleanup_attested': True}),
            dict(original, member=original['member'] | {'lease_id': 'foreign'}),
            dict(original, member=original['member'] | {'lease': original['member']['lease'] | {'memory_mb': 999}})]
        for projection in variations:
            with self.subTest(projection=projection):
                self.projections[self.launches[-1].lease_id] = projection
                await self.reader.refresh()
                self.assertFalse(self.reader.may_start_ranks())
                self.assertFalse(self.reader.may_serve_requests())

    async def test_partial_member_read_failure_cannot_retain_previous_authority(self):
        await self.reader.refresh()
        entered, release = asyncio.Event(), asyncio.Event()
        async def hook(lease_id):
            if lease_id == self.launches[-1].lease_id:
                entered.set()
                await release.wait()
                raise httpx.ReadError('synthetic-coordinator-secret private parser body')
        self.hook = hook
        pending = asyncio.create_task(self.reader.refresh())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertFalse(self.reader.may_serve_requests())
            self.assertIsNone(self.reader.current)
        finally:
            release.set()
        with self.assertRaises(AuthorityReadUnavailable) as caught:
            await pending
        self.assertNotIn('secret', str(caught.exception))
        self.assertFalse(self.reader.may_start_ranks())

    async def test_slow_other_member_reads_exhaust_the_earliest_authority(self):
        self.projections[self.launches[0].lease_id]['authority_expires_at'] = '2026-11-14T10:05:02+00:00'
        async def delay(_lease_id):
            self.now += 1
        self.hook = delay
        report = await self.reader.refresh()
        self.assertEqual(self.now, 103)
        self.assertFalse(report.permits(self.now))
        self.assertFalse(self.reader.may_serve_requests())

    async def test_event_during_read_cannot_republish_old_authority(self):
        await self.reader.refresh()
        async def event(lease_id):
            if lease_id == self.launches[-1].lease_id:
                self.reader.invalidate()
        self.hook = event
        with self.assertRaises(AuthorityReadUnavailable):
            await self.reader.refresh()
        self.assertIsNone(self.reader.current)

    async def test_host_sleep_or_backward_clock_cannot_extend_authority(self):
        await self.reader.refresh()
        self.now = 399.9
        self.assertTrue(self.reader.may_serve_requests())
        self.now = 400
        self.assertFalse(self.reader.may_serve_requests())
        self.now = 99
        self.assertFalse(self.reader.may_start_ranks())

    async def test_clock_read_failure_withdraws_the_snapshot(self):
        await self.reader.refresh()
        def fail():
            raise OSError('clock unavailable')
        self.reader._clock = fail
        self.assertFalse(self.reader.may_serve_requests())
        self.assertIsNone(self.reader.current)

    async def test_other_principal_or_rotated_credential_requires_verification(self):
        await self.reader.refresh()
        self.header = 'Bearer rotated-secret'
        self.auth_owner = 'user:other'
        before = len(self.requests)
        with self.assertRaises(AuthorityReadUnavailable):
            await self.reader.refresh()
        self.assertEqual(len(self.requests), before + 1)
        self.assertEqual(self.requests[-1].url.path, '/api/v1/auth/verify')
        self.assertFalse(self.reader.may_start_ranks())

    async def test_invalid_credential_is_refused_before_http(self):
        for value in ('', 'Bearer ', 'Bearer bad\nvalue', 'secret-without-scheme'):
            self.header = value
            with self.assertRaises(AuthorityReadUnavailable):
                await self.reader.refresh()
        self.assertEqual(self.requests, [])

    async def test_redirect_type_size_and_duplicate_field_fail_closed(self):
        for response in [httpx.Response(302, headers={'location': 'https://other.invalid/secret'}),
                httpx.Response(200, text='not authority'),
                httpx.Response(200, content=b'x' * (128 * 1024 + 1), headers={'content-type': 'application/json'}),
                httpx.Response(200, content=b'{"group_id":"group-one","group_id":"foreign"}', headers={'content-type': 'application/json'})]:
            self.override = response
            with self.assertRaises(AuthorityReadUnavailable):
                await self.reader.refresh()
            self.assertFalse(self.reader.may_start_ranks())

    async def test_cancelled_observation_cannot_retain_previous_permission(self):
        await self.reader.refresh()
        entered = asyncio.Event()
        async def hang(_lease_id):
            entered.set()
            await asyncio.Event().wait()
        self.hook = hang
        pending = asyncio.create_task(self.reader.refresh())
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(self.reader.may_start_ranks())

    async def test_incomplete_duplicate_generation_or_order_binding_refused(self):
        variants = [self.launches[:2], tuple(reversed(self.launches)),
            (self.launches[0], self.launches[0], self.launches[2]),
            tuple(replace(binding, generation=True) for binding in self.launches),
            self.launches[:2] + (replace(self.launches[2], generation=2),)]
        for launches in variants:
            with self.assertRaises(BindingRefused):
                GroupAuthorityReader(self.plan, launches, lambda: self.header)
