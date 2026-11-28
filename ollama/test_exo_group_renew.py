"""Renewal policy/receipt controls; no physical claims, process or inference."""

import copy
from datetime import timedelta
import unittest

from . import test_exo_group_plan as fixtures
from .exo_groups.renew import GroupRenewals, RenewalUnavailable


class RenewalControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        f = fixtures.GroupBindingControls()
        f.setUp()
        self.plan, self.base_wall = f.plan, f.now
        self.group, self.leases = copy.deepcopy(f.group), copy.deepcopy(f.leases)
        self.now = 100.0
        self.calls = []
        self.no_extension = self.fence_during_renew = False
        self.delay = 0
        controls = self
        class Plane:
            owner_id = controls.plan.owner_id
            async def member(self, group_id, lease_id):
                lease = controls.leases[lease_id]
                view = {'group_id': controls.group['id'], 'owner_id': self.owner_id,
                    'generation': controls.group['generation'], 'phase': controls.group['phase'],
                    'release_requested': controls.group['release_requested'], 'authoritative': False,
                    'observed_at': (controls.base_wall + timedelta(seconds=controls.now-100)).isoformat(),
                    'member': {'resource_id': lease['resource_id'], 'lease_id': lease_id,
                        'cleanup_attested': False, 'settlement_recorded': False,
                        'lease': copy.deepcopy(lease)}}
                controls.now += controls.delay
                return view
            async def group(self, group_id):
                assert group_id == controls.group['id']
                return copy.deepcopy(controls.group)
            async def renew(self, lease_id, ttl_seconds):
                controls.calls.append((lease_id, ttl_seconds))
                if not controls.no_extension:
                    lease = controls.leases[lease_id]
                    expiry = controls.base_wall + timedelta(seconds=controls.now-100+ttl_seconds)
                    lease['expires_at'] = expiry.isoformat()
                if controls.fence_during_renew:
                    controls.group.update(generation=2, phase='releasing', release_requested=True)
                return {'status': 'active'}  # Reply alone is never permission.
        self.plane = Plane()
        self.renewals = GroupRenewals(self.plan, self.plane,
            renew_before_s={m.rank: 120 for m in self.plan.members}, clock=lambda: self.now)

    async def test_fresh_deadlines_do_not_renew_early(self):
        report = await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(report.next_due, 280)
        self.assertEqual(self.calls, [])
        self.assertEqual([r.lease_id for r in report.members], [m['lease_id'] for m in self.group['members']])

    async def test_due_renewal_keeps_exact_ids_and_uses_declared_ttl(self):
        self.now = 281
        report = await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(self.calls, [(m['lease_id'], 600) for m in self.group['members']])
        self.assertEqual(report.next_due, 761)
        self.assertTrue(all(r.deadline == 881 for r in report.members))

    async def test_partial_group_renews_active_member_not_indefinite_wait_or_missing_rank(self):
        self.group['phase'] = 'acquiring'
        queued = self.group['members'][1]
        queued['status'] = 'queued'
        self.leases[queued['lease_id']].update(status='queued', granted_at=None, expires_at=None)
        self.group['members'][2].update(lease_id=None, status='not_requested')
        self.now = 281
        report = await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(self.calls, [(self.group['members'][0]['lease_id'], 600)])
        self.assertEqual(len(report.members), 1)

    async def test_accepted_but_unextended_reply_is_not_success_or_busy_retry(self):
        self.now, self.no_extension = 281, True
        with self.assertRaises(RenewalUnavailable):
            await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(len(self.calls), 1)

    async def test_fence_racing_post_cannot_reauthorize_the_old_generation(self):
        self.now, self.fence_during_renew = 281, True
        with self.assertRaises(RenewalUnavailable):
            await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.group['generation'], 2)
        self.assertTrue(all(lease['status'] == 'active' for lease in self.leases.values()))

    async def test_expired_observation_never_attempts_renewal(self):
        self.now = 401
        with self.assertRaises(RenewalUnavailable):
            await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(self.calls, [])

    async def test_ceiling_has_no_further_renewal_wakeup_and_no_release(self):
        self.now = 6940  # 11:59 UTC, one minute before the explicit 2h ceiling.
        for lease in self.leases.values():
            lease['expires_at'] = '2026-11-14T12:00:00+00:00'
        report = await self.renewals.maintain(self.group, generation=1)
        self.assertIsNone(report.next_due)
        self.assertEqual(report.next_event, 7000)  # End-of-authority notification, not another renewal.
        self.assertTrue(all(r.at_ceiling and r.deadline == 7000 for r in report.members))
        self.assertEqual(self.calls, [])

    async def test_slow_transport_cannot_extend_observed_authority(self):
        self.delay = 301
        with self.assertRaises(RenewalUnavailable):
            await self.renewals.maintain(self.group, generation=1)
        self.assertEqual(self.calls, [])

    async def test_changed_lease_envelope_owner_or_generation_refuses(self):
        lease_id = self.group['members'][0]['lease_id']
        original = copy.deepcopy(self.leases[lease_id])
        self.now = 281
        for bad in (dict(original, agent_id='foreign'), dict(original, memory_mb=999),
                    dict(original, ttl_s=True), dict(original, settled_at='observed')):
            self.leases[lease_id] = bad
            with self.assertRaises(RenewalUnavailable):
                await self.renewals.maintain(self.group, generation=1)
        self.leases[lease_id] = original
        with self.assertRaises(RenewalUnavailable):
            await self.renewals.maintain(self.group, generation=True)
        self.assertEqual(self.calls, [])

    async def test_lead_times_are_explicit_complete_and_not_deadline_defaults(self):
        for leads in ({}, {m.rank: True for m in self.plan.members},
                      {m.rank: 0 for m in self.plan.members},
                      {m.rank: 600 for m in self.plan.members}):
            with self.assertRaises(ValueError):
                GroupRenewals(self.plan, self.plane, renew_before_s=leads)


if __name__ == '__main__':
    unittest.main()
