"""Explicit Slice/Obelisk membership controls; no fleet, compute or deployment."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from . import test_exo_group_plan as fixtures
from .exo_groups.authority import GroupAuthorityReader
from .exo_groups.intents import GroupIntents
from .exo_groups.plan import BindingRefused, GroupPlan, bind_admitted_group, check_rank_manifest
from .inference_arbiter import Arbiter


def two_hosts():
    f = fixtures.GroupBindingControls()
    f.setUp()
    raw = copy.deepcopy(f.raw)
    raw['members'] = [m for m in raw['members'] if m['rank'] != 'wafer']
    for member in raw['members']:
        if member['rank'] == 'slice':
            member['layers'] = [0, 12]
        else:
            member['pipeline_rank'] = 1
    manifest = json.loads(f.manifest)
    manifest['ranks'] = [{'name': 'slice', 'layers': [0, 12]},
                         {'name': 'obelisk', 'layers': [12, 28]}]
    f.manifest = json.dumps(manifest).encode()
    raw['rank_manifest_sha256'] = hashlib.sha256(f.manifest).hexdigest()
    f.raw, f.plan = raw, GroupPlan.parse(raw)
    f.group['members'] = [m for m in f.group['members'] if m['lease_id'] != 'lease-wafer']
    for i, member in enumerate(f.group['members']):
        member['ordinal'] = i
    f.group['id'] = 'group_' + 'c' * 24
    f.group['intent_sha256'] = f.plan.intent_sha256
    f.leases.pop('lease-wafer')
    return f


class TwoHostControls(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.f = two_hosts()

    async def test_exact_two_host_plan_and_manifest_bind_all_required_members(self):
        check_rank_manifest(self.f.plan, self.f.manifest)
        launches = self.f.bind()
        self.assertEqual(self.f.plan.pipeline_ranks, ('slice', 'obelisk'))
        self.assertEqual([(v.rank, v.pipeline_rank, v.layers) for v in launches],
                         [('slice', 0, (0, 12)), ('obelisk', 1, (12, 28))])
        self.assertEqual([m['resource_id'] for m in self.f.plan.request()['members']],
                         ['obelisk-services/gpu-0', 'claude-services-slice/gpu-0'])

    async def test_single_host_wrong_subset_or_gap_cannot_be_a_two_host_plan(self):
        scenarios = []
        one = copy.deepcopy(self.f.raw); one['members'] = one['members'][:1]; scenarios.append(one)
        wrong = copy.deepcopy(self.f.raw)
        wrong['members'][0].update(rank='wafer', resource_id='wafer-services/gpu-metal')
        scenarios.append(wrong)
        gap = copy.deepcopy(self.f.raw); gap['members'][1]['layers'] = [0, 11]; scenarios.append(gap)
        rank = copy.deepcopy(self.f.raw); rank['members'][0]['pipeline_rank'] = 2; scenarios.append(rank)
        for raw in scenarios:
            with self.subTest(raw=raw), self.assertRaises(BindingRefused):
                GroupPlan.parse(raw)

    async def test_two_host_authority_reads_every_member_and_rejects_missing_observation(self):
        launches = self.f.bind()
        seen = []
        missing = False
        async def handle(request):
            if request.url.path.endswith('/auth/verify'):
                return httpx.Response(200, json={'valid': True, 'owner_id': self.f.plan.owner_id})
            binding = next(b for b in launches if b.lease_id == request.url.path.split('/')[-2])
            seen.append(binding.rank)
            if missing and binding.rank == 'obelisk':
                return httpx.Response(200, json={})
            return httpx.Response(200, json={
                'group_id': self.f.group['id'], 'owner_id': self.f.plan.owner_id,
                'generation': 1, 'phase': 'serving', 'release_requested': False,
                'authoritative': True, 'observed_at': self.f.now.isoformat(),
                'authority_expires_at': '2026-11-14T10:10:00+00:00',
                'member': {'resource_id': binding.resource_id, 'lease_id': binding.lease_id,
                           'cleanup_attested': False, 'settlement_recorded': False,
                           'lease': self.f.leases[binding.lease_id]}})
        reader = GroupAuthorityReader(self.f.plan, launches, lambda: 'Bearer synthetic',
            clock=lambda: 100, transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(reader.close)
        await reader.refresh()
        self.assertEqual(seen, ['slice', 'obelisk'])
        self.assertTrue(reader.may_serve_requests())
        missing = True
        await reader.refresh()
        self.assertFalse(reader.may_serve_requests())

    async def test_two_host_group_with_extra_or_partial_member_never_admits(self):
        for members in (self.f.group['members'][:1], self.f.group['members'] + [
                {'resource_id': 'wafer-services/gpu-metal', 'lease_id': 'lease-wafer'}]):
            group = self.f.group | {'members': members}
            with self.assertRaises(BindingRefused):
                bind_admitted_group(self.f.plan, group, self.f.leases,
                                    expected_generation=1, now=self.f.now)

    async def test_two_host_launches_freeze_in_the_existing_queue_journal(self):
        with tempfile.TemporaryDirectory() as d:
            journal = Arbiter(Path(d)/'queue.sqlite')
            self.addCleanup(journal.close)
            intents = GroupIntents(journal)
            grant = journal.submit('caller', 'test', 'claude-services-slice/gpu-0',
                                   self.f.plan.model_id, lifetime='execution')
            call = journal.enqueue(grant['id'], 'caller', 'test', {'operation': 'chat'})
            intents.prepare(call['id'], self.f.plan, self.f.manifest)
            intents.record_staging(self.f.plan.owner_id, self.f.plan.request_key,
                                   fixtures.staging_fixture(self.f.plan))
            intents.begin_create(self.f.plan.owner_id, self.f.plan.request_key)
            intents.record_group(self.f.plan.owner_id, self.f.plan.request_key, self.f.group)
            saved = intents.bind_launches(self.f.plan.owner_id, self.f.plan.request_key, self.f.bind())
            self.assertEqual([b['rank'] for b in saved['launches']], ['slice', 'obelisk'])
            self.assertEqual(journal.execution(call['id'], 'caller')['state'], 'queued')
