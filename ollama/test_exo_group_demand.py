"""Existing-supervisor demand hook and shutdown ownership, no real providers."""

import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx

from . import test_exo_group_plan as fixtures
from .exo_groups.acquire import GroupAcquisition
from .exo_groups.demand import AcquisitionDemand
from .exo_groups.intents import GroupIntents
from .exo_groups.plane import CoordinatorPlane
from .inference_supervisor import Supervisor


class DemandControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend_calls = []
        async def backend(grant, request):
            self.backend_calls.append(grant)
            return {'answer': 'fixture'}
        self.supervisor = Supervisor(Path(self.temp.name) / 'queue.sqlite', backend, lambda _: False)
        fixture = fixtures.GroupBindingControls()
        fixture.setUp()
        self.plan = fixture.plan
        self.requests = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        async def transport(request):
            self.requests.append(request)
            self.entered.set()
            await self.release.wait()
            return httpx.Response(200, json={'valid': True, 'owner_id': self.plan.owner_id})
        plane = CoordinatorPlane(self.plan.owner_id, lambda: 'Bearer fixture',
            transport=httpx.MockTransport(transport))
        def unexpected_authority(*_):
            raise AssertionError('no admitted group in this control')
        self.actor = GroupAcquisition(GroupIntents(self.supervisor.journal), self.plan,
            fixture.manifest, plane, unexpected_authority, verify_staging=fixtures.staged_fixture)

    async def asyncTearDown(self):
        self.release.set()
        await self.actor.close()
        self.supervisor.close()
        self.temp.cleanup()

    def enqueue(self, model=None):
        journal = self.supervisor.journal
        grant = journal.submit('caller', 'one', 'claude-services-slice/gpu-0',
            model or self.plan.model_id, lifetime='execution')
        return journal.enqueue(grant['id'], 'caller', 'one', {'operation': 'chat'})

    async def test_capacity_wait_still_prepares_demand_and_repeated_pump_does_not_loop(self):
        AcquisitionDemand(self.supervisor, {self.plan.model_id: self.actor})
        call = self.enqueue()
        self.assertIsNone(self.supervisor.pump('claude-services-slice/gpu-0'))
        saved_epoch = self.actor._epoch
        self.supervisor.pump('claude-services-slice/gpu-0')
        self.supervisor.pump('claude-services-slice/gpu-0')
        self.assertEqual(self.actor._epoch, saved_epoch)
        self.assertEqual(self.supervisor.journal.db.execute(
            'SELECT execution_id FROM exo_group_demands').fetchone()[0], call['id'])
        self.assertEqual(self.requests, [])  # Event cursor not yet confirmed.
        self.assertEqual(self.backend_calls, [])

    async def test_unrelated_local_model_is_not_added_to_the_group(self):
        AcquisitionDemand(self.supervisor, {self.plan.model_id: self.actor})
        self.enqueue('local-fixture')
        self.supervisor.pump('claude-services-slice/gpu-0')
        self.assertEqual(self.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM exo_group_demands').fetchone()[0], 0)
        self.assertEqual(self.requests, [])

    async def test_preparation_exception_keeps_call_queued_without_fallback(self):
        call = self.enqueue()
        self.supervisor.permitted = lambda _: True
        def fail(_resource):
            raise ValueError('private credential and prompt')
        self.supervisor.prepare_demand = fail
        with self.assertLogs('ollama.inference_supervisor', level='WARNING') as logs:
            self.assertIsNone(self.supervisor.pump('claude-services-slice/gpu-0'))
            self.assertIsNone(self.supervisor.pump('claude-services-slice/gpu-0'))
        self.assertNotIn('private', repr(logs.output))
        self.assertEqual(len(logs.output), 1)
        failures = self.supervisor.journal.db.execute("SELECT state,reason FROM metadata_events "
            "WHERE kind='capacity'").fetchall()
        self.assertEqual([tuple(row) for row in failures], [('unknown', 'demand_preparation_failed')])
        self.supervisor.prepare_demand = lambda _: None
        self.supervisor.permitted = lambda _: False
        self.supervisor.pump('claude-services-slice/gpu-0')
        self.assertEqual(self.supervisor.journal.db.execute("SELECT reason FROM metadata_events "
            "WHERE kind='capacity' ORDER BY seq DESC LIMIT 1").fetchone()[0],
            'demand_preparation_reconciled')
        self.assertEqual(self.supervisor.journal.execution(call['id'], 'caller')['state'], 'queued')
        self.assertEqual(self.backend_calls, [])

    async def test_installation_refuses_empty_duplicate_or_wrong_model_registration(self):
        for actors in ({}, {'wrong-model': self.actor}):
            with self.assertRaises(ValueError):
                AcquisitionDemand(self.supervisor, actors)
        self.assertIsNone(self.supervisor.prepare_demand)
        AcquisitionDemand(self.supervisor, {self.plan.model_id: self.actor})
        with self.assertRaises(ValueError):
            AcquisitionDemand(self.supervisor, {self.plan.model_id: self.actor})

    async def test_supervisor_owns_pending_acquisition_and_cannot_close_over_it(self):
        AcquisitionDemand(self.supervisor, {self.plan.model_id: self.actor})
        self.release.clear()
        self.enqueue()
        self.actor.events_reconciled()
        self.supervisor.pump('claude-services-slice/gpu-0')
        await asyncio.wait_for(self.entered.wait(), 2)
        self.assertTrue(self.supervisor.preparations)
        with self.assertRaisesRegex(RuntimeError, 'live backend work'):
            self.supervisor.close()
        closer = asyncio.create_task(self.actor.close())
        await asyncio.sleep(0)
        self.assertFalse(closer.done())
        self.release.set()
        await closer
        self.assertEqual([r.method for r in self.requests], ['GET'])
        self.assertEqual(self.backend_calls, [])


if __name__ == '__main__':
    unittest.main()
