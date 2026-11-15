"""Host role uses real QueueRuntime/capacity/journal; no host/lease/network effects."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fastapi import FastAPI

from . import server
from . import test_exo_group_rank_runtime as runtime_fixtures
from .exo_groups.rank_process import ObservationUnknown
from .exo_groups.rank_runtime import record_host_attempt
from .inference_bootstrap import QueueRuntime, QueueSettings


class HostInstallControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.parts = runtime_fixtures.RankRuntimeControls()
        await self.parts.asyncSetUp()
        self.owner = self.parts.runtime
        self.resource = self.owner.actor.spec.binding.resource_id
        self.manager = SimpleNamespace(models={}, residency_renewal=None,
            sc=SimpleNamespace(request_lease=Mock(side_effect=AssertionError('no extra lease'))),
            ollama=SimpleNamespace(memory_estimate_mb=Mock(side_effect=AssertionError('no local load'))))
        service = SimpleNamespace(app=FastAPI(), mgr=self.manager, ChatRequest=server.ChatRequest,
            config=SimpleNamespace(WHISPER_MAX_UPLOAD_MB=1, SC_RESOURCE_ID=self.resource,
                                   SC_SERVICE_ID='host-fixture'))
        self.runtime = QueueRuntime.install(service, QueueSettings(
            Path(self.parts.fixture.root) / 'queue.sqlite',
            {'local': self.resource, 'group-model': self.resource}),
            registry_owner=self.owner.adapter.reader.owner_id)
        self.runtime.bind_capacity_source(work_permitted=True)
        self.source = self.runtime.capacity_source
        self.source.observed_residency[self.resource,'local'] = ('local',None,None)
        self.source.publish(self.resource,'available','fixture cold observation',model='local')

    async def asyncTearDown(self):
        await self.parts.asyncTearDown()
        self.runtime.close()

    async def test_binding_blocks_local_dispatch_without_requesting_another_lease(self):
        self.assertTrue(self.source.permitted_model(self.resource,'local'))
        self.runtime.bind_exo_rank(self.owner)
        self.assertFalse(self.source.permitted_model(self.resource,'local'))
        self.source.publish(self.resource,'available','stale local callback',model='local')
        self.assertEqual(self.source.last_model[self.resource,'local']['reason'],'exo_host_rank_owned')
        self.assertFalse(self.source.permitted_model(self.resource,'local'))
        journal = self.runtime.queue.supervisor.journal
        grant = journal.submit('caller','one',self.resource,'local',lifetime='execution')
        call = journal.enqueue(grant['id'],'caller','one',{'operation':'chat'})
        self.assertIsNone(self.runtime.queue.supervisor.pump(self.resource))
        self.assertEqual(journal.execution(call['id'],'caller')['state'],'queued')
        row = journal.db.execute('SELECT * FROM exo_host_rank_attempts').fetchone()
        self.assertEqual(row['lease_id'],self.owner.actor.spec.binding.lease_id)
        self.assertEqual(row['launch_sha256'],self.owner.launch_sha256)
        self.manager.sc.request_lease.assert_not_called()
        self.manager.ollama.memory_estimate_mb.assert_not_called()

    async def test_quiesce_and_close_retain_open_host_owner_without_stopping_or_restarting_it(self):
        self.runtime.bind_exo_rank(self.owner)
        await self.owner.start()
        result = await self.runtime.quiesce()
        self.assertFalse(result['drained'])
        self.assertTrue(result['host_rank_owned'])
        self.assertEqual(self.parts.fixture.child.stop_count,0)
        with self.assertRaises(RuntimeError):
            self.runtime.close()
        with self.assertRaises(ObservationUnknown):
            await self.runtime.close_exo_rank()
        self.assertEqual(self.parts.fixture.spawn_count,1)

    async def test_host_barrier_preserves_separate_registered_group_provider_permission(self):
        self.runtime.bind_exo_rank(self.owner)
        self.source.external['group-model'] = SimpleNamespace(
            permitted=lambda _: True, local_permission=lambda: False)
        self.source.publish(self.resource,'available','group fixture snapshot')
        self.assertFalse(self.source.permitted_model(self.resource,'local'))
        self.assertTrue(self.source.permitted_model(self.resource,'group-model'))
        self.assertEqual(self.source.last_model[self.resource,'local']['state'],'unavailable')
        self.assertEqual(self.source.last_model[self.resource,'group-model']['state'],'available')
        self.manager.sc.request_lease.assert_not_called()

    async def test_observed_cleanup_closes_host_role_but_requires_fresh_capacity(self):
        self.runtime.bind_exo_rank(self.owner)
        await self.owner.start()
        self.parts.fixture.child.code = 0
        self.parts.watches[0].notify()
        await self.runtime.close_exo_rank()
        self.assertTrue(self.owner.closed)
        self.assertIsNone(self.source.permitted_model(self.resource,'local'))
        result = await self.runtime.quiesce()
        self.assertTrue(result['drained'])
        self.assertFalse(result['host_rank_owned'])
        self.assertEqual(self.runtime.queue.supervisor.journal.db.execute(
            'SELECT COUNT(*) FROM exo_host_rank_attempts').fetchone()[0],1)
        self.manager.sc.request_lease.assert_not_called()

    async def test_same_member_cannot_be_registered_again_even_after_close(self):
        self.runtime.bind_exo_rank(self.owner)
        await self.runtime.close_exo_rank()  # Even a pre-launch committed attempt is not replayed.
        with self.assertRaises(ObservationUnknown):
            record_host_attempt(self.owner,self.runtime.queue.supervisor.journal)
        self.assertEqual(self.parts.fixture.spawn_count,0)

    async def test_committed_host_attempt_survives_reopening_the_same_journal(self):
        from .inference_arbiter import Arbiter
        self.runtime.bind_exo_rank(self.owner)
        reopened = Arbiter(self.runtime.settings.journal)
        try:
            with self.assertRaises(ObservationUnknown):
                record_host_attempt(self.owner,reopened)
            self.assertEqual(reopened.db.execute(
                'SELECT COUNT(*) FROM exo_host_rank_attempts').fetchone()[0],1)
        finally:
            reopened.close()

    async def test_unverified_host_identity_or_existing_model_refuses_before_binding(self):
        self.owner.adapter.reader.owner_id = 'user:foreign'
        with self.assertRaises(RuntimeError):
            self.runtime.bind_exo_rank(self.owner)
        self.owner.adapter.reader.owner_id = self.manager.initial_reservations.owner
        self.manager.models['local'] = object()
        with self.assertRaises(RuntimeError):
            self.runtime.bind_exo_rank(self.owner)
        self.manager.models.clear()
        self.assertIsNone(getattr(self.runtime,'host_rank',None))
        self.assertIsNone(self.runtime.queue.supervisor.journal.db.execute(
            "SELECT name FROM sqlite_master WHERE name='exo_host_rank_attempts'").fetchone())
