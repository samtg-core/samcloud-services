"""Durable no-launch decisions; physical counters injected, no fleet calls."""

import asyncio
from dataclasses import replace
import hashlib
import json
import unittest
from unittest import mock

from . import test_exo_group_rank_runtime as fixtures
from .exo_groups.cleanup import validate_cleanup
from .exo_groups.rank_process import ObservationUnknown
from .exo_groups.rank_runtime import record_host_attempt
from .inference_arbiter import Arbiter


class NoStartControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.parts=fixtures.RankRuntimeControls()
        self.parts.initial_phase='acquiring'
        await self.parts.asyncSetUp()
        self.runtime,self.actor,self.journal=self.parts.runtime,self.parts.actor,self.parts.journal

    async def asyncTearDown(self):
        self.actor.spec=self.parts.original_spec
        self.parts.observation=self.parts.original_observation
        self.parts.fixture.memory,self.parts.fixture.port_free=100,True
        self.parts.fixture.boot=self.runtime.boot_id
        if self.actor.process is not None:
            self.parts.fixture.child.code=0
            self.actor.process_exited()
        if self.actor.receipt is None:
            try:
                await self.runtime.attest_never_started()
            except ObservationUnknown:
                # Dispose only the injected CPU fixture's observer. The test
                # does NOT mark the owner closed or its journal claim settled.
                self.runtime.adapter.alarm.close()
                await self.runtime.adapter.reader.close()
                self.journal.close()
                self.parts.fixture.tearDown()
                return
        await self.runtime.close_after_cleanup()
        self.journal.close()
        self.parts.fixture.tearDown()

    def phase(self):
        return self.journal.db.execute('SELECT phase FROM exo_host_rank_attempts').fetchone()[0]

    async def test_partial_member_fences_without_child_and_publishes_exact_evidence(self):
        self.parts.context.projection.update(phase='acquiring',authoritative=False)
        await self.runtime.on_heartbeat()
        self.assertFalse(self.actor.draining)
        receipt=await self.runtime.attest_never_started()
        self.assertEqual(self.phase(),'launch_fenced')
        self.assertFalse(self.actor.launch_attempted)
        self.assertIsNone(self.actor.process)
        self.assertEqual(self.parts.fixture.spawn_count,0)
        raw=(self.actor.state/'cleanup-evidence.json').read_bytes()
        value=validate_cleanup(raw,self.runtime.prepared)
        self.assertEqual(value['kind'],'observed_rank_never_started')
        self.assertNotIn('child',value['observation'])
        self.assertEqual(value['observation']['processes_started'],0)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),receipt['evidence_sha256'])
        self.assertFalse(receipt['lease_released'])
        self.assertFalse(receipt['settlement_recorded'])
        self.parts.context.projection.update(phase='admitted',authoritative=True)
        await self.runtime.on_heartbeat()
        with self.assertRaises(ObservationUnknown):
            await self.runtime.start()
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.parts.fixture.spawn_count,0)

    async def test_unregistered_owner_cannot_start_from_valid_authority(self):
        self.parts.context.projection.update(phase='admitted',authoritative=True)
        await self.runtime.on_heartbeat()
        journal=self.runtime._journal
        self.runtime._journal=None
        try:
            with self.assertRaises(ObservationUnknown):
                await self.runtime.start()
        finally:
            self.runtime._journal=journal
        self.assertEqual(self.parts.fixture.spawn_count,0)

    async def test_registered_prelaunch_owner_cannot_close_without_negative_cleanup(self):
        with self.assertRaises(ObservationUnknown):
            await self.runtime.close_after_cleanup()
        self.assertFalse(self.runtime.closed)
        self.assertEqual(self.phase(),'prepared')

    async def test_memory_and_listener_failure_keep_fence_until_actual_inventory_change(self):
        self.parts.fixture.memory=1000
        self.parts.fixture.port_free=False
        self.assertIsNone(await self.runtime.attest_never_started())
        self.assertEqual(self.phase(),'launch_fenced')
        self.assertFalse((self.actor.state/'cleanup-evidence.json').exists())
        self.parts.fixture.memory=100
        self.assertIsNone(self.runtime.inventory_changed())
        self.parts.fixture.port_free=True
        self.assertIsNotNone(self.runtime.inventory_changed())
        self.assertEqual(self.parts.fixture.spawn_count,0)

    async def test_boot_or_runtime_change_refuses_without_negative_proof(self):
        original=self.parts.observation
        for changed in (replace(original,boot_id='00000000-0000-0000-0000-000000000099'),
                        replace(original,uid=original.uid+1),
                        replace(original,runtime_sha256='a'*64)):
            self.parts.observation=changed
            with self.assertRaises(ObservationUnknown):
                await self.runtime.attest_never_started()
            self.assertEqual(self.phase(),'prepared')
        self.parts.observation=original

    async def test_durable_launch_decision_with_zero_observed_process_is_not_no_start(self):
        self.runtime._transition('prepared','launch_committed')
        self.assertIsNone(self.actor.process)
        self.assertFalse(self.actor.launch_attempted)
        with self.assertRaises(ObservationUnknown):
            await self.runtime.attest_never_started()
        self.assertFalse((self.actor.state/'cleanup-evidence.json').exists())

    async def test_uncertain_spawn_never_gets_negative_cleanup(self):
        self.parts.context.projection.update(phase='admitted',authoritative=True)
        await self.runtime.on_heartbeat()
        def unknown(*_args,**_kw):
            raise OSError('injected unknown spawn outcome')
        self.actor.spawner=unknown
        with self.assertRaises(OSError):
            await self.runtime.start()
        self.assertEqual(self.phase(),'spawning')
        self.assertTrue(self.actor.launch_attempted)
        with self.assertRaises(ObservationUnknown):
            await self.runtime.attest_never_started()
        self.assertFalse((self.actor.state/'cleanup-evidence.json').exists())

    async def test_existing_launch_intent_blocks_negative_cleanup_even_without_handle(self):
        (self.actor.state/'launch-intent.json').write_text('{}')
        with self.assertRaises(ObservationUnknown):
            await self.runtime.attest_never_started()
        self.assertFalse((self.actor.state/'cleanup-evidence.json').exists())

    async def test_launch_and_negative_attestation_serialize_one_decision(self):
        result=await asyncio.gather(self.runtime.attest_never_started(),self.runtime.start(),
                                   return_exceptions=True)
        self.assertIsInstance(result[1],ObservationUnknown)
        self.assertEqual(self.phase(),'launch_fenced')
        self.assertEqual(self.parts.fixture.spawn_count,0)

    async def test_failed_audit_after_artifact_reuses_same_bytes_on_reconciliation(self):
        original=self.actor._record
        failed=False
        def record(kind,**details):
            nonlocal failed
            if kind=='cleanup_required' and not failed:
                failed=True
                raise OSError('injected durable notification failure')
            return original(kind,**details)
        with mock.patch.object(self.actor,'_record',side_effect=record):
            self.assertIsNone(await self.runtime.attest_never_started())
        raw=(self.actor.state/'cleanup-evidence.json').read_bytes()
        self.assertTrue(self.actor.audit_write_failures)
        receipt=self.runtime.inventory_changed()
        self.assertEqual((self.actor.state/'cleanup-evidence.json').read_bytes(),raw)
        self.assertEqual(receipt['evidence_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertFalse(self.actor.audit_write_failures)

    async def test_fence_and_no_rebind_survive_host_journal_reopen(self):
        await self.runtime.attest_never_started()
        reopened=Arbiter(self.journal.path if hasattr(self.journal,'path') else self.parts.fixture.root/'host-inference.sqlite')
        try:
            self.assertEqual(reopened.db.execute('SELECT phase FROM exo_host_rank_attempts').fetchone()[0],
                             'launch_fenced')
            with self.assertRaises(ObservationUnknown):
                record_host_attempt(self.runtime,reopened)
        finally:
            reopened.close()

    async def test_negative_artifact_requires_exact_preparation_and_fence(self):
        await self.runtime.attest_never_started()
        raw=(self.actor.state/'cleanup-evidence.json').read_bytes()
        value=json.loads(raw)
        for key,bad in (('launch_fenced',False),('launch_attempted',True),('processes_started',False),
                        ('attempt_sha256','f'*64),('host_boot','other'),('launch_intent_absent',False),
                        ('baseline_in_use_bytes',999),('observed_in_use_bytes',1000)):
            changed=json.loads(raw)
            changed['observation'][key]=bad
            with self.assertRaises(ValueError):
                validate_cleanup(json.dumps(changed).encode(),self.runtime.prepared)
        value['observation']['child']={'pid':123}
        with self.assertRaises(ValueError):
            validate_cleanup(json.dumps(value).encode(),self.runtime.prepared)

    async def test_legacy_attempt_is_unknown_after_schema_migration_not_rebound(self):
        legacy=Arbiter(self.parts.fixture.root/'legacy.sqlite')
        names=('resource_id','lease_id','group_id','generation','runtime_sha256','launch_sha256',
               'boot_id','state_path')
        original=self.runtime._attempt()
        legacy.db.execute('''CREATE TABLE exo_host_rank_attempts (
            resource_id TEXT,lease_id TEXT,group_id TEXT,generation INTEGER,
            runtime_sha256 TEXT,launch_sha256 TEXT,boot_id TEXT,state_path TEXT,
            PRIMARY KEY(resource_id,lease_id),UNIQUE(group_id,generation,resource_id))''')
        legacy.db.execute('INSERT INTO exo_host_rank_attempts VALUES(?,?,?,?,?,?,?,?)',
                          tuple(original[k] for k in names))
        saved_journal=self.runtime._journal
        self.runtime._journal=None
        try:
            with self.assertRaises(ObservationUnknown):
                record_host_attempt(self.runtime,legacy)
            row=legacy.db.execute('SELECT * FROM exo_host_rank_attempts').fetchone()
            self.assertEqual(row['phase'],'unknown')
            self.assertIsNone(row['prepared_json'])
        finally:
            self.runtime._journal=saved_journal
            legacy.close()

    async def test_direct_lifecycle_cannot_bypass_installed_launch_journal(self):
        self.parts.context.projection.update(phase='admitted',authoritative=True)
        await self.runtime.on_heartbeat()
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.phase(),'prepared')
        self.assertEqual(self.parts.fixture.spawn_count,0)
        self.assertFalse((self.actor.state/'launch-intent.json').exists())

    async def test_live_child_cannot_be_relabelled_never_started(self):
        self.parts.context.projection.update(phase='admitted',authoritative=True)
        await self.runtime.on_heartbeat()
        await self.runtime.start()
        with self.assertRaises(ObservationUnknown):
            await self.runtime.attest_never_started()
        self.assertEqual(self.phase(),'spawning')
        self.assertEqual(self.parts.fixture.spawn_count,1)

    async def test_failed_physical_inventory_keeps_the_fence_and_allows_later_observation(self):
        original=self.actor.memory_reader
        def unreadable():
            raise OSError('injected device inventory failure')
        self.actor.memory_reader=unreadable
        try:
            self.assertIsNone(await self.runtime.attest_never_started())
            self.assertEqual(self.phase(),'launch_fenced')
            self.assertFalse((self.actor.state/'cleanup-evidence.json').exists())
        finally:
            self.actor.memory_reader=original
        self.assertIsNotNone(self.runtime.inventory_changed())
