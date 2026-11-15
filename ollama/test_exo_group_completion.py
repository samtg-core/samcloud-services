"""Exact settlement, lost replies and closure races; CPU-only injected hosts."""

import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import unittest

import httpx

from . import test_exo_group_acquire as acquisition
from .exo_groups.cleanup import CleanupExpectation, validate_cleanup
from .exo_groups.completion import GroupCompletion
from .exo_groups.install import GroupProvider
from .exo_groups.intents import GroupIntents
from .exo_groups.plane import CoordinatorPlane, PlaneUnavailable
from .exo_groups.plan import BindingRefused
from .exo_groups.rank_process import Listener, CleanupObservation, ProcessIdentity, RunBinding, cleanup_evidence
from .inference_arbiter import Arbiter, Conflict


class CompletionControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = acquisition.AcquisitionControls()
        await self.base.asyncSetUp()
        self.addAsyncCleanup(self.base.asyncTearDown)
        b = self.base
        await b.start()
        self.actor, self.plane, self.group = b.actor, b.actor.plane, b.group
        self.completion = GroupCompletion(self.actor)
        self.pins, self.files, self.proofs, self.mutations = {}, {}, {}, []
        self.lost_reply = self.fence_on_blob = self.bad_blob = self.bad_meta = False
        self.keep_hold = False
        self.post_entered, self.post_release = asyncio.Event(), asyncio.Event()
        self.post_release.set()
        runtime = b.intents.staging(b.plan.owner_id, b.plan.request_key)['runtime_sha256']
        for i, expected in enumerate(self.actor.authority.expected):
            run = RunBinding.from_expected(expected, runtime)
            pin = CleanupExpectation(run, ProcessIdentity(100+i, 99, 100+i, 501, 1000+i, 1),
                (Listener('127.0.0.1', 39999+i),), 100, 0)
            self.completion.pin_launch(pin)
            raw, digest = cleanup_evidence(CleanupObservation(run, pin.child, 0, (), (True,),
                100, 100, 0, '2026-11-14T10:05:00+00:00'))
            file_id = i+1
            self.files[file_id] = raw
            self.pins[run.lease_id] = pin
            self.proofs[run.lease_id] = (file_id, digest)
        self.lease_id = next(iter(self.pins))

        async def artifact(file_id):
            raw = self.files[file_id]
            return {'id': file_id, 'sha256': hashlib.sha256(raw).hexdigest(),
                    'bytes': 9999999 if self.bad_meta else len(raw)}
        async def artifact_bytes(file_id):
            if self.fence_on_blob:
                self.group.update(generation=2, phase='releasing', release_requested=True)
            raw = self.files[file_id]
            return raw+b' ' if self.bad_blob else raw
        async def begin_release(group_id, generation):
            self.assertEqual(generation, self.group['generation'])
            self.mutations.append(('group_release', group_id, generation))
            self.group.update(phase='releasing', release_requested=True)
            return copy.deepcopy(self.group)
        async def release_member(group_id, lease_id, generation, proof):
            self.assertEqual(generation, self.group['generation'])
            saved = b.journal.db.execute('SELECT intent_json FROM exo_group_settlements '
                'WHERE lease_id=? AND generation=?', (lease_id, generation)).fetchone()
            self.assertEqual(json.loads(saved['intent_json'])['evidence_sha256'], proof['evidence_sha256'])
            self.mutations.append(('member_release', lease_id, generation))
            self.post_entered.set()
            await self.post_release.wait()
            lease = b.leases[lease_id]
            if not self.keep_hold:
                lease.update(status='released', released_at='2026-11-14T10:05:00+00:00')
            member = next(m for m in self.group['members'] if m['lease_id'] == lease_id)
            member['cleanup_evidence'] = dict(proof, lease_id=lease_id, generation=generation)
            member['release_receipt'] = {'lease_id': lease_id, **{k:lease[k]
                for k in ('status', 'released_at', 'settled_at')}}
            member['status'] = lease['status']
            terminal = all(m.get('release_receipt') for m in self.group['members'])
            self.group.update(phase='released' if terminal else 'releasing', release_requested=True)
            if self.lost_reply:
                self.lost_reply = False
                raise OSError('unknown after commit')
            return copy.deepcopy(self.group)
        self.plane.artifact, self.plane.artifact_bytes = artifact, artifact_bytes
        self.plane.begin_release, self.plane.release_member = begin_release, release_member

    def end_call(self):
        self.base.journal.request_release(self.base.grant['id'], 'caller')

    async def settle(self, lease_id=None, generation=1):
        lease_id = lease_id or self.lease_id
        file_id, digest = self.proofs[lease_id]
        return await self.completion.settle_member(lease_id, generation=generation,
                                                 file_id=file_id, sha256=digest)

    def releases(self):
        return [m for m in self.mutations if m[0] == 'member_release']

    async def test_three_exact_receipts_required_before_group_confirmed(self):
        self.end_call()
        report = await self.completion.begin_release(generation=1)
        self.assertEqual(report['state'], 'host_cleanup_required')
        for i, lease_id in enumerate(self.pins):
            self.assertEqual((await self.settle(lease_id))['state'], 'exact_member_settlement_confirmed')
            self.assertEqual(sum(lease['status']=='active' for lease in self.base.leases.values()), 2-i)
        self.assertEqual((await self.completion.begin_release(generation=1))['state'],
                         'group_settlement_confirmed')
        self.assertEqual(len(self.releases()), 3)
        self.assertEqual(self.actor.report.group_phase, 'released')
        self.assertEqual(self.base.journal.db.execute('SELECT COUNT(*) FROM exo_group_settlements '
            'WHERE confirmed_json IS NOT NULL').fetchone()[0], 3)

    async def test_observer_disconnect_cannot_cancel_owned_settlement(self):
        self.end_call()
        self.post_release.clear()
        waiter = asyncio.create_task(self.settle())
        await self.post_entered.wait()
        owned = self.completion._task
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(owned.done())
        self.post_release.set()
        self.assertEqual((await self.completion.wait())['state'], 'exact_member_settlement_confirmed')
        self.assertEqual(len(self.releases()), 1)

    async def test_lost_post_reply_reconciles_saved_receipt_after_journal_reopen(self):
        self.end_call()
        self.lost_reply = True
        with self.assertRaises(OSError):
            await self.settle()
        b = self.base
        path = Path(b.temp.name)/'queue.sqlite'
        b.journal.close()
        b.journal = Arbiter(path)
        b.intents = self.actor.intents = GroupIntents(b.journal)
        self.completion = GroupCompletion(self.actor)
        self.assertEqual((await self.settle())['state'], 'exact_member_settlement_confirmed')
        self.assertEqual(len(self.releases()), 1)

    async def test_expired_hold_without_exact_settlement_does_not_pass(self):
        self.end_call()
        self.base.leases[self.lease_id]['status'] = 'expired'
        self.keep_hold = True
        with self.assertRaises(BindingRefused):
            await self.settle()
        self.assertIsNone(self.base.journal.db.execute('SELECT confirmed_json '
            'FROM exo_group_settlements').fetchone()[0])
        self.assertEqual(self.group['phase'], 'releasing')

    async def test_fence_during_evidence_read_does_not_write_or_adopt_generation(self):
        self.end_call()
        self.fence_on_blob = True
        with self.assertRaises(BindingRefused):
            await self.settle()
        self.assertEqual(self.releases(), [])
        self.assertEqual(self.base.saved()['generation'], 1)
        self.fence_on_blob = False
        self.assertEqual((await self.settle(generation=2))['launch_generation'], 1)

    async def test_saved_proof_cannot_change_even_after_fence(self):
        self.end_call()
        self.lost_reply = True
        with self.assertRaises(OSError):
            await self.settle()
        old_id, _ = self.proofs[self.lease_id]
        raw = json.loads(self.files[old_id])
        raw['observation']['observed_at'] = '2026-11-14T10:06:00+00:00'
        self.files[10] = json.dumps(raw).encode()
        self.proofs[self.lease_id] = (10, hashlib.sha256(self.files[10]).hexdigest())
        self.group['generation'] = 2
        with self.assertRaises(Conflict):
            await self.settle(generation=2)
        self.assertEqual(len(self.releases()), 1)

    async def test_queued_running_and_uncertain_work_block_closure_without_withdrawal(self):
        for state in ('queued', 'running', 'uncertain'):
            self.base.journal.db.execute('UPDATE executions SET state=? WHERE id=?',
                (state, self.base.call['id']))
            with self.assertRaises(Conflict):
                await self.completion.begin_release(generation=1)
            self.assertFalse(self.base.intents.closing(self.base.plan.owner_id, self.base.plan.request_key))
            self.assertTrue(self.actor.may_start_ranks())
        self.assertEqual(self.mutations, [])

    async def test_closing_session_rejects_new_demand_but_never_retries_compute(self):
        self.end_call()
        await self.completion.begin_release(generation=1)
        b = self.base
        grant = b.journal.submit('caller', 'two', b.grant['resource'], b.plan.model_id)
        call = b.journal.enqueue(grant['id'], 'caller', 'two', {'operation':'chat'})
        with self.assertRaises(Conflict):
            self.actor.note_demand(call['id'])
        self.assertEqual(b.journal.execution(call['id'], 'caller')['state'], 'queued')
        self.assertFalse(self.actor.may_start_ranks())
        self.assertEqual(sum(c[0]=='advance' for c in b.calls if isinstance(c,tuple)), 0)

    async def test_wrong_receipt_owner_size_or_exact_identity_refuses(self):
        self.end_call()
        lease = self.base.leases[self.lease_id]
        for key, value in (('agent_id','other'), ('memory_mb',999), ('exclusive',False),
                           ('ttl_s',True), ('resource_id','other/gpu-0')):
            old = lease[key]
            lease[key] = value
            with self.assertRaises(BindingRefused):
                await self.settle()
            lease[key] = old
        self.assertEqual(self.releases(), [])

    async def test_wrong_artifact_metadata_or_bytes_never_settle(self):
        self.end_call()
        for flag in ('bad_meta','bad_blob'):
            setattr(self,flag,True)
            with self.assertRaises(BindingRefused):
                await self.settle()
            setattr(self,flag,False)
        self.assertEqual(self.releases(), [])

    async def test_actual_launch_pin_is_immutable_and_not_posthoc(self):
        pinned = self.pins[self.lease_id]
        with self.assertRaises(Conflict):
            self.completion.pin_launch(replace(pinned,tolerance_bytes=999))
        self.end_call()
        await self.completion.begin_release(generation=1)
        other = list(self.pins)[1]
        self.base.journal.db.execute('DELETE FROM exo_group_cleanup_pins WHERE lease_id=?',(other,))
        with self.assertRaises(Conflict):
            self.completion.pin_launch(self.pins[other])
        with self.assertRaises(BindingRefused):
            await self.settle(other)

    async def test_cancelled_terminal_group_unblocks_local_provider(self):
        self.end_call()
        self.group.update(phase='cancelled', release_requested=True)
        for member in self.group['members']:
            member.update(status='not_requested',lease_id=None)
        self.actor.wake()
        self.assertEqual((await self.actor.wait()).phase, 'draining')
        provider = GroupProvider.__new__(GroupProvider)
        provider.actor, provider.resource = self.actor, 'claude-services-slice/gpu-0'
        self.assertIs(provider.local_permission(),True)
        self.assertFalse(provider.owns_group())
        self.assertFalse(self.actor.may_start_ranks())

    async def test_no_create_is_needed_to_close_unsubmitted_demand(self):
        self.end_call()
        # A distinct unsubmitted intent uses its existing journal and actor.
        self.base.journal.db.execute('UPDATE exo_group_intents SET group_id=NULL,create_started=0')
        self.actor.group_snapshot = None
        count = len(self.base.calls)
        result = await self.completion.begin_release(generation=1)
        self.assertEqual(result['state'],'closed_without_submission')
        self.assertFalse(any(isinstance(c,tuple) and c[0]=='create' for c in self.base.calls[count:]))

    async def test_invalid_cleanup_observations_cannot_authorize_settlement(self):
        pinned = self.pins[self.lease_id]
        raw = self.files[self.proofs[self.lease_id][0]]
        for field,value in (('exit_code',None),('group_members',[{'pid':55}]),
            ('listeners_closed',[False]),('listeners_closed',[True,True]),
            ('observed_in_use_bytes',101),('baseline_in_use_bytes',999),
            ('tolerance_bytes',1000),('observed_at','2026-11-14T10:00:00')):
            item=json.loads(raw)
            item['observation'][field]=value
            with self.assertRaises(BindingRefused):
                validate_cleanup(json.dumps(item).encode(),pinned)
        item=json.loads(raw)
        item['observation']['child']['started_microseconds']=2
        with self.assertRaises(BindingRefused):
            validate_cleanup(json.dumps(item).encode(),pinned)
        with self.assertRaises(BindingRefused):
            validate_cleanup(raw[:-1]+b',"kind":"observed_rank_cleanup"}',pinned)


class PlaneEvidenceControls(unittest.IsolatedAsyncioTestCase):
    async def test_raw_redirect_does_not_forward_owner_bearer(self):
        calls=[]
        def handle(request):
            calls.append(str(request.url))
            if request.url.path.endswith('/auth/verify'):
                return httpx.Response(200,json={'valid':True,'owner_id':'user:58'})
            return httpx.Response(302,headers={'location':'https://other.invalid/steal'})
        plane=CoordinatorPlane('user:58',lambda:'Bearer own-fixture',transport=httpx.MockTransport(handle))
        try:
            with self.assertRaises(PlaneUnavailable):
                await plane.artifact_bytes(1)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(c.startswith('https://cloud.samtg.xyz/') for c in calls))
        finally:
            await plane.close()

    async def test_oversize_artifact_and_nonexact_ids_refuse(self):
        def handle(request):
            if request.url.path.endswith('/auth/verify'):
                return httpx.Response(200,json={'valid':True,'owner_id':'user:58'})
            return httpx.Response(200,content=b'x'*(128*1024+1))
        plane=CoordinatorPlane('user:58',lambda:'Bearer own-fixture',transport=httpx.MockTransport(handle))
        try:
            with self.assertRaises(PlaneUnavailable):
                await plane.artifact_bytes(1)
            for value in (True,0,-1,'1','https://other.invalid'):
                with self.assertRaises(ValueError):
                    await plane.artifact_bytes(value)
        finally:
            await plane.close()


class ClosureRaceControls(unittest.IsolatedAsyncioTestCase):
    async def test_closing_during_owned_create_does_not_advance_or_forget_group(self):
        base=acquisition.AcquisitionControls()
        await base.asyncSetUp()
        self.addAsyncCleanup(base.asyncTearDown)
        base.group['phase']='acquiring'
        base.release.clear()
        base.actor.note_demand(base.call['id'])
        base.actor.events_reconciled()
        await base.entered.wait()
        base.journal.request_release(base.grant['id'],'caller')
        completion=GroupCompletion(base.actor)
        async def release(group_id,generation):
            base.calls.append(('release',group_id,generation))
            base.group.update(phase='cancelled',release_requested=True)
            for member in base.group['members']:
                member.update(lease_id=None,status='not_requested')
            return copy.deepcopy(base.group)
        base.actor.plane.begin_release=release
        waiter=asyncio.create_task(completion.begin_release(generation=1))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        base.release.set()
        report=await waiter
        self.assertEqual(report['state'],'group_settlement_confirmed')
        self.assertEqual(base.saved()['group_id'],base.group['id'])
        self.assertEqual(sum(c[0]=='create' for c in base.calls if isinstance(c,tuple)),1)
        self.assertEqual(sum(c[0]=='advance' for c in base.calls if isinstance(c,tuple)),0)
        self.assertFalse(base.actor.may_start_ranks())


class PreparedSettlementControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from .exo_groups.cleanup import NoStartExpectation
        from uuid import UUID
        self.parts=CompletionControls()
        await self.parts.asyncSetUp()
        self.addAsyncCleanup(self.parts.base.asyncTearDown)
        p=self.parts
        self.journal=p.base.journal
        self.journal.db.execute('DELETE FROM exo_group_cleanup_pins')
        run=p.pins[p.lease_id].binding
        self.pin=NoStartExpectation(run,str(UUID(int=1)),'a'*64,'b'*64,
            p.pins[p.lease_id].listeners,100,0,'2026-11-14T10:00:00+00:00')

    async def prepare(self):
        await self.parts.completion.pin_prepared(self.pin,generation=1)

    async def test_prepared_partial_member_settles_without_fictional_child(self):
        p=self.parts
        p.group['phase']='acquiring'
        self.journal.db.execute('UPDATE exo_group_intents SET launch_json=NULL')
        await self.prepare()
        # Inject the host's separately observed negative artifact, not the pin.
        observation={k:v for k,v in asdict(self.pin).items() if k!='listeners'}
        observation.update(launch_fenced=True,launch_intent_absent=True,launch_attempted=False,
            processes_started=0,attempt_phase='launch_fenced',listeners_closed=[True],
            observed_in_use_bytes=100,observed_at='2026-11-14T10:05:00+00:00')
        raw=json.dumps({'kind':'observed_rank_never_started',
            **asdict(self.pin.binding),'observation':observation,
            'lease_released':False,'settlement_recorded':False}).encode()
        file_id,digest=p.proofs[p.lease_id]
        p.files[file_id]=raw
        p.proofs[p.lease_id]=(file_id,hashlib.sha256(raw).hexdigest())
        p.end_call()
        result=await p.settle()
        self.assertEqual(result['state'],'exact_member_settlement_confirmed')
        self.assertEqual(len(p.releases()),1)
        self.assertNotIn('child',json.loads(raw)['observation'])
        self.assertEqual(self.journal.db.execute('SELECT COUNT(*) FROM exo_group_settlements').fetchone()[0],1)

    async def test_prepare_pin_is_immutable_and_cannot_be_learned_from_cleanup_after_drain(self):
        await self.prepare()
        with self.assertRaises(Conflict):
            await self.parts.completion.pin_prepared(replace(self.pin,attempt_sha256='c'*64),generation=1)
        self.parts.end_call()
        await self.parts.completion.begin_release(generation=1)
        self.journal.db.execute('DELETE FROM exo_group_cleanup_pins')
        with self.assertRaises(Conflict):
            await self.prepare()

    async def test_actual_start_specializes_prepared_pin_one_way_before_drain(self):
        await self.prepare()
        p=self.parts
        p.completion.pin_launch(p.pins[p.lease_id])
        self.assertIsInstance(p.completion._pin(p.lease_id),CleanupExpectation)
        with self.assertRaises(Conflict):
            await self.prepare()
        p.end_call()
        self.assertEqual((await p.settle())['state'],'exact_member_settlement_confirmed')

    async def test_foreign_preparation_or_implicit_generation_refuses(self):
        p=self.parts
        with self.assertRaises(BindingRefused):
            await p.completion.pin_prepared(replace(self.pin,
                binding=replace(self.pin.binding,runtime_sha256='f'*64)),generation=1)
        with self.assertRaises(BindingRefused):
            await p.completion.pin_prepared(self.pin,generation=True)
        with self.assertRaises(BindingRefused):
            await p.completion.pin_prepared(replace(self.pin,
                host_boot='00000000-0000-0000-0000-000000000099'),generation=1)
        self.assertEqual(self.journal.db.execute('SELECT COUNT(*) FROM exo_group_cleanup_pins').fetchone()[0],0)
