"""Real journal transitions; temporary SQLite, no models or physical claims."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .inference_arbiter import Arbiter, Conflict, NotFound, StintPolicy
from .test_shared_resource_handoff import SharedResourceHandoff


class RequesterJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'jobs.sqlite'
        self.a = Arbiter(self.path, stint_policy=StintPolicy(idle_grace_seconds=1, fairness_seconds=1))

    def tearDown(self):
        self.a.close()
        self.tmp.cleanup()

    def call(self, owner, grant_key, call_key, *, job=None, model='text', resource='gpu',
             lifetime='execution'):
        grant = self.a.submit(owner, grant_key, resource, model, lifetime)
        call = self.a.enqueue(grant['id'], owner, call_key, {'fixture': call_key}, job_key=job)
        return grant, call

    def dispatch(self, expected, resource='gpu'):
        item = self.a.dispatch_next(resource)
        self.assertIsNotNone(item)
        self.assertEqual(item['execution']['id'], expected['id'])
        return item

    def test_text_compaction_vision_text_share_one_place_until_parent_finishes(self):
        root, main = self.call('dev', 'main', 'm1', job='turn-1', lifetime='work_session')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        other, waiting = self.call('other', 'other', 'waiting')
        for name, model in (('compaction', 'text'), ('vision', 'vision')):
            _, child = self.call('dev', name, name, job='turn-1', model=model)
            self.dispatch(child)
            self.a.finish(child['id'], {})
        _, followup = self.call('dev', 'main', 'm2', job='turn-1', lifetime='work_session')
        with patch('ollama.inference_arbiter.time.time', return_value=10**12):
            self.assertIsNone(self.a.next_handoff_at('gpu'))
            self.dispatch(followup)
            self.a.finish(followup['id'], {})
            self.assertIsNone(self.a.dispatch_next('gpu'))  # long tool gap holds its place
        self.a.end_job('dev', 'turn-1')
        self.dispatch(waiting)

    def test_same_identity_independent_job_does_not_inherit_ownership(self):
        _, main = self.call('seat', 'first', 'first', job='first')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        _, independent = self.call('seat', 'second', 'second', job='independent')
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.end_job('seat', 'first')
        self.dispatch(independent)

    def test_other_owner_cannot_end_or_attach_to_an_owned_job(self):
        _, main = self.call('one', 'one', 'one', job='same-key')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        self.assertEqual(self.a.end_job('two', 'same-key', finish=False), [])
        _, other = self.call('two', 'two', 'two', job='same-key')
        self.assertNotEqual(main['job_id'], other['job_id'])
        self.assertIsNone(self.a.dispatch_next('gpu'))

    def test_replay_closed_job_finds_same_execution_without_reacquiring(self):
        grant, main = self.call('one', 'one', 'one', job='job')
        self.dispatch(main)
        self.a.finish(main['id'], {'value': 'kept'})
        self.a.end_job('one', 'job')
        replay = self.a.enqueue(grant['id'], 'one', 'one', {'fixture': 'one'}, job_key='job')
        self.assertEqual(replay['id'], main['id'])
        self.assertEqual(replay['state'], 'complete')
        self.assertFalse(self.a.owns_job('gpu'))
        another = self.a.submit('one', 'new-grant', 'gpu', 'text')
        with self.assertRaises(Conflict):
            self.a.enqueue(another['id'], 'one', 'new-call', {}, job_key='job')

    def test_legacy_receipt_cannot_be_reinterpreted_as_job_work(self):
        grant, old = self.call('one', 'one', 'one')
        with self.assertRaises(Conflict):
            self.a.enqueue(grant['id'], 'one', 'one', {'fixture': 'one'}, job_key='invented')
        self.assertIsNone(self.a.execution(old['id'], 'one')['job_id'])
        self.assertEqual(self.a.db.execute('SELECT count(*) FROM requester_jobs').fetchone()[0], 0)

    def test_changed_job_on_same_key_conflicts_without_mutating_intake(self):
        grant, main = self.call('one', 'one', 'one', job='first')
        with self.assertRaises(Conflict):
            self.a.enqueue(grant['id'], 'one', 'one', {'fixture': 'one'}, job_key='second')
        self.assertEqual(self.a.execution(main['id'], 'one')['job_id'], main['job_id'])

    def test_restart_quarantines_child_and_keeps_parent_until_confirmed_end(self):
        _, main = self.call('one', 'main', 'main', job='job')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        _, child = self.call('one', 'vision', 'vision', job='job', model='vision')
        self.dispatch(child)
        _, other = self.call('two', 'other', 'other')
        self.a.close()
        self.a = Arbiter(self.path, stint_policy=StintPolicy())
        self.a.quarantine_running()
        self.a.end_job('one', 'job')
        self.assertTrue(self.a.owns_job('gpu'))
        self.assertEqual(self.a.execution(child['id'], 'one')['state'], 'uncertain')
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.finish(child['id'], {'reconciled': True})
        self.assertFalse(self.a.owns_job('gpu'))
        self.dispatch(other)

    def test_finish_drains_accepted_child_and_rejects_new_children(self):
        _, main = self.call('one', 'main', 'main', job='job')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        _, other = self.call('two', 'other', 'other')
        _, child = self.call('one', 'child', 'child', job='job')
        self.a.end_job('one', 'job')
        refused = self.a.submit('one', 'refused', 'gpu', 'text')
        with self.assertRaises(Conflict):
            self.a.enqueue(refused['id'], 'one', 'refused', {}, job_key='job')
        self.dispatch(child)
        self.a.finish(child['id'], {})
        self.assertEqual(self.a.job('one', 'job', 'gpu')['state'], 'closed')
        self.dispatch(other)

    def test_explicit_yield_preserves_job_identity_and_lets_other_owner_run(self):
        _, main = self.call('one', 'main', 'main', job='job')
        ident = main['job_id']
        self.dispatch(main)
        self.a.finish(main['id'], {})
        _, other = self.call('two', 'other', 'other')
        self.a.end_job('one', 'job', finish=False)
        _, resumed = self.call('one', 'resumed', 'resumed', job='job')
        self.assertEqual(resumed['job_id'], ident)
        self.dispatch(other)
        self.a.finish(other['id'], {})
        self.dispatch(resumed)

    def test_cancelled_queued_child_does_not_leave_finish_stuck(self):
        _, main = self.call('one', 'main', 'main', job='job')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        grant, _ = self.call('one', 'child', 'child', job='job')
        self.a.end_job('one', 'job')
        self.a.request_release(grant['id'], 'one')
        self.assertFalse(self.a.owns_job('gpu'))
        self.assertEqual(self.a.job('one', 'job', 'gpu')['state'], 'closed')

    def test_model_capacity_refusal_does_not_give_parent_place_to_other_owner(self):
        _, main = self.call('one', 'main', 'main', job='job')
        self.dispatch(main)
        self.a.finish(main['id'], {})
        self.call('two', 'other', 'other')
        _, child = self.call('one', 'child', 'child', job='job', model='vision')
        self.assertIsNone(self.a.dispatch_next('gpu', permitted_model=lambda _, model: model == 'text'))
        self.assertTrue(self.a.owns_job('gpu'))
        self.assertEqual(self.a.execution(child['id'], 'one')['state'], 'queued')

    def test_cross_resource_work_does_not_lock_the_other_resource(self):
        _, one = self.call('one', 'one', 'one', job='job', resource='gpu-one')
        _, two = self.call('one', 'two', 'two', job='job', resource='gpu-two')
        self.dispatch(one, 'gpu-one')
        self.dispatch(two, 'gpu-two')
        self.assertNotEqual(one['job_id'], two['job_id'])
        self.a.end_job('one', 'job')
        self.a.finish(one['id'], {})
        self.assertFalse(self.a.owns_job('gpu-one'))
        self.assertTrue(self.a.owns_job('gpu-two'))

    def test_scheduler_downgrade_refuses_unfinished_job_without_rebinding(self):
        _, call = self.call('one', 'one', 'one', job='job')
        self.dispatch(call)
        self.a.finish(call['id'], {})
        with self.assertRaises(Conflict):
            Arbiter(self.path)  # no job-aware atomic policy
        self.assertTrue(self.a.owns_job('gpu'))
        self.a.end_job('one', 'job')
        old = Arbiter(self.path)
        try:
            self.assertEqual(old.execution(call['id'], 'one')['id'], call['id'])
        finally:
            old.close()

    def test_end_before_intake_seals_job_and_never_creates_capacity_ownership(self):
        self.assertEqual(self.a.end_job('one', 'late'), [])
        self.assertFalse(self.a.owns_job('gpu'))
        grant = self.a.submit('one', 'late-grant', 'gpu', 'text')
        with self.assertRaises(Conflict):
            self.a.enqueue(grant['id'], 'one', 'late-call', {}, job_key='late')
        self.assertEqual(self.a.db.execute('SELECT count(*) FROM executions').fetchone()[0], 0)

    def test_finished_parent_cannot_reopen_on_previously_unseen_resource(self):
        _, call = self.call('one', 'main', 'main', job='job')
        self.dispatch(call)
        self.a.finish(call['id'], {})
        self.a.end_job('one', 'job')
        grant = self.a.submit('one', 'elsewhere', 'another-gpu', 'text')
        with self.assertRaises(Conflict):
            self.a.enqueue(grant['id'], 'one', 'new-child', {}, job_key='job')

    def test_delayed_old_finish_cannot_end_new_job_on_same_conversation_grant(self):
        _, first = self.call('one', 'conversation', 'first', job='first', lifetime='work_session')
        self.dispatch(first)
        self.a.finish(first['id'], {})
        self.a.end_job('one', 'first')
        _, next_call = self.call('one', 'conversation', 'next', job='next', lifetime='work_session')
        self.dispatch(next_call)
        self.a.end_job('one', 'first')
        self.assertEqual(self.a.job('one', 'next', 'gpu')['state'], 'owned')
        self.assertEqual(self.a.execution(next_call['id'], 'one')['state'], 'running')
        self.a.finish(next_call['id'], {})
        self.assertTrue(self.a.owns_job('gpu'))


class PhysicalJobOwnership(SharedResourceHandoff):
    def own_idle_parent_with_absent_model(self):
        self.journal.stint_policy = StintPolicy()
        grant = self.journal.submit('dev', 'main', 'gpu', 'qwen', 'work_session')
        call = self.journal.enqueue(grant['id'], 'dev', 'main', {}, job_key='turn')
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'], {})
        self.models.clear()
        self.running = []

    async def test_owned_parent_reconciles_absent_handed_off_reservation(self):
        self.own_idle_parent_with_absent_model()
        self.assertEqual(self.initial.snapshot('qwen')['state'], 'handed_off')
        self.assertEqual((await self.actor.begin())['phase'], 'idle')
        self.assertEqual(self.initial.snapshot('qwen')['state'], 'retired')
        self.client.release_lease.assert_called_once_with('owned')
        self.manager.unload.assert_not_called()
        self.assertTrue(self.journal.owns_job('gpu'))

    async def test_owned_parent_retains_unknown_allocation_and_blocks_dispatch(self):
        self.own_idle_parent_with_absent_model()
        with self.initial._owned() as db:
            db.execute("UPDATE initial_reservations SET state='unknown',lease_id=NULL")
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.assertEqual(self.initial.snapshot('qwen')['state'], 'unknown')
        self.client.release_lease.assert_not_called()
        self.assertFalse(self.capacity.permitted('gpu'))
        self.assertTrue(self.journal.owns_job('gpu'))

    async def test_owned_parent_does_not_infer_absence_from_missing_inventory(self):
        self.own_idle_parent_with_absent_model()
        self.running = [{'name': 'qwen'}]
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.assertEqual(self.initial.snapshot('qwen')['state'], 'handed_off')
        self.client.release_lease.assert_not_called()
        self.assertTrue(self.journal.owns_job('gpu'))

    async def test_owned_parent_recovers_lost_release_reply_on_same_reservation(self):
        self.own_idle_parent_with_absent_model()
        def lost_reply(lease_id):
            self.release(lease_id)
            raise OSError('lost exact DELETE receipt')
        self.client.release_lease.side_effect = lost_reply
        self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.assertEqual(self.initial.snapshot('qwen')['state'], 'retiring')
        self.assertEqual((await self.actor.begin())['phase'], 'idle')
        self.assertEqual(self.initial.snapshot('qwen')['state'], 'retired')
        self.client.release_lease.assert_called_once_with('owned')
        self.assertTrue(self.journal.owns_job('gpu'))

    async def test_retained_recovery_requires_fresh_exact_active_registry_charge(self):
        from types import SimpleNamespace
        model = SimpleNamespace(name='qwen', lease_id='owned', memory_mb=100)
        self.models['qwen'] = model
        self.actor._confirm_retained_vision(model)
        original = dict(self.row)
        for change in ({'id': 'different'}, {'agent_id': 'other'}, {'status': 'queued'},
                       {'status': 'expired'}, {'memory_mb': 99},
                       {'expires_at': '2000-01-01T00:00:00Z'}):
            with self.subTest(change=change):
                self.row.clear(); self.row.update(original | change)
                with self.assertRaises(Conflict):
                    self.actor._confirm_retained_vision(model)
        self.client.release_lease.assert_not_called()

    async def test_dead_vision_child_recovers_inside_owned_claim_without_handoff(self):
        from .manager import VLM_MODELS
        from types import SimpleNamespace
        from unittest.mock import Mock
        vision = next(iter(VLM_MODELS.values()))['default']
        self.journal.stint_policy = StintPolicy()
        grant = self.journal.submit('dev', 'main', 'gpu', vision, 'work_session')
        call = self.journal.enqueue(grant['id'], 'dev', 'main', {}, job_key='turn')
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'], {})
        child = Mock(); child.poll.return_value = 1
        model = SimpleNamespace(name=vision, lease_id='owned', memory_mb=100,
                                managed=True, vlm_process=child)
        self.models.clear(); self.models[vision] = model
        self.runtime.settings = SimpleNamespace(models={vision: 'gpu'})
        def lost_readiness(mm, *, confirm_admission, resume):
            self.assertFalse(resume)
            confirm_admission()
            child.poll.return_value = None
            raise RuntimeError('health observation lost after replacement spawn')
        with patch('ollama.vision_lifecycle.recover_owned_vision', side_effect=lost_readiness):
            self.assertEqual((await self.actor.begin())['phase'], 'uncertain')
        self.assertEqual(self.actor.snapshot()['recovering'], [vision])
        self.client.release_lease.assert_not_called()
        self.actor = type(self.actor)(self.runtime)  # durable intent survives actor recreation
        def recover(mm, *, confirm_admission, resume):
            self.assertIs(mm, model)
            self.assertTrue(resume)
            confirm_admission()
            child.poll.return_value = None
        with patch('ollama.vision_lifecycle.recover_owned_vision', side_effect=recover, create=True) as restart:
            result = await self.actor.begin()
        restart.assert_called_once()
        self.assertEqual(result['phase'], 'idle')
        self.manager.unload.assert_not_called()
        self.client.release_lease.assert_not_called()
        self.assertEqual(self.models[vision].lease_id, 'owned')
        self.assertTrue(self.journal.owns_job('gpu'))

    async def test_automatic_cooldown_keeps_parent_residency_across_long_tool_gap(self):
        from .manager import ModelManager
        from .inference_capacity_source import ManagerCapacitySource
        from types import SimpleNamespace
        from unittest.mock import Mock
        self.journal.stint_policy = StintPolicy()
        grant = self.journal.submit('dev', 'main', 'gpu', 'qwen', 'work_session')
        call = self.journal.enqueue(grant['id'], 'dev', 'main', {}, job_key='turn')
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'], {})
        source = object.__new__(ManagerCapacitySource)
        source.runtime = self.runtime
        manager = object.__new__(ModelManager)
        manager.models = {'qwen': SimpleNamespace(managed=True, in_flight=0, last_used=0)}
        manager.durable_inference = True
        manager.inference_capacity_source = source
        manager.offer_reading = Mock(return_value={'memory_device_inuse_mb': 0})
        manager.own_device_mb = Mock(return_value=0)
        manager.work_in_progress = Mock(return_value=True)
        manager.unload = Mock(return_value={'status': 'unloaded'})
        self.assertEqual(manager.check_cooldowns(), [])
        manager.unload.assert_not_called()
        self.journal.end_job('dev', 'turn')
        self.assertEqual(manager.check_cooldowns(), [{'status': 'unloaded'}])
        manager.unload.assert_called_once_with('qwen')

    async def test_foreign_waiter_cannot_end_an_owned_parent_between_calls(self):
        self.journal.stint_policy = StintPolicy()
        grant = self.journal.submit('dev', 'main', 'gpu', 'qwen', 'work_session')
        call = self.journal.enqueue(grant['id'], 'dev', 'main', {}, job_key='turn')
        self.journal.dispatch_next('gpu')
        self.journal.finish(call['id'], {})
        await self.actor.begin()
        self.manager.unload.assert_not_called()
        self.client.release_lease.assert_not_called()
        self.assertTrue(self.journal.owns_job('gpu'))
        self.journal.end_job('dev', 'turn')
        await self.actor.begin()
        self.manager.unload.assert_called_once()
        self.client.release_lease.assert_called_once_with('owned')


if __name__ == '__main__':
    unittest.main()
