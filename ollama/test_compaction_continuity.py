"""Offline parent -> compaction -> parent scheduling regressions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import inference_arbiter as arbiter


class CompactionContinuity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'queue.sqlite'
        self.a = arbiter.Arbiter(self.path, stint_policy=arbiter.StintPolicy())

    def tearDown(self):
        self.a.close()
        self.temp.cleanup()

    def submit(self, owner, key, lifetime='work_session'):
        return self.a.submit(owner, key, 'gpu', 'model', lifetime)

    def initial_parent(self):
        p = self.submit('hs', 'native-turn')
        e = self.a.enqueue(p['id'], 'hs', 'main-1', {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], e['id'])
        self.a.finish(e['id'], {'done': True})
        return p

    def summary(self, parent):
        s = self.submit('hs', 'summary-1', 'execution')
        e = self.a.enqueue(s['id'], 'hs', 'summary-call', {},
                           yield_grant_keys=(parent['client_key'],))
        return s, e

    def competitor(self):
        other = self.submit('producer', 'other-turn')
        e = self.a.enqueue(other['id'], 'producer', 'other-call', {})
        return other, e

    def test_enter_compaction_keeps_incumbent_position(self):
        p = self.initial_parent()
        self.competitor()
        _, summary = self.summary(p)
        next_call = self.a.dispatch_next('gpu')
        self.assertEqual(next_call['execution']['id'], summary['id'],
                         'dependent summary must inherit its incumbent parent position')

    def test_summary_completion_restores_parent_before_next_submit(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], summary['id'])
        self.competitor()
        self.a.finish(summary['id'], {'summary': 'finished'})
        parent = self.a.get(p['id'], 'hs')
        self.assertEqual(parent['state'], 'granted',
                         'continuation must survive the gap before the parent next call')
        self.assertIsNone(parent['active_execution'])
        self.assertIsNone(self.a.dispatch_next('gpu'),
                          'a competitor must not enter during summary handback')

    def test_parent_resume_stays_ahead_of_waiting_competitor(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.finish(summary['id'], {'summary': 'finished'})
        continuation = self.a.enqueue(p['id'], 'hs', 'main-2', {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], continuation['id'],
                         'compaction completion must hand scheduling affinity back to parent')

    def test_ordinary_long_tool_yield_still_returns_position(self):
        p = self.initial_parent()
        _, competitor = self.competitor()
        self.a.request_yield(p['id'], 'hs')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], competitor['id'])


    def test_return_survives_journal_reopen(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.finish(summary['id'], {})
        self.a.close()
        self.a = arbiter.Arbiter(self.path, stint_policy=arbiter.StintPolicy())
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'granted')
        self.assertIsNone(self.a.dispatch_next('gpu'))
        call = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], call['id'])

    def test_uncertain_summary_never_resumes_parent(self):
        p = self.initial_parent()
        child, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.uncertain(summary['id'])
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'queued')
        self.assertEqual(self.a.get(child['id'], 'hs')['state'], 'uncertain')

    def test_terminal_failure_returns_parent_without_resampling(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.finish(summary['id'], {}, failed=True)
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'granted')
        self.assertEqual(self.a.execution(summary['id'], 'hs')['state'], 'failed')
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertEqual(self.a.db.execute('SELECT count(*) FROM executions').fetchone()[0], 3)

    def test_released_parent_is_not_revived(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        _, other = self.competitor()
        self.a.request_release(p['id'], 'hs')
        self.a.finish(summary['id'], {})
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'cancelled')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_explicit_yield_during_summary_waives_return(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        _, other = self.competitor()
        self.a.request_yield(p['id'], 'hs')
        self.a.finish(summary['id'], {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_cancel_never_started_summary_keeps_parent(self):
        p = self.initial_parent()
        child, _ = self.summary(p)
        self.competitor()
        self.a.request_release(child['id'], 'hs')
        call = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], call['id'])

    def test_replay_old_summary_cannot_yield_later_parent(self):
        p = self.initial_parent()
        child, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.a.finish(summary['id'], {})
        call = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.a.dispatch_next('gpu')
        self.a.finish(call['id'], {})
        self.competitor()
        replay = self.a.enqueue(child['id'], 'hs', 'summary-call', {},
                                yield_grant_keys=(p['client_key'],))
        self.assertEqual(replay['id'], summary['id'])
        self.assertEqual(self.a.get(p['id'], 'hs')['yield_requested'], 0)
        self.assertIsNone(self.a.dispatch_next('gpu'))

    def test_explicit_yield_before_summary_uses_ordinary_fifo(self):
        p = self.initial_parent()
        _, other = self.competitor()
        self.summary(p)
        self.a.request_yield(p['id'], 'hs')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_running_and_uncertain_parent_block_summary(self):
        p = self.submit('hs', 'native-turn')
        call = self.a.enqueue(p['id'], 'hs', 'main', {})
        self.a.dispatch_next('gpu')
        self.summary(p)
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.uncertain(call['id'])
        self.assertIsNone(self.a.dispatch_next('gpu'))

    def test_capacity_deferral_keeps_same_child(self):
        p = self.initial_parent()
        self.competitor()
        _, summary = self.summary(p)
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], summary['id'])
        self.a.defer_capacity(summary['id'], 'capacity_unavailable')
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.capacity_available('gpu')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], summary['id'])
        self.a.finish(summary['id'], {})
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'granted')

    def test_return_credit_then_normal_fairness(self):
        now = [1000.0]
        with patch.object(arbiter.time, 'time', side_effect=lambda: now[0]):
            self.a.stint_policy = arbiter.StintPolicy(fairness_seconds=10)
            p = self.initial_parent()
            _, other = self.competitor()
            _, summary = self.summary(p)
            self.a.dispatch_next('gpu')
            now[0] += 100
            self.a.finish(summary['id'], {})
            call = self.a.enqueue(p['id'], 'hs', 'resume', {})
            self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], call['id'])
            self.a.finish(call['id'], {})
            self.a.enqueue(p['id'], 'hs', 'after-tool', {})
            self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_deferred_summary_precedes_an_already_queued_parent_call(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.a.defer_capacity(summary['id'], 'capacity_unavailable')
        self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.a.capacity_available('gpu')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], summary['id'])

    def test_unknown_summary_capacity_keeps_incumbent(self):
        p = self.initial_parent()
        self.competitor()
        self.summary(p)
        self.assertIsNone(self.a.dispatch_next('gpu', permitted_model=lambda *args: None))
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'granted')

    def test_idle_return_still_expires_under_existing_policy(self):
        now = [1000.0]
        with patch.object(arbiter.time, 'time', side_effect=lambda: now[0]):
            self.a.stint_policy = arbiter.StintPolicy(idle_grace_seconds=300)
            p = self.initial_parent()
            _, summary = self.summary(p)
            self.a.dispatch_next('gpu')
            _, other = self.competitor()
            self.a.finish(summary['id'], {})
            now[0] += 301
            self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_second_connection_cannot_dispatch_during_return_gap(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.finish(summary['id'], {})
        other = arbiter.Arbiter(self.path, stint_policy=arbiter.StintPolicy())
        try:
            self.assertIsNone(other.dispatch_next('gpu'))
            self.assertEqual(other.get(p['id'], 'hs')['state'], 'granted')
        finally:
            other.close()

    def test_cancel_deferred_child_cannot_displace_an_admitted_competitor(self):
        p = self.initial_parent()
        child = self.a.submit('hs', 'summary-1', 'gpu', 'summary-model', 'execution')
        summary = self.a.enqueue(child['id'], 'hs', 'summary-call', {},
                                 yield_grant_keys=(p['client_key'],))
        self.a.dispatch_next('gpu')
        self.a.defer_capacity(summary['id'], 'capacity_unavailable')
        _, other = self.competitor()
        self.a.capacity_available('gpu')
        permission = lambda resource, model: model != 'summary-model'
        self.assertEqual(self.a.dispatch_next('gpu', permitted_model=permission)['execution']['id'], other['id'])
        self.a.request_release(child['id'], 'hs')
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'queued')
        self.assertEqual(self.a.execution(other['id'], 'producer')['state'], 'running')
        self.assertEqual(self.a.execution(summary['id'], 'hs')['state'], 'cancelled')

    def test_return_wakeup_uses_idle_grace_not_expired_fairness(self):
        now = [1000.0]
        with patch.object(arbiter.time, 'time', side_effect=lambda: now[0]):
            self.a.stint_policy = arbiter.StintPolicy(idle_grace_seconds=300, fairness_seconds=10)
            p = self.initial_parent()
            _, summary = self.summary(p)
            self.a.dispatch_next('gpu')
            self.competitor()
            now[0] += 100
            self.a.finish(summary['id'], {})
            self.assertEqual(self.a.next_handoff_at('gpu'), 1400)

    def test_audited_terminal_child_restores_parent_on_next_dispatch(self):
        p = self.initial_parent()
        child, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.uncertain(summary['id'])
        self.assertIsNone(self.a.dispatch_next('gpu'))
        # Simulate the terminal projection made by a separately fenced operator;
        # dispatch never performs a recovery or manufactures backend-death proof.
        with self.a.transaction():
            self.a.db.execute("UPDATE executions SET state='failed' WHERE id=?", (summary['id'],))
            self.a.db.execute("UPDATE grants SET state='released',active_execution=NULL WHERE id=?", (child['id'],))
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'granted')

    def test_additive_schema_opens_pre_continuation_journal(self):
        p = self.initial_parent()
        with self.a.transaction():
            self.a.db.execute('DROP TABLE dependency_continuations')
        self.a.close()
        self.a = arbiter.Arbiter(self.path, stint_policy=arbiter.StintPolicy())
        self.assertEqual(self.a.get(p['id'], 'hs')['state'], 'granted')
        _, summary = self.summary(p)
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], summary['id'])

    def test_return_keeps_order_through_physical_capacity_loss(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        _, other = self.competitor()
        self.a.finish(summary['id'], {})
        before = self.a.get(p['id'], 'hs')
        resume = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.assertIsNone(self.a.dispatch_next('gpu', permitted_model=lambda *args: False))
        waiting = self.a.get(p['id'], 'hs')
        self.assertEqual(waiting['queue_order'], before['queue_order'])
        self.assertEqual(waiting['stint_started'], before['stint_started'])
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], resume['id'])

    def test_return_gap_survives_capacity_loss_before_parent_post(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.finish(summary['id'], {})
        before = self.a.get(p['id'], 'hs')
        self.assertIsNone(self.a.dispatch_next('gpu', permitted_model=lambda *args: False))
        resume = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.assertEqual(self.a.get(p['id'], 'hs')['queue_order'], before['queue_order'])
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], resume['id'])

    def test_capacity_wait_can_run_available_other_model_without_preemption(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        other = self.a.submit('producer', 'other-turn', 'gpu', 'other-model', 'execution')
        other_call = self.a.enqueue(other['id'], 'producer', 'other-call', {})
        self.a.finish(summary['id'], {})
        resume = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.assertEqual(self.a.dispatch_next('gpu', permitted_model=lambda r,m: m == 'other-model')['execution']['id'], other_call['id'])
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.finish(other_call['id'], {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], resume['id'])

    def test_capacity_gap_without_ready_call_does_not_extend_idle_grace(self):
        now = [1000.0]
        with patch.object(arbiter.time, 'time', side_effect=lambda: now[0]):
            self.a.stint_policy = arbiter.StintPolicy(idle_grace_seconds=10)
            p = self.initial_parent()
            _, summary = self.summary(p)
            self.a.dispatch_next('gpu')
            _, other = self.competitor()
            self.a.finish(summary['id'], {})
            self.a.dispatch_next('gpu', permitted_model=lambda *args: False)
            now[0] += 11
            self.a.enqueue(p['id'], 'hs', 'late-resume', {})
            self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_ready_return_wait_does_not_reset_fairness_clock(self):
        now = [1000.0]
        with patch.object(arbiter.time, 'time', side_effect=lambda: now[0]):
            self.a.stint_policy = arbiter.StintPolicy(fairness_seconds=10)
            p = self.initial_parent()
            _, summary = self.summary(p)
            self.a.dispatch_next('gpu')
            _, other = self.competitor()
            self.a.finish(summary['id'], {})
            resume = self.a.enqueue(p['id'], 'hs', 'resume', {})
            self.a.dispatch_next('gpu', permitted_model=lambda *args: False)
            now[0] += 100
            self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], resume['id'])
            self.a.finish(resume['id'], {})
            self.a.enqueue(p['id'], 'hs', 'after-tool', {})
            self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])

    def test_capacity_wait_return_survives_reopen(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        self.competitor()
        self.a.finish(summary['id'], {})
        self.a.dispatch_next('gpu', permitted_model=lambda *args: False)
        self.a.close()
        self.a = arbiter.Arbiter(self.path, stint_policy=arbiter.StintPolicy())
        resume = self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], resume['id'])

    def test_explicit_yield_waives_capacity_wait_return_position(self):
        p = self.initial_parent()
        _, summary = self.summary(p)
        self.a.dispatch_next('gpu')
        _, other = self.competitor()
        self.a.finish(summary['id'], {})
        self.a.enqueue(p['id'], 'hs', 'resume', {})
        self.a.dispatch_next('gpu', permitted_model=lambda *args: False)
        self.a.request_yield(p['id'], 'hs')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], other['id'])


if __name__ == "__main__":
    unittest.main()
