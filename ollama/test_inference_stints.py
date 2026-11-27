"""Offline conversation-affinity policy: elapsed time never cancels inference."""
import asyncio
import concurrent.futures
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .inference_arbiter import Arbiter, Conflict, NotFound, StintPolicy
from .inference_supervisor import Supervisor


class StintJournal(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'jobs.sqlite'
        self.now = 1000.0
        self.clock = patch('ollama.inference_arbiter.time.time', side_effect=lambda: self.now)
        self.clock.start()
        # Explicit experiment configuration, not a new live default.
        self.policy = StintPolicy(idle_grace_seconds=300, fairness_seconds=1800)
        self.a = Arbiter(self.path, stint_policy=self.policy)

    def tearDown(self):
        self.a.close()
        self.clock.stop()
        self.temp.cleanup()

    def queued(self, owner='creative', scope='1777/main', key='one', lifetime='work_session'):
        grant = self.a.submit(owner, scope, 'gpu', 'qwen', lifetime=lifetime)
        call = self.a.enqueue(grant['id'], owner, key, {'key': key})
        return grant, call

    def finish(self):
        item = self.a.dispatch_next('gpu')
        self.assertIsNotNone(item)
        self.a.finish(item['execution']['id'], {'answer': 'done'})
        return item

    def test_short_tools_keep_conversation_affinity(self):
        first, _ = self.queued()
        other, _ = self.queued('assistant')
        self.finish()
        self.now += 30
        self.a.enqueue(first['id'], 'creative', 'two', {})
        self.assertEqual(self.finish()['grant']['id'], first['id'])
        self.assertEqual(self.a.get(other['id'], 'assistant')['state'], 'queued')

    def test_unknown_capacity_keeps_affinity_but_blocks_incumbent_call(self):
        first, _ = self.queued()
        other, _ = self.queued('assistant')
        self.finish()
        call=self.a.enqueue(first['id'],'creative','two',{})
        self.assertIsNone(self.a.dispatch_next('gpu',permitted_model=lambda *a:None))
        self.assertEqual(self.a.get(first['id'],'creative')['state'],'granted')
        self.assertIsNone(self.a.get(first['id'],'creative')['handoff_reason'])
        self.assertEqual(self.a.execution(call['id'],'creative')['state'],'queued')
        self.assertEqual(self.a.get(other['id'],'assistant')['state'],'queued')

    def test_affirmative_model_unavailability_can_handoff_to_fitting_model(self):
        first,_=self.queued()
        self.finish()
        other=self.a.submit('vision','vision','gpu','small-vision')
        self.a.enqueue(other['id'],'vision','call',{})
        self.a.enqueue(first['id'],'creative','two',{})
        permission=lambda resource,model:model=='small-vision'
        item=self.a.dispatch_next('gpu',permitted_model=permission)
        self.assertEqual(item['grant']['id'],other['id'])
        self.assertEqual(self.a.get(first['id'],'creative')['handoff_reason'],'capacity_changed')

    def test_event_only_stints_have_no_age_deadline_and_yield_on_explicit_event(self):
        self.a.stint_policy = StintPolicy()
        first, call = self.queued()
        other, _ = self.queued('assistant')
        self.finish()
        self.now += 86400 * 3
        self.assertIsNone(self.a.next_handoff_at('gpu'))
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertEqual(self.a.get(other['id'], 'assistant')['state'], 'queued')
        self.assertEqual(self.a.execution(call['id'], 'creative')['state'], 'complete')
        self.a.request_yield(first['id'], 'creative')
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], other['id'])

    def test_lifecycle_event_is_fenced_to_its_call_not_a_later_call(self):
        first, _ = self.queued()
        self.finish()
        self.a.enqueue(first['id'], 'creative', 'second', {})
        self.a.dispatch_next('gpu')
        with self.assertRaises(Conflict):
            self.a.yield_by_key('creative', first['client_key'], 'one')
        self.assertEqual(self.a.get(first['id'], 'creative')['yield_requested'], 0)
        with self.assertRaises(NotFound):
            self.a.yield_by_key('assistant', first['client_key'], 'second')
        self.a.yield_by_key('creative', first['client_key'], 'second')
        self.assertEqual(self.a.get(first['id'], 'creative')['yield_requested'], 1)
        self.assertIsNone(self.a.dispatch_next('gpu'))

    def test_dependency_handoff_commits_with_auxiliary_and_replay_does_not_yield_again(self):
        parent, _ = self.queued()
        self.finish()
        auxiliary = self.a.submit('creative', 'summary', 'gpu', 'qwen', 'execution')
        call = self.a.enqueue(auxiliary['id'], 'creative', 'summary-call', {},
                              yield_grant_keys=(parent['client_key'],))
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], call['id'])
        self.a.finish(call['id'], {})
        main = self.a.enqueue(parent['id'], 'creative', 'next', {})
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], main['id'])
        self.a.finish(main['id'], {})
        self.a.enqueue(auxiliary['id'], 'creative', 'summary-call', {},
                       yield_grant_keys=(parent['client_key'],))
        self.assertEqual(self.a.get(parent['id'], 'creative')['yield_requested'], 0)
        with self.assertRaises(Conflict):
            self.a.enqueue(auxiliary['id'], 'creative', 'summary-call', {})

    def test_dependency_never_preempts_running_or_unknown_parent(self):
        parent, main = self.queued()
        self.a.dispatch_next('gpu')
        aux = self.a.submit('creative', 'summary', 'gpu', 'qwen', 'execution')
        self.a.enqueue(aux['id'], 'creative', 'call', {}, yield_grant_keys=(parent['client_key'],))
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.uncertain(main['id'])
        self.now += 10000
        self.assertIsNone(self.a.dispatch_next('gpu'))

    def test_dependency_cannot_yield_other_owner_or_other_resource(self):
        parent, _ = self.queued()
        self.finish()
        aux = self.a.submit('assistant', 'summary', 'gpu', 'qwen', 'execution')
        self.a.enqueue(aux['id'], 'assistant', 'call', {}, yield_grant_keys=(parent['client_key'],))
        self.assertIsNone(self.a.dispatch_next('gpu'))
        remote = self.a.submit('creative', 'remote-summary', 'other-gpu', 'vision', 'execution')
        self.a.enqueue(remote['id'], 'creative', 'call', {}, yield_grant_keys=(parent['client_key'],))
        self.assertEqual(self.a.get(parent['id'], 'creative')['yield_requested'], 0)
        self.assertEqual(self.a.dispatch_next('other-gpu')['grant']['id'], remote['id'])

    def test_invalid_dependency_rolls_back_without_yield_or_enqueue(self):
        parent, _ = self.queued()
        self.finish()
        aux = self.a.submit('creative', 'summary', 'gpu', 'qwen', 'execution')
        with self.assertRaises(Conflict):
            self.a.enqueue(aux['id'], 'creative', 'call', {},
                           yield_grant_keys=(parent['client_key'], aux['client_key']))
        self.assertEqual(self.a.get(parent['id'], 'creative')['yield_requested'], 0)
        self.assertIsNone(self.a.db.execute('SELECT 1 FROM executions WHERE grant_id=?', (aux['id'],)).fetchone())

    def test_idle_handoff_keeps_grant_and_saved_results(self):
        first, call = self.queued()
        second, _ = self.queued('assistant')
        self.finish()
        self.assertEqual(self.a.next_handoff_at('gpu'), 1300)
        self.now = 1299
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.now = 1300
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], second['id'])
        old = self.a.get(first['id'], 'creative')
        self.assertEqual((old['state'], old['handoff_reason']), ('queued', 'idle_grace'))
        self.assertEqual(self.a.execution(call['id'], 'creative')['state'], 'complete')
        self.assertEqual(self.a.enqueue(first['id'], 'creative', 'one', {'key': 'one'})['id'], call['id'])
        self.a.enqueue(first['id'], 'creative', 'two', {})

    def test_late_return_cannot_beat_an_already_waiting_conversation(self):
        first, _ = self.queued()
        other, _ = self.queued('assistant')
        self.finish()
        self.now += 301
        pending = self.a.enqueue(first['id'], 'creative', 'late', {})
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], other['id'])
        self.assertEqual(self.a.execution(pending['id'], 'creative')['state'], 'queued')

    def test_running_and_uncertain_work_never_expires(self):
        first, call = self.queued()
        self.queued('assistant')
        self.a.dispatch_next('gpu')
        self.now += 10**8
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertIsNone(self.a.next_handoff_at('gpu'))
        self.a.request_yield(first['id'], 'creative')
        self.a.uncertain(call['id'])
        self.now += 10**8
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertIsNone(self.a.next_handoff_at('gpu'))
        self.assertEqual(self.a.get(first['id'], 'creative')['state'], 'uncertain')

    def test_long_tool_explicit_yield_and_return(self):
        first, _ = self.queued()
        second, _ = self.queued('assistant')
        self.finish()
        self.a.request_yield(first['id'], 'creative')
        self.a.request_yield(first['id'], 'creative')
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], second['id'])
        pending = self.a.enqueue(first['id'], 'creative', 'after-build', {})
        active = self.a.get(second['id'], 'assistant')['active_execution']
        self.a.finish(active, {})
        self.a.request_yield(second['id'], 'assistant')
        self.assertEqual(self.a.dispatch_next('gpu')['execution']['id'], pending['id'])

    def test_yield_during_execution_drains_before_handoff(self):
        first, call = self.queued()
        second, _ = self.queued('assistant')
        self.a.dispatch_next('gpu')
        self.a.request_yield(first['id'], 'creative')
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.finish(call['id'], {})
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], second['id'])
        self.assertEqual(self.a.get(first['id'], 'creative')['handoff_reason'], 'explicit_yield')

    def test_yield_does_not_cancel_a_ready_execution(self):
        first, _ = self.queued()
        second, _ = self.queued('assistant')
        self.finish()
        pending = self.a.enqueue(first['id'], 'creative', 'two', {})
        self.a.request_yield(first['id'], 'creative')
        self.assertEqual(self.a.execution(pending['id'], 'creative')['state'], 'queued')
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], second['id'])

    def test_fairness_hands_off_only_after_backend_end(self):
        first, call = self.queued()
        # A sibling conversation of the SAME identity is a separate requester.
        sibling, _ = self.queued(scope='1788/main')
        self.a.dispatch_next('gpu')
        self.now += 1801
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.a.finish(call['id'], {})
        self.a.enqueue(first['id'], 'creative', 'two', {})
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], sibling['id'])
        self.assertEqual(self.a.get(first['id'], 'creative')['handoff_reason'], 'fair_share')

    def test_no_ready_competitor_means_no_forced_handoff(self):
        first, _ = self.queued()
        empty = self.a.submit('assistant', 'not-ready', 'gpu', 'qwen')
        self.finish()
        self.now += 10**8
        self.a.enqueue(first['id'], 'creative', 'two', {})
        self.assertEqual(self.finish()['grant']['id'], first['id'])
        self.assertEqual(self.a.get(empty['id'], 'assistant')['state'], 'queued')

    def test_empty_grant_does_not_bank_queue_priority(self):
        empty = self.a.submit('creative', 'future', 'gpu', 'qwen')
        ready, _ = self.queued('assistant')
        self.a.enqueue(empty['id'], 'creative', 'later', {})
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], ready['id'])

    def test_reads_and_saved_retries_do_not_extend_idle_grace(self):
        first, call = self.queued()
        self.queued('assistant')
        self.finish()
        self.now += 299
        self.a.get(first['id'], 'creative')
        self.a.execution(call['id'], 'creative')
        self.a.enqueue(first['id'], 'creative', 'one', {'key': 'one'})
        self.assertEqual(self.a.next_handoff_at('gpu'), 1300)

    def test_restart_preserves_idle_deadline_and_quarantines_running(self):
        first, _ = self.queued()
        other, call = self.queued('assistant')
        self.finish()
        self.a.close()
        self.now += 301
        self.a = Arbiter(self.path, stint_policy=self.policy)
        self.assertEqual(self.a.dispatch_next('gpu')['grant']['id'], other['id'])
        self.a.close()
        self.a = Arbiter(self.path, stint_policy=self.policy)
        self.a.quarantine_running()
        self.now += 10**8
        self.assertIsNone(self.a.dispatch_next('gpu'))
        self.assertEqual(self.a.execution(call['id'], 'assistant')['state'], 'uncertain')

    def test_handoff_is_atomic_across_connections(self):
        self.queued()
        other, _ = self.queued('assistant')
        self.finish()
        self.now += 301
        def dispatch(_):
            a = Arbiter(self.path, stint_policy=self.policy)
            try:
                return a.dispatch_next('gpu')
            finally:
                a.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            items = [r for r in pool.map(dispatch, range(4)) if r is not None]
        self.assertEqual([r['grant']['id'] for r in items], [other['id']])

    def test_yield_is_owner_scoped_and_not_an_execution_grant_release(self):
        grant, _ = self.queued()
        with self.assertRaises(NotFound):
            self.a.request_yield(grant['id'], 'assistant')
        execution, _ = self.queued('assistant', lifetime='execution')
        with self.assertRaises(Conflict):
            self.a.request_yield(execution['id'], 'assistant')
        with self.assertRaises(Conflict):
            self.a.admit_next('gpu')
        with self.assertRaises(Conflict):
            self.a.start(grant['id'], 'creative', 'bypass', {})

    def test_policy_requires_explicit_finite_service_bounds(self):
        for value in (0, -1, True, float('nan'), float('inf'), '300'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StintPolicy(value, 1800)
            with self.subTest(value=value), self.assertRaises(ValueError):
                StintPolicy(300, value)


class AutomaticStintWakeup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.allowed = True
        self.entered = []
        self.gate = asyncio.Event()
        self.gate.set()
        async def execute(grant, body):
            self.entered.append(grant['owner'])
            await self.gate.wait()
            return body
        self.s = Supervisor(Path(self.temp.name) / 'jobs.sqlite', execute,
            lambda _: self.allowed, stint_policy=StintPolicy(.04, 1))

    async def asyncTearDown(self):
        self.allowed = False
        self.gate.set()
        await asyncio.gather(*list(self.s.workers.values()))
        self.s.close()
        self.temp.cleanup()

    def queued(self, owner):
        grant = self.s.journal.submit(owner, 'conversation/main', 'gpu', 'qwen')
        call = self.s.journal.enqueue(grant['id'], owner, 'one', {})
        return grant, call

    async def test_idle_timer_hands_off_without_a_new_http_request(self):
        first, a = self.queued('creative')
        second, b = self.queued('assistant')
        self.s.pump('gpu')
        await asyncio.wait_for(self.s.observe(b['id'], 'assistant'), 2)
        self.assertEqual(self.entered, ['creative', 'assistant'])
        self.assertEqual(self.s.journal.get(first['id'], 'creative')['state'], 'queued')
        self.assertEqual(self.s.journal.execution(a['id'], 'creative')['state'], 'complete')

    async def test_timer_rechecks_capacity_permission_before_dispatch(self):
        self.queued('creative')
        _, b = self.queued('assistant')
        self.s.pump('gpu')
        async with asyncio.timeout(2):
            while not self.s.wakeups:
                await asyncio.sleep(.001)
        self.allowed = False
        await asyncio.sleep(.08)
        self.assertEqual(self.entered, ['creative'])
        self.assertEqual(self.s.journal.execution(b['id'], 'assistant')['state'], 'queued')
        self.allowed = True
        self.s.pump('gpu')  # owner signals capacity is available again
        await asyncio.wait_for(self.s.observe(b['id'], 'assistant'), 2)

    async def test_observer_cancellation_and_yield_do_not_stop_worker(self):
        self.gate.clear()
        first, a = self.queued('creative')
        _, b = self.queued('assistant')
        self.s.pump('gpu')
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(self.s.observe(a['id'], 'creative'), .02)
        self.s.yield_grant(first['id'], 'creative')
        await asyncio.sleep(.08)
        self.assertEqual(self.entered, ['creative'])
        self.assertEqual(self.s.journal.execution(a['id'], 'creative')['state'], 'running')
        self.gate.set()
        await asyncio.wait_for(self.s.observe(b['id'], 'assistant'), 2)
        self.assertEqual(self.entered, ['creative', 'assistant'])


if __name__ == '__main__':
    unittest.main()
