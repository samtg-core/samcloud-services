"""Completion accounting uses real transition metadata, never busy time."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .inference_arbiter import Arbiter, StintPolicy
from .inference_health import read_report


class Health(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'jobs.sqlite'
        self.now = 100
        self.clock = patch('ollama.inference_arbiter.time.time', side_effect=lambda:self.now)
        self.clock.start()
        self.a = Arbiter(self.path, stint_policy=StintPolicy())

    def tearDown(self):
        self.a.close()
        self.clock.stop()
        self.tmp.cleanup()

    def enqueue(self, key):
        grant = self.a.submit('owner', key, 'gpu', 'qwen', 'execution')
        call = self.a.enqueue(grant['id'], 'owner', key, {'prompt':'PRIVATE'})
        return grant, call

    def result(self, **bounds):
        return read_report(self.path, **({'start':100, 'end':200, 'as_of':200}|bounds))

    def test_open_requests_are_in_denominator_and_retries_count_once(self):
        g, first = self.enqueue('one')
        self.a.enqueue(g['id'], 'owner', 'one', {'prompt':'PRIVATE'})
        self.enqueue('two')
        self.now = 120
        self.a.dispatch_next('gpu')
        self.now = 130
        self.a.finish(first['id'], {'answer':'SECRET'})
        result = self.result()
        self.assertEqual(result['requests']['accepted'], 2)
        self.assertEqual(result['requests']['completed_percent'], 50)
        self.assertEqual(result['requests']['open_or_unobserved'], 1)
        self.assertEqual(result['queue_wait_seconds_dispatched_only']['total'], 20)
        self.assertNotIn('PRIVATE', str(result))
        self.assertNotIn('SECRET', str(result))

    def test_failure_is_terminal_but_not_success_and_uncertain_is_open(self):
        _, first = self.enqueue('one')
        self.enqueue('two')
        self.a.dispatch_next('gpu')
        self.a.finish(first['id'], {}, failed=True)
        item = self.a.dispatch_next('gpu')
        self.a.uncertain(item['execution']['id'])
        self.a.uncertain(item['execution']['id'])
        result = self.result()
        self.assertEqual(result['requests']['terminal_percent'], 50)
        self.assertEqual(result['requests']['completed_percent'], 0)
        self.assertEqual(result['requests']['states']['uncertain'], 1)
        self.assertIsNone(result['watchdog_cancellations'])

    def test_historical_cutoff_does_not_leak_later_completion(self):
        _, call = self.enqueue('one')
        self.now = 110
        self.a.dispatch_next('gpu')
        self.now = 210
        self.a.finish(call['id'], {})
        self.assertEqual(self.result()['requests']['states'], {'running':1})
        self.assertEqual(self.result(as_of=220)['requests']['states'], {'complete':1})

    def test_end_exclusive_and_carry_in_are_separate(self):
        self.now = 90
        self.enqueue('carry')
        self.now = 100
        self.enqueue('in')
        self.now = 200
        self.enqueue('out')
        result = self.result()
        self.assertEqual(result['requests']['accepted'], 1)
        self.assertEqual(result['carry_in_states_as_of'], {'queued':1})

    def test_cancelled_is_not_completed(self):
        grant, _ = self.enqueue('cancel')
        self.a.request_release(grant['id'], 'owner')
        result = self.result()['requests']
        self.assertEqual(result['terminal_percent'], 100)
        self.assertEqual(result['completed_percent'], 0)

    def test_capacity_deferral_is_one_open_request_then_wait_not_service(self):
        _, call = self.enqueue('capacity')
        self.now = 110
        self.a.dispatch_next('gpu')
        self.now = 112
        self.a.defer_capacity(call['id'], 'capacity_unavailable')
        waiting = self.result()['requests']
        self.assertEqual(waiting['accepted'], 1)
        self.assertEqual(waiting['states'], {'queued': 1})
        self.assertEqual(waiting['terminal_percent'], 0)
        self.now = 150
        self.a.capacity_available('gpu')
        self.a.dispatch_next('gpu')
        self.now = 160
        self.a.finish(call['id'], {})
        result = self.result()
        self.assertEqual(result['requests']['completed_percent'], 100)
        self.assertEqual(result['queue_wait_seconds_dispatched_only']['total'], 48)
        self.assertEqual(result['service_elapsed_seconds_confirmed_ended_only']['total'], 12)

    def test_already_ended_before_window_is_not_carry_in(self):
        self.now = 80
        _, call = self.enqueue('past')
        self.a.dispatch_next('gpu')
        self.now = 90
        self.a.finish(call['id'], {})
        self.assertEqual(self.result()['carry_in_states_as_of'], {})

    def test_missing_history_is_not_invented(self):
        self.enqueue('old')
        self.a.db.execute('DELETE FROM execution_lifecycle')
        result = self.result()
        self.assertEqual(result['requests']['states'], {'unobserved':1})
        self.assertEqual(result['queue_wait_seconds_dispatched_only']['count'], 0)

    def test_empty_denominator_is_null_and_missing_db_not_created(self):
        self.assertIsNone(self.result()['requests']['completed_percent'])
        missing = Path(self.tmp.name)/'missing'
        with self.assertRaises(Exception):
            read_report(missing, start=1, end=2, as_of=2)
        self.assertFalse(missing.exists())

    def test_switches_do_not_claim_cache_cost(self):
        _, one = self.enqueue('one')
        self.enqueue('two')
        self.a.dispatch_next('gpu')
        self.a.finish(one['id'], {})
        self.a.dispatch_next('gpu')
        result = self.result()['interleaving']
        self.assertEqual(result['different_grant_pairs'], 1)
        self.assertIsNone(result['cache_prefill_cost'])

    def test_reopen_keeps_events_and_invalid_bounds_refuse(self):
        self.enqueue('one')
        self.a.close()
        self.a = Arbiter(self.path)
        self.assertEqual(self.result()['requests']['accepted'], 1)
        for bounds in ({'start':200}, {'as_of':199}, {'start':float('nan')}):
            with self.assertRaises(ValueError):
                self.result(**bounds)


if __name__ == '__main__':
    unittest.main()
