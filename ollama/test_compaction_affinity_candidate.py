"""Offline real identity/arbiter boundary; no provider, network or model."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from .hermes_queue_identity import HostBindings, HeldSource
from .inference_identity import IdentityStore
from .inference_arbiter import Arbiter, StintPolicy, Conflict

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'identity.sqlite';self.host=HostBindings(self.path,'profile')
        self.history=[{'role':'user','_row_id':1,'content':'fixture'}]
        self.body={'model':'qwen','messages':[{'role':'user','content':'fixture'}]}
        self.summary={'model':'qwen','messages':[{'role':'user','content':'summarize fixture'}]}
        self.now=1000
        self.clock=patch('ollama.inference_arbiter.time.time',side_effect=lambda:self.now);self.clock.start();self.addCleanup(self.clock.stop)
        self.a=Arbiter(Path(self.tmp.name)/'queue.sqlite',stint_policy=StintPolicy(idle_grace_seconds=300,fairness_seconds=1800));self.addCleanup(self.a.close)
    def main(self,session='session',row=1):
        self.host.pre_api_request(session_id=session,turn_id='turn',api_request_id='api',conversation_history=[self.history[0]|{'_row_id':row}])
        return self.host.main_stint_keys(self.body)
    def enqueue(self,keys,body):
        g=self.a.submit('agent',keys.grant_key,'gpu',body['model'],'work_session' if keys.work_session else 'execution')
        job=self.a.enqueue(g['id'],'agent',keys.call_key,body,yield_grant_keys=keys.yield_grant_keys)
        if keys.on_accepted:keys.on_accepted(SimpleNamespace(grant_id=g['id'],execution_id=job['id']))
        return job
    def incumbent(self):
        keys=self.main();g=self.a.submit('agent',keys.grant_key,'gpu','qwen','work_session')
        job=self.a.enqueue(g['id'],'agent',keys.call_key,self.body)
        keys.on_accepted(SimpleNamespace(grant_id=g['id'],execution_id=job['id']))
        self.a.finish(self.a.dispatch_next('gpu')['execution']['id'],{})
        return keys
    def other(self):
        g=self.a.submit('other','other','gpu','qwen','work_session')
        return self.a.enqueue(g['id'],'other','other-call',{})
    def compact(self,owner=lambda:('turn','qwen'),body=None,session='session'):
        with self.host.compression_scope(session,self.history,owner_provider=owner):
            return self.host.auxiliary_keys(body or self.summary)
    def test_continuation_precedes_waiter_and_finishes_with_exact_summary_fence(self):
        main=self.incumbent();other=self.other();summary=self.compact();job=self.enqueue(summary,self.summary)
        self.assertEqual(main.grant_key,summary.grant_key);self.assertNotEqual(main.call_key,summary.call_key)
        self.assertEqual(summary.yield_grant_keys,());self.assertTrue(summary.work_session)
        self.assertEqual(job['id'],self.a.dispatch_next('gpu')['execution']['id']);self.a.finish(job['id'],{})
        self.host.finish_turn('session','turn');event=self.host.pending_events()[0]
        self.assertEqual(event['expected_call_key'],summary.call_key)
        self.a.yield_by_key('agent',event['grant_key'],event['expected_call_key'])
        self.assertEqual(other['id'],self.a.dispatch_next('gpu')['execution']['id'])
    def test_normal_fairness_still_yields_to_waiter(self):
        self.incumbent();other=self.other();self.now+=1801
        self.enqueue(self.compact(),self.summary)
        self.assertEqual(other['id'],self.a.dispatch_next('gpu')['execution']['id'])
    def test_first_foreground_call_can_be_compaction(self):
        keys=self.compact();main=self.main()
        self.assertEqual(keys.grant_key,main.grant_key);self.assertNotEqual(keys.call_key,main.call_key)
    def test_other_session_does_not_borrow_parent(self):
        main=self.incumbent();keys=self.compact(session='other-session')
        self.assertNotEqual(keys.grant_key,main.grant_key);self.assertEqual(keys.yield_grant_keys,())
    def test_different_summary_model_is_separate_dependency(self):
        main=self.incumbent();keys=self.compact(body=self.summary|{'model':'vision'})
        self.assertNotEqual(main.grant_key,keys.grant_key);self.assertEqual(keys.yield_grant_keys,(main.grant_key,));self.assertFalse(keys.work_session)
    def test_no_live_owner_keeps_execution_policy(self):
        main=self.incumbent();keys=self.compact(owner=None)
        self.assertFalse(keys.work_session);self.assertEqual(keys.yield_grant_keys,(main.grant_key,))
    def test_owner_ending_before_submission_refuses(self):
        with self.assertRaises(ValueError):self.compact(owner=lambda:None)
    def test_legacy_bound_execution_is_not_reinterpreted(self):
        main=self.incumbent()
        old=HeldSource.capture('profile','session','compression',self.history).bind(self.path,self.summary)
        s=IdentityStore(self.path)
        try:old=s.bind_handoff(old,(main.grant_key,))
        finally:s.close()
        new=self.compact();self.assertEqual(new,old);self.assertFalse(new.work_session)
    def test_reopen_preserves_operation_call_and_updates_exact_new_turn_fence(self):
        history=[{'role':'user','content':'unpersisted fixture'}]
        with self.host.compression_scope('session',history,persist_unstored=True,owner_provider=lambda:('turn','qwen')) as source:
            first=self.host.auxiliary_keys(self.summary);operation=source.operation
        restored=HostBindings(self.path,'profile')
        with restored.resume_compression('session',operation) as held:
            with restored.compression_scope('session',held,persist_unstored=True,owner_provider=lambda:('resumed-turn','qwen')):
                second=restored.auxiliary_keys(self.summary)
        self.assertEqual(first,second)
        second.on_accepted(SimpleNamespace(grant_id='grant',execution_id=second.call_key))
        restored.finish_turn('session','resumed-turn')
        self.assertEqual(restored.pending_events()[0]['expected_call_key'],first.call_key)
    def test_existing_continuation_without_owner_refuses_instead_of_new_grant(self):
        self.compact()
        with self.assertRaises(Conflict):self.compact(owner=None)
    def test_changed_input_refuses(self):
        self.compact()
        with self.assertRaises(Conflict):self.compact(body=self.summary|{'temperature':0.1})

    def test_late_acknowledgement_keeps_old_event_and_adds_exact_new_fence(self):
        from .inference_arbiter import LifecycleStale
        main=self.incumbent();summary=self.compact()
        grant=self.a.submit('agent',summary.grant_key,'gpu','qwen','work_session')
        job=self.a.enqueue(grant['id'],'agent',summary.call_key,self.summary)
        self.host.finish_turn('session','turn')
        original=self.host.pending_events()[0]
        self.assertEqual(original['expected_call_key'],main.call_key)
        self.a.finish(self.a.dispatch_next('gpu')['execution']['id'],{})
        receipt=SimpleNamespace(grant_id=grant['id'],execution_id=job['id'])
        summary.on_accepted(receipt);summary.on_accepted(receipt)
        events=self.host.pending_events();self.assertEqual(len(events),2)
        self.assertEqual(events[0],original)
        self.assertEqual(events[1]['expected_call_key'],summary.call_key)
        with self.assertRaises(LifecycleStale):self.a.yield_by_key('agent',original['grant_key'],original['expected_call_key'])
        self.a.yield_by_key('agent',events[1]['grant_key'],events[1]['expected_call_key'])

if __name__=='__main__':unittest.main()
