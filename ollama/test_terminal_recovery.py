"""Synthetic offline journals; attestations here are fixtures, not measurements."""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib,json,sqlite3,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx

from ollama.inference_arbiter import Arbiter,Conflict
from ollama.inference_recovery import OfflineRecovery,StoppedLocalBackend,BackendProcess,DeadLocalBackend
from ollama.inference_supervisor import Supervisor
from ollama.inference_consumer import DurableConsumer,ExecutionStopped,Job
from ollama.inference_sdk import _error_response

@contextmanager
def fixture(lifetime='work_session',partial=False):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'queue.sqlite';a=Arbiter(path)
        g=a.submit('fixture-owner','grant-key','fixture-host/pool','fixture-model',lifetime=lifetime)
        e=a.enqueue(g['id'],'fixture-owner','call-key',{'prompt':'SYNTHETIC-INPUT-DO-NOT-LOG'})
        a.dispatch_next(g['resource'])
        if partial:a.append_event(e['id'],'synthetic partial')
        a.uncertain(e['id']);a.close()
        process=BackendProcess(12345,502,'fixture-birth-token','/fixture/ollama')
        proof=StoppedLocalBackend(e['id'],g['resource'],'fixture-host','3c17741b-ea8f-4430-b3b3-39b4964574d1',
            (process,),(process,),True,True,True,'a'*64)
        f=SimpleNamespace(path=path,g=g,e=e,proof=proof,r=OfflineRecovery(path))
        try:yield f
        finally:
            if f.r is not None:f.r.close()

def fail(f,expected=None,proof=None,code='backend_attempt_dead'):
    return f.r.fail_dead_attempt(expected or f.r.inspect(f.e['id']),proof=proof or f.proof,
                                 operator='codex:offline-fixture',code=code)

def audit_count(f):
    return f.r.journal.db.execute('SELECT COUNT(*) FROM backend_attempt_terminations').fetchone()[0]

class TerminalRecovery(unittest.TestCase):
    def test_work_session_fails_exact_attempt_preserves_input_and_audits(self):
        with fixture() as f:
            expected=f.r.inspect(f.e['id'])
            with patch.object(f.r.journal,'finish',side_effect=AssertionError('must not nest finish')):
                tid=fail(f,expected)
            e=f.r.journal.execution(f.e['id'],'fixture-owner');g=f.r.journal.get(f.g['id'],'fixture-owner')
            self.assertEqual(e['state'],'failed');self.assertEqual(e['request'],f.e['request'])
            result=json.loads(e['result']);self.assertEqual(result['status'],502)
            self.assertEqual(result['code'],'backend_attempt_dead');self.assertEqual(result['type'],'provider_error')
            self.assertEqual(result['kind'],'error');self.assertIn(tid,result['detail'])
            self.assertNotIn('SYNTHETIC',json.dumps(result));self.assertEqual(g['state'],'granted')
            self.assertIsNone(g['active_execution']);self.assertIsNotNone(g['idle_since'])
            row=f.r.journal.db.execute('SELECT * FROM backend_attempt_terminations').fetchone()
            self.assertEqual(row['execution_id'],f.e['id']);self.assertEqual(row['previous_lifecycle_seq'],expected['lifecycle_seq'])
            self.assertEqual(row['request_sha256'],hashlib.sha256(e['request'].encode()).hexdigest())
            self.assertEqual(json.loads(row['evidence'])['attestation_type'],'StoppedLocalBackend')
            self.assertEqual(f.r.journal.db.execute('SELECT COUNT(*) FROM backend_attempt_recoveries').fetchone()[0],0)
            self.assertEqual(f.r.journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0],1)
            self.assertGreater(f.r.inspect(f.e['id'])['lifecycle_seq'],expected['lifecycle_seq'])

    def test_execution_grant_is_logically_released_without_any_registry_action(self):
        with fixture('execution') as f:
            fail(f);g=f.r.journal.get(f.g['id'],'fixture-owner')
            self.assertEqual(g['state'],'released');self.assertIsNone(g['active_execution']);self.assertIsNone(g['idle_since'])

    def test_terminal_audit_survives_reopen_and_replay_is_rejected(self):
        with fixture() as f:
            old=f.r.inspect(f.e['id']);tid=fail(f,old);f.r.close();f.r=OfflineRecovery(f.path)
            self.assertEqual(f.r.inspect(f.e['id'])['state'],'failed')
            self.assertEqual(f.r.journal.db.execute('SELECT id FROM backend_attempt_terminations').fetchone()[0],tid)
            for expected in (old,f.r.inspect(f.e['id'])):
                with self.assertRaises(Conflict):fail(f,expected)
            self.assertEqual(audit_count(f),1)

    def test_stale_cas_and_wrong_work_proof_leave_uncertainty(self):
        with fixture() as f:
            old=f.r.inspect(f.e['id']);stale=dict(old,updated=old['updated']-1)
            with self.assertRaises(Conflict):fail(f,stale)
            for proof in (replace(f.proof,execution_id='other'),replace(f.proof,resource='other')):
                with self.assertRaises(Conflict):fail(f,proof=proof)
            self.assertEqual(f.r.inspect(f.e['id'])['state'],'uncertain');self.assertEqual(audit_count(f),0)

    def test_complete_fenced_process_proof_and_birth_identities_are_mandatory(self):
        with fixture() as f:
            proofs=[replace(f.proof,complete_inventory=False),replace(f.proof,revival_fenced=False),
                    replace(f.proof,listener_closed=False),replace(f.proof,processes=(),observed_gone=()),
                    replace(f.proof,observed_gone=()),replace(f.proof,observed_gone=(replace(f.proof.processes[0],birth_token='reused'),)),
                    replace(f.proof,processes=(replace(f.proof.processes[0],birth_token=''),),observed_gone=f.proof.observed_gone)]
            for proof in proofs:
                with self.subTest(proof=proof),self.assertRaises((Conflict,ValueError)):fail(f,proof=proof)
            with self.assertRaises(TypeError):f.r.fail_dead_attempt(f.r.inspect(f.e['id']),proof=None,operator='fixture',code='backend_attempt_dead')
            self.assertEqual(audit_count(f),0)

    def test_legacy_proof_still_requires_a_new_boot(self):
        with fixture() as f:
            p=DeadLocalBackend(f.e['id'],f.g['resource'],'fixture-host',f.proof.boot_id,f.proof.boot_id,'b'*64)
            with self.assertRaises(Conflict):fail(f,proof=p)
            fail(f,proof=replace(p,new_boot='d34b663c-db45-44a7-8950-8300c67b31bb'))
            self.assertEqual(f.r.inspect(f.e['id'])['state'],'failed')

    def test_output_result_release_or_lost_ownership_cannot_be_overwritten(self):
        for kind in ('output','result','release','ownership','running'):
            with self.subTest(kind=kind),fixture(partial=kind=='output') as f:
                if kind=='output':pass
                elif kind=='result':f.r.journal.finish(f.e['id'],{'answer':'existing'})
                elif kind=='release':f.r.journal.request_release(f.g['id'],'fixture-owner')
                elif kind=='ownership':f.r.journal.db.execute('UPDATE grants SET active_execution=NULL WHERE id=?',(f.g['id'],))
                else:f.r.journal.db.execute("UPDATE executions SET state='running' WHERE id=?",(f.e['id'],))
                before=f.r.inspect(f.e['id'])
                with self.assertRaises(Conflict):fail(f)
                self.assertEqual(f.r.inspect(f.e['id']),before);self.assertEqual(audit_count(f),0)

    def test_audit_execution_and_grant_rollback_together(self):
        with fixture() as f:
            expected=f.r.inspect(f.e['id'])
            f.r.journal.db.execute("CREATE TRIGGER reject_fixture BEFORE UPDATE ON grants BEGIN SELECT RAISE(ABORT,'fixture rejection'); END")
            with self.assertRaises(sqlite3.IntegrityError):fail(f,expected)
            self.assertEqual(f.r.inspect(f.e['id']),expected);self.assertEqual(audit_count(f),0)

    def test_recovery_lock_excludes_an_active_supervisor(self):
        with fixture() as f:
            f.r.close();f.r=None
            s=Supervisor(f.path,lambda *a:None,lambda resource:False)
            try:
                with self.assertRaises(BlockingIOError):OfflineRecovery(f.path)
            finally:s.close();f.r=OfflineRecovery(f.path)

    def test_only_a_fixed_error_code_can_enter_the_result(self):
        with fixture() as f:
            for code in ('SECRET-CANARY','backend_attempt_dead\nsecret',None,True):
                with self.assertRaises(ValueError):fail(f,code=code)
            self.assertEqual(audit_count(f),0)

    def test_waiter_gets_one_safe_provider_failure_with_no_retry(self):
        with fixture() as f:
            fail(f);row=f.r.journal.execution(f.e['id'],'fixture-owner');result=json.loads(row['result'])
            async def run():
                requests=[]
                def answer(request):
                    requests.append((request.method,request.url.path))
                    self.assertEqual(request.method,'GET')
                    if request.url.path.endswith('/events'):
                        return httpx.Response(200,json={'state':'failed','events':[],'next_cursor':0,'has_more':False})
                    return httpx.Response(200,json={'state':'failed','result':result})
                async with httpx.AsyncClient(base_url='http://offline',transport=httpx.MockTransport(answer)) as http:
                    consumer=DurableConsumer(http);job=Job(f.g['id'],f.e['id'])
                    with self.assertRaises(ExecutionStopped) as caught:await consumer.result(job)
                    self.assertEqual(len(requests),1)
                    response=_error_response(httpx.Request('POST','http://offline/v1/chat/completions'),caught.exception)
                    self.assertEqual(response.status_code,502);self.assertEqual(response.headers['x-should-retry'],'false')
                    self.assertEqual(response.headers['X-Inference-Execution'],f.e['id'])
                    self.assertEqual(response.json()['error']['code'],'backend_attempt_dead')
                    self.assertNotIn('SYNTHETIC',response.text)
                    requests.clear()
                    with self.assertRaises(ExecutionStopped):
                        async for _ in consumer.events(job):pass
                    self.assertEqual(len(requests),2)
            asyncio.run(run())

if __name__=='__main__':unittest.main()
