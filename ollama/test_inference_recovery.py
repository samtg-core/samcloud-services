"""Offline lifecycle tests only: no provider, plane, credentials or real input."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from .inference_arbiter import Arbiter, Conflict
from .inference_recovery import DeadLocalBackend, OfflineRecovery
from .inference_supervisor import Supervisor


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'queue.sqlite'
        a=Arbiter(self.path)
        self.grant=a.submit('creative','one','slice','qwen')
        self.call=a.enqueue(self.grant['id'],'creative','one',{'prompt':'synthetic fixture'})
        a.dispatch_next('slice')
        if self._testMethodName == 'test_partial_output_is_never_retried':
            a.append_event(self.call['id'],'synthetic delta')
        a.uncertain(self.call['id']);a.close()
        self.r=OfflineRecovery(self.path)
        self.proof=DeadLocalBackend(self.call['id'],'slice','slice','d34b663c-db45-44a7-8950-8300c67b31bb',
                                   '3c17741b-ea8f-4430-b3b3-39b4964574d1','a'*64)

    def tearDown(self):
        self.r.close();self.tmp.cleanup()

    def retry(self, expected=None, proof=None):
        return self.r.retry_dead_attempt(expected or self.r.inspect(self.call['id']),
            proof=proof or self.proof,operator='codex:test-fixture')

    def test_same_logical_input_and_single_audited_recovery(self):
        before=self.r.inspect(self.call['id']);rid=self.retry(before)
        row=self.r.journal.execution(self.call['id'],'creative')
        self.assertEqual(row['state'],'queued');self.assertEqual(row['request'],self.call['request'])
        self.assertIsNone(row['result'])
        self.assertEqual(self.r.journal.enqueue(self.grant['id'],'creative','one',
                         {'prompt':'synthetic fixture'})['id'],self.call['id'])
        audit=self.r.journal.db.execute('SELECT * FROM backend_attempt_recoveries').fetchall()
        self.assertEqual(len(audit),1);self.assertEqual(audit[0]['id'],rid)
        self.assertEqual(json.loads(audit[0]['evidence'])['new_boot'],self.proof.new_boot)
        with self.assertRaises(Conflict):self.retry(before)
        self.assertEqual(self.r.journal.db.execute('SELECT COUNT(*) FROM backend_attempt_recoveries').fetchone()[0],1)

    def test_same_boot_is_not_death(self):
        with self.assertRaises(Conflict):
            self.retry(proof=replace(self.proof,new_boot=self.proof.old_boot))
        self.assertEqual(self.r.inspect(self.call['id'])['state'],'uncertain')

    def test_missing_proof_is_not_an_escape_hatch(self):
        with self.assertRaises(TypeError):
            self.r.retry_dead_attempt(self.r.inspect(self.call['id']),proof=None,operator='codex')

    def test_proof_is_bound_to_execution_and_resource(self):
        for proof in (replace(self.proof,execution_id='different'),replace(self.proof,resource='other')):
            with self.assertRaises(Conflict):self.retry(proof=proof)

    def test_recovery_and_audit_survive_reopen_without_dispatch(self):
        self.retry();self.r.close();self.r=OfflineRecovery(self.path)
        self.assertEqual(self.r.inspect(self.call['id'])['state'],'queued')
        self.assertEqual(self.r.journal.db.execute('SELECT COUNT(*) FROM backend_attempt_recoveries').fetchone()[0],1)
        self.assertEqual(self.r.journal.db.execute("SELECT COUNT(*) FROM executions WHERE state='running'").fetchone()[0],0)

    def test_audit_and_retry_rollback_together(self):
        self.r.journal.db.execute("CREATE TRIGGER reject_fixture_retry BEFORE UPDATE ON grants "
            "BEGIN SELECT RAISE(ABORT,'fixture rejection'); END")
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):self.retry()
        self.assertEqual(self.r.inspect(self.call['id'])['state'],'uncertain')
        self.assertEqual(self.r.journal.db.execute('SELECT COUNT(*) FROM backend_attempt_recoveries').fetchone()[0],0)

    def test_partial_output_is_never_retried(self):
        with self.assertRaises(Conflict):self.retry()

    def test_saved_result_is_never_retried(self):
        self.r.journal.finish(self.call['id'],{'answer':'saved'})
        with self.assertRaises(Conflict):self.retry()

    def test_user_release_is_not_resurrected(self):
        self.r.journal.request_release(self.grant['id'],'creative')
        with self.assertRaises(Conflict):self.retry()

    def test_stale_snapshot_refused_even_without_state_change(self):
        before=self.r.inspect(self.call['id'])
        self.r.journal.uncertain(self.call['id'])
        with self.assertRaises(Conflict):self.retry(before)

    def test_other_queued_work_survives_and_precedes_retry(self):
        grant=self.r.journal.submit('assistant','two','slice','qwen')
        call=self.r.journal.enqueue(grant['id'],'assistant','two',{'prompt':'second fixture'})
        self.retry()
        item=self.r.journal.dispatch_next('slice')
        self.assertEqual(item['execution']['id'],call['id'])

    def test_does_not_create_a_missing_journal(self):
        with self.assertRaises(FileNotFoundError):OfflineRecovery(Path(self.tmp.name)/'missing')

    def test_excludes_another_recovery_or_scheduler(self):
        with self.assertRaises(BlockingIOError):OfflineRecovery(self.path)
        with self.assertRaises(BlockingIOError):Supervisor(self.path,None,lambda _:False)

    def test_scheduler_excludes_recovery(self):
        self.r.close();self.r=None
        s=Supervisor(self.path,None,lambda _:False)
        try:
            with self.assertRaises(BlockingIOError):OfflineRecovery(self.path)
        finally:
            s.close();self.r=OfflineRecovery(self.path)


class WaitingConsumerRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_consumer_survives_offline_recovery_without_resubmission(self):
        import asyncio
        import httpx
        from fastapi import FastAPI, Request
        from .inference_api import build_router
        from .inference_consumer import DurableConsumer, Job

        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'queue.sqlite';gate=asyncio.Event();calls=[];states=[]
            async def dead_attempt(grant,body):
                calls.append('old-attempt')
                raise ConnectionError('synthetic unknown outcome')
            async def new_attempt(grant,body):
                calls.append('new-attempt');await gate.wait()
                return {'answer':'same saved fixture'}
            s=Supervisor(path,dead_attempt,lambda _:True)
            g=s.journal.submit('creative','same-grant','slice','qwen')
            e=s.journal.enqueue(g['id'],'creative','same-call',{'prompt':'fixture'})
            s.pump('slice');await s.observe(e['id'],'creative')
            def make_app(supervisor):
                app=FastAPI()
                @app.middleware('http')
                async def identity(request:Request,next_handler):
                    request.state.caller={'username':'creative'}
                    return await next_handler(request)
                app.include_router(build_router(supervisor,lambda m:('slice',m),lambda m,b:b))
                return app
            active=httpx.ASGITransport(app=make_app(s));methods=[]
            class SwitchableService(httpx.AsyncBaseTransport):
                async def handle_async_request(self,request):
                    methods.append(request.method)
                    if active is None:return httpx.Response(503,request=request)
                    return await active.handle_async_request(request)
            async with httpx.AsyncClient(transport=SwitchableService(),base_url='http://fixture') as http:
                consumer=DurableConsumer(http,poll_interval=.002,progress=states.append)
                reader=asyncio.create_task(consumer.result(Job(g['id'],e['id'])))
                async def until(predicate):
                    async with asyncio.timeout(2):
                        while not predicate():await asyncio.sleep(.001)
                try:
                    await until(lambda:'uncertain' in states)
                    self.assertFalse(reader.done())
                    active=None;s.close();s=None
                    await until(lambda:'service_unavailable' in states)
                    r=OfflineRecovery(path)
                    try:
                        proof=DeadLocalBackend(e['id'],'slice','slice',
                            'd34b663c-db45-44a7-8950-8300c67b31bb',
                            '3c17741b-ea8f-4430-b3b3-39b4964574d1','a'*64)
                        r.retry_dead_attempt(r.inspect(e['id']),proof=proof,operator='test-fixture')
                    finally:r.close()
                    s=Supervisor(path,new_attempt,lambda _:True)
                    active=httpx.ASGITransport(app=make_app(s))
                    await until(lambda:'queued' in states)
                    s.pump('slice');await until(lambda:'running' in states)
                    gate.set()
                    self.assertEqual(await asyncio.wait_for(reader,2),{'answer':'same saved fixture'})
                    self.assertEqual(calls,['old-attempt','new-attempt'])
                    self.assertEqual(set(methods),{'GET'})
                    self.assertEqual(s.journal.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0],1)
                finally:
                    gate.set()
                    if not reader.done():reader.cancel()
                    await asyncio.gather(reader,return_exceptions=True)
                    if s is not None:
                        await asyncio.gather(*list(s.workers.values()));s.close()


if __name__=='__main__':unittest.main()
