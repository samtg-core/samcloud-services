"""Model-only recovery observes the original admitted job; it never re-grants."""
import asyncio
from contextlib import nullcontext
import json,os,subprocess,sys
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import httpx
from openai import APIConnectionError
from .hermes_queue_identity import HostBindings
from .inference_sdk import QueuedClient
from .test_inference_sdk import QueueFixture

class ReceiptFixture(QueueFixture):
    wrong_receipt=False
    def handle(self,request):
        response=super().handle(request)
        if request.url.path.startswith('/v1/inference/executions/') and not request.url.path.endswith('/events'):
            body=response.json()
            body.update(id=request.url.path.rsplit('/',1)[1],grant_id='wrong' if self.wrong_receipt else 'grant')
            return httpx.Response(response.status_code,json=body)
        return response

class ModelRecovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'identity.sqlite'
        self.host=HostBindings(self.path,'profile');self.api=ReceiptFixture();self.keys=[];self.drop_ack=False
        self.history=[{'role':'user','_row_id':1,'content':'fixture'}]
        self.body={'model':'qwen','messages':[{'role':'user','content':'summary fixture'}]}
    async def asyncTearDown(self):self.tmp.cleanup()
    async def call(self,model='qwen',sync=False,owner_model=None,persist_unstored=False,operation=None,**changes):
        def keys(body):
            value=self.host.auxiliary_keys(body);self.keys.append(value)
            return replace(value,on_accepted=None) if self.drop_ack else value
        client=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
            http_factory=self.api.http,key_provider=keys,grant_lifetime='execution',complete_responses=True)
        body=self.body|{'model':model}|changes
        try:
            with self.host.resume_compression('session',operation) if operation else nullcontext(self.history) as held:
                with self.host.compression_scope('session',held,persist_unstored=persist_unstored,
                        owner_provider=lambda:('turn:'+(owner_model or model),owner_model or model)) as source:
                    self.operation=getattr(source,'operation',None)
                    if sync:return await asyncio.to_thread(lambda:client.create(**body))
                    return await client.create(**body)
        finally:await client.aclose()
    async def test_model_switch_reads_original_job_without_any_post_sync_and_async(self):
        await self.call();original=self.keys[-1]
        for sync in (False,True):
            with self.subTest(sync=sync):
                before=len(self.api.requests);answer=await self.call('qwen-new',sync=sync)
                recovered=self.keys[-1]
                self.assertEqual((recovered.grant_key,recovered.call_key),(original.grant_key,original.call_key))
                self.assertEqual(recovered.resume_job,('grant',original.call_key))
                self.assertEqual(answer.choices[0].message.content,'hello')
                self.assertTrue(all(method=='GET' for method,_,_ in self.api.requests[before:]))
                self.assertEqual(len(self.api.calls),1)
        self.host.finish_turn('session','turn:qwen-new')
        self.assertEqual(self.host.pending_events()[0]['expected_call_key'],original.call_key)
    async def test_private_compaction_source_and_receipt_reopen_in_fresh_process(self):
        self.history=[{'role':'user','content':'not-yet-persisted fixture'}]
        await self.call(persist_unstored=True);operation=self.operation;original=self.keys[-1]
        code="""import json,sys
from ollama.hermes_queue_identity import HostBindings
h=HostBindings(sys.argv[1],'profile')
with h.resume_compression('session',sys.argv[2]) as held:
 with h.compression_scope('session',held,persist_unstored=True,owner_provider=lambda:('new-process','qwen-new')):
  k=h.auxiliary_keys({'model':'qwen-new','messages':[{'role':'user','content':'summary fixture'}]})
print(json.dumps({'grant':k.grant_key,'call':k.call_key,'receipt':k.resume_job}))
"""
        result=subprocess.run([sys.executable,'-c',code,str(self.path),operation],
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])),capture_output=True,text=True,check=True)
        recovered=json.loads(result.stdout)
        self.assertEqual(recovered,{'grant':original.grant_key,'call':original.call_key,'receipt':['grant',original.call_key]})
        self.host=HostBindings(self.path,'profile');before=len(self.api.requests)
        await self.call('qwen-new',persist_unstored=True,operation=operation)
        self.assertTrue(all(method=='GET' for method,_,_ in self.api.requests[before:]))
        self.assertEqual(len(self.api.calls),1)
    async def test_same_model_reconnect_uses_receipt_reads_without_reposting(self):
        await self.call();before=len(self.api.requests);original=self.keys[-1]
        self.host=HostBindings(self.path,'profile')
        await self.call()
        self.assertEqual(self.keys[-1],original)
        self.assertEqual(self.keys[-1].resume_job,('grant',original.call_key))
        self.assertTrue(all(method=='GET' for method,_,_ in self.api.requests[before:]))
        self.assertEqual(len(self.api.calls),1)
    async def test_main_model_switch_with_pinned_aux_model_reads_original_job(self):
        await self.call();original=self.keys[-1];before=len(self.api.requests)
        answer=await self.call('qwen',owner_model='qwen-new')
        self.assertEqual(answer.choices[0].message.content,'hello')
        self.assertEqual(self.keys[-1].resume_job,('grant',original.call_key))
        self.assertTrue(all(method=='GET' for method,_,_ in self.api.requests[before:]))
        self.assertEqual(len(self.api.calls),1)
    async def test_changed_prompt_is_not_silently_replaced_by_old_summary(self):
        await self.call();before=len(self.api.requests)
        with self.assertRaises(APIConnectionError):
            await self.call('qwen-new',messages=[{'role':'user','content':'different focus/window'}])
        self.assertEqual(len(self.api.requests),before)
        self.assertEqual(len(self.api.calls),1)
        await self.call()  # exact original model/request remains recoverable
        self.assertEqual(len(self.api.calls),1)
    async def test_unknown_intake_restores_original_model_and_same_keys(self):
        self.drop_ack=True;await self.call();original=self.keys[-1]
        self.drop_ack=False;before=len(self.api.requests)
        with self.assertRaises(APIConnectionError):await self.call('qwen-new')
        self.assertEqual(len(self.api.requests),before)
        await self.call();restored=self.keys[-1]
        self.assertEqual((restored.grant_key,restored.call_key),(original.grant_key,original.call_key))
        self.assertEqual(len(self.api.calls),1)  # second POST reattaches, no second execution
    async def test_mismatched_service_receipt_never_falls_back_to_submission(self):
        await self.call();self.api.wrong_receipt=True;before=len(self.api.requests)
        with self.assertRaises(APIConnectionError):await self.call('qwen-new')
        self.assertTrue(all(method=='GET' for method,_,_ in self.api.requests[before:]))
        self.assertEqual(len(self.api.calls),1)

if __name__=='__main__':unittest.main()
