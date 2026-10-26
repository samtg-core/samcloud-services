"""Continuation lifetime/capability on both real SDK transports, mocked HTTP only."""
import asyncio
import unittest
import httpx
from openai import APIConnectionError
from .inference_identity import CallKeys
from .inference_sdk import QueuedClient
from .test_inference_sdk import QueueFixture,arguments

class SDKContinuation(unittest.IsolatedAsyncioTestCase):
    async def test_auxiliary_continuation_uses_work_session_on_both_transports(self):
        for asynchronous in (False,True):
            api=QueueFixture()
            client=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
                http_factory=api.http,grant_lifetime='execution',complete_responses=True,
                key_provider=lambda _:CallKeys('parent','summary',work_session=True))
            body=arguments();body.pop('extra_headers')
            try:
                if asynchronous:answer=await client.create(**body)
                else:answer=await asyncio.to_thread(lambda:client.create(**body))
                self.assertEqual(answer.choices[0].message.content,'hello')
                self.assertEqual(api.requests[0][:2],('GET','/v1/inference/capabilities'))
                grants=[r[2] for r in api.requests if r[:2]==('POST','/v1/inference/grants')]
                self.assertEqual(grants,[{'key':'parent','model':'qwen','lifetime':'work_session'}])
            finally:await client.aclose()
    async def test_continuation_requires_lifecycle_capability_before_any_write(self):
        requests=[]
        def handle(request):
            requests.append((request.method,request.url.path))
            return httpx.Response(200,json={'stints':{'version':1,'enabled':True}})
        client=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
            http_factory=lambda:httpx.AsyncClient(base_url='https://queue.invalid',transport=httpx.MockTransport(handle)),
            grant_lifetime='execution',key_provider=lambda _:CallKeys('parent','summary',work_session=True))
        body=arguments();body.pop('extra_headers')
        try:
            with self.assertRaises(APIConnectionError):await client.create(**body)
            self.assertEqual(requests,[('GET','/v1/inference/capabilities')])
        finally:await client.aclose()


class IntakeFence(unittest.IsolatedAsyncioTestCase):
    async def test_refused_summary_keeps_last_actual_main_call_fence(self):
        import tempfile
        from pathlib import Path
        from .hermes_queue_identity import HostBindings
        from .inference_arbiter import Arbiter,StintPolicy
        with tempfile.TemporaryDirectory() as tmp:
            host=HostBindings(Path(tmp)/'identity.sqlite','profile')
            history=[{'role':'user','_row_id':1,'content':'fixture'}]
            body={'model':'qwen','messages':[{'role':'user','content':'fixture'}]}
            host.pre_api_request(session_id='session',turn_id='turn',api_request_id='api',conversation_history=history)
            main=host.main_stint_keys(body);api=QueueFixture()
            main_client=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
                http_factory=api.http,key_provider=lambda _:main)
            try:await main_client.create(**body)
            finally:await main_client.aclose()
            original=api.handle
            def refuse(request):
                if request.url.path.endswith('/capabilities'):
                    api.requests.append((request.method,request.url.path,None))
                    return httpx.Response(200,json={'stints':{'version':1,'enabled':True}})
                return original(request)
            api.handle=refuse
            aux=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
                http_factory=api.http,grant_lifetime='execution',key_provider=host.auxiliary_keys)
            try:
                with host.compression_scope('session',history,owner_provider=lambda:('turn','qwen')):
                    with self.assertRaises(APIConnectionError):await aux.create(**body)
            finally:await aux.aclose()
            host.finish_turn('session','turn');event=host.pending_events()[0]
            self.assertEqual(event['expected_call_key'],main.call_key)
            self.assertEqual(len(api.calls),1)
            arbiter=Arbiter(Path(tmp)/'queue.sqlite',stint_policy=StintPolicy())
            try:
                grant=arbiter.submit('agent',main.grant_key,'gpu','qwen','work_session')
                arbiter.enqueue(grant['id'],'agent',main.call_key,body)
                arbiter.finish(arbiter.dispatch_next('gpu')['execution']['id'],{})
                arbiter.yield_by_key('agent',event['grant_key'],event['expected_call_key'])
                self.assertEqual(arbiter.get(grant['id'],'agent')['state'],'queued')
            finally:arbiter.close()

if __name__=='__main__':unittest.main()
