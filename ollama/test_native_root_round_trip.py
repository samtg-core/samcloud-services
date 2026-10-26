"""Actual Hermes/plugins/SDK -> root runtime HTTP; synthetic physical providers.

No model weights, GPU, production profile/identity or external registry is used.
Real registry admission/restart is covered separately by the staged contract.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
import httpx
import pytest
import uvicorn

from . import manager as module,server
from .manager import Backend,ModelManager
from .inference_bootstrap import QueueSettings
from .inference_root_bootstrap import RootQueueStartup
from .inference_root_client import RegistryRootClient,RootIntentStore
from .test_root_models import Ollama,Child
from .test_native_job_round_trip import test_native_parent_tool_and_recovery_keep_place_until_real_finish as _native_scenario


@pytest.fixture
def root_queue(tmp_path,monkeypatch):
    base='https://registry.invalid/api/v1';resource='synthetic-gpu'
    observed=[];state={};roots={};keys={};children=[];main_calls=0
    monkeypatch.setattr(module,'RESOURCE_ID',resource);monkeypatch.setattr(module,'SERVICE_ID','models')
    def transport(request):
        if request.url.path.endswith('capabilities/lease-roots'):
            return httpx.Response(200,json={'protocol':'samcloud.lease-roots','version':1,'actor':'provider',
                'features':{name:True for name in RegistryRootClient.FEATURES}})
        body=json.loads(request.content) if request.content else {}
        if request.method=='POST' and request.url.path.endswith('/roots'):
            key=body['key']
            if key not in keys:
                assert not any(row['lease']['status']=='active' for row in roots.values())
                ident='root-'+str(len(roots)+1);lease='lease-'+ident;keys[key]=ident
                roots[ident]={'root_id':ident,'initial_lease_id':lease,'current_lease_id':lease,
                    'resource_id':resource,'lifetime':'explicit_end','state':'open','health_ttl_s':120,
                    'lease':{'id':lease,'resource_id':resource,'agent_id':'provider','service_id':'models',
                             'exclusive':1,'memory_mb':49152,'status':'active','expires_at':'2099-01-01T00:00:00+00:00'}}
            value=roots[keys[key]]
        elif request.url.path.endswith('/roots/end'):
            value=roots[keys[body['key']]];value['state']='ending'
        else:
            ident=request.url.path.split('/lease-roots/')[1].split('/')[0];value=roots[ident]
            if request.url.path.endswith('/settle'):
                assert body['lease_id']==value['current_lease_id']
                value['state']='complete';value['lease']['status']='released'
        return httpx.Response(200,json=deepcopy(value))
    root_client=RegistryRootClient(RootIntentStore(tmp_path/'root-client.sqlite',registry=base,actor='provider',service='models'),
        token='synthetic-provider-only',transport=httpx.MockTransport(transport))
    def resource_view(*_):
        return {'id':resource,'allocation_mode':'shared','expiry_policy':'hold',
            'specs':{'unified_memory_mb':49152},'leases':[deepcopy(r['lease']) for r in roots.values() if r['lease']['status']=='active'],
            'unsettled':[]}
    ordinary=SimpleNamespace(base=base,verify_token=lambda:{'valid':True,'owner_id':'provider'},
        get_resource=resource_view,request_lease=Mock(),release_lease=Mock(),renew_lease=Mock())
    m=ModelManager.__new__(ModelManager);m.sc=ordinary;m.models={};m.root_binding=None;m.durable_inference=False
    m.initial_reservations=None;m.residency_renewal=None;m.inference_capacity_source=None;m._background_started=False
    m._pool_lock=threading.Lock();m._pool_leases=set();m._unreleased={};m.ollama=Ollama();m.refuse_if_device_in_use=lambda *_:None
    m.ollama.list_models=lambda:[{'name':'qwen'}]
    for name in m.MAINTENANCE_TASKS:setattr(m,name,None)
    def physical():
        used=sum(m.ollama.running.values())+sum(5700 for child in children if child.alive)
        return {'memory_available_mb':64000-used,'memory_device_inuse_mb':used}
    monkeypatch.setattr(module.capacity,'collect',physical)
    monkeypatch.setattr(module,'match_vlm_model',lambda name:('vision',5700))
    def spawn(args,**kwargs):child=Child(args);children.append(child);return child
    proxy=SimpleNamespace(**vars(subprocess));proxy.Popen=spawn
    monkeypatch.setattr(module,'subprocess',proxy)  # never replace native/core subprocess globally
    actual_get=httpx.get
    health=f'http://{module.VLM_HOST}:{module.VLM_PORT}/health'
    monkeypatch.setattr(httpx,'get',lambda url,*a,**kw:SimpleNamespace(status_code=200) if url==health else actual_get(url,*a,**kw))
    from . import vision_lifecycle,inference_shared_resource
    monkeypatch.setattr(vision_lifecycle,'stop_owned_vision',lambda mm:mm.vlm_process.terminate())
    monkeypatch.setattr(inference_shared_resource,'cold_vision_absent',lambda:True)
    service=SimpleNamespace(mgr=m,config=SimpleNamespace(SC_RESOURCE_ID=resource,SC_SERVICE_ID='models',
        WHISPER_MAX_UPLOAD_MB=1,EXO_ENABLED=False),inference_runtime=None,inference_dispatcher=None,
        ChatRequest=server.ChatRequest,CompletionRequest=server.CompletionRequest,Backend=Backend)
    journal=tmp_path/'service.sqlite'
    async def execute(grant,call):
        nonlocal main_calls
        body=call.get('body',call);ident=grant['active_execution'];supervisor=state['supervisor']
        is_summary=state.get('allow_compaction',False) and any(
            isinstance(message.get('content'),str) and 'TURNS TO SUMMARIZE:' in message['content']
            for message in body.get('messages',[]))
        observed.append({'owner':grant['owner'],'model':grant['model'],'execution':ident,
                         'lease_id':m.models[grant['model']].lease_id,
                         'job_id':supervisor.journal.execution(ident,grant['owner'])['job_id'],
                         'purpose':'summary' if is_summary else 'ordinary'})
        if is_summary:
            state.setdefault('summary_executions',[]).append(ident)
            other=supervisor.journal.submit('other','other',resource,'qwen','execution')
            waiting=supervisor.journal.enqueue(other['id'],'other','other-call',{'model':'qwen','messages':[]})
            state['other_execution']=waiting['id']
            content='## Goal\nComplete the synthetic tool request.\n## State\nEarlier synthetic history was reviewed. No external work happened.'
            if body.get('stream'):
                frame={'id':ident,'object':'chat.completion.chunk','created':1,'model':grant['model'],
                    'choices':[{'index':0,'delta':{'role':'assistant','content':content},'finish_reason':'stop'}]}
                supervisor.journal.append_event(ident,'data: '+json.dumps(frame)+'\n\n',terminal=True)
                supervisor.journal.append_event(ident,'data: [DONE]\n\n',terminal=True)
                return {'kind':'stream','event_count':2,'media_type':'text/event-stream'}
            return {'kind':'json','body':{'id':ident,'object':'chat.completion','created':1,'model':grant['model'],
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':content}}]}}
        if body.get('stream'):
            main_calls+=1
            if main_calls==1 and state.get('tool_first',True):
                delta={'role':'assistant','tool_calls':[{'index':index,'id':f'native-job-tool-{index}',
                    'type':'function','function':{'name':'job_round_trip_probe','arguments':json.dumps({'slot':index})}}
                    for index in range(state.get('child_count',1))]};finish='tool_calls'
            else:
                delta={'role':'assistant','content':'done'};finish='stop'
                if main_calls==1:
                    other=supervisor.journal.submit('other','other',resource,'qwen','execution')
                    waiting=supervisor.journal.enqueue(other['id'],'other','other-call',{'model':'qwen','messages':[]})
                    state['other_execution']=waiting['id']
            frame={'id':ident,'object':'chat.completion.chunk','created':1,'model':grant['model'],
                   'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}
            supervisor.journal.append_event(ident,'data: '+json.dumps(frame)+'\n\n',terminal=True)
            supervisor.journal.append_event(ident,'data: [DONE]\n\n',terminal=True)
            return {'kind':'stream','event_count':2,'media_type':'text/event-stream'}
        if grant['model']=='vision' and state.get('child_delay'):await asyncio.sleep(state['child_delay'])
        return {'kind':'json','body':{'id':ident,'object':'chat.completion','created':1,'model':grant['model'],
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'child done'}}]}}
    @asynccontextmanager
    async def lifespan(app):
        service.app=app
        rt=await RootQueueStartup(QueueSettings(journal,{'qwen':resource,'vision':resource}),root_client,
            {'qwen':Backend.OLLAMA,'vision':Backend.VLM},
            lambda:not m.ollama.running and not any(c.alive for c in children),True).start(service)
        state['supervisor']=rt.queue.supervisor;state['runtime']=rt
        rt.queue.supervisor.execute=execute
        await rt.capacity_source.refresh(physical());await rt.capacity_source.drain()
        yield
        outcome=await rt.quiesce();assert outcome['drained'],outcome
        await m.shutdown_async();rt.close()
    app=FastAPI(lifespan=lifespan)
    monkeypatch.setattr(server,'AUTH_ENABLED',True)
    identities={'Bearer native-test':{'username':'native'},'Bearer other-test':{'username':'other'}}
    monkeypatch.setattr(server.SamcloudAuthMiddleware,'_get_cached',lambda middleware,token:identities[token])
    app.add_middleware(server.SamcloudAuthMiddleware,verify_url='https://not-called.invalid',required_scope='services')
    listener=socket.socket();listener.bind(('127.0.0.1',0));url='http://127.0.0.1:'+str(listener.getsockname()[1])
    host=uvicorn.Server(uvicorn.Config(app,log_level='error',lifespan='on'))
    thread=threading.Thread(target=lambda:host.run(sockets=[listener]),daemon=True);thread.start()
    limit=time.monotonic()+10  # fixture startup observation only
    while not host.started:
        assert thread.is_alive() and time.monotonic()<limit,'synthetic root HTTP fixture did not start'
        time.sleep(.01)
    state['roots']=roots;state['ordinary']=ordinary
    try:yield url,observed,journal,state
    finally:
        host.should_exit=True;thread.join(10);listener.close();root_client.close()
        assert not thread.is_alive()


@pytest.mark.parametrize('provider',['samcloud-queue','samcloud-queue-stint'])
@pytest.mark.parametrize('child_count,interrupt_after_child,durable_wait',[
    (1,False,False),(1,True,False),(0,False,False),(2,False,False),(1,False,True)])
def test_native_turn_tool_recovery_holds_same_physical_root(tmp_path,monkeypatch,root_queue,
        provider,child_count,interrupt_after_child,durable_wait):
    _native_scenario(tmp_path,monkeypatch,root_queue,provider,child_count,interrupt_after_child,durable_wait)
    _,observed,_,state=root_queue
    native={row['lease_id'] for row in observed if row['owner']=='native'}
    other={row['lease_id'] for row in observed if row['owner']=='other'}
    assert len(native)==len(other)==1 and native.isdisjoint(other)
    assert len(state['roots'])==2
    first=next(iter(state['roots'].values()));assert first['state']=='complete' and first['lease']['status']=='released'
    state['ordinary'].request_lease.assert_not_called();state['ordinary'].release_lease.assert_not_called()
