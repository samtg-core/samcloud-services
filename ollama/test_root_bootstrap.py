"""Actual runtime/API/lifecycle wiring; fake registry/provider, no GPU/network."""
import asyncio
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading

from fastapi import FastAPI
import httpx
import pytest

from . import manager as module,server
from .manager import ModelManager,Backend
from .inference_arbiter import Arbiter,Conflict
from .inference_bootstrap import QueueSettings
from .inference_root_client import RootIntentStore,RegistryRootClient,RootProtocolError
from .inference_root_bootstrap import RootQueueStartup
from .test_root_models import Ollama

BASE='https://registry.test/api/v1'


@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'RESOURCE_ID','host/gpu');monkeypatch.setattr(module,'SERVICE_ID','models')
    state={'cold':True,'capable':True,'active':False,'posts':[]}
    claim={'id':'lease-root','resource_id':'host/gpu','service_id':'models','agent_id':'provider',
           'exclusive':1,'memory_mb':49152,'status':'active','expires_at':'2099-01-01T00:00:00+00:00'}
    body={'root_id':'root-id','initial_lease_id':'lease-root','current_lease_id':'lease-root',
          'resource_id':'host/gpu','state':'open','lifetime':'explicit_end','health_ttl_s':120,'lease':claim}
    def transport(request):
        if request.url.path.endswith('capabilities/lease-roots'):
            return httpx.Response(200,json={'protocol':'samcloud.lease-roots','version':1,'actor':'provider',
                'features':{name:state['capable'] for name in RegistryRootClient.FEATURES}})
        if request.method=='POST':
            state['posts'].append(request.url.path)
            if request.url.path.endswith('/roots'):state['active']=True
            if request.url.path.endswith('/end'):body['state']='ending'
            if request.url.path.endswith('/settle'):
                body['state']='complete';claim['status']='released'
        return httpx.Response(200,json=deepcopy(body))
    roots=RegistryRootClient(RootIntentStore(tmp_path/'root.sqlite',registry=BASE,actor='provider',service='models'),
                             token='fixture-only',transport=httpx.MockTransport(transport))
    sc=SimpleNamespace(base=BASE,verify_token=Mock(return_value={'valid':True,'owner_id':'provider'}),
        get_resource=lambda *_:{'id':'host/gpu','allocation_mode':'shared','expiry_policy':'hold',
            'specs':{'unified_memory_mb':49152},'leases':[deepcopy(claim)] if state['active'] else [],'unsettled':[]},
        request_lease=Mock(),release_lease=Mock(),renew_lease=Mock())
    m=ModelManager.__new__(ModelManager);m.sc=sc;m.models={};m.root_binding=None;m.durable_inference=False
    m.initial_reservations=None;m.residency_renewal=None;m.inference_capacity_source=None;m._background_started=False
    m._pool_lock=threading.Lock();m._pool_leases=set();m._unreleased={};m.ollama=Ollama()
    m.refuse_if_device_in_use=lambda *_:None;m.discover=Mock();m.claim_leases=Mock();m.start_background_tasks=Mock()
    for name in m.MAINTENANCE_TASKS:setattr(m,name,None)
    def physical():
        used=sum(m.ollama.running.values())
        return {'memory_available_mb':64000-used,'memory_device_inuse_mb':used}
    monkeypatch.setattr(module.capacity,'collect',physical)
    app=FastAPI()
    @app.middleware('http')
    async def own_identity(request,call_next):
        request.state.caller={'username':'requester'};return await call_next(request)
    config=SimpleNamespace(SC_RESOURCE_ID='host/gpu',SC_SERVICE_ID='models',EXO_ENABLED=False,WHISPER_MAX_UPLOAD_MB=1)
    svc=SimpleNamespace(mgr=m,app=app,config=config,inference_dispatcher=None,inference_runtime=None,
        ChatRequest=server.ChatRequest,CompletionRequest=server.CompletionRequest,Backend=Backend)
    gate={'entered':None,'release':None,'ambiguous':False}
    async def resolve(model,**kwargs):return m.load_ollama_model(model)
    async def chat(req,**kwargs):
        if gate['entered'] is not None:gate['entered'].set();await gate['release'].wait()
        if gate['ambiguous']:raise TimeoutError('fixture unknown completion')
        return {'choices':[{'message':{'content':'fixture'},'finish_reason':'stop'}]}
    svc._resolve_model=resolve;svc._chat_completions_direct=chat
    settings=QueueSettings(tmp_path/'jobs.sqlite',{'text':'host/gpu'})
    startup=RootQueueStartup(settings,roots,{'text':Backend.OLLAMA},lambda:state['cold'],True)
    yield SimpleNamespace(m=m,sc=sc,roots=roots,state=state,body=body,service=svc,settings=settings,startup=startup,
                          physical=physical,gate=gate)
    rt=svc.inference_runtime
    if rt is not None and not rt._closed:rt.queue.supervisor.close()  # fixture handles only, no physical compute
    roots.close()


async def drain(rt):
    s=rt.queue.supervisor
    while tasks:=tuple(t for t in (*s.preparations,*s.workers.values()) if not t.done()):
        await asyncio.gather(*tasks,return_exceptions=True)


def enqueue(rt,key='call'):
    a=rt.queue.supervisor.journal;g=a.submit('requester','grant','host/gpu','text','execution')
    call=a.enqueue(g['id'],'requester',key,{'operation':'chat','body':{'model':'text','messages':[]},'release_after':False},job_key='parent')
    rt.queue.supervisor.pump('host/gpu');return call


def test_start_is_cold_unobserved_and_does_not_allocate_on_intake(env):
    async def scenario():
        e=env;rt=await e.startup.start(e.service);call=enqueue(rt);await drain(rt)
        assert e.service.inference_dispatcher is rt.queue and e.m.inference_capacity_source is rt.capacity_source
        assert e.m.initial_reservations is None and e.m.residency_renewal is None
        assert rt.queue.supervisor.journal.execution(call['id'],'requester')['state']=='queued'
        assert e.state['posts']==[] and e.m.ollama.loads==[]
        assert not rt.capacity_source.snapshot()['host_known']
        assert (await rt.quiesce())['drained'];await e.m.shutdown_async();rt.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('bad',['identity','capability','cold','maintenance','exo'])
def test_startup_refuses_before_mount_or_manager_mutation(env,bad):
    async def scenario():
        e=env
        if bad=='identity':e.sc.verify_token.return_value={'valid':True,'owner_id':'another'}
        if bad=='capability':e.state['capable']=False
        if bad=='cold':e.state['cold']=False
        if bad=='exo':e.service.config.EXO_ENABLED=True
        if bad=='maintenance':e.m._background_started=True
        with pytest.raises((Conflict,RootProtocolError)):await e.startup.start(e.service)
        assert not e.settings.journal.exists() and e.service.inference_runtime is None
        assert e.m.sc is e.sc and e.m.root_binding is None and not e.m.durable_inference
        assert e.state['posts']==[] and not any(r.path.startswith('/v1/inference') for r in e.service.app.routes)
    asyncio.run(scenario())


def test_failed_mount_restores_manager_routes_and_releases_scheduler_lock(env,monkeypatch):
    async def scenario():
        e=env;before=list(e.service.app.router.routes)
        def fail(router):
            e.service.app.router.routes.extend(router.routes)
            raise RuntimeError('fixture mount failed')
        with monkeypatch.context() as patch:
            patch.setattr(e.service.app,'include_router',fail)
            with pytest.raises(RuntimeError,match='mount failed'):await e.startup.start(e.service)
        assert e.service.app.router.routes==before and e.m.sc is e.sc and not e.m.durable_inference
        assert e.m.root_binding is None and e.m.inference_capacity_source is None
        rt=await e.startup.start(e.service)
        assert (await rt.quiesce())['drained'];await e.m.shutdown_async();rt.close()
    asyncio.run(scenario())


def test_legacy_queued_work_requires_explicit_migration_and_keeps_exact_ids(env):
    e=env;a=Arbiter(e.settings.journal)
    g=a.submit('legacy','grant','host/gpu','text');call=a.enqueue(g['id'],'legacy','call',{})
    a.close()
    async def scenario():
        with pytest.raises(Conflict,match='legacy queue'):await e.startup.start(e.service)
        assert e.m.root_binding is None and e.service.inference_runtime is None and e.state['posts']==[]
        a=Arbiter(e.settings.journal)
        try:assert a.execution(call['id'],'legacy')['state']=='queued'
        finally:a.close()
    asyncio.run(scenario())


def test_real_api_dispatch_then_runtime_stop_retains_parent_and_exact_root(env):
    async def scenario():
        e=env;rt=await e.startup.start(e.service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=e.service.app),base_url='http://fixture') as http:
            grant=await http.post('/v1/inference/grants',json={'key':'api-grant','model':'text','lifetime':'execution'})
            assert grant.status_code==202
            call=await http.post('/v1/inference/grants/'+grant.json()['id']+'/executions',json={
                'key':'api-call','job_key':'parent','request':{'model':'text','messages':[],'stream':False}})
            assert call.status_code==202;ident=call.json()['id']
            await rt.capacity_source.refresh(e.physical());await drain(rt)
            result=await http.get('/v1/inference/executions/'+ident)
            assert result.json()['state']=='complete',rt.capacity_source.snapshot()
            assert result.json()['result']['body']['choices'][0]['message']['content']=='fixture'
            a=rt.queue.supervisor.journal;prep=dict(a._current_preparation('host/gpu'))
            assert a.job('requester','parent','host/gpu')['state']=='owned'
            assert (await rt.quiesce())['drained'];await e.m.shutdown_async();rt.close()
            assert e.body['lease']['status']=='active' and e.body['state']=='open'
            assert e.state['posts'].count('/api/v1/resources/host/gpu/roots')==1
            assert not any(path.endswith(('/end','/settle')) for path in e.state['posts'])
            assert prep['current_lease_id']=='lease-root' and e.m.ollama.unloads==['text']
        e.sc.request_lease.assert_not_called();e.sc.release_lease.assert_not_called()
    asyncio.run(scenario())


def test_unknown_execution_refuses_quiescence_and_keeps_control_and_journal(env):
    async def scenario():
        e=env;rt=await e.startup.start(e.service);e.gate['ambiguous']=True
        call=enqueue(rt);await rt.capacity_source.refresh(e.physical());await drain(rt)
        assert rt.queue.supervisor.journal.execution(call['id'],'requester')['state']=='uncertain'
        state=await rt.quiesce();assert not state['drained'] and state['uncertain']==1
        with pytest.raises(Conflict):await e.m.shutdown_async()
        with pytest.raises(Conflict):rt.close()
        assert not e.m.root_binding._health_stopped and e.body['lease']['status']=='active'
        assert 'text' in e.m.models and rt.queue.supervisor.journal.db.execute('SELECT 1').fetchone()[0]==1
    asyncio.run(scenario())


def test_cancelled_quiesce_observer_keeps_owned_drain_and_same_backend_call(env):
    async def scenario():
        e=env;rt=await e.startup.start(e.service)
        e.gate.update(entered=asyncio.Event(),release=asyncio.Event());call=enqueue(rt)
        await rt.capacity_source.refresh(e.physical());await asyncio.wait_for(e.gate['entered'].wait(),2)
        observer=asyncio.create_task(rt.quiesce());await asyncio.sleep(0);observer.cancel()
        with pytest.raises(asyncio.CancelledError):await observer
        owned=rt._quiesce_task;assert not owned.done()
        with pytest.raises(Conflict):rt.resume()
        e.gate['release'].set();state=await rt.quiesce()
        assert state['drained'] and rt._quiesce_task is owned
        assert rt.queue.supervisor.journal.execution(call['id'],'requester')['state']=='complete'
        await e.m.shutdown_async();rt.close();assert e.body['lease']['status']=='active'
    asyncio.run(scenario())


def test_refreshed_capability_failure_discards_old_positive_and_blocks_later_begin(env):
    e=env;e.roots.negotiate();assert e.roots._verified
    e.state['capable']=False
    async def scenario():
        with pytest.raises(RootProtocolError):await e.startup.start(e.service)
        assert not e.roots._verified
        with pytest.raises(RootProtocolError):e.roots.begin('future-intent','host/gpu')
        assert e.state['posts']==[] and e.service.inference_runtime is None
    asyncio.run(scenario())


def test_false_cold_suspension_keeps_control_alive_and_explicit_retry_reuses_root(env):
    async def scenario():
        e=env;rt=await e.startup.start(e.service);enqueue(rt)
        await rt.capacity_source.refresh(e.physical());await drain(rt)
        e.state['cold']=False
        assert not (await rt.quiesce())['drained']
        with pytest.raises(Conflict):await e.m.shutdown_async()
        assert not e.m.root_binding._health_stopped and e.body['lease']['status']=='active'
        e.state['cold']=True
        assert (await rt.quiesce())['drained'];await e.m.shutdown_async();rt.close()
        assert e.state['posts'].count('/api/v1/resources/host/gpu/roots')==1
        assert not any(path.endswith(('/end','/settle')) for path in e.state['posts'])
    asyncio.run(scenario())


def test_actual_server_lifespan_accepts_explicit_root_binding_before_maintenance(env,monkeypatch):
    async def scenario():
        e=env;e.m.status=Mock(return_value={})
        with monkeypatch.context() as patch:
            for name,value in (('app',e.service.app),('mgr',None),('inference_dispatcher',None),
                ('inference_runtime',None),('inference_startup',e.startup),
                ('SamcloudClient',Mock(return_value=e.sc)),('ModelManager',Mock(return_value=e.m)),('SC_TOKEN','')):
                patch.setattr(server,name,value)
            patch.setattr(server.config,'SC_RESOURCE_ID','host/gpu')
            patch.setattr(server.config,'SC_SERVICE_ID','models')
            patch.setattr(server.config,'EXO_ENABLED',False)
            def maintenance():
                assert server.inference_runtime is not None and e.m.root_binding is not None
                assert e.m.inference_capacity_source is server.inference_runtime.capacity_source
            e.m.start_background_tasks.side_effect=maintenance
            async with server.lifespan(e.service.app):
                rt=server.inference_runtime
                assert server.inference_dispatcher is rt.queue
                assert not rt.capacity_source.snapshot()['host_known']
                status=await server.status()
                assert status['requester_root']['mode']=='requester_root'
                assert status['requester_root']['preparation'] is None
                e.m.discover.assert_not_called();e.m.claim_leases.assert_not_called()
            assert rt._closed and e.m.root_binding._health_stopped
            assert e.state['posts']==[]  # Empty boot/shutdown never buys a physical claim.
    asyncio.run(scenario())


@pytest.mark.parametrize('drift',['other_root','removed_model'])
def test_restart_scope_cannot_leave_other_root_or_accepted_model_behind(env,drift):
    e=env
    if drift=='other_root':e.roots.store.prepare('other-root-intent','host/other-gpu')
    else:
        from .inference_root_scheduler import RootPreparingArbiter
        a=RootPreparingArbiter(e.settings.journal,registry=BASE,registry_actor='provider',service_id='models')
        try:
            g=a.submit('requester','saved','host/gpu','removed-vision')
            a.enqueue(g['id'],'requester','saved-call',{},job_key='saved-parent')
        finally:a.close()
    async def scenario():
        with pytest.raises(Conflict,match='scope|model'):await e.startup.start(e.service)
        assert e.m.root_binding is None and e.service.inference_runtime is None and e.state['posts']==[]
    asyncio.run(scenario())
