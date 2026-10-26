"""Actual ModelManager accounting paths, deterministic providers, no GPU/network."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
import threading

import httpx
import pytest

from . import manager as module
from .manager import ModelManager,Backend
from .inference_arbiter import Conflict,CapacityDeferred
from .inference_root_scheduler import RootPreparingArbiter
from .inference_root_client import RootIntentStore,RegistryRootClient
from .inference_root_models import RootModelBinding

BASE='https://registry.test/api/v1'


class Ollama:
    def __init__(self):self.running={};self.actual=46000;self.loads=[];self.unloads=[];self.crash=False
    def memory_estimate_mb(self,name):return 46000
    def list_models(self):return [{'name':'text'}]
    def num_ctx_for(self,name):return None
    def load_model(self,name,**kwargs):
        self.loads.append(name)
        if self.crash:raise TimeoutError('synthetic unknown startup')
        self.running[name]=self.actual
    def list_running(self):return [{'name':name,'size':size*1024*1024} for name,size in self.running.items()]
    def unload_model(self,name,**kwargs):
        assert kwargs=={'service_owned':True}
        self.unloads.append(name);self.running.pop(name,None);return {'done':True}


class Child:
    def __init__(self,args):self.args=args;self.pid=12345;self.alive=True;self.returncode=None
    def poll(self):return None if self.alive else 0
    def terminate(self):self.alive=False;self.returncode=0


@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'RESOURCE_ID','host/gpu');monkeypatch.setattr(module,'SERVICE_ID','models')
    a=RootPreparingArbiter(tmp_path/'jobs.sqlite',registry=BASE,registry_actor='provider',service_id='models')
    grant=a.submit('one','grant','host/gpu','text','execution')
    call=a.enqueue(grant['id'],'one','call',{'fixture':True},job_key='parent')
    prep=a.prepare_root('host/gpu')
    lease={'id':'root-lease','resource_id':'host/gpu','service_id':'models','agent_id':'provider',
           'exclusive':1,'memory_mb':49152,'status':'active','expires_at':'2099-01-01T00:00:00+00:00'}
    body={'root_id':'root-id','initial_lease_id':lease['id'],'current_lease_id':lease['id'],
          'resource_id':'host/gpu','state':'open','lifetime':'explicit_end','health_ttl_s':120,'lease':lease}
    def view(*_):return {'id':'host/gpu','specs':{'unified_memory_mb':49152},'allocation_mode':'shared',
                         'expiry_policy':'hold','leases':[deepcopy(body['lease'])],'unsettled':[]}
    control=[];release_failure={'enabled':False};cold={'confirmed':True}
    def registry(request):
        control.append((request.method,request.url.path))
        if request.url.path.endswith('capabilities/lease-roots'):
            return httpx.Response(200,json={'protocol':'samcloud.lease-roots','version':1,'actor':'provider',
                'features':{name:True for name in RegistryRootClient.FEATURES}})
        if request.url.path.endswith('/roots/end'):
            body['state']='ending' if body['state']!='complete' else 'complete'
        if request.url.path.endswith('/settle'):
            if release_failure['enabled']:raise httpx.ReadTimeout('synthetic lost release',request=request)
            body['state']='complete';body['lease']['status']='released'
        return httpx.Response(200,json=body)
    store=RootIntentStore(tmp_path/'root.sqlite',registry=BASE,actor='provider',service='models')
    store.prepare(prep['id'],'host/gpu');store.record(prep['id'],'host/gpu',body)
    client=RegistryRootClient(store,token='synthetic',transport=httpx.MockTransport(registry))
    sc=SimpleNamespace(base=BASE,get_resource=view,request_lease=Mock(),release_lease=Mock(),renew_lease=Mock())
    m=object.__new__(ModelManager);m.sc=sc;m.models={};m.root_binding=None
    m.durable_inference=True;m.initial_reservations=None;m.residency_renewal=None;m._unreleased={}
    m.ollama=Ollama();m.refuse_if_device_in_use=lambda *_:None
    child=[]
    monkeypatch.setattr(module.capacity,'collect',lambda:{'memory_available_mb':64000-sum(m.ollama.running.values())-sum(5700 for p in child if p.alive)})
    monkeypatch.setattr(module,'match_vlm_model',lambda name:('vision',5700))
    from . import inference_shared_resource,vision_lifecycle
    monkeypatch.setattr(inference_shared_resource,'cold_vision_absent',lambda:True)
    def spawn(args,**kwargs):
        result=Child(args);child.append(result);return result
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    monkeypatch.setattr(httpx,'get',lambda *args,**kwargs:SimpleNamespace(status_code=200))
    monkeypatch.setattr(vision_lifecycle,'stop_owned_vision',lambda mm:mm.vlm_process.terminate())
    binding=RootModelBinding(m,a,client,resource='host/gpu',models={'text':Backend.OLLAMA,'vision':Backend.VLM},confirm_cold=lambda:cold['confirmed'])
    binding.publish(prep['id'],body,view())
    yield SimpleNamespace(a=a,m=m,sc=sc,client=client,store=store,binding=binding,body=body,view=view,prep=prep,child=child,control=control,call=call,release_failure=release_failure,cold=cold)
    client.close();a.close()


def test_large_text_to_vision_switch_reuses_full_root_without_self_claim(env):
    e=env;text=e.m.load_ollama_model('text')
    assert text.lease_id=='root-lease' and text.memory_mb==46000
    vision=e.m.load_vlm_model('vision')
    assert e.m.ollama.unloads==['text'] and set(e.m.models)=={'vision'}
    assert vision.lease_id==text.lease_id=='root-lease'
    assert [r['state'] for r in e.binding._rows()]==['resident']
    e.sc.request_lease.assert_not_called();e.sc.release_lease.assert_not_called()
    e.m.unload('vision',require_confirmed=True)
    assert not e.binding._rows() and not e.m.models
    e.sc.release_lease.assert_not_called()


def test_ordinary_lease_mutation_cannot_escape_root_mode(env):
    for name,args in [('request_lease',()),('release_lease',('root-lease',)),('renew_lease',('root-lease',)),('rollover_lease',('root-lease',))]:
        with pytest.raises(RuntimeError):getattr(env.m.sc,name)(*args)
    env.sc.request_lease.assert_not_called();env.sc.release_lease.assert_not_called()


def test_unknown_ollama_startup_keeps_exact_debt_and_prevents_repeat_load(env):
    e=env;e.m.ollama.crash=True
    with pytest.raises(TimeoutError):e.m.load_ollama_model('text')
    rows=e.binding._rows();assert len(rows)==1 and rows[0]['state']=='loading' and rows[0]['lease_id']=='root-lease'
    with pytest.raises(Conflict):e.m.load_ollama_model('text')
    assert e.m.ollama.loads==['text'];e.sc.release_lease.assert_not_called()
    with pytest.raises(Conflict):
        e.binding.reconcile_gone(rows[0]['id'],observation='missing-no-work-proof',confirm_absent=lambda _:True,confirm_no_work=lambda:False)
    e.binding.reconcile_gone(rows[0]['id'],observation='fresh-absence-and-no-work',confirm_absent=lambda _:True,confirm_no_work=lambda:True)
    e.m.ollama.crash=False
    assert e.m.load_ollama_model('text').lease_id=='root-lease'
    e.sc.request_lease.assert_not_called()


def test_post_load_oversize_preserves_exact_handle_for_cleanup(env):
    e=env;e.m.ollama.actual=50000
    with pytest.raises(Conflict):e.m.load_ollama_model('text')
    assert e.m.models['text'].root_model_intent and e.binding._rows()[0]['state']=='unknown'
    assert e.m.models['text'].lease_id=='root-lease'
    e.m.unload('text',require_confirmed=True)
    assert not e.binding._rows();e.sc.request_lease.assert_not_called();e.sc.release_lease.assert_not_called()


def test_switch_never_stops_inflight_model(env):
    e=env;text=e.m.load_ollama_model('text');text.in_flight=1
    with pytest.raises(RuntimeError):e.m.load_vlm_model('vision')
    assert not e.child and e.m.ollama.unloads==[] and e.m.models['text'] is text
    e.sc.release_lease.assert_not_called()


def test_vision_handle_is_owned_before_health_wait_and_failed_startup_keeps_claim(env,monkeypatch):
    e=env
    def unhealthy(*args,**kwargs):
        mm=e.m.models['vision']
        assert mm.root_model_intent and mm.vlm_process is e.child[0]
        mm.vlm_process.terminate();return SimpleNamespace(status_code=503)
    monkeypatch.setattr(httpx,'get',unhealthy);monkeypatch.setattr(module.time,'sleep',lambda *_:None)
    with pytest.raises(RuntimeError):e.m.load_vlm_model('vision')
    assert e.m.models['vision'].vlm_process is e.child[0]
    assert e.binding._rows()[0]['state']=='unknown'
    e.m.unload('vision',require_confirmed=True)
    assert not e.binding._rows();e.sc.release_lease.assert_not_called()


def test_provider_worker_thread_uses_its_own_sqlite_connections(env):
    values=[]
    def run():
        try:values.append(env.m.load_ollama_model('text').lease_id)
        except BaseException as exc:values.append(exc)
    thread=threading.Thread(target=run);thread.start();thread.join(5)
    assert not thread.is_alive() and values==['root-lease']


def test_persistent_withdrawal_and_client_end_refuse_resident_reuse(env):
    e=env;e.m.load_ollama_model('text');e.a.withdraw_root(e.prep['id'])
    with pytest.raises(CapacityDeferred):e.m.load_ollama_model('text')
    e.binding.publish(e.prep['id'],e.body,e.view());e.store.request_end(e.prep['id'],'host/gpu')
    with pytest.raises(CapacityDeferred):e.m.load_ollama_model('text')
    # Cleanup of known owned work remains possible after withdrawal/end.
    e.m.unload('text',require_confirmed=True);e.sc.release_lease.assert_not_called()


def test_root_health_renews_once_for_parent_not_each_model(env):
    e=env
    result=asyncio.run(e.m.renew_model_leases())
    assert len(result)==1 and result[0]['current_lease_id']=='root-lease'
    assert len([p for method,p in e.control if method=='POST' and p.endswith('/renew')])==1
    e.sc.renew_lease.assert_not_called();e.sc.request_lease.assert_not_called()


@pytest.mark.parametrize('entry,args',[('load_llama_model',('file.gguf',)),('load_whisper_model',('tiny',)),('resolve_exo_tier',('think',))])
def test_unqualified_backends_refuse_before_provider_work(env,entry,args):
    with pytest.raises(RuntimeError):getattr(env.m,entry)(*args)
    assert not env.binding._rows();env.sc.request_lease.assert_not_called()


def test_admission_lost_after_provider_load_is_never_safe_to_repeat(env):
    e=env;load=e.m.ollama.load_model
    def lose_after_load(*args,**kwargs):
        load(*args,**kwargs);e.a.withdraw_root(e.prep['id'])
    e.m.ollama.load_model=lose_after_load
    with pytest.raises(Conflict):e.m.load_ollama_model('text')
    assert e.m.models['text'].root_model_intent and e.binding._rows()[0]['state']=='unknown'
    assert e.m.ollama.loads==['text'];e.sc.release_lease.assert_not_called()


def test_spawn_handle_survives_metadata_link_failure(env,monkeypatch):
    e=env
    def fail_link(mm):raise RuntimeError('synthetic metadata failure')
    monkeypatch.setattr(e.binding,'bind_handle',fail_link)
    with pytest.raises(RuntimeError):e.m.load_vlm_model('vision')
    assert e.m.models['vision'].vlm_process is e.child[0]
    assert e.child[0].alive and e.binding._rows()[0]['state']=='loading'
    e.sc.release_lease.assert_not_called()


def drained(e):
    assert e.a.dispatch_next('host/gpu',permitted_model=lambda *_:True)['execution']['id']==e.call['id']
    e.a.finish(e.call['id'],{'fixture':'complete'});e.a.end_job('one','parent')
    assert e.a.preparation(e.prep['id'])['state']=='ending'


def test_explicit_end_unloads_then_observes_then_settles_before_logical_finish(env):
    e=env;e.m.load_ollama_model('text');drained(e)
    final=asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='fresh-provider-absence'))
    assert final['state']=='complete' and e.m.ollama.unloads==['text']
    assert e.a.job('one','parent','host/gpu')['state']=='closed' and not e.binding._rows()
    e.sc.release_lease.assert_not_called()
    assert len([path for method,path in e.control if path.endswith('/settle')])==1


def test_unconfirmed_cold_absence_retains_root_and_end_debt(env):
    e=env;e.m.load_ollama_model('text');drained(e);e.cold['confirmed']=False
    with pytest.raises(Conflict):asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='unknown-absence'))
    assert e.body['lease']['status']=='active' and e.a.owns_job('host/gpu')
    assert e.a.preparation(e.prep['id'])['state']=='ending'
    assert not any(path.endswith('/settle') for _,path in e.control)
    e.cold['confirmed']=True
    assert asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='fresh-absence'))['state']=='complete'


def test_lost_release_does_not_close_parent_or_create_another_claim(env):
    e=env;e.m.load_ollama_model('text');drained(e);e.release_failure['enabled']=True
    with pytest.raises(httpx.ReadTimeout):asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='fresh-absence'))
    assert e.body['lease']['status']=='active' and e.a.owns_job('host/gpu')
    assert e.store.prepare(e.prep['id'],'host/gpu')['end_requested']==1
    e.release_failure['enabled']=False
    assert asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='fresh-absence'))['state']=='complete'
    e.sc.request_lease.assert_not_called();e.sc.release_lease.assert_not_called()


def test_end_refuses_before_accepted_work_drains(env):
    e=env
    with pytest.raises(Conflict):asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='not-drained'))
    assert e.control==[] and e.a.execution(e.call['id'],'one')['state']=='queued'


def test_ending_can_reconcile_start_intent_only_with_fresh_positive_absence(env):
    e=env;e.m.ollama.crash=True
    with pytest.raises(TimeoutError):e.m.load_ollama_model('text')
    assert e.binding._rows()[0]['state']=='loading'
    drained(e);e.cold['confirmed']=False
    with pytest.raises(Conflict):asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='unknown'))
    assert e.binding._rows()[0]['state']=='loading'
    e.cold['confirmed']=True
    assert asyncio.run(e.binding.end_drained_root(e.prep['id'],observation='fresh-global-absence'))['state']=='complete'
    assert not e.binding._rows()


def test_newer_expired_registry_receipt_invalidates_old_cached_permission(env):
    e=env;e.m.load_ollama_model('text')
    expired=deepcopy(e.body);expired['lease']['status']='expired';expired['lease']['expires_at']='2000-01-01T00:00:00+00:00'
    e.store.record(e.prep['id'],'host/gpu',expired)
    with pytest.raises(CapacityDeferred):e.m.load_ollama_model('text')
    e.sc.request_lease.assert_not_called()


def test_failed_refresh_withdraws_resident_permission_without_releasing(env):
    e=env;e.m.load_ollama_model('text')
    def unavailable(*args):raise TimeoutError('synthetic observer unavailable')
    e.client.observe=unavailable
    with pytest.raises(TimeoutError):asyncio.run(e.binding.refresh())
    with pytest.raises(CapacityDeferred):e.m.load_ollama_model('text')
    assert e.a.preparation(e.prep['id'])['state']=='prepared'
    e.sc.release_lease.assert_not_called()


def test_background_clock_cooldown_cannot_release_parent_models(env):
    e=env;text=e.m.load_ollama_model('text');text.last_used=0
    e.m.work_in_progress=lambda *_:True;e.m.offer_reading=lambda:{'memory_device_inuse_mb':46000}
    e.m.own_device_mb=lambda:46000
    assert e.m.check_cooldowns()==[] and e.m.models['text'] is text
    e.sc.release_lease.assert_not_called()


def test_heartbeat_uses_root_contract_and_status_reads_cannot_postpone_it(env):
    import time
    e=env;due=e.binding._health_due
    assert 0<due-time.monotonic()<=30
    for _ in range(3):e.binding.publish(e.prep['id'],e.body,e.view())
    assert e.binding._health_due==due
    e.binding._health_due=time.monotonic()-1
    asyncio.run(e.binding.wait_for_renewal())
    assert not e.control  # This is only a health wakeup, not inference or release.


def test_failed_heartbeat_keeps_nonzero_health_retry_delay_and_owned_claim(env):
    import time
    e=env
    def fail(*args):raise TimeoutError('synthetic registry observation failure')
    e.client.observe=fail
    with pytest.raises(TimeoutError):asyncio.run(e.m.renew_model_leases())
    assert 0<e.binding._health_due-time.monotonic()<=30
    assert e.a.owns_job('host/gpu');e.sc.request_lease.assert_not_called();e.sc.release_lease.assert_not_called()


def test_background_loop_is_not_allowed_before_root_mode_attachment(env):
    e=env;e.m._background_started=True
    # New binding cannot be hot-installed over a running legacy maintenance clock.
    e.m.root_binding=None
    with pytest.raises(Conflict):
        RootModelBinding(e.m,e.a,e.client,resource='host/gpu',models={'text':Backend.OLLAMA},confirm_cold=lambda:True)


def test_profile_backend_must_be_explicit_enum_before_attachment(env):
    e=env;e.m.root_binding=None
    with pytest.raises(Conflict):
        RootModelBinding(e.m,e.a,e.client,resource='host/gpu',models={'text':'ollama'},confirm_cold=lambda:True)
