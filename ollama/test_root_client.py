"""Durable client intent and transport refusal, without a production registry."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from .inference_root_client import RegistryRootClient, RootIntentStore, RootProtocolError

BASE='https://registry.test/api/v1/'


def store(tmp_path,**changes):
    return RootIntentStore(tmp_path/'client.sqlite',**dict(registry=BASE,actor='owner',service='models',**changes))


def capability(**changes):
    value={'protocol':'samcloud.lease-roots','version':1,'actor':'owner',
           'features':{key:True for key in RegistryRootClient.FEATURES}}
    value.update(changes)
    return value


def receipt(**changes):
    value={'root_id':'root_saved','initial_lease_id':'lease_initial','current_lease_id':'lease_initial',
           'resource_id':'host/gpu','lifetime':'explicit_end','state':'open','health_ttl_s':120,
           'lease':{'id':'lease_initial','resource_id':'host/gpu','agent_id':'owner','service_id':'models','status':'queued'}}
    value.update(changes)
    return value


def client(tmp_path,handler):
    return RegistryRootClient(store(tmp_path),token='test-secret',transport=httpx.MockTransport(handler))


def test_lost_submit_reply_reuses_persisted_key_after_fresh_process(tmp_path):
    seen=[]
    def lost(request):
        if request.method=='GET':return httpx.Response(200,json=capability())
        seen.append(json.loads(request.content))
        assert store(tmp_path).prepare('job','host/gpu')['submit_key']==seen[-1]['key']
        raise httpx.ReadTimeout('observer disconnected',request=request)
    first=client(tmp_path,lost)
    with pytest.raises(httpx.ReadTimeout):first.begin('job','host/gpu')
    first.close()
    def reply(request):
        if request.url.path.endswith('capabilities/lease-roots'):return httpx.Response(200,json=capability())
        seen.append(json.loads(request.content))
        return httpx.Response(200,json=receipt())
    second=client(tmp_path,reply)
    assert second.begin('job','host/gpu')['root_id']=='root_saved'
    second.close()
    assert seen[0]==seen[1]
    # New Python process reads the same committed intent, not an in-memory clone.
    code="from ollama.inference_root_client import RootIntentStore; import sys; s=RootIntentStore(sys.argv[1],registry=sys.argv[2],actor='owner',service='models'); print(s.prepare('job','host/gpu')['submit_key'])"
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'client.sqlite'),BASE],
                          cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()==seen[0]['key']


def test_end_is_sticky_even_if_late_open_ack_arrives(tmp_path):
    s=store(tmp_path);s.prepare('job','host/gpu');s.request_end('job','host/gpu')
    s.record('job','host/gpu',receipt())
    seen=[]
    def handler(request):
        seen.append((request.method,request.url.path))
        if request.method=='GET':return httpx.Response(200,json=capability())
        return httpx.Response(200,json=receipt(state='ending'))
    c=client(tmp_path,handler)
    assert c.begin('job','host/gpu')['state']=='ending'
    c.close()
    assert seen[-1][1].endswith('/roots/end') and all(not p.endswith('/roots') for _,p in seen)
    assert store(tmp_path).prepare('job','host/gpu')['end_requested']==1


def test_failed_end_persists_before_handshake_or_intake(tmp_path):
    seen=[]
    def handler(request):
        seen.append(request.method)
        return httpx.Response(404)
    c=client(tmp_path,handler)
    with pytest.raises(httpx.HTTPStatusError):c.end('job','host/gpu')
    c.close()
    assert seen==['GET'] and store(tmp_path).prepare('job','host/gpu')['end_requested']==1


@pytest.mark.parametrize('cap',[capability(actor='other'),capability(version=True),
    capability(features={}),capability(protocol='ordinary-leases')])
def test_unsupported_contract_never_falls_back_to_ordinary_admission(tmp_path,cap):
    seen=[]
    def handler(request):
        seen.append(request.method);return httpx.Response(200,json=cap)
    c=client(tmp_path,handler)
    with pytest.raises(RootProtocolError):c.begin('job','host/gpu')
    c.close()
    assert seen==['GET'] and store(tmp_path).prepare('job','host/gpu')['root_id'] is None


def test_saved_root_missing_or_redirected_never_replaced(tmp_path):
    s=store(tmp_path);s.prepare('job','host/gpu');s.record('job','host/gpu',receipt())
    seen=[]
    def handler(request):
        seen.append(request.method)
        if request.url.path.endswith('capabilities/lease-roots'):return httpx.Response(200,json=capability())
        return httpx.Response(307,headers={'Location':'https://different.test/steal'})
    c=client(tmp_path,handler)
    with pytest.raises(httpx.HTTPStatusError):c.begin('job','host/gpu')
    c.close()
    assert seen==['GET','GET']


def test_close_does_not_infer_job_end_or_release(tmp_path):
    seen=[]
    c=client(tmp_path,lambda r:seen.append(r))
    c.store.prepare('job','host/gpu');c.close()
    assert seen==[] and store(tmp_path).prepare('job','host/gpu')['end_requested']==0


def test_concurrent_prepare_and_identity_binding(tmp_path):
    s=store(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(lambda _:s.prepare('job','host/gpu'),range(24)))
    assert len({row['submit_key'] for row in rows})==1
    assert s.prepare('new-job','host/gpu')['submit_key']!=rows[0]['submit_key']
    for field,value in [('actor','other'),('service','other'),('registry','https://other.test/api/v1')]:
        args={'registry':BASE,'actor':'owner','service':'models',field:value}
        with pytest.raises(PermissionError):RootIntentStore(tmp_path/'client.sqlite',**args)
    assert 'test-secret' not in (tmp_path/'client.sqlite').read_bytes().decode(errors='replace')


@pytest.mark.parametrize('base',['http://registry.test/api/v1','https://user:secret@registry.test/api/v1',
    'https://registry.test/api/v1?token=x','https://registry.test/other'])
def test_unsafe_origins_refused_before_http(tmp_path,base):
    with pytest.raises(ValueError):RootIntentStore(tmp_path/'client.sqlite',registry=base,actor='owner',service='models')


def test_nonprivate_and_symlink_journals_are_refused(tmp_path):
    path=tmp_path/'client.sqlite';path.touch(mode=0o644)
    with pytest.raises(PermissionError):store(tmp_path)
    path.unlink();target=tmp_path/'target';target.touch(mode=0o600);path.symlink_to(target)
    with pytest.raises(OSError):store(tmp_path)


def test_saved_identity_and_terminal_end_cannot_be_overwritten(tmp_path):
    s=store(tmp_path);s.prepare('job','host/gpu');s.record('job','host/gpu',receipt())
    with pytest.raises(RootProtocolError):s.record('job','host/gpu',receipt(root_id='replacement'))
    s.record('job','host/gpu',receipt(state='complete'))
    with pytest.raises(RootProtocolError):s.record('job','host/gpu',receipt())


def test_absence_operation_uses_same_saved_key_after_lost_reply(tmp_path):
    s=store(tmp_path);s.prepare('job','host/gpu');s.record('job','host/gpu',receipt())
    bodies=[]
    def handler(request):
        if request.method=='GET':return httpx.Response(200,json=capability())
        bodies.append(json.loads(request.content));raise httpx.ReadTimeout('lost reply',request=request)
    for _ in range(2):
        c=client(tmp_path,handler)
        with pytest.raises(httpx.ReadTimeout):c.settle('job','host/gpu',lease_id='lease_initial',observation='fresh-absence')
        c.close()
    assert bodies[0]==bodies[1]
    assert bodies[0]['lease_id']=='lease_initial' and bodies[0]['key']


def test_end_arriving_during_submit_is_flushed_before_open_ack_returns(tmp_path):
    seen=[]
    def handler(request):
        seen.append(request.url.path)
        if request.method=='GET':return httpx.Response(200,json=capability())
        if request.url.path.endswith('/roots'):
            store(tmp_path).request_end('job','host/gpu')
            return httpx.Response(200,json=receipt())
        return httpx.Response(200,json=receipt(state='ending'))
    c=client(tmp_path,handler)
    assert c.begin('job','host/gpu')['state']=='ending'
    assert seen[-1].endswith('/roots/end')
    c.close()


def test_end_intent_forbids_recovery_readmission_even_if_end_http_never_arrived(tmp_path):
    s=store(tmp_path);s.prepare('job','host/gpu');s.record('job','host/gpu',receipt())
    s.request_end('job','host/gpu');seen=[]
    c=client(tmp_path,lambda request:seen.append(request))
    with pytest.raises(RootProtocolError):
        c.recover_absent('job','host/gpu',lease_id='lease_initial',observation='absent')
    assert seen==[];c.close()


@pytest.mark.parametrize('body',[receipt(resource_id='other/gpu'),receipt(lease=None),
    receipt(current_lease_id='different'),receipt(lifetime='execution')])
def test_wrong_receipt_never_overwrites_saved_intent(tmp_path,body):
    def handler(request):
        return httpx.Response(200,json=capability() if request.method=='GET' else body)
    c=client(tmp_path,handler)
    with pytest.raises(RootProtocolError):c.begin('job','host/gpu')
    assert c.store.prepare('job','host/gpu')['root_id'] is None
    c.close()
