"""Real native Relay ownership seam; no turn/model, live profile or network."""
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import relay_runtime
from agent.context_compressor import ContextCompressor
from ollama.hermes_queue_context import QueuedContextCompressor
from ollama.hermes_queue_provider import host_for_home
from ollama.inference_arbiter import Conflict

@pytest.fixture
def bound(tmp_path,monkeypatch):
    home=tmp_path/'own-profile';home.mkdir();monkeypatch.setenv('HERMES_HOME',str(home))
    host=host_for_home(home)
    engine=QueuedContextCompressor(model='qwen',provider='samcloud-queue-stint',
        config_context_length=65536,quiet_mode=True)
    engine.bind_session_state(session_id='session')
    coordinator=relay_runtime.RelaySessionCoordinator()
    lease=relay_runtime.ConversationLease(str(home.resolve()),'session','cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()),'offline test'),None)
    turn=coordinator.begin_turn(lease,turn_id='native-turn',task_id='task')
    history=[{'role':'user','_row_id':1,'content':'fixture'}]
    body={'model':'qwen','messages':[{'role':'user','content':'summary fixture'}]}
    try:
        with patch.object(ContextCompressor,'compress',lambda self,*a,**kw:host.auxiliary_keys(body)):
            yield host,engine,coordinator,lease,turn,history
    finally:
        coordinator.end_turn(turn,outcome='success');coordinator.release_conversation(lease)

def test_real_native_owner_binds_summary_and_turn_finish(bound):
    host,engine,_,_,turn,history=bound
    keys=engine.compress(history)
    host.pre_api_request(session_id='session',turn_id=turn.turn_id,api_request_id='api',conversation_history=history)
    main=host.main_stint_keys({'model':'qwen','messages':[{'role':'user','content':'main fixture'}]})
    assert keys.work_session and keys.grant_key==main.grant_key and keys.call_key!=main.call_key
    # Compaction is again the last call, so turn-end is fenced to that exact call.
    again=engine.compress(history)
    again.on_accepted(SimpleNamespace(grant_id="grant",execution_id=again.call_key))
    assert again==keys
    host.finish_turn('session',turn.turn_id)
    assert host.pending_events()[0]['expected_call_key']==keys.call_key

def test_copied_context_cannot_borrow_closed_native_turn(bound):
    host,engine,coordinator,lease,turn,history=bound
    keys=engine.compress(history);ctx=copy_context()
    coordinator.end_turn(turn,outcome='success');coordinator.release_conversation(lease)
    with pytest.raises(Conflict):ctx.run(engine.compress,history)
    # A different source outside the turn uses the independent execution policy.
    other=ctx.run(engine.compress,[history[0]|{'_row_id':2}])
    assert not other.work_session and other.grant_key!=keys.grant_key

def test_other_session_and_detached_compressor_do_not_borrow(bound):
    host,engine,_,_,_,history=bound
    main=engine.compress(history)
    engine.bind_session_state(session_id='other-session')
    other=engine.compress(history)
    assert not other.work_session and other.grant_key!=main.grant_key
    engine.bind_session_state(session_id='')
    with pytest.raises(ValueError):engine.compress(history)

from .test_hermes_queue_client import queued_profile

@pytest.mark.parametrize('job_mode,provider', [
    (False, 'samcloud-queue-stint'),
    (True, 'samcloud-queue-stint'),
    (True, 'samcloud-queue'),
])
def test_stock_compressor_worker_inherits_owner_and_uses_continuation_wire(tmp_path,monkeypatch,queued_profile,job_mode,provider):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import hermes_cli.config as config
    from .inference_sdk import QueuedClient
    home=tmp_path/'profile';home.mkdir(exist_ok=True)
    if job_mode:
        (home/'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    host=host_for_home(home)
    original_handle = queued_profile.handle
    def job_capabilities(request):
        import httpx
        response = original_handle(request)
        if request.url.path.endswith('/capabilities'):
            data = response.json()
            data['jobs'] = {'version': 1, 'enabled': True,
                            'same_service_children': True, 'explicit_finish': True,
                            'explicit_yield': True}
            return httpx.Response(200, json=data)
        if request.method == 'POST' and request.url.path.endswith('/executions'):
            import json
            body = json.loads(request.content)
            if body.get('job_key'):
                return httpx.Response(200, json=response.json() | {'job_id': 'parent-job'})
        return response
    monkeypatch.setattr(queued_profile, 'handle', job_capabilities)
    original_load=config.load_config
    def cfg(*a,**kw):
        value=original_load(*a,**kw)
        return value|{'auxiliary':{'compression':{'provider':'queue-test','model':'qwen',
            'base_url':'https://queue.invalid/v1','timeout':1}}}
    monkeypatch.setattr(config,'load_config',cfg)
    clients=[]
    def client():
        c=QueuedClient(api_key='synthetic',base_url='https://queue.invalid/v1',
            http_factory=queued_profile.http,poll_interval=.01,complete_responses=True,
            grant_lifetime='execution',key_provider=host.auxiliary_keys)
        clients.append(c);return c
    monkeypatch.setattr(queued_profile,'client',client)
    import providers
    from providers.base import ProviderProfile
    from hermes_cli.auth_plugin_providers import register_plugin_provider
    class StintProfile(ProviderProfile):
        def create_client(self,**kwargs):return client()
    profile=StintProfile(name=provider,auth_type='api_key',
        env_vars=('SYNTHETIC_QUEUE_TEST_KEY',),base_url='https://queue.invalid/v1',default_aux_model='qwen')
    providers.register_provider(profile);register_plugin_provider(profile)
    engine=QueuedContextCompressor(model='qwen',provider=provider,
        api_key='synthetic',base_url='https://queue.invalid/v1',
        config_context_length=65536,quiet_mode=True)
    engine.bind_session_state(session_id='stock-owner')
    engine.threshold_tokens=1000;engine.tail_token_budget=256
    history=[{'role':'user' if i%2==0 else 'assistant','_row_id':i+1,
              'content':('synthetic archive detail '*200)+str(i)} for i in range(40)]
    host.pre_api_request(session_id='stock-owner',turn_id='stock-turn',api_request_id='api',conversation_history=history)
    if job_mode:
        host.begin_foreground('stock-owner', 'stock-turn')
    main=host.main_stint_keys({'model':'qwen','messages':[{'role':'user','content':'main fixture'}]})
    coordinator=relay_runtime.RelaySessionCoordinator()
    lease=relay_runtime.ConversationLease(str(home.resolve()),'stock-owner','cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()),'offline test'),None)
    turn=coordinator.begin_turn(lease,turn_id='stock-turn',task_id='task')
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result=pool.submit(copy_context().run,lambda:engine.compress(history,current_tokens=12000,force=True)).result(timeout=15)
        assert engine._last_summary_fallback_used is False, engine._last_summary_error
        assert len(result)<len(history)
        grants=[r[2] for r in queued_profile.requests if r[:2]==('POST','/v1/inference/grants')]
        assert len(grants)==1
        assert grants[0]['key']==main.grant_key and grants[0]['lifetime']=='work_session'
        calls=[r[2] for r in queued_profile.requests if r[:2]==('POST','/v1/inference/grants/grant/executions')]
        assert len(calls)==1 and calls[0]['key']!=main.call_key
        assert not calls[0].get('yield_grant_keys')
        if job_mode:
            assert calls[0]['job_key'] == main.job_key
        host.finish_turn('stock-owner','stock-turn')
        if job_mode:
            assert host.pending_events()[0]['kind'] == 'job_finish'
            assert host.pending_events()[0]['job_key'] == main.job_key
        else:
            assert host.pending_events()[0]['expected_call_key']==calls[0]['key']
        # A fresh native turn/model recovers the SAME accepted summary through
        # authenticated receipt reads, even with the aux model pinned to qwen.
        coordinator.end_turn(turn,outcome='success');coordinator.release_conversation(lease)
        lease=relay_runtime.ConversationLease(str(home.resolve()),'stock-owner','cli',
            relay_runtime.NoopRelayRuntime(str(home.resolve()),'offline test'),None)
        turn=coordinator.begin_turn(lease,turn_id='stock-resumed',task_id='resume-task')
        old_handle=queued_profile.handle
        def receipt_handle(request):
            import httpx
            response=old_handle(request)
            if request.url.path.startswith('/v1/inference/executions/') and not request.url.path.endswith('/events'):
                data=response.json();data.update(id=request.url.path.rsplit('/',1)[1],grant_id='grant')
                return httpx.Response(response.status_code,json=data)
            return response
        monkeypatch.setattr(queued_profile,'handle',receipt_handle)
        restored=QueuedContextCompressor(model='qwen-new',provider=provider,
            api_key='synthetic',base_url='https://queue.invalid/v1',config_context_length=65536,quiet_mode=True)
        restored.bind_session_state(session_id='stock-owner')
        restored.threshold_tokens=1000;restored.tail_token_budget=256
        before=len(queued_profile.requests)
        with ThreadPoolExecutor(max_workers=1) as pool:
            recovered=pool.submit(copy_context().run,lambda:restored.compress(history,current_tokens=12000,force=True)).result(timeout=15)
        assert restored._last_summary_fallback_used is False,restored._last_summary_error
        assert len(recovered)<len(history)
        assert all(method=='GET' for method,_,_ in queued_profile.requests[before:])
        assert len(queued_profile.calls)==1
        host.finish_turn('stock-owner','stock-resumed')
        if job_mode:
            # The old summary receipt stays on its original, completed parent.
            # The later foreground attempt may also seal its unused new parent;
            # outbox ordering is not the recovered summary's identity.
            events = host.pending_events()
            assert all(event['kind'] == 'job_finish' for event in events)
            assert main.job_key in {event['job_key'] for event in events}
        else:
            assert host.pending_events()[-1]['expected_call_key']==calls[0]['key']
    finally:
        coordinator.end_turn(turn,outcome='success');coordinator.release_conversation(lease)
        for c in clients:asyncio.run(c.aclose())
