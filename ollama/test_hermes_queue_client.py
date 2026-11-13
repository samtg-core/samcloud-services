"""Actual Hermes client-resolution and auxiliary execution contract.

Run through the matching Hermes checkout's scripts/run_tests.sh, with this
absolute file path and its PM-built test interpreter. No installed profile or
production inference. Includes cold native plugin discovery, actual plugin
hooks/execution middleware, full persisted turns and stock compaction with
explicit held-source reopen and durable unpersisted preflight against a synthetic
API. Does not qualify automatic cold-turn reconstruction, summary quality or
real-provider deadlines.
"""
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('stream', [False, True])
def test_saved_provider_refusal_reaches_native_overflow_classifier(stream):
    from agent.error_classifier import classify_api_error, FailoverReason
    from openai import BadRequestError
    from ollama.test_inference_sdk import FailedQueueFixture, arguments
    client = FailedQueueFixture().client()
    try:
        with pytest.raises(BadRequestError) as caught:
            response = client.create(**arguments(stream=stream))
            if stream:
                with response:
                    list(response)
        verdict = classify_api_error(caught.value, provider='samcloud-queue', model='qwen')
        assert verdict.reason == FailoverReason.context_overflow
        assert verdict.should_compress
    finally:
        client.close()


def test_cold_native_discovery_resolves_aux_without_registry_warmup(tmp_path):
    """Files exist before the first native import, unlike hot-install fixtures."""
    home = tmp_path / 'cold-profile'
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
    for component in ('samcloud-queue-provider', 'samcloud-queue-hooks', 'samcloud-queue-context'):
        shutil.copytree(source / component, home / 'plugins' / component)
    (home / 'config.yaml').write_text(json.dumps({'plugins': {'isolation': 'in_process',
        'enabled': ['samcloud-queue-hooks']}, 'context': {'engine': 'samcloud-queue-context'},
        'model': {'default': 'qwen', 'context_length': 65536},
        'compression': {'threshold': .75, 'threshold_tokens': None}}))
    script = '''
import asyncio
from agent.auxiliary_client import resolve_provider_client, _resolve_task_provider_model
from ollama.inference_sdk import QueuedClient
from ollama.hermes_queue_context import QueuedContextCompressor
from plugins.context_engine import load_context_engine
from hermes_cli import plugins, lifecycle
client, model = resolve_provider_client('samcloud-queue-aux', 'qwen', task='compression')
assert isinstance(client, QueuedClient) and client.complete_responses
asyncio.run(client.aclose())
engine = load_context_engine('samcloud-queue-context')
assert isinstance(engine, QueuedContextCompressor)
plugins.discover_plugins()
assert lifecycle.has_hook('pre_api_request')
print('cold discovery passed')
'''
    native = Path(importlib.util.find_spec('run_agent').origin).parent
    env = dict(os.environ, HERMES_HOME=str(home), SAMCLOUD_QUEUE_API_KEY='synthetic',
        PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-c', script], cwd=native, env=env,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'cold discovery passed' in result.stdout


def test_cold_rollout_plan_uses_native_wait_policy_without_threshold_retune(tmp_path):
    from ollama.inference_native_profile import profile_plan
    home=tmp_path/'planned-profile'
    source=Path(__file__).resolve().parents[1]/'hermes-plugins'
    for component in ('samcloud-queue-provider','samcloud-queue-hooks','samcloud-queue-context'):
        shutil.copytree(source/component,home/'plugins'/component)
    original={'model':{'default':'qwen','provider':'custom','context_length':65536,
                      'base_url':'https://queue.invalid/v1'},
              'compression':{'enabled':True,'threshold':.75},'context':{'engine':'compressor'}}
    (home/'config.yaml').write_text(json.dumps(profile_plan(original)))
    script='''
import asyncio,socket
def forbidden(*args,**kwargs): raise AssertionError('No network during configuration qualification')
socket.socket.connect=forbidden
socket.socket.connect_ex=forbidden
from agent.conversation_compression import resolve_context_compression_timeouts
from agent.auxiliary_client import resolve_provider_client, _resolve_task_provider_model
from plugins.context_engine import load_context_engine
from ollama.inference_sdk import QueuedClient
assert resolve_context_compression_timeouts()[0]==0
engine=load_context_engine('samcloud-queue-context')
assert engine.context_length==65536
assert engine.threshold_percent==.75
assert engine.threshold_tokens_cap is None
client,model=resolve_provider_client('samcloud-queue-aux','qwen',task='compression')
assert isinstance(client,QueuedClient) and client.complete_responses
asyncio.run(client.aclose())
provider,model,base,key,_=_resolve_task_provider_model('compression')
assert provider=='samcloud-queue-aux' and key is None
client,_=resolve_provider_client(provider,model,explicit_base_url=base,task='compression')
assert isinstance(client,QueuedClient) and client.complete_responses
asyncio.run(client.aclose())
print('planned native wait policy passed')
'''
    native=Path(importlib.util.find_spec('run_agent').origin).parent
    env=dict(os.environ,HERMES_HOME=str(home),SAMCLOUD_QUEUE_API_KEY='synthetic',
             PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result=subprocess.run([sys.executable,'-c',script],cwd=native,env=env,
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'planned native wait policy passed' in result.stdout


def test_native_explicit_tool_wait_preserves_command_and_originating_api(tmp_path, monkeypatch):
    home = tmp_path/'profile'
    monkeypatch.setenv('HERMES_HOME',str(home))
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    source = Path(__file__).resolve().parents[1]/'hermes-plugins'/'samcloud-queue-hooks'
    shutil.copytree(source,home/'plugins'/'samcloud-queue-hooks')
    (home/'config.yaml').write_text('plugins:\n  isolation: in_process\n  enabled:\n    - samcloud-queue-hooks\n')
    from hermes_cli import plugins
    from ollama.hermes_queue_provider import host_for_home
    plugins.discover_plugins()
    host = host_for_home(home)
    def bind(row,api):
        host.pre_api_request(session_id='session',turn_id='turn',api_request_id=api,
            conversation_history=[{'role':'user','_row_id':row,'content':'private'}])
        keys = host.main_stint_keys({'model':'qwen','messages':['private']})
        keys.on_accepted(SimpleNamespace(grant_id='fixture-grant',execution_id=keys.call_key))
        host.post_api_request()
        plugins.invoke_hook('post_api_request',provider='samcloud-queue-stint',
                            session_id='session',turn_id='turn',api_request_id=api,
                            response={'assistant_message':{'tool_calls':[{'id':'tool-one'}]}})
        return keys
    first = bind(1,'api-one')
    latest = bind(2,'api-two')
    tail = 'printf "%s" "quoted value"; printf done'
    original = {'command':'sc-queue-wait -- '+tail,'timeout':33}
    block, modified = plugins._dispatch_pre_tool_call_hooks('terminal',original,
        session_id='session',turn_id='turn',api_request_id='api-one',tool_call_id='tool-one')
    assert block is None
    assert modified == {'command':tail,'timeout':33}
    assert original['command'].startswith('sc-queue-wait -- ')
    events = host.pending_events()
    assert len(events) == 1
    assert events[0]['expected_call_key'] == first.call_key
    assert events[0]['expected_call_key'] != latest.call_key
    block, modified = plugins._dispatch_pre_tool_call_hooks('terminal',original,
        session_id='sibling',turn_id='turn',api_request_id='api-one',tool_call_id='tool-one')
    assert block and 'nothing started' in block
    assert modified is None
    assert host.pending_events() == events
    block, modified = plugins._dispatch_pre_tool_call_hooks('terminal',original,
        session_id='session',turn_id='turn',api_request_id='api-one',tool_call_id='invented-tool')
    assert block and 'nothing started' in block
    assert modified is None and host.pending_events() == events
    block, modified = plugins._dispatch_pre_tool_call_hooks('terminal',{'command':'sc-queue-wait --   '},
        session_id='session',turn_id='turn',api_request_id='api-one',tool_call_id='tool-one')
    assert block and 'requires a command' in block and modified is None
    block, modified = plugins._dispatch_pre_tool_call_hooks('terminal',{'command':'printf short'},
        session_id='session',turn_id='turn',api_request_id='api-two',tool_call_id='short')
    assert block is None and modified is None
    host.finish_turn('session','turn')
    assert len(host.pending_events()) == 2


def test_context_engine_persists_unstored_source_before_summary_without_rewriting(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    from ollama.hermes_queue_context import QueuedContextCompressor
    engine = QueuedContextCompressor(model='qwen', config_context_length=65536)
    engine.bind_session_state(session_id='explicit-session')
    messages = [{'role': 'user', 'content': 'not persisted yet'}]
    engine.compress(messages, current_tokens=12000, force=True)
    from ollama.hermes_queue_source import CompactionSource
    source=CompactionSource.load(tmp_path/'profile'/'samcloud-queue'/'compaction-sources',
        str((tmp_path/'profile').resolve()),'explicit-session',engine.last_queue_source_id)
    assert source.messages()==messages
    assert messages == [{'role': 'user', 'content': 'not persisted yet'}]
    assert engine.compression_count == 0


@pytest.mark.parametrize('wait_for_service', [False, True])
@pytest.mark.parametrize('gateway_accepted', [False, True])
@pytest.mark.parametrize('compression_trigger', ['preflight', 'provider_overflow'])
@pytest.mark.parametrize('stint', [False, True])
def test_native_preflight_keeps_input_and_can_continue(tmp_path, monkeypatch, wait_for_service, gateway_accepted, compression_trigger, stint):
    """Preflight or saved-overflow recovery, including accepted row adoption.

    Uses the stock whole-turn compressor/error loop, not an actual RPC process.
    """
    import httpx
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent import secret_scope
    from hermes_cli import plugins
    from ollama import hermes_queue_provider as plugin
    from ollama.hermes_queue_context import QueuedContextCompressor
    home = tmp_path / 'profile'
    provider = 'samcloud-queue-stint' if stint else 'samcloud-queue'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
    for component in ('samcloud-queue-provider', 'samcloud-queue-hooks', 'samcloud-queue-context'):
        shutil.copytree(source / component, home / 'plugins' / component)
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'default':'qwen', 'provider':provider, 'context_length':65536,
                  'base_url':'https://queue.invalid/v1'},
        'plugins': {'isolation':'in_process', 'enabled':['samcloud-queue-hooks']},
        'context': {'engine':'samcloud-queue-context'},
        'compression': {'threshold_tokens':1000 if compression_trigger=='preflight' else 60000,
                        'context_timeout_seconds':0},
        'auxiliary': {'compression': {'provider':'samcloud-queue-aux', 'model':'qwen',
                                     'base_url':'https://queue.invalid/v1', 'timeout':1}}}))
    import providers
    assert providers.get_provider_profile(provider) is not None
    plugins.discover_plugins()
    class FinalAnswerAPI(QueueFixture):
        failed_execution = None
        def handle(self, request):
            if request.method=='POST' and request.url.path.endswith('/executions'):
                submitted=json.loads(request.content)
                body=submitted['request']
                if (compression_trigger=='provider_overflow' and body.get('stream')
                        and self.failed_execution is None):
                    self.failed_execution=submitted['key']
                operation=getattr(agent.context_compressor,'last_queue_source_id',None)
                if not body.get('stream'):
                    snapshot = {'messages': []}
                    if operation:
                        manifest=home/'samcloud-queue'/'compaction-sources'/(operation+'.json')
                        snapshot=json.loads(manifest.read_text())
                    summary_ownership.append((operation,
                        any(m.get('content')=='NEW INPUT BEFORE PREFLIGHT' and not m.get('_row_id')
                            for m in snapshot['messages']),
                        not any(m.get('content')=='NEW INPUT BEFORE PREFLIGHT'
                            for m in db.get_messages_as_conversation('preflight',include_row_ids=True))))
            if (self.failed_execution is not None and
                    request.url.path.removesuffix('/events').endswith('/'+self.failed_execution)):
                if request.url.path.endswith('/events'):
                    return httpx.Response(200,json={'state':'failed','events':[],
                        'next_cursor':0,'has_more':False})
                return httpx.Response(200,json={'state':'failed','result':{'kind':'error',
                    'status':400,'detail':'context length exceeded','code':'context_length_exceeded'}})
            if request.url.path.endswith('/events'):
                frame = {'id':'answer','object':'chat.completion.chunk','created':1,'model':'qwen',
                    'choices':[{'index':0,'delta':{'content':'hello'},'finish_reason':'stop'}]}
                return httpx.Response(200,json={'state':'complete','events':[
                    {'seq':1,'data':'data: '+json.dumps(frame)+'\n\n'},
                    {'seq':2,'data':'data: [DONE]\n\n'}], 'next_cursor':2,'has_more':False})
            return super().handle(request)
    api, clients, attempts, summary_ownership = FinalAnswerAPI(), [], [], []
    if wait_for_service:
        api.ready.clear()
    real_client = plugin.QueuedClient
    def service_client(**kwargs):
        client=real_client(**kwargs,http_factory=api.http,poll_interval=.01)
        clients.append(client)
        return client
    monkeypatch.setattr(plugin,'QueuedClient',service_client)
    compress=QueuedContextCompressor.compress
    def observe(self,messages,*args,**kwargs):
        attempts.append([m.get('_row_id') for m in messages if m.get('role')!='system'])
        return compress(self,messages,*args,**kwargs)
    monkeypatch.setattr(QueuedContextCompressor,'compress',observe)
    token=secret_scope.set_secret_scope({'SAMCLOUD_QUEUE_API_KEY':'synthetic'})
    db=SessionDB(tmp_path/'preflight.db')
    db.create_session('preflight',source='cli')
    original=[{'role':'user' if n%2==0 else 'assistant',
               'content':f'Previous exchange {n}: '+'detail '*200} for n in range(40)]
    db.append_messages_batch('preflight',original)
    history=db.get_messages_as_conversation('preflight',include_row_ids=True)
    original_ids=[m['_row_id'] for m in history]
    agent=AIAgent(api_key='synthetic',base_url='https://queue.invalid/v1',provider=provider,
        model='qwen',session_id='preflight',session_db=db,quiet_mode=True,
        skip_context_files=True,skip_memory=True)
    agent._cached_system_prompt='Synthetic system.'
    agent._compression_feasibility_checked=True
    agent.save_trajectories=False
    agent.context_compressor.tail_token_budget=256
    agent.context_compressor.update_from_response({'prompt_tokens':12000,'completion_tokens':1})
    accepted_row = None
    if gateway_accepted:
        from contextlib import nullcontext
        from tui_gateway import server as gateway
        monkeypatch.setattr(gateway, '_session_db', lambda session: nullcontext(db))
        session = {'session_key':'preflight', 'agent':agent}
        staged = gateway._write_submit_user_row(session, 'NEW INPUT BEFORE PREFLIGHT', None)
        assert staged and type(staged.get('_row_id')) is int
        accepted_row = staged['_row_id']
        session['_submit_user_row'] = staged
        gateway._adopt_submit_user_row(session, agent,
            'NEW INPUT BEFORE PREFLIGHT', 'NEW INPUT BEFORE PREFLIGHT')
    try:
        assert agent.context_compressor.context_length==65536, (
            agent._config_context_length, agent.context_compressor._config_context_length,
            agent.context_compressor._queue_constructor_options.get('config_context_length'))
        if wait_for_service:
            with ThreadPoolExecutor(max_workers=1) as pool:
                call=pool.submit(copy_context().run,lambda: agent.run_conversation(
                    'NEW INPUT BEFORE PREFLIGHT',conversation_history=history))
                try:
                    assert api.observed.wait(5)
                    assert not api.ready.wait(2.2)
                    assert not call.done()
                finally:
                    api.ready.set()
                result=call.result(timeout=5)
        else:
            result=agent.run_conversation('NEW INPUT BEFORE PREFLIGHT',conversation_history=history)
        assert attempts
        if gateway_accepted or compression_trigger=='provider_overflow':
            # Once API dispatch has started, even a direct caller's input is
            # persisted; overflow must compact that accepted row, not duplicate it.
            assert None not in attempts[0]
        if gateway_accepted:
            assert None not in attempts[0] and accepted_row in attempts[0]
        elif compression_trigger=='preflight':
            assert None in attempts[0]  # direct caller's actual unpersisted preflight
        assert result['final_response']=='hello',result
        assert summary_ownership and summary_ownership[0][1:]==(
            (False,False) if gateway_accepted or compression_trigger=='provider_overflow' else (True,True))
        if gateway_accepted or compression_trigger=='provider_overflow':
            assert summary_ownership[0][0] is None
        if compression_trigger=='provider_overflow':
            assert api.failed_execution is not None
            assert api.failed_execution in api.calls
            main_calls=[body for body in api.calls.values() if body.get('stream')]
            assert len(main_calls)==2
            assert len(json.dumps(main_calls[1]['messages'])) < len(json.dumps(main_calls[0]['messages']))
            assert sum(m.get('content')=='NEW INPUT BEFORE PREFLIGHT'
                       for m in main_calls[1]['messages'])==1
        assert agent.context_compressor.compression_count>=1
        assert any(body.get('stream') for body in api.calls.values())
        assert any(not body.get('stream') for body in api.calls.values())
        active=db.get_messages_as_conversation('preflight',include_row_ids=True)
        assert sum(m.get('content')=='NEW INPUT BEFORE PREFLIGHT' for m in active)==1
        all_rows=db.get_messages('preflight',include_inactive=True,include_compacted=True)
        assert set(original_ids).issubset({row['id'] for row in all_rows})
        assert len(api.calls)>=1
    finally:
        agent.close()
        db.close()
        secret_scope.reset_secret_scope(token)
        for client in clients:
            asyncio.run(client.aclose())


def test_same_model_detached_review_queues_without_rewriting_parent_history(tmp_path,monkeypatch):
    import httpx
    import model_tools
    from hermes_state import SessionDB
    from hermes_cli import plugins
    from run_agent import AIAgent
    from agent.background_review import build_cache_parity_fork
    from ollama import hermes_queue_provider as plugin
    from ollama.inference_sdk import QueuedClient
    home=tmp_path/'profile'
    monkeypatch.setenv('HERMES_HOME',str(home))
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    monkeypatch.setattr(model_tools,'get_tool_definitions',lambda *a,**k:[])
    monkeypatch.setattr(model_tools,'check_toolset_requirements',lambda *a,**k:{})
    source=Path(__file__).resolve().parents[1]/'hermes-plugins'
    for component in ('samcloud-queue-provider','samcloud-queue-hooks'):
        shutil.copytree(source/component,home/'plugins'/component)
    (home/'config.yaml').write_text('plugins:\n  isolation: in_process\n  enabled:\n    - samcloud-queue-hooks\n')
    plugins.discover_plugins()
    class AnswerAPI(QueueFixture):
        def handle(self,request):
            if request.url.path.endswith('/events'):
                frame={'id':'answer','object':'chat.completion.chunk','created':1,'model':'qwen',
                       'choices':[{'index':0,'delta':{'content':'hello'},'finish_reason':'stop'}]}
                return httpx.Response(200,json={'state':'complete','events':[
                    {'seq':1,'data':'data: '+json.dumps(frame)+'\n\n'},
                    {'seq':2,'data':'data: [DONE]\n\n'}],'next_cursor':2,'has_more':False})
            return super().handle(request)
    api=AnswerAPI()
    def fake_service(**kwargs):
        return QueuedClient(**kwargs,http_factory=api.http,poll_interval=.01)
    monkeypatch.setattr(plugin,'QueuedClient',fake_service)
    db=SessionDB(tmp_path/'state.db')
    parent=AIAgent(api_key='synthetic',base_url='https://queue.invalid/v1',
        provider='samcloud-queue-stint',model='qwen',session_db=db,quiet_mode=True,
        skip_context_files=True,skip_memory=True,skip_background_review=True)
    parent._cached_system_prompt='Synthetic stable prefix.'
    parent._use_prompt_caching=parent.compression_enabled=parent.save_trajectories=False
    monkeypatch.setattr(parent,'_cleanup_task_resources',lambda *a,**k:None)
    fork=None
    try:
        result=parent.run_conversation('synthetic foreground')
        assert result['final_response']=='hello'
        before=list(db._conn.execute('SELECT id,role,content FROM messages ORDER BY id'))
        fork,_,routed=build_cache_parity_fork(parent,max_iterations=1,task_cfg={})
        assert not routed and fork._persist_disabled
        assert fork.session_id==parent.session_id
        assert fork._cached_system_prompt==parent._cached_system_prompt
        fork.compression_enabled=fork.save_trajectories=False
        monkeypatch.setattr(fork,'_cleanup_task_resources',lambda *a,**k:None)
        reviewed=fork.run_conversation('synthetic detached review',conversation_history=result['messages'])
        assert reviewed['final_response']=='hello'
        assert list(db._conn.execute('SELECT id,role,content FROM messages ORDER BY id'))==before
        grants=[row[2] for row in api.requests if row[0]=='POST' and row[1].endswith('/grants')]
        assert [row['lifetime'] for row in grants]==['work_session','execution']
        assert grants[0]['key']!=grants[1]['key']
        assert len(api.calls)==2
    finally:
        if fork is not None:fork.release_clients()
        parent.close();db.close()


def test_native_vision_auxiliary_crosses_tool_worker_with_own_queue_identity(tmp_path,monkeypatch):
    """Real tool middleware, thread hop and async aux resolver; synthetic service."""
    home=tmp_path/'vision-profile'
    monkeypatch.setenv('HERMES_HOME',str(home))
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    source=Path(__file__).resolve().parents[1]/'hermes-plugins'
    for component in ('samcloud-queue-provider','samcloud-queue-hooks'):
        shutil.copytree(source/component,home/'plugins'/component)
    (home/'config.yaml').write_text(json.dumps({'plugins':{'isolation':'in_process',
        'enabled':['samcloud-queue-hooks']},'auxiliary':{'vision':{
            'provider':'samcloud-queue-tool','model':'vision-7b',
            'base_url':'https://queue.invalid/v1'}}}))
    from hermes_cli import plugins
    from hermes_cli.middleware import run_tool_execution_middleware
    from agent import secret_scope,auxiliary_client
    import model_tools
    from ollama import hermes_queue_provider as plugin
    from ollama.hermes_queue_identity import HeldSource
    from ollama.inference_sdk import QueuedClient
    plugins.discover_plugins()
    api,clients=QueueFixture(),[]
    def service(**kwargs):
        client=QueuedClient(**kwargs,http_factory=api.http,poll_interval=.001)
        clients.append(client);return client
    monkeypatch.setattr(plugin,'QueuedClient',service)
    from providers import get_provider_profile
    # The foreground client resolves its profile before it can issue this tool.
    assert get_provider_profile('samcloud-queue-stint') is not None
    assert get_provider_profile('samcloud-queue-tool') is not None
    secret=secret_scope.set_secret_scope({'SAMCLOUD_QUEUE_API_KEY':'synthetic'})
    host=plugin.host_for_home(home)
    source=HeldSource.capture(str(home),'session','main',
        [{'role':'user','content':'synthetic','_row_id':1}],turn_id='turn',api_request_id='api')
    parent=source.bind_stint(host.path,{'model':'qwen','messages':[{'role':'user','content':'synthetic'}]})
    parent.on_accepted(SimpleNamespace(grant_id='fixture-grant',execution_id=parent.call_key))
    host.complete_api('session','turn','api',('vision-tool',))
    assert plugins.has_middleware('tool_execution')
    with host.tool_scope('session','turn','api','vision-tool'):
        assert host.tool.get() is not None
    async def exercise():
        from hermes_cli.providers import get_provider
        assert get_provider('samcloud-queue-tool') is not None
        resolved=auxiliary_client._resolve_task_provider_model('vision')
        assert resolved[0]=='samcloud-queue-tool',resolved[:3]
        _,probe,_=auxiliary_client.resolve_vision_provider_client()
        assert isinstance(probe,QueuedClient),(resolved[:3],type(probe).__name__)
        def dispatch(args):
            assert host.tool.get() is not None, 'tool middleware failed to bind before dispatch'
            return model_tools._run_async(auxiliary_client.async_call_llm(task='vision',
                messages=[{'role':'user','content':'synthetic image request'}],max_tokens=4))
        for _ in range(2):
            response=run_tool_execution_middleware('vision_analyze',{},dispatch,
                session_id='session',turn_id='turn',api_request_id='api',tool_call_id='vision-tool')
            assert response.choices[0].message.content=='hello'
    try:
        asyncio.run(exercise())
        calls=[body for method,path,body in api.requests if method=='POST' and path.endswith('/executions')]
        grants=[body for method,path,body in api.requests if method=='POST' and path.endswith('/grants')]
        assert len(calls)==2 and calls[0]['key']!=calls[1]['key']
        assert all(call['request']['model']=='vision-7b' for call in calls)
        assert all(call['yield_grant_keys']==[parent.grant_key] for call in calls)
        assert all(grant['lifetime']=='execution' for grant in grants)
        assert host.tool.get() is None and host.tool_api.get() is None
    finally:
        secret_scope.reset_secret_scope(secret)
        for client in clients:asyncio.run(client.aclose())


def test_filesystem_provider_discovery_uses_own_home_and_refuses_unbound_calls(tmp_path, monkeypatch):
    """Actual native discovery, no register_provider call in the test."""
    home = tmp_path / 'profile'
    monkeypatch.setenv('HERMES_HOME', str(home))
    import providers
    from ollama.hermes_queue_provider import host_for_home
    from ollama.inference_sdk import QueuedClient
    from openai import APIConnectionError
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins' / 'samcloud-queue-provider'
    shutil.copytree(source, home / 'plugins' / 'samcloud-queue-provider')
    main = providers.get_provider_profile('samcloud-queue')
    auxiliary = providers.get_provider_profile('samcloud-queue-aux')
    assert main is not None and auxiliary is not None
    api = QueueFixture()
    clients = []
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'synthetic'}]}
    try:
        for profile in (main, auxiliary):
            client = profile.create_client(api_key='synthetic', base_url='https://queue.invalid/v1',
                http_factory=api.http, poll_interval=.001)
            assert isinstance(client, QueuedClient)
            clients.append(client)
            with pytest.raises(APIConnectionError) as rejected:
                client.create(**body)
            assert 'no owning persisted call boundary' in str(rejected.value.__cause__)
        assert not api.requests  # no fallback inference, grant or observer I/O
        main_client, auxiliary_client = clients
        assert not main_client.complete_responses and auxiliary_client.complete_responses
        host = host_for_home(home)
        messages = [{'role': 'user', 'content': 'synthetic', '_row_id': 1}]
        host.pre_api_request(session_id='session', conversation_history=messages)
        assert main_client.create(**body).choices[0].message.content == 'hello'
        with pytest.raises(APIConnectionError) as rejected:
            auxiliary_client.create(**body)
        assert 'no owning persisted call boundary' in str(rejected.value.__cause__)
        with host.compression_scope('session', messages):
            assert auxiliary_client.create(**body).choices[0].message.content == 'hello'
        host.post_api_request()
        # A different profile cannot inherit either boundary or its identity DB.
        other = tmp_path / 'other-profile'
        monkeypatch.setenv('HERMES_HOME', str(other))
        assert providers.get_provider_profile('samcloud-queue') is None
        assert host_for_home(other) is not host
        assert not list((other / 'samcloud-queue').glob('*.sqlite'))
        assert len(api.calls) == 2
    finally:
        for client in clients:
            asyncio.run(client.aclose())


@pytest.mark.parametrize('stint', [False, True])
def test_discovered_provider_native_main_and_auxiliary_resolution(tmp_path, monkeypatch, stint):
    """Resolve both native clients; substitute only the service HTTP fixture."""
    home = tmp_path / 'profile'
    monkeypatch.setenv('HERMES_HOME', str(home))
    import providers
    from openai import APIConnectionError
    from agent import secret_scope
    from agent.agent_runtime_helpers import create_openai_client
    from agent.auxiliary_client import resolve_provider_client, _create_with_progress
    from ollama import hermes_queue_provider as plugin
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins' / 'samcloud-queue-provider'
    shutil.copytree(source, home / 'plugins' / 'samcloud-queue-provider')
    assert providers.get_provider_profile('samcloud-queue') is not None
    hooks = source.parent / 'samcloud-queue-hooks'
    shutil.copytree(hooks, home / 'plugins' / 'samcloud-queue-hooks')
    (home / 'config.yaml').write_text('plugins:\n  isolation: in_process\n  enabled:\n    - samcloud-queue-hooks\n')
    from hermes_cli import lifecycle, plugins
    from hermes_cli.middleware import run_llm_execution_middleware
    plugins.discover_plugins()
    assert lifecycle.has_hook('pre_api_request')
    api, clients = QueueFixture(), []
    real_client = plugin.QueuedClient
    def fake_service(**kwargs):
        client = real_client(**kwargs, http_factory=api.http, poll_interval=.001)
        clients.append(client)
        return client
    monkeypatch.setattr(plugin, 'QueuedClient', fake_service)
    token = secret_scope.set_secret_scope({'SAMCLOUD_QUEUE_API_KEY': 'synthetic'})
    host = plugin.host_for_home(home)
    messages = [{'role': 'user', 'content': 'synthetic', '_row_id': 1}]
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'synthetic'}]}
    provider = 'samcloud-queue-stint' if stint else 'samcloud-queue'
    try:
        agent = SimpleNamespace(provider=provider, _client_log_context=lambda: '',
            _build_keepalive_http_client=lambda *a, **k: None)
        main = create_openai_client(agent, {'api_key': 'synthetic',
            'base_url': 'https://queue.invalid/v1'}, reason='test', shared=False)
        lifecycle.invoke_hook('pre_api_request', provider=provider,
            session_id='session', api_request_id='attempt-1', conversation_history=messages)
        with pytest.raises(ValueError, match='no owning persisted call boundary'):
            host.main_keys(body)  # observer worker never binds its caller
        with pytest.raises(APIConnectionError):
            run_llm_execution_middleware(body, lambda request: main.create(**request),
                provider=provider, session_id='sibling-session', api_request_id='attempt-1')
        assert not api.requests  # same attempt ID on a sibling is not authority
        result = run_llm_execution_middleware(body, lambda request: main.create(**request),
            provider=provider, session_id='session', api_request_id='attempt-1')
        assert result.choices[0].message.content == 'hello'
        aux, model = resolve_provider_client('samcloud-queue-aux', 'qwen', task='compression')
        assert aux.complete_responses and not main.complete_responses
        with host.compression_scope('session', messages):
            result = _create_with_progress(aux, body, task='compression', force_stream=True)
        assert result.choices[0].message.content == 'hello'
        assert len(api.calls) == 2
        submissions = [body for method,path,body in api.requests
                       if method=='POST' and path.endswith('/executions')]
        grants = [body for method,path,body in api.requests
                  if method=='POST' and path.endswith('/grants')]
        assert submissions[-1].get('yield_grant_keys', []) == ([grants[0]['key']] if stint else [])
        lifecycle.invoke_hook('post_api_request', provider=provider,
            session_id='session', api_request_id='attempt-1')
        with pytest.raises(ValueError, match='no owning persisted call boundary'):
            host.main_keys(body)
    finally:
        host.post_api_request()
        secret_scope.reset_secret_scope(token)
        for client in clients:
            asyncio.run(client.aclose())


@pytest.mark.parametrize('asynchronous', [False, True])
def test_auxiliary_hooks_scope_identity_without_leaking_main_boundary(tmp_path, monkeypatch, asynchronous):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    import hermes_cli.lifecycle as lifecycle
    from agent import auxiliary_client as aux
    from ollama.inference_identity import IdentityStore
    from ollama.inference_sdk import QueuedClient
    api = QueueFixture()
    boundary = ContextVar('aux_identity', default='main-sentinel')
    tokens = []
    seen = []
    def capture(name, **kw):
        if name == 'pre_auxiliary_call':
            seen.append((kw['aux_task'], kw['api_request_id']))
            tokens.append(boundary.set(kw['api_request_id']))
        elif name == 'post_auxiliary_call':
            boundary.reset(tokens.pop())
        return []
    monkeypatch.setattr(lifecycle, 'has_hook', lambda name: name in ('pre_auxiliary_call', 'post_auxiliary_call'))
    monkeypatch.setattr(lifecycle, 'invoke_hook', capture)
    def keys(body):
        call = boundary.get()
        assert call.startswith('aux-')  # never accidentally borrow main identity
        store = IdentityStore(tmp_path / 'aux-calls.sqlite')
        try:
            return store.bind('profile/conversation', 'explicit-parent-input', call, body)
        finally:
            store.close()
    client = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                          http_factory=api.http, key_provider=keys)
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'summarize synthetic text'}]}
    async def exercise():
        # Two physical attempts in one logical aux scope, then identical NEW
        # auxiliary work. This does not claim scope persistence across restart.
        for attempts in (2, 1):
            with aux._relay_aux_call_scope(('compression',), {}):
                for _ in range(attempts):
                    if asynchronous:
                        result = await aux._relay_async_completion(client, body, create=lambda request: client.create(**request))
                    else:
                        result = aux._relay_sync_completion(client, body, create=lambda request: client.sync.chat.completions.create(**request))
                    assert result.choices[0].message.content == 'hello'
                    assert boundary.get() == 'main-sentinel'
    try:
        asyncio.run(exercise())
        assert [purpose for purpose, _ in seen] == ['compression'] * 3
        assert seen[0][1] == seen[1][1] != seen[2][1]
        assert len(api.calls) == 2
        assert not tokens
    finally:
        asyncio.run(client.aclose())


def test_native_stint_retry_after_terminal_failure_is_a_new_sample(tmp_path, monkeypatch, queued_profile):
    """0b800389's shape: a stint call the service ends `failed` (backend_tool_parse, 502),
    then the native retry of the same persisted rows. The retry must reach the service
    as a new call on the same stint grant, not replay the saved failure."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    import httpx
    from hermes_state import SessionDB
    from run_agent import AIAgent
    import model_tools
    from hermes_cli import plugins
    from ollama import hermes_queue_provider as plugin
    from ollama.inference_sdk import QueuedClient
    monkeypatch.setattr(model_tools, 'get_tool_definitions', lambda *a, **k: [])
    monkeypatch.setattr(model_tools, 'check_toolset_requirements', lambda *a, **k: {})
    class ToolParseFailureAPI(QueueFixture):
        failed_key = None
        def handle(self, request):
            if request.method == 'POST' and request.url.path.endswith('/executions'):
                key = json.loads(request.content)['key']
                if self.failed_key is None:
                    self.failed_key = key
            path = request.url.path
            if self.failed_key is not None and path.removesuffix('/events').endswith('/' + self.failed_key):
                with self.lock:
                    self.requests.append((request.method, path, None))
                if path.endswith('/events'):
                    return httpx.Response(200, json={'state': 'failed', 'events': [],
                                                     'next_cursor': 0, 'has_more': False})
                return httpx.Response(200, json={'state': 'failed', 'result': {'kind': 'error',
                    'status': 502, 'code': 'backend_tool_parse',
                    'detail': 'The model ended on a tool call the backend could not parse'}})
            if path.endswith('/events'):
                frame = {'id': 'answer', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'qwen', 'choices': [{'index': 0, 'delta': {'content': 'hello'},
                                                     'finish_reason': 'stop'}]}
                return httpx.Response(200, json={'state': 'complete', 'events': [
                    {'seq': 1, 'data': 'data: ' + json.dumps(frame) + '\n\n'},
                    {'seq': 2, 'data': 'data: [DONE]\n\n'}], 'next_cursor': 2, 'has_more': False})
            return super().handle(request)
    api = ToolParseFailureAPI()
    home = tmp_path / 'profile'
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
    for component in ('samcloud-queue-provider', 'samcloud-queue-hooks'):
        shutil.copytree(source / component, home / 'plugins' / component)
    (home / 'config.yaml').write_text('plugins:\n  isolation: in_process\n  enabled:\n    - samcloud-queue-hooks\n')
    plugins.discover_plugins()
    monkeypatch.setattr(plugin, 'QueuedClient',
                        lambda **kwargs: QueuedClient(**kwargs, http_factory=api.http, poll_interval=.01))
    db = SessionDB(tmp_path / 'state.db')
    agent = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
                    provider='samcloud-queue-stint', model='qwen', session_db=db, quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = 'Synthetic test.'
    agent._use_prompt_caching = agent.compression_enabled = agent.save_trajectories = False
    monkeypatch.setattr(agent, '_cleanup_task_resources', lambda *a, **k: None)
    try:
        result = agent.run_conversation('same request')
        assert result['final_response'] == 'hello', result
        main = [body for method, path, body in api.requests
                if method == 'POST' and path.endswith('/executions')]
        keys = [body['key'] for body in main]
        assert keys[0] == api.failed_key
        # One failed call, then exactly one new call. Hermes may retry streamed (the seat's
        # 0b800389 replayed its saved failure) or non-streamed (a changed digest on the same
        # boundary was a Conflict); either way the same persisted input is re-sampled.
        assert len(set(keys)) == 2, keys
        assert main[0]['request']['messages'] == main[-1]['request']['messages']
        stints = {body['key'] for method, path, body in api.requests
                  if method == 'POST' and path.endswith('/grants') and body['lifetime'] == 'work_session'}
        assert len(stints) == 1  # the conversation stint is kept
    finally:
        agent.close()
        db.close()


@pytest.mark.parametrize('wait_for_service,native_plugins,stint,contact_loss', [
    (wait, plugins, stint, 'none')
    for wait in (False, True) for plugins in (False, True) for stint in (False, True)
] + [(True, True, True, 'before'), (True, True, True, 'poll'), (True, True, True, 'quarantine')])
def test_native_turn_hook_to_durable_sdk_identity(tmp_path, monkeypatch, queued_profile,
                                                wait_for_service, native_plugins, stint, contact_loss):
    """Real turn/persistence/worker/SDK, synthetic service; no runtime install."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setenv('HERMES_STREAM_STALE_TIMEOUT', '1')
    monkeypatch.setenv('HERMES_STREAM_RETRIES', '0')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    import httpx
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.chat_completion_helpers import _StreamingCall
    import model_tools
    import hermes_cli.lifecycle as lifecycle
    from ollama.hermes_queue_identity import HostBindings
    from ollama.inference_sdk import QueuedClient
    monkeypatch.setattr(model_tools, 'get_tool_definitions', lambda *a, **k: [])
    monkeypatch.setattr(model_tools, 'check_toolset_requirements', lambda *a, **k: {})
    class FinalAnswerAPI(QueueFixture):
        def handle(self, request):
            if not self.ready.is_set():
                if contact_loss == 'before' and request.url.path.endswith('/capabilities'):
                    self.observed.set()
                    raise httpx.ReadError('service contact lost before acceptance', request=request)
                if contact_loss == 'poll' and request.url.path.endswith('/events'):
                    self.observed.set()
                    raise httpx.ReadError('service contact lost during observation', request=request)
                if contact_loss == 'quarantine' and request.url.path.endswith('/events'):
                    self.observed.set()
                    return httpx.Response(200, json={'state': 'uncertain', 'events': [],
                        'next_cursor': 0, 'has_more': False})
            if request.url.path.endswith('/events'):
                if not self.ready.is_set():
                    return super().handle(request)
                frame = {'id': 'answer', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'qwen', 'choices': [{'index': 0, 'delta': {'content': 'hello'},
                                                     'finish_reason': 'stop'}]}
                return httpx.Response(200, json={'state': 'complete', 'events': [
                    {'seq': 1, 'data': 'data: ' + json.dumps(frame) + '\n\n'},
                    {'seq': 2, 'data': 'data: [DONE]\n\n'}], 'next_cursor': 2, 'has_more': False})
            return super().handle(request)
    api = FinalAnswerAPI()
    resolved_timeouts = []
    resolve_timeout = _StreamingCall._resolve_stale_timeout
    def observe_timeout(call):
        resolve_timeout(call)
        resolved_timeouts.append(call._stream_stale_timeout)
    monkeypatch.setattr(_StreamingCall, '_resolve_stale_timeout', observe_timeout)
    if native_plugins:
        from hermes_cli import plugins
        from ollama import hermes_queue_provider as plugin
        home = tmp_path / 'profile'
        source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
        for component in ('samcloud-queue-provider', 'samcloud-queue-hooks'):
            shutil.copytree(source / component, home / 'plugins' / component)
        (home / 'config.yaml').write_text('plugins:\n  isolation: in_process\n  enabled:\n    - samcloud-queue-hooks\n')
        plugins.discover_plugins()
        def fake_service(**kwargs):
            return QueuedClient(**kwargs, http_factory=api.http, poll_interval=.05)
        monkeypatch.setattr(plugin, 'QueuedClient', fake_service)
        provider = 'samcloud-queue-stint' if stint else 'samcloud-queue'
    else:
        host = HostBindings(tmp_path / 'identity.sqlite', 'synthetic-profile')
        def capture(name, **kw):
            if name == 'pre_api_request':
                host.pre_api_request(**kw)
            elif name == 'post_api_request':
                host.post_api_request(**kw)
            return []
        monkeypatch.setattr(lifecycle, 'has_hook', lambda name: name in ('pre_api_request','post_api_request'))
        monkeypatch.setattr(lifecycle, 'invoke_hook', capture)
        monkeypatch.setattr(queued_profile, 'client', lambda: QueuedClient(
            api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=api.http, key_provider=host.main_stint_keys if stint else host.main_keys,
            poll_interval=.05, state_frames=wait_for_service,
            grant_lifetime='work_session' if stint else 'execution'))
        provider = 'queue-test'
    db = SessionDB(tmp_path / 'state.db')
    agent = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
                    provider=provider, model='qwen', session_db=db, quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = 'Synthetic test.'
    agent._use_prompt_caching = agent.compression_enabled = agent.save_trajectories = False
    monkeypatch.setattr(agent, '_cleanup_task_resources', lambda *a, **k: None)
    try:
        history = None
        for _ in range(2):
            deltas = []
            if wait_for_service:
                api.ready.clear()
                api.observed.clear()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    call = pool.submit(agent.run_conversation, 'same request',
                                       conversation_history=history, stream_callback=deltas.append)
                    try:
                        assert api.observed.wait(3)
                        assert resolved_timeouts[-1] == 1.0
                        assert all(body['stream'] is True for body in api.calls.values())
                        # Real native turn + persisted input + registered client
                        # factory: waiting crosses two actual stale windows.
                        assert not api.ready.wait(2.2)
                        assert not call.done()
                        assert getattr(agent, '_consecutive_stale_streams', 0) == 0
                        assert not deltas  # queue status is not generated answer text
                    finally:
                        api.ready.set()
                    result = call.result(timeout=5)
            else:
                result = agent.run_conversation('same request', conversation_history=history,
                                                stream_callback=deltas.append)
            assert result['final_response'] == 'hello'
            assert ''.join(deltas) == 'hello'
            history = result['messages']
        assert len(api.calls) == 2  # identical text, distinct committed inputs
        if native_plugins and stint:
            from ollama.inference_identity import IdentityStore
            store = IdentityStore(home/'samcloud-queue'/'identity.sqlite')
            try:
                events = store.db.execute('SELECT turn_id,grant_key,call_key FROM lifecycle_outbox').fetchall()
            finally:
                store.close()
            assert len(events) == 2
            assert events[0][0] != events[1][0]  # exact native turns, not a session-latest guess
            assert events[0][1] == events[1][1]  # conversation/model stint retained
            assert events[0][2] != events[1][2]  # each turn has its own persisted call
        assert all(body['lifetime'] == ('work_session' if stint else 'execution')
                   for method, path, body in api.requests
                   if method == 'POST' and path.endswith('/grants'))
        grants = {body['key'] for method, path, body in api.requests
                  if method == 'POST' and path.endswith('/grants')}
        assert len(grants) == (1 if stint else 2)
        assert sum(method == 'POST' and path.endswith('/executions')
                   for method, path, _ in api.requests) == 2
    finally:
        agent.close()
        db.close()


@pytest.mark.parametrize('transient_error', [False, True])
def test_pre_request_hook_exposes_committed_distinct_turn_boundaries(tmp_path, monkeypatch, transient_error):
    """Check real persistence, not a latest-row query or prompt-hash identity."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    from unittest.mock import MagicMock
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _make_tool_defs, _mock_response, _mock_tool_call
    import model_tools
    import hermes_cli.lifecycle as lifecycle
    import agent.turn_api_error as api_errors
    import httpx
    from openai import APIConnectionError
    from ollama.hermes_queue_identity import HostBindings
    monkeypatch.setattr(api_errors, 'interruptible_backoff_sleep', lambda *a, **kw: None)
    monkeypatch.setattr(model_tools, 'get_tool_definitions', lambda *a, **k: _make_tool_defs('web_search'))
    monkeypatch.setattr(model_tools, 'check_toolset_requirements', lambda *a, **k: {})
    monkeypatch.setattr(model_tools, 'handle_function_call', lambda *a, **k: 'synthetic result')
    db = SessionDB(tmp_path / 'state.db')
    agent = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
                    model='test/model', session_db=db, quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = 'Synthetic test.'
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    monkeypatch.setattr(agent, '_cleanup_task_resources', lambda *a, **k: None)
    snapshots = []
    request_ids = []
    call_boundary = ContextVar('synthetic_queue_boundary', default=None)
    physical_boundaries = []
    physical_keys = []
    host = HostBindings(tmp_path / 'native-tool-call-identities.sqlite','synthetic-profile')
    def capture(name, **kw):
        if name == 'pre_api_request':
            host.pre_api_request(**kw)
            snapshots.append([(m.get('role'), m.get('_row_id')) for m in kw['conversation_history']])
            request_ids.append(kw['api_request_id'])
            call_boundary.set(tuple(snapshots[-1]))
        elif name == 'post_api_request':
            host.post_api_request(**kw)
        return []
    monkeypatch.setattr(lifecycle, 'has_hook', lambda name: name in ('pre_api_request','post_api_request'))
    monkeypatch.setattr(lifecycle, 'invoke_hook', capture)
    try:
        history = None
        for n in range(2):
            tc = _mock_tool_call(name='web_search', arguments='{}', call_id=f'synthetic-{n}')
            responses = [
                _mock_response(content='', finish_reason='tool_calls', tool_calls=[tc]),
                _mock_response(content='done', finish_reason='stop')]
            if transient_error and n == 0:
                responses.insert(0, APIConnectionError(request=httpx.Request('POST', 'https://queue.invalid/v1/chat/completions')))
            def provider_call(**kwargs):
                physical_boundaries.append(call_boundary.get())
                physical_keys.append(host.main_keys({'model':kwargs['model'],'messages':kwargs['messages']}))
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
            agent.client.chat.completions.create.side_effect = provider_call
            result = agent.run_conversation('same intentional request', conversation_history=history)
            assert result['final_response'] == 'done'
            history = result['messages']
        assert len(snapshots) == 4 + int(transient_error)
        # The actual native request-worker thread receives the hook's scoped
        # boundary; no process-global "latest request" lookup is necessary.
        assert physical_boundaries == [tuple(rows) for rows in snapshots]
        if transient_error:
            assert snapshots[0] == snapshots[1]  # retry: same committed intent
            # This transport-error path retains its hook request ID too. Do
            # not confuse it with the relay's separate physical-attempt ID.
            assert request_ids[0] == request_ids[1]
            assert request_ids[1] != request_ids[2]  # next post-tool request
            snapshots.pop(0)
            assert physical_keys[0] == physical_keys[1]
            physical_keys.pop(0)
        assert len(set(physical_keys)) == 4  # new tool prefix or accepted input
        user_ids = [[rid for role, rid in rows if role == 'user'][-1] for rows in snapshots]
        assert all(type(rid) is int and rid > 0 for rid in user_ids)
        assert user_ids[0] == user_ids[1]
        assert user_ids[2] == user_ids[3]
        assert user_ids[0] != user_ids[2]  # same prompt is NOT the same work
        for before, after in ((snapshots[0], snapshots[1]), (snapshots[2], snapshots[3])):
            assert after[-1][0] == 'tool'
            assert type(after[-1][1]) is int
            assert before[-1][1] != after[-1][1]
    finally:
        agent.close()
        db.close()


@pytest.mark.parametrize('through_sdk', [False, True])
def test_native_stream_waits_through_status_frames_until_real_answer(tmp_path, monkeypatch, through_sdk):
    """Real native streaming worker/watchdog; synthetic provider, no network."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_STREAM_STALE_TIMEOUT', '1')
    monkeypatch.setenv('HERMES_STREAM_RETRIES', '0')
    from unittest.mock import MagicMock
    from openai.types.chat import ChatCompletionChunk
    from run_agent import AIAgent
    from agent.chat_completion_helpers import interruptible_streaming_api_call, _StreamingCall
    resolved_timeouts = []
    resolve_timeout = _StreamingCall._resolve_stale_timeout
    def observe_timeout(call):
        resolve_timeout(call)
        resolved_timeouts.append(call._stream_stale_timeout)
    monkeypatch.setattr(_StreamingCall, '_resolve_stale_timeout', observe_timeout)
    agent = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
                    model='test/model', quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = 'chat_completions'
    agent._interrupt_requested = False
    ready, release, first_delta = threading.Event(), threading.Event(), threading.Event()
    def chunk(choices, **extra):
        return ChatCompletionChunk.model_validate({'id': 'synthetic-execution',
            'object': 'chat.completion.chunk', 'created': 1, 'model': 'test/model',
            'choices': choices, **extra})
    def frames():
        while not release.is_set():
            yield chunk([], model_service={'kind': 'queue_status', 'state': 'queued'})
            ready.set()
            release.wait(.05)
        yield chunk([{'index': 0, 'delta': {'content': 'real answer'}, 'finish_reason': 'stop'}])
        yield chunk([], usage={'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12})
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kwargs: frames()
    sdk = api = None
    call_args = {'model': 'test/model', 'messages': []}
    if through_sdk:
        from ollama.inference_sdk import QueuedClient
        api = QueueFixture()
        api.ready.clear()
        ready, release = api.observed, api.ready
        sdk = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
                          http_factory=api.http, poll_interval=.05, state_frames=True)
        client = sdk.sync
        call_args = arguments(model='test/model')
    monkeypatch.setattr(agent, '_create_request_openai_client', lambda *a, **k: client)
    monkeypatch.setattr(agent, '_close_request_openai_client', lambda *a, **k: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        call = pool.submit(interruptible_streaming_api_call, agent,
                           call_args, on_first_delta=first_delta.set)
        try:
            assert ready.wait(3)
            assert resolved_timeouts == [1.0]
            # More than two stale windows, but the service keeps reporting a
            # real waiting state. None of those frames is an answer delta.
            assert not first_delta.wait(2.2)
            assert not call.done()
            assert getattr(agent, '_consecutive_stale_streams', 0) == 0
            release.set()
            answer = call.result(timeout=3)
            assert first_delta.is_set()
            if through_sdk:
                assert answer.choices[0].message.content == 'hello'
                assert answer.choices[0].finish_reason == 'tool_calls'
                assert answer.choices[0].message.tool_calls[0].function.name == 'terminal'
                assert len(api.calls) == 1
                assert sum(method == 'POST' and path.endswith('/executions')
                           for method, path, _ in api.requests) == 1
            else:
                assert answer.choices[0].message.content == 'real answer'
                assert answer.choices[0].finish_reason == 'stop'
                assert answer.usage.completion_tokens == 2
                assert client.chat.completions.create.call_count == 1
        finally:
            release.set()
            if sdk is not None:
                call.result(timeout=3)
                asyncio.run(sdk.aclose())


def test_queue_status_frame_is_liveness_not_text_usage_or_completion(tmp_path, monkeypatch):
    """Compatibility probe only: not a complete native waiting-turn acceptance."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    import httpx
    from openai import OpenAI
    from agent.chat_completion_helpers import _StreamingCall, _estimate_chunk_bytes
    frame = {'id': 'execution-test', 'object': 'chat.completion.chunk',
             'created': 1, 'model': 'qwen', 'choices': [],
             'model_service': {'kind': 'queue_status', 'state': 'queued'}}
    wire = 'data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n'
    http = httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=wire)))
    client = OpenAI(api_key='synthetic', base_url='https://queue.invalid/v1', http_client=http)
    try:
        with client.chat.completions.create(model='qwen', messages=[], stream=True) as stream:
            chunks = list(stream)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.choices == []
        assert chunk.usage is None
        assert chunk.model_service == {'kind': 'queue_status', 'state': 'queued'}
        activity = []
        collector = object.__new__(_StreamingCall)
        collector.agent = SimpleNamespace(_touch_activity=activity.append)
        collector.last_chunk_time = {'t': 0}
        diag = {}
        collector._count_chunk(diag, chunk)
        assert collector.last_chunk_time['t'] > 0
        assert activity
        assert diag['chunks'] == 1  # transport frame, not one generated token
        # Diagnostics count framing bytes even with no delta. That is transport
        # overhead, not generated text/tokens; compare an empty control frame.
        assert diag['bytes'] == _estimate_chunk_bytes(SimpleNamespace(choices=[]))
        assert collector._choiceless_chunk(chunk, None) == (None, None)
    finally:
        client.close()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollama.test_inference_sdk import QueueFixture, arguments


@pytest.fixture
def queued_profile(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    import providers
    import hermes_cli.auth as auth
    from providers.base import ProviderProfile
    from hermes_cli.auth_plugin_providers import register_plugin_provider
    from agent import secret_scope
    providers._discover_providers()
    monkeypatch.setattr(providers, '_REGISTRY', dict(providers._REGISTRY))
    monkeypatch.setattr(providers, '_ALIASES', dict(providers._ALIASES))
    monkeypatch.setattr(providers, '_PROVIDER_LIST_CACHE', None)
    monkeypatch.setattr(auth, 'PROVIDER_REGISTRY', dict(auth.PROVIDER_REGISTRY))
    api = QueueFixture()
    clients = []
    class QueueProfile(ProviderProfile):
        def create_client(self, **kwargs):
            client = api.client()
            clients.append(client)
            return client
    profile = QueueProfile(name='queue-test', auth_type='api_key',
        env_vars=('SYNTHETIC_QUEUE_TEST_KEY',), base_url='https://queue.invalid/v1',
        default_aux_model='qwen')
    token = secret_scope.set_secret_scope({'SYNTHETIC_QUEUE_TEST_KEY': 'synthetic'})
    providers.register_provider(profile)
    register_plugin_provider(profile)
    yield api
    for client in clients:
        asyncio.run(client.aclose())
    secret_scope.reset_secret_scope(token)


def test_main_client_resolution_executes_through_queue(queued_profile):
    from agent.agent_runtime_helpers import create_openai_client
    agent = SimpleNamespace(provider='queue-test', _client_log_context=lambda: '',
                            _build_keepalive_http_client=lambda *a, **k: None)
    client = create_openai_client(agent, {'api_key': 'synthetic',
        'base_url': 'https://queue.invalid/v1'}, reason='test', shared=False)
    result = client.chat.completions.create(**arguments())
    assert result.choices[0].message.content == 'hello'
    assert len(queued_profile.calls) == 1


@pytest.mark.parametrize('stream', [False, True])
def test_native_sync_auxiliary_execution(queued_profile, stream):
    from agent.auxiliary_client import resolve_provider_client, _create_with_progress
    client, model = resolve_provider_client('queue-test', 'qwen', task='compression')
    result = _create_with_progress(client, arguments(), task='compression', force_stream=stream)
    assert result.choices[0].message.content == 'hello'
    if stream:
        assert result.choices[0].message.tool_calls[0].function.name == 'terminal'
        assert result.choices[0].finish_reason == 'tool_calls'
    assert len(queued_profile.calls) == 1


@pytest.mark.parametrize('stream', [False, True])
def test_native_async_auxiliary_execution(queued_profile, stream):
    from agent.auxiliary_client import resolve_provider_client, _acreate_with_progress
    async def run():
        client, model = resolve_provider_client('queue-test', 'qwen', async_mode=True)
        result = await _acreate_with_progress(client, arguments(), task='compression', force_stream=stream)
        assert result.choices[0].message.content == 'hello'
        if stream:
            assert result.choices[0].message.tool_calls[0].function.name == 'terminal'
            assert result.choices[0].finish_reason == 'tool_calls'
        await client.aclose()
    asyncio.run(run())
    assert len(queued_profile.calls) == 1


@pytest.mark.parametrize('asynchronous', [False, True])
def test_auxiliary_complete_response_waits_without_summary_watchdog(
        tmp_path, monkeypatch, queued_profile, asynchronous):
    """Native auxiliary resolution and aggregation; not the outer compaction host."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_STREAM_STALE_TIMEOUT', '1')
    from agent import auxiliary_client as aux
    from agent import auxiliary_stream_watchdog as watch
    from ollama.inference_sdk import QueuedClient
    api = QueueFixture()
    api.ready.clear()
    watchdogs = []
    original = watch.ChatStreamWatchdog
    class ObservedWatchdog(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            watchdogs.append(self)
    monkeypatch.setattr(watch, 'ChatStreamWatchdog', ObservedWatchdog)
    monkeypatch.setattr(queued_profile, 'client', lambda: QueuedClient(
        api_key='synthetic', base_url='https://queue.invalid/v1', http_factory=api.http,
        poll_interval=.05, complete_responses=True, grant_lifetime='execution'))
    clients = []
    kwargs = arguments(timeout=1)
    async def async_call():
        client, _ = aux.resolve_provider_client('queue-test','qwen',task='compression',async_mode=True)
        client = client.with_options(timeout=1)
        clients.append(client)
        return await aux._acreate_with_progress(client,kwargs,task='compression',force_stream=True)
    def run():
        if asynchronous:
            return asyncio.run(async_call())
        client, _ = aux.resolve_provider_client('queue-test','qwen',task='compression')
        client = client.with_options(timeout=1)
        clients.append(client)
        return aux._create_with_progress(client,kwargs,task='compression',force_stream=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Auxiliary resolution consumes the owning profile's scoped secret,
            # just as Hermes's protected provider worker copies its context.
            call = pool.submit(copy_context().run, run)
            try:
                observed = api.observed.wait(3)
                if not observed and call.done():
                    call.result()  # surface resolution errors instead of a generic wait failure
                assert observed
                assert not api.ready.wait(2.2)
                assert not call.done()
                assert not watchdogs  # a complete-response adapter owns its wait
                assert all(body.get('stream') is False for body in api.calls.values())
                assert all('stream_options' not in body for body in api.calls.values())
            finally:
                api.ready.set()
            result = call.result(timeout=5)
        assert result.choices[0].message.content == 'hello'
        assert result.choices[0].finish_reason == 'stop'
        assert 'stream' not in kwargs  # request policy did not rewrite caller state
        assert all(body['lifetime'] == 'execution' for method, path, body in api.requests
                   if method == 'POST' and path.endswith('/grants'))
        assert len(api.calls) == 1
        assert sum(method=='POST' and path.endswith('/executions')
                   for method,path,_ in api.requests) == 1
    finally:
        api.ready.set()
        for client in clients:
            asyncio.run(client.aclose())


@pytest.mark.parametrize('interrupt', [False, True])
def test_native_compaction_host_can_wait_without_clock_and_keep_stop_fence(
        tmp_path, monkeypatch, queued_profile, interrupt):
    """Real host/fence + custom summarizer; not stock summary quality or installation."""
    import hermes_cli.config as config
    from run_agent import AIAgent
    from agent import auxiliary_client as aux
    from ollama.inference_sdk import QueuedClient
    api = QueueFixture()
    api.ready.clear()
    clients = []
    def client():
        instance = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=api.http, poll_interval=.05, complete_responses=True)
        clients.append(instance)
        return instance
    monkeypatch.setattr(queued_profile, 'client', client)
    load = config.load_config
    def load_without_host_deadline(*args, **kwargs):
        cfg = load(*args, **kwargs)
        return {**cfg, 'compression': {**cfg.get('compression', {}), 'context_timeout_seconds': 0}}
    monkeypatch.setattr(config, 'load_config', load_without_host_deadline)
    agent = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
        provider='queue-test', model='qwen', quiet_mode=True,
        skip_context_files=True, skip_memory=True)
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = 'Synthetic system.'
    agent.save_trajectories = False
    original = [{'role': 'user', 'content': 'synthetic ' * 100} for _ in range(20)]
    def summarize(messages, **kwargs):
        resolved, _ = aux.resolve_provider_client('queue-test', 'qwen', task='compression')
        # Use the real dispatch boundary, not only its inner aggregation helper:
        # the relay owns protected cancellation while aggregation owns streaming.
        result = aux._relay_sync_completion(resolved, arguments(timeout=1),
            create=lambda request: aux._create_with_progress(
                resolved, request, task='compression', force_stream=True))
        return [{'role': 'user', 'content': result.choices[0].message.content}]
    monkeypatch.setattr(agent.context_compressor, 'compress', summarize)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            call = pool.submit(copy_context().run, lambda: agent._compress_context(
                original, 'Synthetic system.', approx_tokens=2000, force=True))
            try:
                observed = api.observed.wait(3)
                if not observed and call.done():
                    call.result()
                assert observed
                fence = agent._active_compression_commit_fence
                assert not api.ready.wait(2.2)
                assert not call.done()
                assert not fence.is_cancelled
                if interrupt:
                    agent.hard_interrupt('synthetic stop')
                    assert fence.is_cancelled
                    # Stop must return before the provider answers. The saved
                    # inference can outlive observation; it cannot commit context.
                    result, _ = call.result(timeout=5)
                    assert not api.ready.is_set()
                else:
                    api.ready.set()
                    result, _ = call.result(timeout=5)
                if interrupt:
                    assert result == original
                    assert not any(m.get('content') == 'hello' for m in result)
                else:
                    assert any(m.get('content') == 'hello' for m in result)
                assert len(api.calls) == 1
            finally:
                api.ready.set()
    finally:
        for instance in clients:
            asyncio.run(instance.aclose())


@pytest.mark.parametrize('compression,window', [
    ({'threshold': .75, 'threshold_tokens': None}, 65536),
    ({'threshold': .75, 'threshold_tokens': 70000, 'protect_first_n': 0,
      'protect_last_n': 12, 'target_ratio': .3, 'tail_mode': 'legacy',
      'min_tail_user_messages': 2, 'model_thresholds': {'qwen': .65},
      'proactive_prune_tokens': 9000, 'proactive_prune_min_result_chars': 800,
      'proactive_prune_min_reclaim_tokens': 500, 'abort_on_summary_failure': True}, 262144),
])
def test_discovered_context_engine_preserves_current_qwen_policy(
        tmp_path, monkeypatch, queued_profile, compression, window):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from ollama.hermes_queue_context import QueuedContextCompressor
    home = tmp_path / 'profile'
    home.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins' / 'samcloud-queue-context'
    shutil.copytree(source, home / 'plugins' / source.name)
    cfg = {'model': {'default': 'qwen', 'provider': 'queue-test', 'context_length': window},
           'compression': compression, 'context': {'engine': 'compressor'}}
    path = home / 'config.yaml'
    path.write_text(json.dumps(cfg))
    def agent():
        return AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
            provider='queue-test', model='qwen', quiet_mode=True,
            skip_context_files=True, skip_memory=True)
    ordinary = agent()
    cfg['context']['engine'] = 'samcloud-queue-context'
    path.write_text(json.dumps(cfg))
    queued = agent()
    db = SessionDB(tmp_path / 'clone.db')
    try:
        expected, actual = ordinary.context_compressor, queued.context_compressor
        assert isinstance(actual, QueuedContextCompressor)  # not a silent fallback
        for field in ('context_length', 'threshold_percent', 'threshold_tokens',
                      'threshold_tokens_cap', 'protect_first_n', 'protect_last_n',
                      'summary_target_ratio', 'tail_mode', 'min_tail_user_messages',
                      'max_tokens', 'proactive_prune_tokens', 'proactive_prune_min_result_chars',
                      'proactive_prune_min_reclaim_tokens', 'abort_on_summary_failure'):
            assert getattr(actual, field) == getattr(expected, field), field
        db.create_session('bound', source='cli')
        actual.bind_session_state(db, 'bound')
        clone = actual.clone_for_agent()
        assert isinstance(clone, QueuedContextCompressor)
        assert clone._session_db is None and clone._session_id == ''
        clone.update_model(model='qwen', context_length=131072)
        assert actual.context_length == window  # clone never changes parent's budget
    finally:
        ordinary.close()
        queued.close()
        db.close()


@pytest.mark.parametrize('native_plugins', [False, True])
@pytest.mark.parametrize('interrupt', [False, True])
def test_stock_compaction_waits_and_persists_without_losing_later_input(
        tmp_path, monkeypatch, queued_profile, interrupt, native_plugins):
    """Stock engine/host/SQLite + synthetic summary: persistence, not summary quality."""
    import hermes_cli.config as config
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from ollama.hermes_queue_identity import HeldSource
    from ollama.inference_sdk import QueuedClient
    api = QueueFixture()
    api.ready.clear()
    clients = []
    db = SessionDB(tmp_path / 'stock-state.db')
    session = 'stock-queue-compaction'
    db.create_session(session, source='cli')
    original = [{'role': 'user' if n % 2 == 0 else 'assistant',
                 'content': f'Archived synthetic exchange {n}: ' + 'detail ' * 200}
                for n in range(40)]
    db.append_messages_batch(session, original)
    messages = db.get_messages_as_conversation(session, include_row_ids=True)
    source_rows = [m['_row_id'] for m in messages]
    source_contents = [m['content'] for m in messages]
    held_source = tmp_path / 'held-source.json'
    held_source.write_text(json.dumps({'session':session, 'rows':source_rows}))
    def keys(body):
        held = json.loads(held_source.read_text())
        # Frozen explicit row IDs, not a newest-row lookup by a late worker.
        source = HeldSource.capture('synthetic-profile', held['session'], 'compression',
            [{'role':'user','_row_id':row} for row in held['rows']])
        return source.bind(tmp_path / 'stock-queue-identity.sqlite',body)
    def client():
        instance = QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=api.http, poll_interval=.05, key_provider=keys,
            complete_responses=True, grant_lifetime='execution')
        clients.append(instance)
        return instance
    monkeypatch.setattr(queued_profile, 'client', client)
    load = config.load_config
    def queued_config(*args, **kwargs):
        cfg = load(*args, **kwargs)
        return {**cfg, 'compression': {**cfg.get('compression', {}), 'context_timeout_seconds': 0},
                'auxiliary': {**cfg.get('auxiliary', {}), 'compression': {
                    'provider': 'queue-test', 'model': 'qwen',
                    'base_url': 'https://queue.invalid/v1', 'timeout': 1}}}
    if native_plugins:
        from agent import secret_scope
        from ollama import hermes_queue_provider as plugin
        home = tmp_path / 'profile'
        home.mkdir(exist_ok=True)
        source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
        for component in ('samcloud-queue-provider', 'samcloud-queue-context'):
            shutil.copytree(source / component, home / 'plugins' / component)
        # This fixture installed into an already-imported native process; ask
        # for the NEW provider name so native discovery refreshes the home layer
        # rather than using the short cached scan for queue-test.
        import providers
        assert providers.get_provider_profile('samcloud-queue-aux') is not None
        cfg = queued_config()
        cfg['context'] = {'engine': 'samcloud-queue-context'}
        cfg['auxiliary']['compression']['provider'] = 'samcloud-queue-aux'
        (home / 'config.yaml').write_text(json.dumps(cfg))
        def service_client(**kwargs):
            instance = QueuedClient(**kwargs, http_factory=api.http, poll_interval=.05)
            clients.append(instance)
            return instance
        monkeypatch.setattr(plugin, 'QueuedClient', service_client)
        secret_token = secret_scope.set_secret_scope({'SAMCLOUD_QUEUE_API_KEY': 'synthetic',
            'SYNTHETIC_QUEUE_TEST_KEY': 'synthetic'})
    else:
        monkeypatch.setattr(config, 'load_config', queued_config)
    def make_agent():
        instance = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1', provider='queue-test',
            model='qwen', session_id=session, session_db=db, quiet_mode=True,
            skip_context_files=True, skip_memory=True)
        instance._compression_feasibility_checked = True
        instance._cached_system_prompt = 'Synthetic system.'
        instance.save_trajectories = False
        instance.context_compressor.threshold_tokens = 1000
        instance.context_compressor.tail_token_budget = 256
        if native_plugins:
            from ollama.hermes_queue_context import QueuedContextCompressor
            assert isinstance(instance.context_compressor, QueuedContextCompressor)
        return instance
    agent = make_agent()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            call = pool.submit(copy_context().run, lambda: agent._compress_context(
                messages, 'Synthetic system.', approx_tokens=12000, force=True))
            try:
                observed = api.observed.wait(5)
                if not observed and call.done():
                    call.result()
                assert observed
                assert not api.ready.wait(2.2)
                assert not call.done()
                # This new accepted input must not replace the held source IDs
                # or be silently swallowed by a later summary commit.
                db.append_message(session, 'user', 'NEW INPUT WHILE SUMMARY QUEUED')
                if interrupt:
                    agent.hard_interrupt('synthetic stop during stock compaction')
                    result, _ = call.result(timeout=5)
                    assert not api.ready.is_set()
                else:
                    api.ready.set()
                    result, _ = call.result(timeout=5)
            finally:
                api.ready.set()
        assert agent.session_id == session
        active = db.get_messages_as_conversation(session, include_row_ids=True)
        assert sum(m.get('content') == 'NEW INPUT WHILE SUMMARY QUEUED' for m in active) == 1
        assert len(api.calls) == 1
        assert not any('NEW INPUT WHILE SUMMARY QUEUED' in json.dumps(body)
                       for body in api.calls.values())
        if interrupt:
            assert [m['_row_id'] for m in active[:-1]] == source_rows
            assert [m['content'] for m in active[:-1]] == source_contents
            assert not any('hello' in str(m.get('content')) for m in active)
            agent.close()
            agent = make_agent()
            # Reconstruct only the explicitly persisted held source. The newer
            # accepted input is not silently promoted into the interrupted work.
            held = json.loads(held_source.read_text())
            recovered = [m for m in db.get_messages_as_conversation(session, include_row_ids=True)
                         if m['_row_id'] in held['rows']]
            result, _ = agent._compress_context(recovered, 'Synthetic system.',
                                                approx_tokens=12000, force=True)
            active = db.get_messages_as_conversation(session, include_row_ids=True)
            assert len(result) < len(original)
            assert any('hello' in str(m.get('content')) for m in active)
            assert sum(m.get('content') == 'NEW INPUT WHILE SUMMARY QUEUED' for m in active) == 1
            assert len(api.calls) == 1  # same durable inference, not a second dispatch
        else:
            assert len(result) < len(original)
            assert any('hello' in str(m.get('content')) for m in active)
            assert agent.context_compressor._last_summary_fallback_used is False
            assert len(db.get_messages(session, include_inactive=True, include_compacted=True)) > len(original)
        reopened = SessionDB(tmp_path / 'stock-state.db')
        try:
            assert reopened.get_messages_as_conversation(session, include_row_ids=True) == active
        finally:
            reopened.close()
    finally:
        agent.close()
        db.close()
        for instance in clients:
            asyncio.run(instance.aclose())
        if native_plugins:
            secret_scope.reset_secret_scope(secret_token)
