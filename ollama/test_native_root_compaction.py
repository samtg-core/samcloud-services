"""Real automatic compaction inference/commit -> root HTTP, synthetic backend.

No production identity/queue, weights or GPU. External HTTP is forbidden.
"""
import asyncio
import json
from pathlib import Path
import shutil
import time

import httpx
import pytest

from .test_native_root_round_trip import root_queue  # noqa: F401


@pytest.mark.parametrize('provider',['samcloud-queue','samcloud-queue-stint'])
@pytest.mark.parametrize('in_place',[True,False])
def test_auto_compaction_request_tool_and_rotating_owner_keep_root(
        tmp_path,monkeypatch,root_queue,provider,in_place):
    from agent import secret_scope
    from agent.auxiliary_client import resolve_provider_client
    from hermes_state import SessionDB
    from hermes_cli import plugins
    import model_tools
    from run_agent import AIAgent
    from tools.registry import registry
    from .inference_identity import IdentityStore
    from .hermes_queue_provider import host_for_home

    url,observed,_,state=root_queue;state['allow_compaction']=True
    original_send=httpx.Client.send;original_async_send=httpx.AsyncClient.send
    def send(self,request,*args,**kwargs):
        assert request.url.host in ('127.0.0.1','localhost','registry.invalid'),'External network forbidden in fixture'
        return original_send(self,request,*args,**kwargs)
    async def async_send(self,request,*args,**kwargs):
        assert request.url.host in ('127.0.0.1','localhost','registry.invalid'),'External network forbidden in fixture'
        return await original_async_send(self,request,*args,**kwargs)
    monkeypatch.setattr(httpx.Client,'send',send);monkeypatch.setattr(httpx.AsyncClient,'send',async_send)
    home=tmp_path/'compacting-profile';home.mkdir()
    monkeypatch.setenv('HERMES_HOME',str(home));monkeypatch.setattr(Path,'home',lambda:tmp_path)
    source=Path(__file__).resolve().parents[1]/'hermes-plugins'
    for name in ('samcloud-queue-hooks','samcloud-queue-provider','samcloud-queue-context'):
        shutil.copytree(source/name,home/'plugins'/name)
    config={'plugins':{'isolation':'in_process','enabled':['samcloud-queue-hooks']},
        'samcloud_queue':{'job_continuity':True},'context':{'engine':'samcloud-queue-context'},
        'model':{'provider':provider,'default':'qwen','base_url':url+'/v1','context_length':65536},
        'auxiliary':{'compression':{'provider':'samcloud-queue-aux','model':'qwen','base_url':url+'/v1'}},
        'compression':{'protect_first_n':0,'protect_last_n':2,'threshold_tokens':1000}}
    (home/'config.yaml').write_text(json.dumps(config));plugins.discover_plugins()
    token=secret_scope.set_secret_scope({'SAMCLOUD_QUEUE_API_KEY':'native-test'})
    schema={'name':'job_round_trip_probe','description':'Synthetic same-root vision tool',
            'parameters':{'type':'object','properties':{'slot':{'type':'integer'}}}}
    monkeypatch.setattr(model_tools,'get_tool_definitions',lambda *a,**k:[{'type':'function','function':schema}])
    monkeypatch.setattr(model_tools,'check_toolset_requirements',lambda *a,**k:{})
    database=SessionDB(tmp_path/'native-state.sqlite');database.create_session('native-session',source='cli')
    history=[]
    for index in range(40):
        role='user' if index%2==0 else 'assistant';content=('synthetic archive detail '*200)+str(index)
        row=database.append_message('native-session',role,content)
        history.append({'role':role,'content':content,'_row_id':row})
    agent=AIAgent(api_key='native-test',base_url=url+'/v1',provider=provider,model='qwen',
        session_id='native-session',session_db=database,quiet_mode=True,skip_context_files=True,skip_memory=True)
    agent._cached_system_prompt='Synthetic system.';agent._use_prompt_caching=False
    agent.compression_enabled=True;agent.compression_in_place=in_place;agent.save_trajectories=False;agent.tool_delay=0
    engine=agent.context_compressor;engine.threshold_tokens=1000;engine.tail_token_budget=256
    clients=[]
    def tool(args,**kwargs):
        child,_=resolve_provider_client('samcloud-queue-tool','vision',task='vision',explicit_base_url=url+'/v1')
        clients.append(child)
        result=child.create(model='vision',messages=[{'role':'user','content':'synthetic vision request'}])
        assert result.choices[0].message.content=='child done'
        assert all(row['owner']=='native' for row in observed)
        return 'child finished'
    registry.register(name='job_round_trip_probe',toolset='utility',schema=schema,handler=tool,override=True)
    try:
        result=agent.run_conversation('Use the synthetic tool, then finish.',conversation_history=history)
        assert result['completed'] and result['final_response']=='done'
        assert state.get('summary_executions'),'No real compaction inference was submitted'
        assert not engine._last_summary_fallback_used,engine._last_summary_error
        summary_positions=[i for i,row in enumerate(observed) if row['purpose']=='summary']
        assert max(i for i,row in enumerate(observed) if row['owner']=='native')>min(summary_positions),observed
        native=[row for row in observed if row['owner']=='native']
        assert len({row['job_id'] for row in native})==1 and native[0]['job_id'] is not None
        assert len({row['lease_id'] for row in native})==1
        assert any(row['model']=='vision' for row in native)
        if in_place:assert agent.session_id=='native-session'
        else:
            assert agent.session_id!='native-session'
            assert database.get_session('native-session')['end_reason']=='compression'
            assert host_for_home(home).owner_session(agent.session_id)=='native-session'
        identity=IdentityStore(home/'samcloud-queue'/'identity.sqlite')
        try:
            keys=identity.db.execute('SELECT DISTINCT job_key FROM call_identity WHERE job_key IS NOT NULL').fetchall()
            assert len(keys)==1
            assert identity.db.execute("SELECT COUNT(*) FROM lifecycle_outbox WHERE kind='job_finish'").fetchone()[0]==1
        finally:identity.close()
        limit=time.monotonic()+15  # synthetic fixture scheduling bound, not a job deadline
        with httpx.Client(base_url=url,headers={'Authorization':'Bearer other-test'}) as http:
            while http.get('/v1/inference/executions/'+state['other_execution']).json()['state']!='complete':
                assert time.monotonic()<limit
                time.sleep(.01)
        assert [row['owner'] for row in observed]==['native']*len(native)+['other']
        assert native[0]['lease_id']!=observed[-1]['lease_id'] and len(state['roots'])==2
    finally:
        for client in clients:asyncio.run(client.aclose())
        if getattr(agent,'client',None) is not None:agent.client.close()
        secret_scope.reset_secret_scope(token);database.close()
