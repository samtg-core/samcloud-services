"""Actual native agent/plugins/SDK → local HTTP journal, synthetic backend only.

Run with the matching Hermes scripts/run_tests.sh. No model, GPU, production
token/profile, registry lease or external HTTP is used. Physical service and
image usability qualification remain separate.
"""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from .inference_api import build_router
from .inference_arbiter import StintPolicy
from .inference_supervisor import Supervisor


@pytest.fixture
def local_queue(tmp_path, monkeypatch):
    from . import server as model_service
    journal = tmp_path / 'service.sqlite'
    observed = []
    state = {}
    main_calls = [0]

    async def execute(grant, body):
        ident = grant['active_execution']
        observed.append({'owner': grant['owner'], 'model': grant['model'], 'execution': ident})
        if body.get('stream'):
            main_calls[0] += 1
            if main_calls[0] == 1 and state.get('tool_first', True):
                delta = {'role': 'assistant', 'tool_calls': [
                    {'index': index, 'id': f'native-job-tool-{index}', 'type': 'function',
                     'function': {'name': 'job_round_trip_probe',
                                  'arguments': json.dumps({'slot': index})}}
                    for index in range(state.get('child_count', 1))]}
                finish = 'tool_calls'
            else:
                delta, finish = {'role': 'assistant', 'content': 'done'}, 'stop'
                if main_calls[0] == 1:
                    supervisor = state['supervisor']
                    other = supervisor.journal.submit('other', 'other', 'synthetic-gpu', 'qwen', 'execution')
                    waiting = supervisor.journal.enqueue(other['id'], 'other', 'other-call', {
                        'model': 'qwen', 'messages': [{'role': 'user', 'content': 'other requester'}]})
                    state['other_execution'] = waiting['id']
            frame = {'id': ident, 'object': 'chat.completion.chunk', 'created': 1,
                     'model': grant['model'], 'choices': [{'index': 0, 'delta': delta,
                                                          'finish_reason': finish}]}
            state['supervisor'].journal.append_event(ident, 'data: ' + json.dumps(frame) + '\n\n', terminal=True)
            state['supervisor'].journal.append_event(ident, 'data: [DONE]\n\n', terminal=True)
            return {'kind': 'stream', 'event_count': 2, 'media_type': 'text/event-stream'}
        return {'kind': 'json', 'body': {'id': ident, 'object': 'chat.completion', 'created': 1,
            'model': grant['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                                                'message': {'role': 'assistant', 'content': 'child done'}}]}}

    @asynccontextmanager
    async def lifespan(app):
        supervisor = Supervisor(journal, execute, lambda resource: True, stint_policy=StintPolicy())
        state['supervisor'] = supervisor
        app.include_router(build_router(supervisor,
            lambda model: ('synthetic-gpu', model),
            lambda model, body: model_service.ChatRequest(**body).model_dump()))
        yield
        if supervisor.preparations:
            await asyncio.gather(*tuple(supervisor.preparations))
        if supervisor.workers:
            await asyncio.gather(*tuple(supervisor.workers.values()))
        supervisor.close()

    app = FastAPI(lifespan=lifespan)
    monkeypatch.setattr(model_service, 'AUTH_ENABLED', True)
    identities = {'Bearer native-test': {'username': 'native'},
                  'Bearer other-test': {'username': 'other'}}
    monkeypatch.setattr(model_service.SamcloudAuthMiddleware, '_get_cached',
                        lambda middleware, token: identities[token])
    app.add_middleware(model_service.SamcloudAuthMiddleware,
                       verify_url='https://not-called.invalid', required_scope='services')
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    url = 'http://127.0.0.1:' + str(listener.getsockname()[1])
    service = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    thread = threading.Thread(target=lambda: service.run(sockets=[listener]), daemon=True)
    thread.start()
    limit = time.monotonic() + 10  # harness startup bound, never a product inference deadline
    while not service.started:
        if not thread.is_alive() or time.monotonic() >= limit:
            raise AssertionError('local synthetic HTTP fixture did not start')
        time.sleep(.01)
    try:
        yield url, observed, journal, state
    finally:
        service.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive()


@pytest.mark.parametrize('provider', ['samcloud-queue', 'samcloud-queue-stint'])
@pytest.mark.parametrize('child_count,interrupt_after_child', [(1, False), (1, True), (0, False), (2, False)])
def test_native_parent_tool_and_recovery_keep_place_until_real_finish(
    tmp_path, monkeypatch, local_queue, provider, child_count, interrupt_after_child,
):
    from agent import secret_scope
    from agent.auxiliary_client import resolve_provider_client
    from hermes_state import SessionDB
    from hermes_cli import plugins
    import model_tools
    from run_agent import AIAgent
    from tools.registry import registry
    from .hermes_queue_provider import host_for_home

    url, observed, _, service_state = local_queue
    service_state['tool_first'] = bool(child_count)
    service_state['child_count'] = child_count
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
    for name in ('samcloud-queue-hooks', 'samcloud-queue-provider'):
        shutil.copytree(source / name, home / 'plugins' / name)
    (home / 'config.yaml').write_text(json.dumps({
        'plugins': {'isolation': 'in_process', 'enabled': ['samcloud-queue-hooks']},
        'samcloud_queue': {'job_continuity': True},
    }))
    plugins.discover_plugins()
    token = secret_scope.set_secret_scope({'SAMCLOUD_QUEUE_API_KEY': 'native-test'})
    schema = {'name': 'job_round_trip_probe', 'description': 'Synthetic child round trip',
              'parameters': {'type': 'object', 'properties': {'slot': {'type': 'integer'}}}}
    from agent import tool_dispatch_helpers
    monkeypatch.setattr(tool_dispatch_helpers, '_PARALLEL_SAFE_TOOLS',
                        tool_dispatch_helpers._PARALLEL_SAFE_TOOLS | {'job_round_trip_probe'})
    monkeypatch.setattr(model_tools, 'get_tool_definitions', lambda *a, **k:
                        [{'type': 'function', 'function': schema}])
    monkeypatch.setattr(model_tools, 'check_toolset_requirements', lambda *a, **k: {})
    db = SessionDB(tmp_path / 'native-state.sqlite')
    db.create_session('native-session', source='cli')
    agent = AIAgent(api_key='native-test', base_url=url + '/v1', provider=provider, model='qwen',
        session_id='native-session', session_db=db, quiet_mode=True,
        skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = 'Synthetic system.'
    agent._use_prompt_caching = agent.compression_enabled = agent.save_trajectories = False
    agent.tool_delay = 0
    clients = []
    other = {}
    concurrent_start = threading.Barrier(2) if child_count == 2 else None

    def tool(args, **kwargs):
        if concurrent_start is not None:
            # Prove the actual native executor starts both workers; a sequential
            # execution cannot satisfy this harness-only synchronization bound.
            concurrent_start.wait(timeout=5)
        with httpx.Client(base_url=url, headers={'Authorization': 'Bearer other-test'}) as client:
            grant = client.post('/v1/inference/grants', json={
                'model': 'qwen', 'key': 'other', 'lifetime': 'execution'}).json()
            queued = client.post('/v1/inference/grants/' + grant['id'] + '/executions', json={
                'key': 'other-call', 'request': {'model': 'qwen',
                    'messages': [{'role': 'user', 'content': 'other requester'}]}})
            assert queued.status_code == 202, queued.text
            other['execution'] = queued.json()['id']
            assert queued.json()['state'] == 'queued'
        child, _ = resolve_provider_client('samcloud-queue-tool', 'vision', task='vision',
                                           explicit_base_url=url + '/v1')
        clients.append(child)
        result = child.create(model='vision', messages=[{'role': 'user',
            'content': 'synthetic child ' + str(args['slot'])}])
        assert result.choices[0].message.content == 'child done'
        assert all(row['owner'] == 'native' for row in observed)
        if interrupt_after_child:
            agent.interrupt()
        return 'child finished'

    registry.register(name='job_round_trip_probe', toolset='utility', schema=schema,
                      handler=tool, override=True)
    try:
        first = agent.run_conversation('Run the synthetic child.')
        host = host_for_home(home)
        if interrupt_after_child:
            assert not first['completed'] and first['interrupted']
            assert [row['owner'] for row in observed] == ['native', 'native']
            assert host.pending_events() == []
            result = agent.run_conversation('Continue.', conversation_history=first['messages'])
        else:
            result = first
        assert result['completed'] and result['final_response'] == 'done'
        # The async lifecycle relay may already have accepted the same event.
        from .inference_identity import IdentityStore
        store = IdentityStore(home / 'samcloud-queue' / 'identity.sqlite')
        try:
            kinds = store.db.execute('SELECT kind FROM lifecycle_outbox').fetchall()
            jobs = store.db.execute('SELECT DISTINCT job_key FROM call_identity WHERE job_key IS NOT NULL').fetchall()
            assert kinds == [('job_finish',)]
            assert len(jobs) == 1
        finally:
            store.close()
        if not child_count:
            other['execution'] = service_state['other_execution']
        limit = time.monotonic() + 5  # only waiting on this synthetic fixture's task scheduling
        with httpx.Client(base_url=url, headers={'Authorization': 'Bearer other-test'}) as client:
            while client.get('/v1/inference/executions/' + other['execution']).json()['state'] != 'complete':
                assert time.monotonic() < limit
                time.sleep(.01)
        native_count = child_count + 2 if child_count else 1
        assert [row['owner'] for row in observed] == ['native'] * native_count + ['other']
        assert [row['model'] for row in observed] == (
            ['qwen'] + ['vision'] * child_count + ['qwen', 'qwen'] if child_count else ['qwen', 'qwen'])
    finally:
        for client in clients:
            asyncio.run(client.aclose())
        if getattr(agent, 'client', None) is not None:
            close = getattr(agent.client, 'close', None)
            if close:
                close()
        secret_scope.reset_secret_scope(token)
        db.close()
