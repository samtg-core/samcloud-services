"""Real native Relay/hooks/tool middleware; synthetic receipts and journal only."""
from contextvars import copy_context
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import httpx

from agent import relay_runtime
from hermes_state import SessionDB

from .hermes_queue_context import QueuedContextCompressor
from .hermes_queue_provider import host_for_home, register_hooks
from .inference_arbiter import Arbiter, Conflict, StintPolicy
from .inference_identity import IdentityStore, LocalSourceRefused
from .inference_sdk import QueuedClient
from .hermes_queue_source import CompactionSource


@pytest.fixture(params=['samcloud-queue', 'samcloud-queue-stint'])
def native(tmp_path, monkeypatch, request):
    home = tmp_path / 'profile'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    db = SessionDB(tmp_path / 'state.sqlite'); db.create_session('S1', source='cli')
    row = db.append_message('S1', 'user', 'synthetic')
    host = host_for_home(home)
    engine = QueuedContextCompressor(model='qwen', provider=request.param,
        config_context_length=65536, quiet_mode=True)
    engine.bind_session_state(db, 'S1')
    hooks, middleware = {}, {}
    register_hooks(SimpleNamespace(register_hook=lambda name, fn: hooks.__setitem__(name, fn),
        register_middleware=lambda name, fn: middleware.__setitem__(name, fn)))
    coordinator = relay_runtime.RelaySessionCoordinator()
    lease = relay_runtime.ConversationLease(str(home.resolve()), 'S1', 'cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()), 'authority fixture'), None)
    turn = coordinator.begin_turn(lease, turn_id='T1', task_id='fixture')
    hooks['pre_llm_call'](session_id='S1', turn_id='T1')
    def main(turn_id='T1', message_row=row, api='api'):
        hooks['pre_api_request'](provider=request.param, session_id='S1', turn_id=turn_id,
            api_request_id=api, conversation_history=[{'role':'user', 'content':'synthetic', '_row_id':message_row}])
        body = {'model':'qwen', 'messages':[{'role':'user', 'content':'synthetic'}]}
        def bind(body):
            keys = host.main_stint_keys(body) if request.param.endswith('stint') else host.main_keys(body)
            keys.on_accepted(SimpleNamespace(grant_id='main-grant', execution_id=keys.call_key))
            return keys
        keys = middleware['llm_execution'](request=body, next_call=bind, provider=request.param,
            session_id='S1', turn_id=turn_id, api_request_id=api)
        hooks['post_api_request'](provider=request.param, session_id='S1', turn_id=turn_id,
            api_request_id=api, response={'assistant_message':{'tool_calls':[{'id':'tool'}]}})
        return keys
    parent = main()
    state = SimpleNamespace(host=host, db=db, hooks=hooks, middleware=middleware,
        coordinator=coordinator, lease=lease, turn=turn, main=main, parent=parent, provider=request.param)
    try:
        yield state
    finally:
        coordinator.end_turn(turn, outcome='success'); coordinator.release_conversation(lease)
        db.close()


def captured_tool(state, *, pending=False, accepted=False, completed_child=False):
    held = {}
    def dispatch(args):
        with state.host.tool_api_scope():
            held['context'] = copy_context()
            if pending or completed_child:
                held['keys'] = state.host.tool_keys({'model':'vision', 'messages':[{'role':'user','content':'image'}]})
                if accepted:
                    held['keys'].on_accepted(SimpleNamespace(grant_id='vision-grant', execution_id='vision-execution'))
                if pending:
                    raise RuntimeError('observer disconnected before result')
    try:
        state.middleware['tool_execution'](next_call=dispatch, args={}, session_id='S1',
            turn_id='T1', api_request_id='api', tool_call_id='tool')
    except RuntimeError:
        if not pending:
            raise
    return held


def close_turn(state, *, completed=True):
    state.host.observe_native_outcome('S1', 'T1', completed=completed, failed=False, interrupted=not completed)
    state.coordinator.end_turn(state.turn, outcome='success' if completed else 'interrupted')


@pytest.mark.parametrize('boundary', ['closed', 'completed-hook'])
def test_closed_copied_tool_cannot_create_a_new_intent(native, boundary):
    held = captured_tool(native)
    if boundary == 'closed':
        close_turn(native)
    else:
        native.host.observe_native_outcome('S1', 'T1', completed=True, failed=False, interrupted=False)
    store = IdentityStore(native.host.path)
    try:
        before = store.db.execute('SELECT count(*) FROM call_identity').fetchone()[0]
        with pytest.raises(LocalSourceRefused):
            held['context'].run(native.host.tool_keys, {'model':'vision', 'messages':[{'role':'user','content':'image'}]})
        assert store.db.execute('SELECT count(*) FROM tool_aux_intent').fetchone()[0] == 0
        assert store.db.execute('SELECT count(*) FROM call_identity').fetchone()[0] == before
    finally:
        store.close()


def test_native_owner_ending_during_capture_rolls_back_new_tool_identity(native):
    capture = CompactionSource.capture
    def ended_during_capture(*args, **kwargs):
        source = capture(*args, **kwargs)
        native.coordinator.end_turn(native.turn, outcome='interrupted')
        return source
    with patch.object(CompactionSource, 'capture', side_effect=ended_during_capture):
        with pytest.raises(LocalSourceRefused):
            captured_tool(native, pending=True)
    store = IdentityStore(native.host.path)
    try:
        assert store.db.execute('SELECT count(*) FROM tool_aux_intent').fetchone()[0] == 0
        assert store.db.execute('SELECT count(*) FROM call_identity').fetchone()[0] == 1
    finally:
        store.close()


@pytest.mark.parametrize('accepted', [False, True])
@pytest.mark.parametrize('completed_child', [False, True])
def test_closed_copied_tool_can_only_read_its_exact_acknowledged_child(native, accepted, completed_child):
    held = captured_tool(native, pending=not completed_child, accepted=accepted, completed_child=completed_child)
    close_turn(native)
    body = {'model':'vision', 'messages':[{'role':'user','content':'image'}]}
    if not accepted:
        with pytest.raises(LocalSourceRefused):
            held['context'].run(native.host.tool_keys, body)
    else:
        replay = held['context'].run(native.host.tool_keys, body)
        assert replay.resume_job == ('vision-grant', 'vision-execution')
        assert replay.job_key == held['keys'].job_key
        assert (replay.grant_key, replay.call_key) == (held['keys'].grant_key, held['keys'].call_key)
        requests = []
        def read_saved(request):
            requests.append((request.method, request.url.path))
            assert request.headers['Authorization'] == 'Bearer synthetic'
            assert requests[-1] == ('GET', '/v1/inference/executions/vision-execution')
            return httpx.Response(200, json={'id':'vision-execution','grant_id':'vision-grant',
                'state':'complete','job_id':'same-parent','result':{'kind':'json','body':{
                    'id':'vision-execution','object':'chat.completion','created':1,'model':'vision',
                    'choices':[{'index':0,'finish_reason':'stop',
                                'message':{'role':'assistant','content':'saved image analysis'}}]}}})
        client = QueuedClient(api_key='synthetic', base_url='https://synthetic.invalid/v1',
            http_factory=lambda:httpx.AsyncClient(base_url='https://synthetic.invalid',
                headers={'Authorization':'Bearer synthetic'}, transport=httpx.MockTransport(read_saved)),
            key_provider=lambda body:held['context'].run(native.host.tool_keys, body))
        try:
            answer = client.create(**body)
            assert answer.choices[0].message.content == 'saved image analysis'
            assert len(requests) == 2  # reattach + result; no intake/capability/release
        finally:
            asyncio.run(client.aclose())
        with pytest.raises((Conflict, ValueError)):
            held['context'].run(native.host.tool_keys, body | {'model':'changed'})


def test_overlapping_uninstrumented_turn_refuses_before_intake_then_recovers_same_parent(native, tmp_path):
    journal = Arbiter(tmp_path / 'queue.sqlite', stint_policy=StintPolicy())
    root = journal.submit('native', native.parent.grant_key, 'gpu', 'qwen')
    first = journal.enqueue(root['id'], 'native', native.parent.call_key, {}, job_key=native.parent.job_key)
    journal.dispatch_next('gpu'); journal.finish(first['id'], {})
    other = journal.submit('other', 'other', 'gpu', 'qwen')
    waiting = journal.enqueue(other['id'], 'other', 'waiting', {})
    overlapping = native.coordinator.begin_turn(native.lease, turn_id='T2', task_id='overlap')
    try:
        assert not overlapping.relay_enabled
        native.hooks['pre_llm_call'](session_id='S1', turn_id='T2')
        with pytest.raises(LocalSourceRefused):
            native.main('T2', message_row=2, api='overlap')
        # Native observer hooks are fail-open: even if their refusal is caught,
        # the execution middleware cannot reuse the previous main source.
        bind = native.host.main_stint_keys if native.provider.endswith('stint') else native.host.main_keys
        with pytest.raises(LocalSourceRefused):
            native.middleware['llm_execution'](request={'model':'qwen'}, next_call=bind,
                provider=native.provider, session_id='S1', api_request_id='overlap')
        store = IdentityStore(native.host.path)
        try:
            assert store.db.execute('SELECT count(*) FROM call_identity').fetchone()[0] == 1
        finally:
            store.close()
        assert journal.dispatch_next('gpu') is None
        assert journal.execution(waiting['id'], 'other')['state'] == 'queued'
    finally:
        native.coordinator.end_turn(overlapping, outcome='failed')
    close_turn(native, completed=False)
    resumed = native.coordinator.begin_turn(native.lease, turn_id='T3', task_id='resume')
    try:
        native.hooks['pre_llm_call'](session_id='S1', turn_id='T3')
        keys = native.main('T3', message_row=3, api='resume')
        assert keys.job_key == native.parent.job_key
        grant = journal.submit('native', keys.grant_key, 'gpu', 'qwen')
        call = journal.enqueue(grant['id'], 'native', keys.call_key, {}, job_key=keys.job_key)
        assert call['job_id'] == first['job_id']
        assert journal.dispatch_next('gpu')['execution']['id'] == call['id']
    finally:
        native.coordinator.end_turn(resumed, outcome='success'); journal.close()
