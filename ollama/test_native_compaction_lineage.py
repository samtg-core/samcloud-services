"""Committed native SessionDB rotation keeps the same requester parent.

Real native publication, context-engine notification, Relay turn and plugin
hooks; synthetic inference bindings/journal. No GPU or live service/profile.
Run through the matching Hermes scripts/run_tests.sh.
"""
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import relay_runtime
from agent.context_compressor import ContextCompressor
from agent.conversation_compression import _notify_context_engine_compression_complete
from hermes_state import SessionDB

from .hermes_queue_context import QueuedContextCompressor
from .hermes_queue_identity import HostBindings
from .hermes_queue_provider import host_for_home, register_hooks
from .inference_arbiter import Arbiter, StintPolicy
from .inference_identity import LocalSourceRefused


@pytest.mark.parametrize('provider', ['samcloud-queue', 'samcloud-queue-stint'])
@pytest.mark.parametrize('early_summary', [False, True])
@pytest.mark.parametrize('prior_completed_ancestor', [False, True])
@pytest.mark.parametrize('startup', ['resume', 'deferred', 'adopt'])
def test_process_reopen_recovers_published_child_before_alias_commit(
        tmp_path, monkeypatch, provider, early_summary, prior_completed_ancestor, startup):
    import json
    import subprocess
    import sys
    import hermes_state
    home = tmp_path / 'profile'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    code = '''
import os,sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
from hermes_state import SessionDB
from ollama.hermes_queue_identity import HostBindings
from ollama.inference_arbiter import Arbiter,StintPolicy
root=Path(sys.argv[1]);home=root/'profile'
db=SessionDB(root/'state.sqlite')
if sys.argv[5]=='prior':
 db.create_session('S0',source='cli')
 db.publish_compression_child(parent_session_id='S0',child_session_id='S1',source='cli',
  messages=[{'role':'user','content':'old completed context'}],require_compression_lease=False)
else:db.create_session('S1',source='cli')
row=db.append_message('S1','user','fixture')
host=HostBindings(home/'samcloud-queue/identity.sqlite',str(home.resolve()),job_continuity=True)
host.begin_foreground('S1','T1');host.pre_api_request(session_id='S1',turn_id='T1',
 api_request_id='one',conversation_history=[{'role':'user','content':'fixture','_row_id':row}])
body={'model':'qwen','messages':[{'role':'user','content':'fixture'}]}
keys=host.main_stint_keys(body) if sys.argv[4].endswith('stint') else host.main_keys(body)
j=Arbiter(root/'service.sqlite',stint_policy=StintPolicy())
g=j.submit('native',keys.grant_key,'gpu','qwen');e=j.enqueue(g['id'],'native',keys.call_key,body,job_key=keys.job_key)
j.dispatch_next('gpu');j.finish(e['id'],{})
other=j.submit('other','other','gpu','qwen');j.enqueue(other['id'],'other','waiting',{})
(root/'accepted.json').write_text(json.dumps({'job_key':keys.job_key,'job_id':e['job_id']}))
db.publish_compression_child(parent_session_id='S1',child_session_id='S2',source='cli',
 messages=[{'role':'user','content':'published child'}],require_compression_lease=False)
os._exit(73)
'''
    process = subprocess.run([sys.executable, '-c', code, str(tmp_path),
        str(Path(hermes_state.__file__).parent), str(Path(__file__).resolve().parents[1]), provider,
        'prior' if prior_completed_ancestor else 'none'],
        capture_output=True, text=True, timeout=10)
    assert process.returncode == 73, process.stderr
    original = json.loads((tmp_path / 'accepted.json').read_text())
    db = SessionDB(tmp_path / 'state.sqlite')
    host = host_for_home(home)
    engine = QueuedContextCompressor(model='qwen', provider=provider,
        config_context_length=65536, quiet_mode=True)
    if startup in ('deferred', 'adopt'):
        engine.bind_session_state(db, 'S1')
    if startup == 'deferred':
        assert _notify_context_engine_compression_complete(
            SimpleNamespace(context_compressor=engine, platform='cli'),
            new_session_id='S2', old_session_id='S1')
        assert host.owner_session('S2') == 'S2'  # no live native owner yet
    elif startup == 'resume':
        engine.bind_session_state(db, 'S2')
        engine.on_session_start('S2', boundary_reason='resume')
    hooks = {}
    register_hooks(SimpleNamespace(register_hook=lambda name, fn: hooks.__setitem__(name, fn),
                                   register_middleware=lambda *a: None))
    coordinator = relay_runtime.RelaySessionCoordinator()
    lease = relay_runtime.ConversationLease(str(home.resolve()), 'S2', 'cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()), 'reopened process'), None)
    turn = coordinator.begin_turn(lease, turn_id='T2', task_id='resume')
    journal = Arbiter(tmp_path / 'service.sqlite', stint_policy=StintPolicy())
    try:
        if startup == 'adopt':
            assert _notify_context_engine_compression_complete(
                SimpleNamespace(context_compressor=engine, platform='cli'),
                new_session_id='S2', old_session_id='S1')
        if early_summary:
            with patch.object(ContextCompressor, 'compress', lambda self, *a, **kw:
                    host.auxiliary_keys({'model': 'qwen', 'messages': [{'role': 'user', 'content': 'summary'}]})):
                summary = engine.compress([{'role': 'user', 'content': 'child before turn-start hook'}])
            assert summary.job_key == original['job_key']
        hooks['pre_llm_call'](session_id='S2', turn_id='T2')
        history = [{'role': m['role'], 'content': m['content'], '_row_id': m['id']}
                   for m in db.get_messages('S2')]
        host.pre_api_request(session_id='S2', turn_id='T2', api_request_id='next', conversation_history=history)
        body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'resume'}]}
        keys = host.main_stint_keys(body) if provider.endswith('stint') else host.main_keys(body)
        assert keys.job_key == original['job_key']
        grant = journal.submit('native', keys.grant_key, 'gpu', 'qwen')
        call = journal.enqueue(grant['id'], 'native', keys.call_key, body, job_key=keys.job_key)
        assert call['job_id'] == original['job_id']
        assert journal.dispatch_next('gpu')['execution']['id'] == call['id']
        assert host.owner_session('S2') == 'S1'
    finally:
        coordinator.end_turn(turn, outcome='success'); coordinator.release_conversation(lease)
        db.close(); journal.close()


@pytest.mark.parametrize('boundary', ['reset', 'fork-source', 'foreign-profile'])
def test_unrelated_child_does_not_inherit_or_end_an_unfinished_parent(tmp_path, monkeypatch, boundary):
    from .hermes_queue_source import owner_scope
    from .inference_identity import IdentityStore
    home = tmp_path / 'profile'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    host = host_for_home(home)
    db = SessionDB(tmp_path / 'state.sqlite'); db.create_session('S1', source='cli')
    host.begin_foreground('S1', 'old')
    host.pre_api_request(session_id='S1', turn_id='old', api_request_id='old',
        conversation_history=[{'role': 'user', '_row_id': 1, 'content': 'old'}])
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'fixture'}]}
    original = host.main_keys(body)
    db.end_session('S1', 'reset' if boundary == 'reset' else 'compression')
    db.create_session('S2', source='subagent' if boundary == 'fork-source' else 'cli',
        parent_session_id='S1', profile_name='foreign' if boundary == 'foreign-profile' else None)
    row = db.append_message('S2', 'user', 'new independent task')
    engine = QueuedContextCompressor(model='qwen', provider='samcloud-queue',
        config_context_length=65536, quiet_mode=True)
    engine.bind_session_state(db, 'S2')
    hooks = {}
    register_hooks(SimpleNamespace(register_hook=lambda name, fn: hooks.__setitem__(name, fn),
                                   register_middleware=lambda *a: None))
    coordinator = relay_runtime.RelaySessionCoordinator()
    lease = relay_runtime.ConversationLease(str(home.resolve()), 'S2', 'cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()), 'independent task'), None)
    turn = coordinator.begin_turn(lease, turn_id='new', task_id='independent')
    try:
        hooks['pre_llm_call'](session_id='S2', turn_id='new')
        host.pre_api_request(session_id='S2', turn_id='new', api_request_id='new',
            conversation_history=[{'role': 'user', '_row_id': row, 'content': 'new independent task'}])
        assert host.main_keys(body).job_key != original.job_key
        assert host.owner_session('S2') == 'S2'
        store = IdentityStore(host.path)
        try:
            active = store.db.execute('SELECT job_key FROM native_requester_job WHERE scope=?',
                                      (owner_scope(host.scope, 'S1'),)).fetchone()
            assert active == (original.job_key,)
            assert store.pending_events() == []
        finally:
            store.close()
    finally:
        coordinator.end_turn(turn, outcome='success'); coordinator.release_conversation(lease)
        db.close()


@pytest.mark.parametrize('provider', ['samcloud-queue', 'samcloud-queue-stint'])
@pytest.mark.parametrize('rotations', [1, 2])
def test_committed_rotation_keeps_parent_for_main_tool_summary_and_finish(tmp_path, monkeypatch, provider, rotations):
    home = tmp_path / 'profile'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    host = host_for_home(home)
    db = SessionDB(tmp_path / 'state.sqlite')
    db.create_session('S1', source='cli')
    db.append_message('S1', 'user', 'original fixture')
    engine = QueuedContextCompressor(model='qwen', provider=provider,
        config_context_length=65536, quiet_mode=True)
    engine.bind_session_state(db, 'S1')
    hooks, middleware = {}, {}
    register_hooks(SimpleNamespace(register_hook=lambda name, fn: hooks.__setitem__(name, fn),
                                   register_middleware=lambda name, fn: middleware.__setitem__(name, fn)))
    coordinator = relay_runtime.RelaySessionCoordinator()
    lease = relay_runtime.ConversationLease(str(home.resolve()), 'S1', 'cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()), 'synthetic lineage test'), None)
    turn = coordinator.begin_turn(lease, turn_id='T', task_id='fixture')
    journal = Arbiter(tmp_path / 'service.sqlite', stint_policy=StintPolicy())
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'fixture'}]}

    def history(session):
        return [{'role': m['role'], 'content': m['content'], '_row_id': m['id']}
                for m in db.get_messages(session)]

    def main(session, api, *, begin=True):
        if begin:
            hooks['pre_llm_call'](session_id=session, turn_id='T')
        hooks['pre_api_request'](provider=provider, session_id=session, turn_id='T',
            api_request_id=api, conversation_history=history(session))
        keys = middleware['llm_execution'](request=body, provider=provider,
            session_id=session, turn_id='T', api_request_id=api,
            next_call=lambda request: (host.main_stint_keys(request)
                if provider.endswith('stint') else host.main_keys(request)))
        keys.on_accepted(SimpleNamespace(grant_id='grant-' + api, execution_id='execution-' + api))
        return keys

    try:
        first = main('S1', 'one')
        grant = journal.submit('native', first.grant_key, 'gpu', 'qwen')
        call = journal.enqueue(grant['id'], 'native', first.call_key, body, job_key=first.job_key)
        journal.dispatch_next('gpu'); journal.finish(call['id'], {})
        other = journal.submit('other', 'other', 'gpu', 'qwen')
        waiting = journal.enqueue(other['id'], 'other', 'waiting', {})

        # The same native publication primitive used by rotating compaction.
        agent = SimpleNamespace(context_compressor=engine, platform='cli')
        session = 'S1'
        for index in range(rotations):
            child_session = 'S' + str(index + 2)
            db.publish_compression_child(parent_session_id=session, child_session_id=child_session, source='cli',
                messages=[{'role': 'user', 'content': 'compressed fixture'}],
                require_compression_lease=False)
            assert _notify_context_engine_compression_complete(agent,
                new_session_id=child_session, old_session_id=session)
            session = child_session
        assert turn.lease.session_id == 'S1'  # native Relay deliberately retains its original lease
        # Native turn-start hooks run once, not once per inference request.
        after = main(session, 'two', begin=False)
        assert after.job_key == first.job_key and after.job_key is not None
        child_grant = journal.submit('native', after.grant_key, 'gpu', 'qwen')
        child = journal.enqueue(child_grant['id'], 'native', after.call_key, body, job_key=after.job_key)
        assert journal.dispatch_next('gpu')['execution']['id'] == child['id']
        journal.finish(child['id'], {})

        hooks['post_api_request'](provider=provider, session_id=session, turn_id='T', api_request_id='two',
            response={'assistant_message': {'tool_calls': [{'id': 'tool'}]}})
        with host.tool_scope(session, 'T', 'two', 'tool'), host.tool_api_scope():
            tool = host.tool_keys(body | {'model': 'vision'})
        assert tool.job_key == first.job_key
        with patch.object(ContextCompressor, 'compress',
                          lambda self, *a, **kw: host.auxiliary_keys(body)):
            summary = engine.compress(history(session))
        assert summary.job_key == first.job_key
        assert HostBindings(host.path, host.scope, job_continuity=True).owner_session(session) == 'S1'

        hooks['on_session_end'](session_id=session, turn_id='T', completed=True, failed=False, interrupted=False)
        events = host.pending_events()
        assert len(events) == 1 and events[0]['job_key'] == first.job_key
        journal.end_job('native', events[0]['job_key'])
        assert journal.dispatch_next('gpu')['execution']['id'] == waiting['id']
        coordinator.end_turn(turn, outcome='success')
        coordinator.release_conversation(lease)
        lease = relay_runtime.ConversationLease(str(home.resolve()), session, 'cli',
            relay_runtime.NoopRelayRuntime(str(home.resolve()), 'new native turn'), None)
        turn = coordinator.begin_turn(lease, turn_id='T2', task_id='next')
        hooks['pre_llm_call'](session_id=session, turn_id='T2')
        host.pre_api_request(session_id=session, turn_id='T2', api_request_id='next',
            conversation_history=history(session))
        # Existing completed input retains its frozen receipt; a fresh row is
        # needed for a fresh call, exactly as in the real native transcript.
        db.append_message(session, 'user', 'next job')
        host.pre_api_request(session_id=session, turn_id='T2', api_request_id='next',
            conversation_history=history(session))
        fresh = host.main_stint_keys(body) if provider.endswith('stint') else host.main_keys(body)
        assert fresh.job_key is not None and fresh.job_key != first.job_key
    finally:
        coordinator.end_turn(turn, outcome='success')
        coordinator.release_conversation(lease)
        journal.close(); db.close()


def test_uncommitted_or_closed_compaction_cannot_alias_another_session(tmp_path, monkeypatch):
    home = tmp_path / 'profile'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('samcloud_queue:\n  job_continuity: true\n')
    host = host_for_home(home)
    db = SessionDB(tmp_path / 'state.sqlite'); db.create_session('S1', source='cli')
    engine = QueuedContextCompressor(model='qwen', provider='samcloud-queue',
        config_context_length=65536, quiet_mode=True)
    engine.bind_session_state(db, 'S1')
    coordinator = relay_runtime.RelaySessionCoordinator()
    lease = relay_runtime.ConversationLease(str(home.resolve()), 'S1', 'cli',
        relay_runtime.NoopRelayRuntime(str(home.resolve()), 'synthetic'), None)
    turn = coordinator.begin_turn(lease, turn_id='T', task_id='fixture')
    host.begin_foreground('S1', 'T')
    try:
        with pytest.raises(LocalSourceRefused):
            engine.on_session_start('unknown', boundary_reason='compression', old_session_id='S1')
        assert host.owner_session('unknown') == 'unknown'
        db.publish_compression_child(parent_session_id='S1', child_session_id='S2', source='cli',
            messages=[{'role': 'user', 'content': 'fixture'}], require_compression_lease=False)
        copied = copy_context()
        coordinator.end_turn(turn, outcome='success'); coordinator.release_conversation(lease)
        copied.run(engine.on_session_start, 'S2', boundary_reason='compression', old_session_id='S1')
        assert host.owner_session('S2') == 'S2'
        assert engine._session_id == 'S2'
    finally:
        coordinator.end_turn(turn, outcome='success'); coordinator.release_conversation(lease)
        db.close()
