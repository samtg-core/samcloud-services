"""Native finalization metadata -> exact durable queue yield; no provider calls."""
import asyncio
import logging
import socket
from types import SimpleNamespace

import pytest
import httpx

from agent.conversation_loop import _notify_context_engine_turn_complete
from ollama.hermes_queue_context import QueuedContextCompressor
from ollama.hermes_queue_provider import host_for_home
from ollama.inference_arbiter import Arbiter, Conflict, StintPolicy
from ollama.inference_events import relay_once


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError('no real network in lifecycle regression')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    yield
    assert attempts == []


@pytest.fixture
def owned(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    host = host_for_home(home)
    engine = QueuedContextCompressor(model='qwen', provider='samcloud-queue-stint',
                                    config_context_length=65536, quiet_mode=True)
    engine.bind_session_state(session_id='child')
    history = [{'role': 'user', '_row_id': 1, 'content': 'synthetic'}]
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'synthetic'}]}

    def bind(session, turn, api):
        host.pre_api_request(session_id=session, turn_id=turn, api_request_id=api,
                             conversation_history=history)
        keys = host.main_stint_keys(body)
        accepted = getattr(keys, 'on_accepted', None)
        if accepted is not None:
            accepted(SimpleNamespace(grant_id='accepted', execution_id=keys.call_key))
        return keys

    child = bind('child', 'child-turn', 'child-api')
    parent = bind('parent', 'parent-turn', 'parent-api')
    return host, engine, child, parent, bind


def notify(engine, **meta):
    agent = SimpleNamespace(context_compressor=engine, session_id=engine._session_id)
    _notify_context_engine_turn_complete(agent, [], logger=logging.getLogger(__name__), **meta)


@pytest.mark.parametrize('outcome', [dict(interrupted=True), dict(failed=True), {}])
def test_native_completion_records_exact_child_even_without_response(owned, outcome):
    host, engine, child, parent, _ = owned
    notify(engine, turn_id='child-turn', **outcome)
    events = host.pending_events()
    assert len(events) == 1
    assert events[0]['grant_key'] == child.grant_key != parent.grant_key
    assert events[0]['expected_call_key'] == child.call_key
    # Durable identifiers only; a new observer sees the same event.
    reopened = type(host)(host.path, host.scope)
    assert reopened.pending_events() == events


def test_existing_post_hook_and_completion_callback_are_idempotent(owned):
    host, engine, child, _, _ = owned
    host.finish_turn('child', 'child-turn')
    before = host.pending_events()
    notify(engine, turn_id='child-turn')
    notify(engine, turn_id='child-turn', interrupted=True)
    assert host.pending_events() == before


def test_missing_or_unrelated_binding_never_yields_parent(owned):
    host, engine, _, _, _ = owned
    notify(engine, interrupted=True)
    notify(engine, turn_id='unbound-turn', interrupted=True)
    engine.bind_session_state(session_id='')
    notify(engine, turn_id='parent-turn', interrupted=True)
    assert host.pending_events() == []


@pytest.mark.parametrize('provider', ['samcloud-queue', 'samcloud-queue-aux', 'custom'])
def test_nonstint_engine_does_not_signal_work_session(owned, provider):
    host, engine, _, _, _ = owned
    engine.provider = provider
    notify(engine, turn_id='child-turn', interrupted=True)
    assert host.pending_events() == []


def test_existing_relay_yields_completed_child_without_idle_grace(owned, tmp_path):
    host, engine, child, _, _ = owned
    journal = Arbiter(tmp_path / 'queue.sqlite',
                      stint_policy=StintPolicy(idle_grace_seconds=300))
    try:
        grant = journal.submit('seat', child.grant_key, 'gpu', 'qwen', lifetime='work_session')
        call = journal.enqueue(grant['id'], 'seat', child.call_key, {})
        journal.dispatch_next('gpu')
        journal.finish(call['id'], {})
        other = journal.submit('another-seat', 'ready', 'gpu', 'qwen', lifetime='work_session')
        journal.enqueue(other['id'], 'another-seat', 'ready-call', {})
        assert journal.dispatch_next('gpu') is None
        notify(engine, turn_id='child-turn', interrupted=True)

        def handle(request):
            if request.url.path.endswith('/capabilities'):
                return httpx.Response(200, json={'stints': {'lifecycle_yield': True}})
            import json
            body = json.loads(request.content)
            verdict = journal.yield_by_key('seat', body['grant_key'], body['expected_call_key'])
            return httpx.Response(200, json=verdict)

        async def deliver():
            async with httpx.AsyncClient(base_url='https://synthetic.invalid',
                                         transport=httpx.MockTransport(handle)) as http:
                return await relay_once(SimpleNamespace(http=http, request_timeout=1), host)
        assert asyncio.run(deliver()) == 1
        assert host.pending_events() == []
        assert journal.dispatch_next('gpu')['grant']['id'] == other['id']
        assert journal.get(grant['id'], 'seat')['handoff_reason'] == 'explicit_yield'
    finally:
        journal.close()


@pytest.mark.parametrize('uncertain', [False, True])
def test_event_does_not_cancel_active_or_uncertain_compute(owned, tmp_path, uncertain):
    host, engine, child, _, _ = owned
    notify(engine, turn_id='child-turn', interrupted=True)
    events = host.pending_events()
    assert len(events) == 1
    event = events[0]
    journal = Arbiter(tmp_path / 'queue.sqlite', stint_policy=StintPolicy())
    try:
        grant = journal.submit('seat', child.grant_key, 'gpu', 'qwen', lifetime='work_session')
        call = journal.enqueue(grant['id'], 'seat', child.call_key, {})
        journal.dispatch_next('gpu')
        if uncertain:
            journal.uncertain(call['id'])
        before = journal.execution(call['id'], 'seat')['state']
        journal.yield_by_key('seat', event['grant_key'], event['expected_call_key'])
        assert journal.execution(call['id'], 'seat')['state'] == before
        assert journal.get(grant['id'], 'seat')['active_execution'] == call['id']
        assert journal.dispatch_next('gpu') is None
    finally:
        journal.close()


def test_late_event_cannot_yield_newer_call(owned, tmp_path):
    host, engine, child, _, _ = owned
    notify(engine, turn_id='child-turn', interrupted=True)
    events = host.pending_events()
    assert len(events) == 1
    event = events[0]
    journal = Arbiter(tmp_path / 'queue.sqlite', stint_policy=StintPolicy())
    try:
        grant = journal.submit('seat', child.grant_key, 'gpu', 'qwen', lifetime='work_session')
        call = journal.enqueue(grant['id'], 'seat', child.call_key, {})
        journal.dispatch_next('gpu')
        journal.finish(call['id'], {})
        journal.enqueue(grant['id'], 'seat', 'newer', {})
        journal.dispatch_next('gpu')
        with pytest.raises(Conflict):
            journal.yield_by_key('seat', event['grant_key'], event['expected_call_key'])
        assert journal.get(grant['id'], 'seat')['yield_requested'] == 0
    finally:
        journal.close()
