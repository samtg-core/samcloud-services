"""Actual native recovery and tools over the queue SDK, with an offline service."""
import asyncio
import json
from pathlib import Path
import shutil

import httpx

from .test_hermes_queue_client import queued_profile
from .test_inference_sdk import QueueFixture


def test_native_nudge_survives_a_later_real_tool_round(tmp_path, monkeypatch, queued_profile):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from hermes_cli import plugins
    from ollama import hermes_queue_provider as plugin
    from ollama.inference_sdk import QueuedClient
    from ollama.hermes_queue_recovery import NATIVE_EMPTY_NUDGE
    from agent.conversation_loop import _EMPTY_TOOL_RESPONSE_NUDGE
    assert NATIVE_EMPTY_NUDGE == _EMPTY_TOOL_RESPONSE_NUDGE
    home = tmp_path / 'profile'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('agent.title_generator.maybe_auto_title', lambda *a, **kw: None)
    monkeypatch.setattr('agent.title_generator.start_title_upgrade', lambda *a, **kw: None)
    monkeypatch.setattr('agent.model_metadata.fetch_model_metadata', lambda *a, **kw: {})
    monkeypatch.setattr('model_tools.check_toolset_requirements', lambda *a, **kw: {})
    source = Path(__file__).resolve().parents[1] / 'hermes-plugins'
    for name in ('samcloud-queue-provider', 'samcloud-queue-hooks'):
        shutil.copytree(source / name, home / 'plugins' / name)
    (home / 'config.yaml').write_text('plugins:\n  isolation: in_process\n  enabled:\n    - samcloud-queue-hooks\n')
    plugins.discover_plugins(force=True)
    from hermes_cli.lifecycle import has_hook
    assert has_hook('pre_api_request')
    output = tmp_path / 'result.txt'

    def tool(name, args, ident):
        return {'tool_calls': [{'index': 0, 'id': ident, 'type': 'function',
                               'function': {'name': name, 'arguments': json.dumps(args)}}]}

    responses = [(tool('write_file', {'path': str(output), 'content': 'written once'}, 'write-1'), 'tool_calls'),
                 ({}, 'stop'),
                 (tool('read_file', {'path': str(output)}, 'read-1'), 'tool_calls'),
                 ({'content': 'Done.'}, 'stop')]

    class RecoveryAPI(QueueFixture):
        def handle(self, request):
            if request.url.path.endswith('/events'):
                call = request.url.path.split('/')[-2]
                index = list(self.calls).index(call)
                assert index < len(responses), 'unexpected replacement inference'
                delta, finish = responses[index]
                frame = {'id': 'answer', 'object': 'chat.completion.chunk', 'model': 'qwen',
                         'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                return httpx.Response(200, json={'state': 'complete', 'events': [
                    {'seq': 1, 'data': 'data: ' + json.dumps(frame) + '\n\n'},
                    {'seq': 2, 'data': 'data: [DONE]\n\n'}], 'next_cursor': 2, 'has_more': False})
            return super().handle(request)

    api, clients = RecoveryAPI(), []

    def client(**kwargs):
        value = QueuedClient(**kwargs, http_factory=api.http, poll_interval=.001)
        clients.append(value)
        return value

    monkeypatch.setattr(plugin, 'QueuedClient', client)
    db = SessionDB(tmp_path / 'state.db')
    agent = AIAgent(api_key='synthetic', base_url='https://queue.invalid/v1',
                    provider='samcloud-queue-stint', model='qwen', session_db=db,
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    enabled_toolsets=['file'])
    agent._cached_system_prompt = 'Offline native recovery test.'
    agent._use_prompt_caching = agent.compression_enabled = agent.save_trajectories = False
    agent._api_max_retries = 1
    monkeypatch.setattr(agent, '_cleanup_task_resources', lambda *a, **kw: None)
    try:
        result = agent.run_conversation('Write result.txt, read it back, then say Done.')
        assert result['final_response'] == 'Done.'
        assert output.read_text() == 'written once'
        assert len(api.calls) == 4
        saved = db.get_messages_as_conversation(agent.session_id)
        assert [m['tool_call_id'] for m in saved if m['role'] == 'tool'] == ['write-1', 'read-1']
        assert not any(m.get('_empty_recovery_synthetic') or m.get('content') in ('(empty)', NATIVE_EMPTY_NUDGE)
                       for m in saved)
        last_request = list(api.calls.values())[-1]
        sent = last_request['messages']
        position = next(i for i, m in enumerate(sent) if m.get('content') == NATIVE_EMPTY_NUDGE)
        assert sent[position - 1]['content'] == '(empty)'
        assert any(m['role'] == 'tool' for m in sent[position + 1:])
    finally:
        for value in clients:
            asyncio.run(value.aclose())
        db.close()
