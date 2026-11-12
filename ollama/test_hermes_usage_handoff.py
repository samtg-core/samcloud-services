"""Real gateway frames -> SDK -> Hermes aggregation and compression counter.

Run via the matching Hermes checkout's scripts/run_tests.sh with this absolute
file path. Isolated profile, synthetic backend, no model or registry calls.
"""
import asyncio
import json
from pathlib import Path
import sys

import httpx
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollama import test_ollama_stream_usage as wire


def test_provider_counts_reach_hermes_compaction_decision(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    from agent.auxiliary_client import _create_with_progress
    from agent.context_compressor import ContextCompressor
    frames = asyncio.run(wire.StreamUsage().frames([
        {'message': {'content': 'summary'}, 'done': False},
        {'done': True, 'prompt_eval_count': 82312, 'eval_count': 8061},
    ], {'include_usage': True}))
    body = ''.join('data: ' + (f if isinstance(f, str) else json.dumps(f)) + '\n\n' for f in frames)
    def response(request):
        assert json.loads(request.content)['stream_options']['include_usage'] is True
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body)
    with OpenAI(api_key='synthetic', base_url='http://synthetic.invalid/v1',
                http_client=httpx.Client(transport=httpx.MockTransport(response))) as client:
        result = _create_with_progress(client, {'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'synthetic'}]}, force_stream=True)
    assert result.usage.prompt_tokens == 82312
    assert result.usage.completion_tokens == 8061
    monkeypatch.setattr('agent.context_compressor.get_model_context_length', lambda *a, **k: 65536)
    compressor = ContextCompressor(model='synthetic', threshold_percent=.75, quiet_mode=True)
    assert compressor.context_length == 65536
    assert not compressor.should_compress()
    compressor.update_from_response(result.usage.model_dump())
    assert compressor.last_real_prompt_tokens == 82312
    assert compressor.should_compress()
