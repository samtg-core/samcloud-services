"""Provider token accounting survives the gateway's native-Ollama translation.

Real route and claim lifetime; synthetic terminal frames, no live inference.
"""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import server
from .manager import Backend, ModelManager


class StreamUsage(unittest.IsolatedAsyncioTestCase):
    async def frames(self, chunks, options=None):
        async def chat_stream(*args, **kwargs):
            for chunk in chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        model = SimpleNamespace(name='qwen', backend=Backend.OLLAMA, in_flight=0, last_used=0)
        manager = object.__new__(ModelManager)
        manager.models = {'qwen': model}
        manager.ollama = SimpleNamespace(chat_stream=chat_stream)
        async def resolve(name):
            return model
        with patch.object(server, 'mgr', manager), \
             patch.object(server, '_resolve_model', resolve), \
             patch.object(server.prompt_size, 'count', return_value=(99, 'synthetic estimate')):
            response = await server.chat_completions(server.ChatRequest(
                model='qwen', messages=[{'role': 'user', 'content': 'test'}],
                stream=True, stream_options=options))
            raw = [frame async for frame in response.body_iterator]
        self.assertEqual(model.in_flight, 0)
        return [json.loads(frame[6:]) if frame[6:].strip() != '[DONE]' else '[DONE]' for frame in raw]

    async def test_requested_usage_is_backend_count_before_done_not_local_estimate(self):
        frames = await self.frames([
            {'message': {'content': 'hello'}, 'done': False},
            {'done': True, 'prompt_eval_count': 82312, 'eval_count': 8061},
        ], {'include_usage': True})
        self.assertEqual(frames[-1], '[DONE]')
        self.assertEqual(frames[-2]['choices'], [])
        self.assertEqual(frames[-2]['model'], 'qwen')
        self.assertEqual(frames[-2]['usage'], {
            'prompt_tokens': 82312, 'completion_tokens': 8061, 'total_tokens': 90373})
        self.assertEqual(frames[-3]['choices'][0]['finish_reason'], 'stop')

    async def test_unrequested_usage_does_not_change_stream_shape(self):
        for options in (None, {'include_usage': False}):
            with self.subTest(options=options):
                frames = await self.frames([{'done': True, 'prompt_eval_count': 12, 'eval_count': 3}], options)
                self.assertEqual(frames[-1], '[DONE]')
                self.assertFalse(any(isinstance(f, dict) and f.get('usage') is not None for f in frames))

    async def test_missing_counts_are_not_fabricated_as_zero(self):
        frames = await self.frames([{'done': True}], {'include_usage': True})
        self.assertEqual(frames[-1], '[DONE]')
        self.assertFalse(any(isinstance(f, dict) and f.get('usage') is not None for f in frames))

    async def test_real_zero_counts_are_preserved(self):
        frames = await self.frames([{'done': True, 'prompt_eval_count': 0, 'eval_count': 0}],
                                   {'include_usage': True})
        self.assertEqual(frames[-2]['usage']['total_tokens'], 0)

    async def test_invalid_or_partial_counts_stay_unknown(self):
        for prompt, completion in ((True, 1), (1, False), (-1, 2), (2, -1),
                                   (12, 1.5), (12, '2'), (12, None), (None, 2)):
            with self.subTest(prompt=prompt, completion=completion):
                frames = await self.frames([
                    {'done': True, 'prompt_eval_count': prompt, 'eval_count': completion},
                ], {'include_usage': True})
                self.assertEqual(frames[-1], '[DONE]')
                self.assertFalse(any(isinstance(f, dict) and 'usage' in f for f in frames))

    async def test_tool_finish_and_accounting_are_both_preserved(self):
        frames = await self.frames([
            {'message': {'tool_calls': [{'function': {'name': 'terminal', 'arguments': {'command': 'pwd'}}}]},
             'done': False},
            {'done': True, 'prompt_eval_count': 12, 'eval_count': 3},
        ], {'include_usage': True})
        self.assertEqual(frames[-3]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(frames[-2]['usage']['total_tokens'], 15)

    async def test_backend_failure_has_no_success_usage(self):
        frames = await self.frames([RuntimeError('synthetic failure')], {'include_usage': True})
        self.assertTrue(any(isinstance(f, dict) and f.get('error') for f in frames))
        self.assertFalse(any(isinstance(f, dict) and f.get('usage') is not None for f in frames))


if __name__ == '__main__':
    unittest.main()
