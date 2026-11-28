import json
import unittest

import httpx

from ollama.omlx_client import chat


class NativeTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_result_and_stream_pass_through(self):
        body = dict(model='offering', messages=[dict(role='tool', tool_call_id='call1', content='ok')],
                    tools=[dict(type='function', function=dict(name='read'))],
                    stream=True, stream_options={'include_usage': True},
                    reasoning_effort='medium', max_tokens=8192)
        frames = b'data: {"choices":[{"delta":{"reasoning_content":"a"}}]}\n\ndata: [DONE]\n\n'

        def handle(request):
            if request.method == 'GET':
                self.assertEqual(request.url.path, '/v1/models/status')
                return httpx.Response(200, json={'models': [{'id': 'bound-model', 'reasoning_levels': ['medium']}]})
            self.assertEqual(str(request.url), 'http://127.0.0.1:15123/v1/chat/completions')
            self.assertEqual(request.headers['authorization'], 'Bearer private')
            payload = json.loads(request.content)
            expected = dict(body, model='bound-model', chat_template_kwargs={'reasoning_effort': 'medium'})
            del expected['reasoning_effort']
            self.assertEqual(payload, expected)
            return httpx.Response(200, content=frames)

        async with chat(15123, 'private', 'bound-model', body,
                        transport=httpx.MockTransport(handle)) as response:
            self.assertEqual(await response.aread(), frames)
        self.assertTrue(response.is_closed)
        self.assertNotIn('chat_template_kwargs', body)

    async def test_template_effort_requires_budget_too(self):
        with self.assertRaisesRegex(ValueError, 'positive integer max_tokens'):
            async with chat(15123, 'private', 'bound-model',
                            dict(chat_template_kwargs={'reasoning_effort': 'medium'})):
                self.fail('Invalid request reached backend')

    async def test_native_model_advertised_reasoning_levels(self):
        captured = []
        def handle(request):
            captured.append(json.loads(request.content)['chat_template_kwargs'])
            return httpx.Response(200, json={})

        levels = ['low', 'medium', 'high', 'xhigh', 'max']
        async with chat(15123, 'private', 'inkling', dict(reasoning_effort='max', max_tokens=8192),
                        reasoning_levels=levels, transport=httpx.MockTransport(handle)):
            pass
        self.assertEqual(captured, [{'reasoning_effort': 'max'}])
        for budget in (None, True, 0, '8192'):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                async with chat(15123, 'private', 'inkling', dict(reasoning_effort='max', max_tokens=budget),
                                reasoning_levels=levels):
                    self.fail('Invalid budget reached backend')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            async with chat(15123, 'private', 'other', dict(reasoning_effort='max', max_tokens=8192),
                            reasoning_levels=['low', 'high']):
                self.fail('Unadvertised level reached backend')
