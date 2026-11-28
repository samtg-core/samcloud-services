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
