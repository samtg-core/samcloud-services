import unittest

import httpx

from .ollama_client import OllamaClient


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inkling = 'http://127.0.0.1:22434'
        self.default = 'http://127.0.0.1:11434'
        self.calls = []
        self.residents = {}
        self.client = OllamaClient(base_url=self.default, model_backends={'inkling': self.inkling})
        self.client._http.close()
        self.client._http = httpx.Client(base_url=self.default, transport=httpx.MockTransport(self.answer))
        self.addCleanup(self.client._http.close)

    def answer(self, req):
        port, path = req.url.port, req.url.path
        self.calls.append((port, path))
        if path == '/api/tags':
            names = ['qwen', 'inkling:latest'] if port == 11434 else ['inkling:latest']
            return httpx.Response(200, json={'models': [{'name': n} for n in names]})
        if path == '/api/ps':
            return httpx.Response(200, json={'models': self.residents.get(port, [])})
        return httpx.Response(200, json={})

    def test_catalogue_load_and_release_use_configured_origin(self):
        self.assertEqual([m['name'] for m in self.client.list_models()], ['qwen', 'inkling:latest'])
        self.client.load_model('inkling', num_ctx=65536, service_owned=True)
        self.client.show_model('inkling:latest')
        self.client.unload_model('inkling:latest', service_owned=True)
        self.client.show_model('qwen')
        self.assertEqual(self.calls[-4:], [(22434, '/api/generate'), (22434, '/api/show'),
                                          (22434, '/api/generate'), (11434, '/api/show')])

    def test_default_origin_with_trailing_slash_is_polled_once(self):
        client = OllamaClient(base_url=self.default + '/', model_backends={'inkling': self.default})
        client._http.close()
        client._http = self.client._http
        self.assertEqual(client.list_running(), [])
        self.assertEqual(self.calls, [(11434, '/api/ps')])

    def test_missing_secondary_inventory_is_an_error(self):
        def inventory(request):
            if request.url.port == 11434:
                return httpx.Response(200, json={'models': []})
            return httpx.Response(503)

        with httpx.Client(base_url=self.default, transport=httpx.MockTransport(inventory)) as http:
            from unittest.mock import patch
            with patch.object(self.client, '_http', http), self.assertRaises(httpx.HTTPStatusError):
                self.client.list_running()

    def test_unexpected_resident_prevents_cleanup_on_wrong_origin(self):
        self.residents[11434] = [{'name': 'inkling:latest'}]
        with self.assertRaisesRegex(ValueError, 'resident outside'):
            self.client.list_running()


class StreamingRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_listeners_receive_only_their_models(self):
        import asyncio
        import json
        from aiohttp import web
        from unittest.mock import patch

        async def listener(name):
            async def stream(request):
                body = await request.json()
                record = {'model': body['model'], 'origin': name, 'path': request.path, 'done': True}
                return web.Response(text=json.dumps(record) + '\n', content_type='application/x-ndjson')

            app = web.Application()
            app.router.add_post('/api/{operation}', stream)
            runner = web.AppRunner(app)
            await runner.setup()
            self.addAsyncCleanup(runner.cleanup)
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            return 'http://127.0.0.1:' + str(runner.addresses[0][1])

        default = await listener('default')
        alternate = await listener('alternate')
        client = OllamaClient(base_url=default, model_backends={'inkling': alternate})
        self.addCleanup(client._http.close)

        async def collect(stream):
            return [(row['model'], row['origin'], row['path']) async for row in stream]

        with patch.object(client, 'num_ctx_for', return_value=None):
            actual = await asyncio.gather(
                collect(client.chat_stream('qwen', [], service_owned=True)),
                collect(client.generate_stream('inkling:latest', 'test', service_owned=True)))
        self.assertEqual(actual, [[('qwen', 'default', '/api/chat')],
                                  [('inkling:latest', 'alternate', '/api/generate')]])
