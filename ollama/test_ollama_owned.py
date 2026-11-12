"""The durable worker, not a consumer's silence clock, owns generation life."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from . import config
from .ollama_client import OllamaClient, OllamaRequestTimeout


class OwnedProviderLifetime(unittest.TestCase):
    def test_load_unload_owned_policy_has_no_compute_read_deadline(self):
        seen = []
        def answer(request):
            seen.append((json.loads(request.content),request.extensions['timeout']))
            return httpx.Response(200,json={'done':True})
        client = OllamaClient(base_url='http://fake')
        client._http.close()
        client._http = httpx.Client(base_url='http://fake',transport=httpx.MockTransport(answer))
        try:
            for method,legacy in (('load_model',300),('unload_model',60)):
                for owned in (False,True):
                    self.assertEqual(getattr(client,method)('qwen',service_owned=owned),{'done':True})
                    body,limits = seen[-1]
                    self.assertNotIn('service_owned',body)
                    self.assertEqual(limits['read'],None if owned else legacy)
                    if owned:
                        self.assertFalse(body['stream'])
                        self.assertEqual(limits['connect'],10)
                        self.assertEqual(limits['write'],10)
                    else:
                        self.assertNotIn('stream',body)  # legacy wire shape unchanged
        finally:
            client._http.close()

    def test_real_socket_owned_read_survives_a_legacy_read_timeout(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                requests.append(self.path)
                body = b'{"done":true}\n'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                time.sleep(.12)  # real socket silence, longer than legacy bound
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the deliberately timed-out legacy observer
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = OllamaClient(base_url='http://127.0.0.1:' + str(server.server_port))
        client._stream_timeout = httpx.Timeout(connect=1, read=.025, write=1, pool=1)
        try:
            with self.assertRaises(httpx.ReadTimeout):
                list(client.chat('qwen', []))
            started = time.monotonic()
            self.assertEqual(list(client.chat('qwen', [], service_owned=True)), [{'done': True}])
            self.assertGreater(time.monotonic() - started, .1)
            self.assertEqual(requests, ['/api/chat', '/api/chat'])
        finally:
            client._http.close()
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_sync_owned_policy_removes_read_age_only_and_never_enters_payload(self):
        seen = []
        def answer(request):
            seen.append((request.url.path, json.loads(request.content), request.extensions['timeout']))
            return httpx.Response(200, content=b'{"done":true}\n')
        client = OllamaClient(base_url='http://fake')
        client._http.close()
        client._http = httpx.Client(base_url='http://fake', transport=httpx.MockTransport(answer))
        try:
            for method, source in (('chat', []), ('generate', 'prompt')):
                for owned in (False, True):
                    self.assertEqual(list(getattr(client, method)('qwen', source,
                        service_owned=owned, options={'num_predict': 17})), [{'done': True}])
                    path, body, limits = seen[-1]
                    self.assertNotIn('service_owned', body)
                    self.assertEqual(body['options']['num_predict'], 17)
                    self.assertEqual(limits['read'], None if owned else 300)
                    self.assertEqual(limits['connect'], 10)
                    self.assertEqual(limits['write'], 10)
        finally:
            client._http.close()

    def test_async_owned_wait_outlasts_both_legacy_silence_and_whole_request_budgets(self):
        seen = []
        class Response:
            status = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def raise_for_status(self): pass
            @property
            def content(self):
                async def chunks():
                    await asyncio.sleep(.06)
                    yield b'{"message":{"content":"first"}}\n'
                    await asyncio.sleep(.06)
                    yield b'{"done":true}\n'
                return chunks()
        class Session:
            def post(self, url, json, timeout):
                seen.append((json, timeout))
                return Response()
            async def close(self): pass
        async def run():
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = 'http://fake'
            fake = SimpleNamespace(ClientSession=Session,
                                   ClientTimeout=lambda **kw: kw)
            with patch.dict(sys.modules, {'aiohttp': fake}), \
                    patch.object(config, 'OLLAMA_GENERATE_TIMEOUT', .02), \
                    patch.object(config, 'OLLAMA_STALL_TIMEOUT', .01), \
                    patch.object(config, 'OLLAMA_FIRST_TOKEN_MIN_S', .01), \
                    patch.object(config, 'OLLAMA_FIRST_TOKEN_RATE_TPS', 0):
                with self.assertRaises(OllamaRequestTimeout):
                    _ = [c async for c in client._stream_with_deadlines('/api/chat', {'model':'qwen'})]
                for path in ('/api/chat', '/api/generate'):
                    rows = [c async for c in client._stream_with_deadlines(path,
                        {'model':'qwen'}, service_owned=True)]
                    self.assertEqual(rows[-1], {'done': True})
                    self.assertEqual(seen[-1][1]['total'], None)
                    self.assertEqual(seen[-1][1]['sock_connect'], 10)
                    self.assertNotIn('service_owned', seen[-1][0])
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
