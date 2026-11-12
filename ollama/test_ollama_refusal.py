"""Pre-stream negative acknowledgement only; no inference or live backend."""
import asyncio
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from .ollama_client import OllamaClient, OllamaBackendError, OllamaRefused


class RefusalClient(unittest.TestCase):
    def test_sync_chat_and_generate_distinguish_complete_refusal_from_unknown(self):
        from .ollama_client import OllamaRefused
        cases = [(400, {'error': 'PRIVATE invalid input'}, OllamaRefused),
                 (404, {'error': 'PRIVATE model path'}, OllamaRefused),
                 (400, {'detail': 'not Ollama'}, OllamaBackendError),
                 (500, {'error': 'PRIVATE runner crashed'}, OllamaBackendError)]
        for method in ('chat', 'generate'):
            for status, body, kind in cases:
                with self.subTest(method=method, status=status, body=body):
                    client = OllamaClient(base_url='http://fake')
                    client._http.close()
                    client._http = httpx.Client(base_url='http://fake',
                        transport=httpx.MockTransport(lambda req: httpx.Response(status, json=body)))
                    try:
                        with self.assertRaises(kind) as caught:
                            list(getattr(client, method)('qwen', [] if method == 'chat' else 'prompt'))
                        if kind is OllamaBackendError:
                            self.assertNotIsInstance(caught.exception, OllamaRefused)
                            self.assertNotIn('PRIVATE', str(caught.exception))
                        if kind is OllamaRefused:
                            self.assertEqual(caught.exception.result['status'], status)
                            self.assertNotIn('PRIVATE', str(caught.exception.result))
                    finally:
                        client._http.close()

    def test_sync_truncated_error_body_is_not_confirmed_refusal(self):
        class Broken(httpx.SyncByteStream):
            def __iter__(self):
                yield b'{"error":"invalid'
                raise httpx.ReadError('body connection lost')
        client = OllamaClient(base_url='http://fake')
        client._http.close()
        client._http = httpx.Client(base_url='http://fake',
            transport=httpx.MockTransport(lambda req: httpx.Response(400, stream=Broken())))
        try:
            with self.assertRaises(httpx.ReadError):
                list(client.chat('qwen', []))
        finally:
            client._http.close()

    def test_async_pre_stream_refusal_is_typed(self):
        from .ollama_client import OllamaRefused
        class Response:
            status = 400
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def read(self): return json.dumps({'error': 'PRIVATE invalid input'}).encode()
            def raise_for_status(self): raise AssertionError('typed refusal must precede generic status')
        class Session:
            def post(self, *a, **kw): return Response()
            async def close(self): pass
        async def run():
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = 'http://fake'
            fake = SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **kw: None)
            with patch.dict(sys.modules, {'aiohttp': fake}):
                with self.assertRaises(OllamaRefused) as caught:
                    _ = [c async for c in client._stream_with_deadlines('/api/chat', {'model':'qwen'})]
                self.assertNotIn('PRIVATE', str(caught.exception.result))
        asyncio.run(run())

    def test_async_midstream_error_is_not_a_pre_stream_refusal(self):
        class Response:
            status = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def raise_for_status(self): pass
            @property
            def content(self):
                async def chunks():
                    yield b'{"error":"PRIVATE runner error"}\n'
                return chunks()
        class Session:
            def post(self, *a, **kw): return Response()
            async def close(self): pass
        async def run():
            client = OllamaClient.__new__(OllamaClient)
            client.base_url = 'http://fake'
            fake = SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **kw: None)
            with patch.dict(sys.modules, {'aiohttp': fake}):
                for owned in (False, True):
                    with self.assertRaises(OllamaBackendError) as caught:
                        _ = [c async for c in client._stream_with_deadlines(
                            '/api/chat', {'model':'qwen'}, service_owned=owned)]
                    self.assertNotIsInstance(caught.exception, OllamaRefused)
                    self.assertNotIn('PRIVATE', str(caught.exception))
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
