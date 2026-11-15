"""Actual HTTP client behavior with mock transport; no compute or live socket."""

import os
from pathlib import Path
import tempfile
import unittest

import httpx

from .exo_groups.private import PrivateEXO, PrivateEXOUnavailable, private_key


class PrivateEXOControls(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.key = Path(self.temp.name) / 'key'
        self.key.write_text('a' * 64 + '\n')
        self.key.chmod(0o600)
        self.clients = []

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        self.temp.cleanup()

    def client(self, handler, port=52515):
        client = PrivateEXO(port, self.key, transport=httpx.MockTransport(handler))
        self.clients.append(client)
        return client

    async def test_fixed_loopback_state_uses_separate_key_without_deadlines(self):
        seen = []
        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), 'http://127.0.0.1:52515/state')
            self.assertEqual(request.method, 'GET')
            self.assertEqual(request.headers['authorization'], 'Bearer ' + 'a' * 64)
            self.assertTrue(all(value is None for value in request.extensions['timeout'].values()))
            return httpx.Response(200, json={'instances': {}})
        result = await self.client(handler).state()
        self.assertEqual(result, {'instances': {}})
        self.assertEqual(len(seen), 1)

    async def test_key_rotation_is_read_on_each_observation(self):
        keys = []
        def handler(request):
            keys.append(request.headers['authorization'])
            return httpx.Response(200, json={})
        client = self.client(handler)
        await client.state()
        self.key.write_text('b' * 64)
        await client.state()
        self.assertEqual(keys, ['Bearer ' + 'a' * 64, 'Bearer ' + 'b' * 64])

    async def test_redirects_status_errors_and_ambiguous_state_never_authorize(self):
        cases = [
            httpx.Response(307, headers={'location': 'https://cloud.samtg.xyz/api/v1/auth/verify'}),
            httpx.Response(403, json={'error': 'private diagnostic must stay private'}),
            httpx.Response(200, content=b'{"instances":{},"instances":{}}', headers={'content-type': 'application/json'}),
            httpx.Response(200, content=b'{"x":NaN}', headers={'content-type': 'application/json'}),
            httpx.Response(200, json=[]),
            httpx.Response(200, json={'oversize': 'x' * (128 * 1024)}),
        ]
        for response in cases:
            requests = []
            def handler(request):
                requests.append(request)
                return response
            with self.subTest(status=response.status_code):
                with self.assertRaises(PrivateEXOUnavailable) as error:
                    await self.client(handler).state()
                self.assertEqual(str(error.exception), 'private EXO state unavailable')
                self.assertEqual(len(requests), 1)

    async def test_transport_loss_gets_no_cached_state_or_automatic_retry(self):
        count = 0
        def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(200, json={'runners': {}})
            raise httpx.ReadError('private detail')
        client = self.client(handler)
        self.assertEqual(await client.state(), {'runners': {}})
        with self.assertRaises(PrivateEXOUnavailable):
            await client.state()
        self.assertEqual(count, 2)

    async def test_invalid_key_refuses_before_any_http(self):
        seen = []
        def handler(request):
            seen.append(request)
            raise AssertionError('invalid key must not reach transport')
        client = self.client(handler)
        self.key.chmod(0o644)
        with self.assertRaises(PrivateEXOUnavailable):
            await client.state()
        self.key.chmod(0o600)
        self.key.write_text('Bearer plane-token-not-an-exo-key')
        with self.assertRaises(PrivateEXOUnavailable):
            await client.state()
        self.key.write_text('A' * 64)
        with self.assertRaises(PrivateEXOUnavailable):
            await client.state()
        self.assertEqual(seen, [])

    def test_symlink_hardlink_large_key_and_invalid_port_refuse(self):
        link = self.key.with_name('link')
        link.symlink_to(self.key)
        with self.assertRaises(OSError):
            private_key(link)
        link.unlink()
        os.link(self.key, link)
        with self.assertRaises(ValueError):
            private_key(self.key)
        link.unlink()
        self.key.write_text('a' * 259)
        with self.assertRaises(ValueError):
            private_key(self.key)
        for port in (True, 0, 80, 65536, '52515'):
            with self.subTest(port=port), self.assertRaises(ValueError):
                PrivateEXO(port, self.key)


if __name__ == '__main__':
    unittest.main()
