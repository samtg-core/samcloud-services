"""Service client side of the staged atomic residency rollover contract."""
import json
import unittest

import httpx

from .samcloud import SamcloudClient


class RolloverClient(unittest.TestCase):
    def client(self, handler):
        client = SamcloudClient('synthetic', base='http://registry.invalid/api/v1')
        client._http.close()
        client._http = httpx.Client(base_url=client.base, transport=httpx.MockTransport(handler))
        self.addCleanup(client._http.close)
        return client

    def test_unknown_response_retries_identical_persisted_operation_only(self):
        requests = []
        def handler(request):
            requests.append((request.method, request.url.path, json.loads(request.content)))
            if len(requests) == 1:
                raise httpx.ReadTimeout('synthetic lost reply', request=request)
            return httpx.Response(200, json={'lease_id':'replacement', 'status':'active'})
        client = self.client(handler)
        with self.assertRaises(httpx.ReadTimeout):
            client.rollover_lease('predecessor', key='saved-intent', ttl_seconds=3600)
        result = client.rollover_lease('predecessor', key='saved-intent', ttl_seconds=3600)
        self.assertEqual(result['lease_id'], 'replacement')
        self.assertEqual(requests, [('POST','/api/v1/leases/predecessor/rollover',
            {'key':'saved-intent','ttl_seconds':3600})] * 2)

    def test_error_never_falls_back_to_release_or_fresh_allocation(self):
        for status in (401,403,404,409,422,503):
            with self.subTest(status=status):
                calls = []
                def handler(request):
                    calls.append(request.url.path)
                    return httpx.Response(status,json={'detail':'synthetic refusal'})
                client = self.client(handler)
                with self.assertRaises(httpx.HTTPStatusError):
                    client.rollover_lease('old',key='saved',ttl_seconds=3600)
                self.assertEqual(calls, ['/api/v1/leases/old/rollover'])

    def test_current_terminal_replacement_status_is_not_relabelled_active(self):
        client = self.client(lambda request:httpx.Response(200,json={
            'lease_id':'replacement','status':'expired','expires_at':'2000-01-01T00:00:00Z'}))
        result = client.rollover_lease('old',key='saved',ttl_seconds=3600)
        self.assertEqual(result['status'], 'expired')

    def test_missing_identity_or_invalid_ttl_refuses_before_network(self):
        calls = []
        client = self.client(lambda request:calls.append(request))
        for key, ttl in [('',3600),(' ',3600),('x'*513,3600),('saved',True),('saved',0)]:
            with self.subTest(key_length=len(key),ttl=ttl):
                with self.assertRaises(ValueError):
                    client.rollover_lease('old',key=key,ttl_seconds=ttl)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
