"""Opt-in registry wait metadata; no live network, load or lease mutation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import httpx

from .inference_arbiter import CapacityDeferred
from .inference_reservation import InitialReservations
from .samcloud import SamcloudClient


class SharedQueueClient(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def handle(request):
            self.calls.append(json.loads(request.content))
            return httpx.Response(200, json={'lease_id':'exact','status':'queued','queue_position':1})
        self.client = SamcloudClient('synthetic', base='http://registry.invalid/api/v1')
        self.client._http.close()
        self.client._http = httpx.Client(base_url=self.client.base, transport=httpx.MockTransport(handle))
        self.addCleanup(self.client._http.close)

    def test_existing_callers_do_not_silently_join_the_new_queue(self):
        self.client.request_lease('gpu', memory_mb=100)
        self.assertNotIn('wait_s', self.calls[0])

    def test_shared_queue_setting_is_explicit_and_strict(self):
        from .inference_bootstrap import QueueSettings
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'queue.sqlite'
            self.assertIsNone(QueueSettings(path,{'qwen':'gpu'}).shared_queue_wait_s)
            for invalid in (True, 0, -1, '3600', 1.5):
                with self.subTest(value=invalid), self.assertRaises(ValueError):
                    QueueSettings(path,{'qwen':'gpu'},shared_queue_wait_s=invalid)

    def test_explicit_wait_is_sent_without_changing_compute_ttl(self):
        receipt = self.client.request_lease('gpu', service_id='service', memory_mb=100,
                                           ttl_seconds=9000, max_total_s=21600, wait_s=5400)
        self.assertEqual(self.calls, [{'ttl_seconds':9000,'max_total_s':21600,
                                      'service_id':'service','memory_mb':100,'wait_s':5400}])
        self.assertEqual(receipt['status'], 'queued')
        self.assertEqual(receipt['status_code'], 200)

    def test_invalid_wait_never_sends_a_request(self):
        for value in (True, '600', 1.5, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.client.request_lease('gpu', wait_s=value)
        self.assertEqual(self.calls, [])

    def test_initial_queue_intent_remembers_exact_row_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as folder:
            client = SimpleNamespace(request_lease=Mock(return_value={
                'status_code':200,'lease_id':'exact','status':'queued'}),
                list_leases=Mock(return_value=[{'id':'exact','status':'queued',
                    'resource_id':'gpu','service_id':'service','agent_id':'owner',
                    'memory_mb':100,'exclusive':0,'expires_at':'2099-01-01T00:00:00Z'}]))
            store = InitialReservations(Path(folder)/'intents.sqlite', client,
                registry_owner='owner', service_id='service', models={'qwen':'gpu'}, wait_s=5400)
            args = dict(resource='gpu', service_id='service', memory_mb=100, ttl_seconds=9000)
            with self.assertRaises(CapacityDeferred):
                store.stage('qwen', **args)
            with self.assertRaises(CapacityDeferred):
                store.stage('qwen', **args)
            client.request_lease.assert_called_once_with(resource_id='gpu', service_id='service',
                memory_mb=100, ttl_seconds=9000, wait_s=5400)
            self.assertEqual(store.snapshot('qwen')['lease_id'], 'exact')

    def test_indefinite_wait_survives_reopen_without_deadline_or_new_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'initial.sqlite'
            self.client.list_leases = Mock(return_value=[{
                'id':'exact','status':'queued','resource_id':'gpu',
                'service_id':'service','agent_id':'owner','memory_mb':100,
                'exclusive':0,'expires_at':None,'wait_until':None}])
            args = dict(resource='gpu',service_id='service',memory_mb=100,ttl_seconds=3600)
            for clock in (1, 99999999999):
                store = InitialReservations(path,self.client,registry_owner='owner',
                    service_id='service',models={'qwen':'gpu'},wait=True,clock=lambda:clock)
                with self.assertRaises(CapacityDeferred):
                    store.stage('qwen',**args)
                self.assertEqual(store.snapshot('qwen')['lease_id'],'exact')
            self.assertEqual(self.calls,[{'ttl_seconds':3600,'service_id':'service',
                                         'memory_mb':100,'wait':True}])

    def test_indefinite_setting_round_trip_and_strict_boolean(self):
        from .inference_bootstrap import QueueSettings
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            path.write_text(json.dumps({'journal':str(Path(folder)/'queue.sqlite'),
                'models':{'qwen':'gpu'},'shared_queue_enabled':True}))
            settings=QueueSettings.from_file(path)
            self.assertTrue(settings.shared_queue)
            self.assertIsNone(settings.shared_queue_wait_s)
            for value in (1, 'true', None):
                with self.assertRaises(ValueError):
                    QueueSettings(Path(folder)/'q.sqlite',{'qwen':'gpu'},shared_queue_enabled=value)

    def test_renewal_racing_promotion_needs_a_fresh_grant_read(self):
        with tempfile.TemporaryDirectory() as folder:
            row={'id':'exact','status':'queued','agent_id':'owner','service_id':'service',
                 'resource_id':'gpu','memory_mb':100,'exclusive':0,
                 'expires_at':'1970-01-01T00:01:00Z'}
            client=SimpleNamespace(request_lease=Mock(return_value={
                'status_code':200,'lease_id':'exact','status':'queued'}),
                list_leases=Mock(side_effect=lambda **kwargs:[dict(row)]),
                renew_lease=Mock(return_value={'status':'active','reason':'promoted'}))
            store=InitialReservations(Path(folder)/'initial.sqlite',client,registry_owner='owner',
                service_id='service',models={'qwen':'gpu'},wait_s=60,clock=lambda:50)
            args=dict(resource='gpu',service_id='service',memory_mb=100,ttl_seconds=3600)
            with self.assertRaises(CapacityDeferred):store.stage('qwen',**args)
            with self.assertRaises(CapacityDeferred):store.stage('qwen',**args)
            client.renew_lease.assert_called_once_with('exact',60)
            self.assertEqual(store.snapshot('qwen')['state'],'queued')
            row.update(status='active',expires_at='2099-01-01T00:00:00Z')
            self.assertEqual(store.reserve('qwen',**args),'exact')
            client.request_lease.assert_called_once()


if __name__ == '__main__':
    unittest.main()
