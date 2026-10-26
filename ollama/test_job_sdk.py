"""Real OpenAI SDK/custom transport; synthetic authenticated queue responses."""
import asyncio
import unittest

import httpx
from openai import APIConnectionError

from . import test_inference_sdk as fixtures
from .inference_identity import CallKeys
from .inference_sdk import QueuedClient


JOBS = {'version': 1, 'enabled': True, 'same_service_children': True,
        'explicit_finish': True, 'explicit_yield': True}
REQUEST = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'synthetic'}]}


class JobQueue(fixtures.QueueFixture):
    def __init__(self, policy=JOBS):
        super().__init__()
        self.policy = policy
        self.acknowledge_job = True
        self.lose_ack_once = False

    def handle(self, request):
        response = super().handle(request)
        path = request.url.path
        body = response.json()
        if path.endswith('/capabilities'):
            body['stints'] = {}  # job protocol does not depend on legacy lifecycle yield
            if self.policy is not None:
                body['jobs'] = self.policy
        if path.endswith('/executions') and request.method == 'POST':
            if self.lose_ack_once:
                self.lose_ack_once = False
                raise httpx.ReadTimeout('synthetic lost intake response', request=request)
            if self.acknowledge_job:
                body['job_id'] = 'scheduling-job'
        if path.startswith('/v1/inference/executions/') and not path.endswith('/events'):
            body.update(id=path.rsplit('/', 1)[-1], grant_id='grant', job_id='scheduling-job')
        return httpx.Response(response.status_code, json=body)


class JobSDK(unittest.IsolatedAsyncioTestCase):
    def client(self, api, keys):
        return QueuedClient(api_key='synthetic', base_url='https://queue.invalid/v1',
            http_factory=api.http, poll_interval=.001, key_provider=lambda body: keys)

    async def test_job_linkage_is_sent_with_same_keys_on_sync_and_async_paths(self):
        for asynchronous in (False, True):
            api, receipts = JobQueue(), []
            keys = CallKeys('work', 'call', job_key='parent', work_session=True,
                            on_accepted=receipts.append)
            client = self.client(api, keys)
            try:
                if asynchronous:
                    answer = await client.create(**REQUEST)
                else:
                    answer = await asyncio.to_thread(lambda: client.create(**REQUEST))
                self.assertEqual(answer.choices[0].message.content, 'hello')
                writes = [body for method, path, body in api.requests
                          if method == 'POST' and path.endswith('/executions')]
                self.assertEqual(writes[0]['job_key'], 'parent')
                self.assertEqual(writes[0]['key'], 'call')
                self.assertNotIn('yield_grant_keys', writes[0])
                self.assertEqual(receipts[0].job_id, 'scheduling-job')
            finally:
                await client.aclose()
            self.assertFalse(any(path.endswith('/jobs/end') for _, path, _ in api.requests))

    async def test_unqualified_job_policy_refuses_before_any_admission_write(self):
        for policy in (None, {}, JOBS | {'version': True}, JOBS | {'version': 2},
                       JOBS | {'enabled': False}, JOBS | {'explicit_finish': False},
                       JOBS | {'same_service_children': False}):
            api = JobQueue(policy)
            client = self.client(api, CallKeys('work', 'call', job_key='parent'))
            try:
                with self.assertRaises(APIConnectionError):
                    await client.create(**REQUEST)
                self.assertFalse(any(method == 'POST' for method, _, _ in api.requests))
            finally:
                await client.aclose()

    async def test_explicit_header_cannot_override_frozen_host_job(self):
        api = JobQueue()
        client = self.client(api, CallKeys('work', 'call', job_key='parent'))
        try:
            with self.assertRaises(APIConnectionError):
                await client.create(**REQUEST, extra_headers={'X-Inference-Job-Key': 'another'})
            self.assertEqual(api.requests, [])
        finally:
            await client.aclose()

    async def test_accepted_receipt_reattaches_without_capability_or_new_submission(self):
        api = JobQueue(None)
        client = self.client(api, CallKeys('work', 'call', job_key='parent',
                                         resume_job=('grant', 'saved-execution')))
        try:
            response = await client.create(**REQUEST)
            self.assertEqual(response.choices[0].message.content, 'hello')
            self.assertTrue(all(method == 'GET' and '/executions/saved-execution' in path
                                for method, path, _ in api.requests))
        finally:
            await client.aclose()

    async def test_lost_intake_ack_retries_exact_job_linkage_not_a_new_child(self):
        api = JobQueue()
        api.lose_ack_once = True
        client = self.client(api, CallKeys('work', 'call', job_key='parent'))
        try:
            self.assertEqual((await client.create(**REQUEST)).choices[0].message.content, 'hello')
            writes = [body for method, path, body in api.requests
                      if method == 'POST' and path.endswith('/executions')]
            self.assertEqual(len(writes), 2)
            self.assertEqual(writes[0], writes[1])
            self.assertEqual(len(api.calls), 1)
        finally:
            await client.aclose()

    async def test_missing_job_ack_keeps_original_keys_for_later_recovery(self):
        api = JobQueue()
        api.acknowledge_job = False
        keys = CallKeys('work', 'call', job_key='parent')
        client = self.client(api, keys)
        try:
            with self.assertRaises(APIConnectionError):
                await client.create(**REQUEST)
            api.acknowledge_job = True
            self.assertEqual((await client.create(**REQUEST)).choices[0].message.content, 'hello')
            self.assertEqual(len(api.calls), 1)
        finally:
            await client.aclose()


if __name__ == '__main__':
    unittest.main()
