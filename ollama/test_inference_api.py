"""ASGI contract using real gateway auth middleware with synthetic cached tokens.

No live token/registry/provider, and no server lifespan or model load. These tests
exercise owner isolation and HTTP-independent execution, NOT fleet integration.
"""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import httpx

from . import server
from .inference_api import build_router
from .inference_supervisor import Supervisor


class DurableAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gate = asyncio.Event()
        self.entered = []
        async def execute(grant, call):
            self.entered.append(grant['owner'])
            await self.gate.wait()
            return {'answer': call['messages'][-1]['content']}
        self.s = Supervisor(Path(self.temp.name) / 'queue.sqlite', execute, lambda _: True)

        def resolve(model):
            if model != 'qwen':
                raise ValueError('unknown model')
            return 'slice-gpu-0', model

        def validate(model, body):
            # The actual gateway request model, not an untyped echo transport.
            parsed = server.ChatRequest(**body)
            if parsed.model != model:
                raise ValueError('model must match grant')
            return parsed.model_dump()

        self.app = FastAPI()
        self.app.include_router(build_router(self.s, resolve, validate))
        self.app.add_middleware(server.SamcloudAuthMiddleware,
                                verify_url='https://not-called.invalid', required_scope='services')
        self.auth = patch.object(server, 'AUTH_ENABLED', True)
        self.auth.start()
        identities = {'Bearer creative-test': {'username': 'creative'},
                      'Bearer assistant-test': {'username': 'assistant'}}
        self.cache = patch.object(server.SamcloudAuthMiddleware, '_get_cached',
                                  lambda middleware, token: identities[token])
        self.cache.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                       base_url='http://test', headers={'Authorization': 'Bearer creative-test'})

    async def asyncTearDown(self):
        self.gate.set()
        await asyncio.gather(*list(self.s.workers.values()))
        await self.client.aclose()
        self.cache.stop()
        self.auth.stop()
        self.s.close()
        self.temp.cleanup()

    async def grant(self, key='turn', owner='creative'):
        response = await self.client.post('/v1/inference/grants', json={'key': key, 'model': 'qwen'},
                                          headers={'Authorization': f'Bearer {owner}-test'})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    async def call(self, grant, owner='creative'):
        response = await self.client.post(f'/v1/inference/grants/{grant}/executions', json={
            'key': 'call', 'request': {'model': 'qwen', 'messages': [{'role': 'user', 'content': owner}]}
        }, headers={'Authorization': f'Bearer {owner}-test'})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    async def test_capability_reports_actual_policy_and_requires_verified_owner(self):
        from .inference_arbiter import StintPolicy
        before = await self.client.get('/v1/inference/capabilities')
        self.assertEqual(before.json()['stints'], {'version': 1, 'enabled': False, 'dependency_yield': True,
            'lifecycle_yield': True,
            'idle_grace_seconds': None, 'fairness_seconds': None})
        self.s.journal.stint_policy = StintPolicy(300, 900)
        after = await self.client.get('/v1/inference/capabilities')
        self.assertEqual(after.json()['stints'], {'version': 1, 'enabled': True, 'dependency_yield': True,
            'lifecycle_yield': True,
            'idle_grace_seconds': 300, 'fairness_seconds': 900})
        denied = await self.client.get('/v1/inference/capabilities', headers={'Authorization': ''})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(self.entered, [])

    async def test_http_submit_poll_reconnect_and_exclusive_handoff(self):
        first = await self.grant()
        a = await self.call(first)
        second = await self.grant(owner='assistant')
        b = await self.call(second, owner='assistant')
        self.assertEqual(a['state'], 'running')
        self.assertEqual(b['state'], 'queued')
        await asyncio.sleep(0)
        self.assertEqual(self.entered, ['creative'])
        await self.client.aclose()  # the submitter is gone, not the worker
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                       base_url='http://test', headers={'Authorization': 'Bearer creative-test'})
        self.assertEqual((await self.call(first))['id'], a['id'])
        read = await self.client.get('/v1/inference/executions/' + a['id'])
        self.assertEqual(read.json()['state'], 'running')
        self.assertNotIn('request', read.json())
        await self.client.post(f'/v1/inference/grants/{first}/release')
        self.assertEqual(self.entered, ['creative'])
        self.gate.set()
        async with asyncio.timeout(2):
            while len(self.entered) < 2:
                await asyncio.sleep(.001)
        final = await self.s.observe(a['id'], 'creative')
        self.assertEqual(json.loads(final['result']), {'answer': 'creative'})
        self.assertEqual(self.entered, ['creative', 'assistant'])

    async def test_identity_is_verified_not_submitted_and_other_owner_gets_404(self):
        grant = await self.grant()
        call = await self.call(grant)
        other = {'Authorization': 'Bearer assistant-test'}
        for path in (f'/v1/inference/grants/{grant}', '/v1/inference/executions/' + call['id']):
            response = await self.client.get(path, headers=other)
            self.assertEqual(response.status_code, 404)
        self.assertEqual((await self.client.post(f'/v1/inference/grants/{grant}/release', headers=other)).status_code, 404)
        response = await self.client.post('/v1/inference/grants', json={
            'key': 'spoof', 'model': 'qwen', 'owner': 'assistant', 'resource': 'wafer'})
        self.assertEqual(response.status_code, 422)
        self.client.headers.pop('Authorization')
        self.assertEqual((await self.client.get(f'/v1/inference/grants/{grant}')).status_code, 401)
        with patch.object(server, 'AUTH_ENABLED', False):
            self.assertEqual((await self.client.get(f'/v1/inference/grants/{grant}')).status_code, 401)

    async def test_invalid_model_or_call_never_reaches_backend(self):
        response = await self.client.post('/v1/inference/grants', json={'key': 'x', 'model': 'unknown'})
        self.assertEqual(response.status_code, 422)
        grant = await self.grant()
        response = await self.client.post(f'/v1/inference/grants/{grant}/executions', json={
            'key': 'x', 'request': {'model': 'other', 'messages': []}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.entered, [])

    async def test_yield_is_owner_scoped_and_drains_without_canceling_work(self):
        first = await self.grant()
        call = await self.call(first)
        second = await self.grant(owner='assistant')
        next_call = await self.call(second, owner='assistant')
        path = f'/v1/inference/grants/{first}/yield'
        refused = await self.client.post(path, headers={'Authorization': 'Bearer assistant-test'})
        self.assertEqual(refused.status_code, 404)
        accepted = await self.client.post(path)
        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(accepted.json()['yield_requested'])
        self.assertEqual(self.s.journal.execution(call['id'], 'creative')['state'], 'running')
        self.gate.set()
        await asyncio.wait_for(self.s.observe(next_call['id'], 'assistant'), 2)
        self.assertEqual(self.entered, ['creative', 'assistant'])
        self.assertEqual(self.s.journal.get(first, 'creative')['state'], 'queued')
        self.assertEqual((await self.client.post(path)).status_code, 200)

    async def test_execution_grant_cannot_be_changed_to_a_stint_by_yielding(self):
        response = await self.client.post('/v1/inference/grants', json={
            'key': 'execution', 'model': 'qwen', 'lifetime': 'execution'})
        self.assertEqual(response.status_code, 202)
        response = await self.client.post(f"/v1/inference/grants/{response.json()['id']}/yield")
        self.assertEqual(response.status_code, 409)

    async def test_events_are_owner_scoped_paged_and_resume_from_saved_cursor(self):
        grant = await self.grant()
        call = await self.call(grant)
        path = '/v1/inference/executions/' + call['id'] + '/events'
        for text in ('first', 'second'):
            self.s.journal.append_event(call['id'], text)
        page = (await self.client.get(path, params={'limit': 1})).json()
        self.assertEqual(page['state'], 'running')
        self.assertTrue(page['has_more'])
        self.assertEqual(page['events'][0]['data'], 'first')
        page = (await self.client.get(path, params={'after': page['next_cursor']})).json()
        self.assertFalse(page['has_more'])
        self.assertEqual(page['events'][0]['data'], 'second')
        self.assertEqual((await self.client.get(path, headers={
            'Authorization': 'Bearer assistant-test'})).status_code, 404)
        for params in ({'after': -1}, {'limit': 1001}, {'limit': 0}):
            self.assertEqual((await self.client.get(path, params=params)).status_code, 422)
        # Reading output never invokes the provider a second time.
        await asyncio.sleep(0)
        self.assertEqual(self.entered, ['creative'])


if __name__ == '__main__':
    unittest.main()
