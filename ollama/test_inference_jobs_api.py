"""Authenticated local ASGI with real journal/supervisor; no physical backend."""
import asyncio
import unittest

from . import test_inference_api as fixtures
from .inference_arbiter import StintPolicy


class JobAPI(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.DurableAPI.asyncTearDown
    grant = fixtures.DurableAPI.grant

    async def asyncSetUp(self):
        await fixtures.DurableAPI.asyncSetUp(self)
        self.s.journal.stint_policy = StintPolicy()

    async def submit(self, key, job=None, owner='creative'):
        grant = await self.grant(key=key, owner=owner)
        body = {'key': 'call', 'request': {'model': 'qwen',
                'messages': [{'role': 'user', 'content': owner}]}}
        if job is not None:
            body['job_key'] = job
        response = await self.client.post(f'/v1/inference/grants/{grant}/executions', json=body,
                                         headers={'Authorization': f'Bearer {owner}-test'})
        self.assertEqual(response.status_code, 202, response.text)
        return grant, response.json(), body

    async def settle(self):
        await asyncio.sleep(0)
        while pending := tuple(self.s.workers.values()):
            await asyncio.gather(*pending)
            await asyncio.sleep(0)

    async def test_job_children_reconnect_and_finish_through_authenticated_http(self):
        caps = (await self.client.get('/v1/inference/capabilities')).json()['jobs']
        self.assertTrue(caps['enabled'])
        self.assertFalse(caps['cross_service_delegation'])
        grant, first, body = await self.submit('main', 'turn')
        self.gate.set()
        await self.settle()
        _, waiting, _ = await self.submit('other', owner='assistant')
        _, child, _ = await self.submit('compaction', 'turn')
        await self.settle()
        self.assertEqual(child['job_id'], first['job_id'])
        self.assertEqual(self.entered, ['creative', 'creative'])
        ended = await self.client.post('/v1/inference/jobs/end', json={'key': 'turn'})
        self.assertEqual(ended.status_code, 202)
        self.assertEqual(ended.json()['jobs'][0]['state'], 'closed')
        await asyncio.gather(*tuple(self.s.preparations))
        await self.settle()
        self.assertEqual(self.entered, ['creative', 'creative', 'assistant'])
        replay = await self.client.post(f'/v1/inference/grants/{grant}/executions', json=body)
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()['id'], first['id'])
        self.assertEqual(replay.json()['state'], 'complete')
        self.assertEqual(self.s.journal.execution(waiting['id'], 'assistant')['state'], 'complete')

    async def test_job_end_and_status_are_owner_scoped_not_body_selected(self):
        await self.submit('main', 'private-turn')
        auth = {'Authorization': 'Bearer assistant-test'}
        for key in ('private-turn', 'nonexistent'):
            read = await self.client.get('/v1/inference/jobs', params={'key': key}, headers=auth)
            end = await self.client.post('/v1/inference/jobs/end', json={'key': key}, headers=auth)
            self.assertEqual(read.status_code, 404)
            self.assertEqual(end.status_code, 202)
            self.assertEqual(end.json()['jobs'], [])  # only seals this caller's key
        spoof = await self.client.post('/v1/inference/jobs/end',
            json={'key': 'private-turn', 'owner': 'creative'}, headers=auth)
        self.assertEqual(spoof.status_code, 422)
        self.assertTrue(self.s.journal.owns_job('slice-gpu-0'))

    async def test_finish_with_running_child_drains_then_reconciles_before_other_dispatch(self):
        _, first, _ = await self.submit('main', 'turn')
        _, other, _ = await self.submit('other', owner='assistant')
        async def boundary(resource):
            self.assertEqual(resource, 'slice-gpu-0')
            self.entered.append('boundary')
        self.s.on_job_boundary = boundary
        end = await self.client.post('/v1/inference/jobs/end', json={'key': 'turn'})
        self.assertEqual(end.json()['jobs'][0]['state'], 'owned')
        self.assertEqual(end.json()['jobs'][0]['end_requested'], 'finish')
        self.assertEqual(self.s.journal.execution(other['id'], 'assistant')['state'], 'queued')
        self.gate.set()
        await self.settle()
        self.assertEqual(self.entered, ['creative', 'boundary', 'assistant'])
        self.assertEqual(self.s.journal.execution(first['id'], 'creative')['state'], 'complete')

    async def test_boundary_reconciliation_outlives_disconnected_http_observer(self):
        await self.submit('main', 'turn')
        self.gate.set()
        await self.settle()
        await self.submit('other', owner='assistant')
        entered, release = asyncio.Event(), asyncio.Event()
        async def boundary(resource):
            entered.set()
            await release.wait()
        self.s.on_job_boundary = boundary
        end = await self.client.post('/v1/inference/jobs/end', json={'key': 'turn'})
        self.assertEqual(end.status_code, 202)
        await entered.wait()
        await self.client.aclose()
        self.assertEqual(self.entered, ['creative'])
        release.set()
        await asyncio.gather(*tuple(self.s.preparations))
        await self.settle()
        self.assertEqual(self.entered, ['creative', 'assistant'])

    async def test_policy_disabled_refuses_job_intake_without_leaking_binding(self):
        self.s.journal.stint_policy = None
        caps = (await self.client.get('/v1/inference/capabilities')).json()['jobs']
        self.assertFalse(caps['enabled'])
        grant = await self.grant()
        refused = await self.client.post(f'/v1/inference/grants/{grant}/executions', json={
            'key': 'call', 'job_key': 'turn', 'request': {'model': 'qwen', 'messages': []}})
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(self.s.journal.db.execute('SELECT count(*) FROM requester_jobs').fetchone()[0], 0)

    async def test_failed_boundary_observation_does_not_dispatch_on_old_capacity(self):
        await self.submit('main', 'turn')
        self.gate.set()
        await self.settle()
        _, waiting, _ = await self.submit('other', owner='assistant')
        permitted = [True]
        self.s.permitted = lambda resource: permitted[0]
        async def boundary(resource):
            permitted[0] = False  # real capacity source withdraws before observing
            raise OSError('synthetic unavailable observation')
        self.s.on_job_boundary = boundary
        end = await self.client.post('/v1/inference/jobs/end', json={'key': 'turn'})
        self.assertEqual(end.status_code, 202)
        await asyncio.gather(*tuple(self.s.preparations))
        self.assertEqual(self.s.journal.execution(waiting['id'], 'assistant')['state'], 'queued')
        permitted[0] = True  # trusted fresh capacity event, not a timer or retry
        self.s.capacity_available('slice-gpu-0')
        await self.settle()
        self.assertEqual(self.s.journal.execution(waiting['id'], 'assistant')['state'], 'complete')

    async def test_cancel_last_queued_child_also_wakes_physical_boundary_before_dispatch(self):
        await self.submit('main', 'turn')
        self.gate.set()
        await self.settle()
        # Withdraw dispatch to keep the child accepted but queued.
        self.s.permitted = lambda resource: False
        grant, child, _ = await self.submit('child', 'turn')
        _, other, _ = await self.submit('other', owner='assistant')
        await self.client.post('/v1/inference/jobs/end', json={'key': 'turn'})
        async def boundary(resource):
            self.entered.append('boundary')
        self.s.on_job_boundary = boundary
        self.s.permitted = lambda resource: True
        response = await self.client.post(f'/v1/inference/grants/{grant}/release')
        self.assertEqual(response.status_code, 200)
        await asyncio.gather(*tuple(self.s.preparations))
        await self.settle()
        self.assertEqual(self.s.journal.execution(child['id'], 'creative')['state'], 'cancelled')
        self.assertEqual(self.s.journal.execution(other['id'], 'assistant')['state'], 'complete')
        self.assertEqual(self.entered, ['creative', 'boundary', 'assistant'])

    async def test_real_sdk_and_identity_store_keep_parent_across_child_then_hand_back(self):
        from pathlib import Path
        import httpx
        from .inference_consumer import DurableConsumer
        from .inference_identity import IdentityStore
        from .inference_sdk import QueuedClient
        from .hermes_queue_identity import HostBindings
        from .inference_events import relay_once

        async def execute(grant, call):
            self.entered.append(grant['owner'])
            await self.gate.wait()
            return {'kind': 'json', 'body': {'id': grant['active_execution'],
                'object': 'chat.completion', 'created': 1, 'model': 'qwen',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': 'done'}}]}}
        self.s.execute = execute
        self.gate.set()
        identity_path = Path(self.temp.name) / 'client-identity.sqlite'
        store = IdentityStore(identity_path)
        store.begin_native_job('scope', 'native-turn')
        def http():
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                base_url='http://test', headers={'Authorization': 'Bearer creative-test'})
        main = QueuedClient(api_key='synthetic', base_url='http://test/v1', http_factory=http,
            poll_interval=.001, key_provider=lambda request:
                store.bind('scope', 'main', 'main', request, job_turn='native-turn'))
        child = QueuedClient(api_key='synthetic', base_url='http://test/v1', http_factory=http,
            poll_interval=.001, grant_lifetime='execution', key_provider=lambda request:
                store.bind_execution('scope', 'compression', 'summary', request, job_turn='native-turn'))
        try:
            request = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'fixture'}]}
            self.assertEqual((await main.create(**request)).choices[0].message.content, 'done')
            await self.submit('other', owner='assistant')
            self.assertEqual((await child.create(**request)).choices[0].message.content, 'done')
            self.assertEqual(self.entered, ['creative', 'creative'])
            key = store.db.execute('SELECT job_key FROM native_job_binding').fetchone()[0]
            store.observe_native_outcome('scope', 'native-turn', completed=True, failed=False, interrupted=False)
            source = HostBindings(identity_path, 'profile')
            self.assertEqual(await relay_once(DurableConsumer(self.client), source), 1)
            self.assertEqual(source.pending_events(), [])
            await asyncio.gather(*tuple(self.s.preparations))
            await self.settle()
            self.assertEqual(self.entered, ['creative', 'creative', 'assistant'])
        finally:
            await main.aclose()
            await child.aclose()
            store.close()


if __name__ == '__main__':
    unittest.main()
