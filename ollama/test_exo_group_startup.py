"""Private instance POST ownership through the existing journal, no native work."""

import asyncio
import copy
import json
import unittest
from uuid import UUID

import httpx

from . import test_exo_group_install as controls
from .exo_groups.private import PrivateEXOUnavailable
from .exo_groups.startup import InstanceStarts, InstanceStartup
from .inference_arbiter import Conflict


class StartupControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.control = controls.InstalledControls()
        await self.control.asyncSetUp()
        c = self.control
        self.parts, self.actor = c.parts, c.actor
        self.parts.group['phase'] = 'admitted'
        factory = self.actor.authority_factory
        def authority(*args):
            reader = factory(*args)
            reader.may_serve_requests = lambda: self.parts.group['phase'] == 'serving'
            return reader
        self.actor.authority_factory = authority
        self.call = await c.enqueue()
        self.actor.events_reconciled()
        await c.drain()
        self.assertEqual(self.actor.report.phase, 'admitted')
        self.parts.bindings.bind_instance(self.parts.plan.owner_id,
            self.parts.plan.request_key, self.parts.contract)
        self.starts = InstanceStarts(self.parts.bindings)
        self.startup = InstanceStartup(self.actor, self.parts.private, self.starts,
            self.parts.contract, self.parts.launches, self.parts.verify_runtime)
        self.posts = 0
        self.mode = 'success'
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.initial = copy.deepcopy(self.parts.state)
        for name in ('instances', 'runners', 'tasks', 'downloads',
                     'prefillServerPorts', 'instanceLinks'):
            self.initial[name] = {}
        self.initial['lastEventAppliedIdx'] = 12
        self.initial['topology']['nodes'] = list(
            self.parts.contract._instance['shardAssignments']['nodeToRunner'])
        # Replace HTTP effects only; PrivateEXO commits/validates real records.
        await self.parts.private._client.aclose()
        self.parts.private._client = httpx.AsyncClient(base_url='http://127.0.0.1:52515',
            timeout=None, follow_redirects=False, trust_env=False,
            transport=httpx.MockTransport(self.transport))

    async def asyncTearDown(self):
        self.release.set()
        await self.control.asyncTearDown()

    async def transport(self, request):
        if request.method == 'GET':
            self.assertEqual(request.url.path, '/state')
            if self.mode == 'cursor-loss':
                self.actor.events_lost()
            return httpx.Response(200, json=copy.deepcopy(self.initial))
        self.assertEqual((request.method, request.url.path), ('POST', '/instance'))
        self.posts += 1
        saved = self.starts.snapshot(self.parts.plan.owner_id, self.parts.plan.request_key)
        self.assertEqual(saved['state_event_idx'], 12)
        self.assertEqual(saved['contract_sha256'], self.parts.contract.sha256)
        self.assertIsNone(saved['command_id'])  # Commit precedes delivery.
        self.assertEqual(json.loads(request.content), self.parts.contract.create_request())
        self.assertTrue(all(value is None for value in request.extensions['timeout'].values()))
        self.entered.set()
        await self.release.wait()
        if self.mode == 'reset':
            raise httpx.ReadError('private diagnostic')
        card = next(iter(self.parts.contract._instance['shardAssignments']['runnerToShard'].values()))[
            'PipelineShardMetadata']['modelCard']
        return httpx.Response(200, json={'command_id': str(UUID(int=1, version=4)),
            'model_card': card if self.mode != 'bad-card' else {}, 'message': 'Command received.'})

    async def test_admitted_group_submits_once_but_acceptance_is_not_serving(self):
        result = await self.startup.submit()
        self.assertEqual(result['command_id'], str(UUID(int=1, version=4)))
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.control.state(self.call), 'queued')
        self.assertFalse(self.actor.may_serve_requests())
        again = InstanceStarts(self.parts.bindings)  # New store object, same journal.
        self.startup.starts = again
        with self.assertRaises(Conflict):
            await self.startup.submit()
        self.assertEqual(self.posts, 1)

    async def test_lost_create_reply_and_zero_output_never_replay(self):
        self.mode = 'reset'
        with self.assertRaises(PrivateEXOUnavailable):
            await self.startup.submit()
        self.assertIsNone(self.starts.snapshot(self.parts.plan.owner_id,
            self.parts.plan.request_key)['command_id'])
        self.mode = 'success'
        with self.assertRaises(Conflict):
            await self.startup.submit()
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.control.journal.events(self.call, 'caller')['events'], [])

    async def test_reply_with_different_card_keeps_submission_unknown(self):
        self.mode = 'bad-card'
        with self.assertRaises(PrivateEXOUnavailable):
            await self.startup.submit()
        self.assertIsNone(self.starts.snapshot(self.parts.plan.owner_id,
            self.parts.plan.request_key)['command_id'])
        self.assertEqual(self.posts, 1)

    async def test_cursor_loss_during_private_read_does_not_submit(self):
        self.mode = 'cursor-loss'
        with self.assertRaises(ValueError):
            await self.startup.submit()
        self.assertEqual(self.posts, 0)

    async def test_foreign_namespace_missing_or_duplicate_node_refuses_before_commit(self):
        for field, value in [('instances', self.parts.state['instances']),
                             ('topology', {'nodes': ['foreign']}),
                             ('topology', {'nodes': list(self.initial['topology']['nodes']) * 2}),
                             ('lastEventAppliedIdx', True)]:
            original = copy.deepcopy(self.initial)
            self.initial[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                await self.startup.submit()
            self.assertEqual(self.posts, 0)
            self.initial = original

    async def test_observer_cancel_does_not_silently_repost_create(self):
        self.release.clear()
        # The service owns this task; only a shielded HTTP observer is cancelled.
        owned = self.control.queue.supervisor.prepare(self.startup.submit())
        await asyncio.wait_for(self.entered.wait(), 2)
        async def observe():
            return await asyncio.shield(owned)
        observer = asyncio.create_task(observe())
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(owned.done())
        self.release.set()
        await owned
        self.assertEqual(self.posts, 1)

    async def test_second_start_owner_cannot_post_while_first_response_is_pending(self):
        self.release.clear()
        owned = self.control.queue.supervisor.prepare(self.startup.submit())
        await asyncio.wait_for(self.entered.wait(), 2)
        self.startup.starts = InstanceStarts(self.parts.bindings)
        with self.assertRaises(Conflict):
            await self.startup.submit()
        self.assertEqual(self.posts, 1)
        self.release.set()
        await owned

    async def test_runtime_drift_refuses_before_submission(self):
        async def changed():
            return 'f' * 64
        self.startup.verify_runtime = changed
        with self.assertRaises(ValueError):
            await self.startup.submit()
        self.assertEqual(self.posts, 0)


if __name__ == '__main__':
    unittest.main()
