"""Real native API/SQLite ownership, with synthetic admission and raw receipts.

The oracle metadata is retargeted to the sealed tiny staging fixture only for
this wiring test. Actual numeric-reference controls retain the original oracle.
No real lease, rank, native API or numerical qualification happens here.
"""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from . import test_exo_group_deployment as deployed
from . import test_exo_group_factory as factory_fixture
from .test_exo_group_tokens import enable_trace, receipt, report
from .exo_groups.deployment import HostDeployment
from .exo_groups.host_control import run_identity
from .exo_groups.qualification import ComparisonUnavailable, TokenComparison
from .inference_arbiter import NotFound
from .exo_groups.prepare import canonical, digest


class ComparisonOwnerControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        original = deployed.DeploymentControls.asyncSetUp

        async def traced_setup(context):
            await original(context)
            enable_trace(context)

        def traced_layout(plan, files, **options):
            host = HostDeployment(plan, files, **options)
            return enable_trace(SimpleNamespace(config=host, plan=plan, options=options))

        self.context = c = factory_fixture.FixedFactoryControls()
        c.addAsyncCleanup = self.addAsyncCleanup
        with patch.object(deployed.DeploymentControls, 'asyncSetUp', traced_setup), \
                patch.object(factory_fixture, 'HostDeployment', traced_layout):
            await c.asyncSetUp()
        c.installation.demand.paused = True
        self.caller = {'valid': True, 'username': 'codex', 'owner_id': 'user:58'}
        self.assertEqual(c.plan.owner_id, self.caller['owner_id'])

        @c.runtime.queue.service.app.middleware('http')
        async def verified_identity(request, call_next):
            request.state.caller = self.caller
            return await call_next(request)

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=c.runtime.queue.service.app),
                                       base_url='https://slice.test')
        self.addAsyncCleanup(self.client.aclose)
        value = json.loads((Path(__file__).parent.parent/
            'qualification/fixtures/exo-chat-reference-v1.json').read_bytes())
        value.update(model_id=c.plan.model_id, manifest_sha256=c.plan.rank_manifest_sha256)
        for case in value['cases']:
            case['body']['model'] = c.plan.model_id
            case['body_sha256'] = digest(case['body'])
            case['task']['model'] = c.plan.model_id
        self.reference = canonical(value)
        self.case = value['cases'][0]
        self.comparison = TokenComparison(c.provider, c.fleet, self.reference,
                                         hashlib.sha256(self.reference).hexdigest())
        self.token_reads = []

    async def submitted(self, key='one'):
        c = self.context
        grant = await self.client.post('/v1/inference/grants', json={
            'key': key, 'model': c.plan.model_id, 'lifetime': 'execution'})
        self.assertEqual(grant.status_code, 202, grant.text)
        self.assertEqual(grant.json()['owner'], 'codex')
        call = await self.client.post('/v1/inference/grants/'+grant.json()['id']+'/executions',
            json={'key': key, 'request': self.case['body']})
        self.assertEqual(call.status_code, 202, call.text)
        return call.json()['id']

    async def bound(self, *, end=True, outcome='complete'):
        c = self.context
        execution_id = await self.submitted()
        c.actor.note_demand(execution_id)
        c.actor.events_reconciled()
        self.assertEqual((await c.actor.wait()).phase, 'admitted')
        c.provider.assembly = await c.factory(c.provider)
        backend = c.provider.backend
        journal = c.runtime.queue.supervisor.journal
        journal.dispatch_next(c.layouts[0].files.resource_id)
        backend.bindings.begin(execution_id, backend.contract, self.case['body'], state_event_idx=10)
        command, task = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'test-owned-task'
        backend.bindings.observe_command(execution_id, command)
        proof = {'command_id': command, 'task_id': task, 'group_id': backend.contract.group_id,
            'generation': backend.contract.generation, 'instance_id': backend.contract.instance_id,
            'rank_manifest_sha256': c.plan.rank_manifest_sha256, 'end_event_idx': 20,
            'runner_ids': sorted(backend.contract._instance['shardAssignments']['runnerToShard'])}
        if end:
            backend.bindings.record_end(execution_id, proof)
        if outcome == 'uncertain':
            journal.uncertain(execution_id)
        elif outcome in ('complete', 'failed'):
            journal.finish(execution_id, {'kind': 'json', 'body': {}}, failed=outcome=='failed')
        for endpoint in c.fleet.endpoints:
            host_client = c.fleet.clients[endpoint.rank]
            await host_client._client.aclose()

            async def tokens(request, *, route=endpoint):
                self.token_reads.append(route.rank)
                self.assertEqual(request.url.path, '/v1/exo/host/tokens')
                self.assertEqual(request.headers['authorization'], 'Bearer synthetic-coordinator')
                body = json.loads(request.content)
                binding = next(item.binding for item in c.provider.assembly.commands
                               if item.binding.expected.binding.rank==route.rank)
                self.assertEqual(body['identity'], run_identity(binding))
                self.assertEqual((body['command_id'], body['task_id']), (command, task))
                launch = binding.expected.binding
                identity = {'group_id': binding.group_id, 'generation': binding.generation,
                    'rank': launch.rank, 'device_rank': launch.pipeline_rank,
                    'rank_manifest_sha256': binding.rank_manifest_sha256, 'command_id': command,
                    'task_id': task, 'model': c.plan.model_id}
                return httpx.Response(200, headers={'cache-control': 'no-store'}, json={
                    'request_id': body['request_id'], 'source_sha256': route.source_sha256,
                    'configuration_sha256': route.configuration_sha256,
                    'report': report(binding, receipt(self.case, identity))})
            host_client._client = httpx.AsyncClient(base_url=endpoint.url,
                transport=httpx.MockTransport(tokens))
        return execution_id

    async def test_display_name_queue_owner_compares_under_its_canonical_group_owner(self):
        execution_id = await self.bound()
        journal = self.context.runtime.queue.supervisor.journal
        with self.assertRaises(NotFound):
            journal.execution(execution_id, 'user:58')
        value = await self.comparison.observe(execution_id)
        self.assertTrue(value['matched'])
        self.assertEqual(self.token_reads, ['slice', 'wafer', 'obelisk'])
        self.assertEqual(journal.execution(execution_id, 'codex')['state'], 'complete')
        self.caller = {'valid': True, 'username': 'another', 'owner_id': 'user:999'}
        self.assertEqual((await self.client.get('/v1/inference/executions/'+execution_id)).status_code, 404)

    async def test_unlinked_execution_and_changed_contract_are_not_observed(self):
        execution_id = await self.bound()
        foreign = await self.submitted('unlinked')
        with self.assertRaises(ComparisonUnavailable):
            await self.comparison.observe(foreign)
        backend = self.context.provider.backend
        with patch.object(backend, 'contract', replace(backend.contract, runtime_sha256='0'*64)):
            with self.assertRaises(ComparisonUnavailable):
                await self.comparison.observe(execution_id)
        self.assertFalse(self.token_reads)

    async def test_completed_call_without_all_rank_end_is_not_observed(self):
        execution_id = await self.bound(end=False)
        with self.assertRaises(ComparisonUnavailable):
            await self.comparison.observe(execution_id)
        self.assertFalse(self.token_reads)

    async def test_uncertain_call_is_not_observed_or_finished(self):
        execution_id = await self.bound(outcome='uncertain')
        with self.assertRaises(ComparisonUnavailable):
            await self.comparison.observe(execution_id)
        self.assertEqual(self.context.runtime.queue.supervisor.journal.execution(execution_id, 'codex')['state'], 'uncertain')
        self.assertFalse(self.token_reads)

    async def test_failed_call_is_not_observed(self):
        execution_id = await self.bound(outcome='failed')
        with self.assertRaises(ComparisonUnavailable):
            await self.comparison.observe(execution_id)
        self.assertFalse(self.token_reads)

    async def test_changed_saved_request_cannot_reuse_the_binding(self):
        execution_id = await self.bound()
        journal = self.context.runtime.queue.supervisor.journal
        body = copy.deepcopy(self.case['body'])
        body['messages'][0]['content'] = 'another request'
        with journal.transaction():
            journal.db.execute('UPDATE executions SET request=? WHERE id=?',
                (canonical({'operation': 'chat', 'body': body}).decode(), execution_id))
        with self.assertRaises(ComparisonUnavailable):
            await self.comparison.observe(execution_id)
        self.assertFalse(self.token_reads)

    async def test_wrong_group_scope_or_removed_demand_refuses_before_host_reads(self):
        execution_id = await self.bound()
        backend = self.context.provider.backend
        for owner, key in (('codex', self.context.plan.request_key),
                           ('user:58', 'another-intent')):
            with self.assertRaises(NotFound):
                backend.bindings.completed_snapshot(execution_id, group_owner=owner,
                    request_key=key, contract=backend.contract)
        journal = self.context.runtime.queue.supervisor.journal
        with journal.transaction():
            journal.db.execute('DELETE FROM exo_group_demands WHERE execution_id=?', (execution_id,))
        with self.assertRaises(ComparisonUnavailable):
            await self.comparison.observe(execution_id)
        self.assertFalse(self.token_reads)
