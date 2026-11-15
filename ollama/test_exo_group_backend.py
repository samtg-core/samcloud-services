"""Real supervisor/journal/HTTP adapter; runtime and host effects are injected."""

import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import UUID

import httpx

from . import test_exo_group_plan as fixtures
from .exo_groups.backend import EXOBackend
from .exo_groups.calls import ExecutionBindings
from .exo_groups.instance import InstanceContract
from .exo_groups.intents import GroupIntents
from .exo_groups.plan import GroupPlan, HOST_RESOURCES
from .exo_groups.prepare import canonical, digest
from .exo_groups.private import PrivateEXO
from .inference_supervisor import Supervisor


class Parts(httpx.AsyncByteStream):
    def __init__(self, parts, *, broken=False):
        self.parts, self.broken = parts, broken

    async def __aiter__(self):
        for part in self.parts:
            yield part
            await asyncio.sleep(0)
        if self.broken:
            raise httpx.ReadError('private backend detail')


class PrivateBackendControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.supervisor = Supervisor(Path(self.temp.name) / 'queue.sqlite',
                                     self.execute, lambda _resource: True)
        self.journal = self.supervisor.journal
        self.intents = GroupIntents(self.journal)
        self.bindings = ExecutionBindings(self.intents)
        f = fixtures.GroupBindingControls()
        f.setUp()
        wire = json.loads((Path(__file__).resolve().parents[1] /
                           'qualification/fixtures/exo-instance-v1.json').read_text())
        self.state = wire['state']
        manifest = json.loads(f.manifest)
        manifest['model_id'] = wire['plan']['model_id']
        manifest['revision'] = wire['plan']['revision']
        self.manifest = canonical(manifest)
        raw = f.raw | {'model_id': manifest['model_id'], 'revision': manifest['revision'],
                       'rank_manifest_sha256': hashlib.sha256(self.manifest).hexdigest()}
        self.plan = GroupPlan.parse(raw)
        self.group = f.group | {'id': 'group_' + 'c' * 24}
        self.launches = tuple(replace(b, group_id=self.group['id'],
            rank_manifest_sha256=self.plan.rank_manifest_sha256) for b in f.bind())
        self.runtime = wire['runtime_sha256']
        self.contract = InstanceContract(digest(asdict(self.plan)), self.group['id'], 1,
            self.runtime, canonical(wire['request']).decode(),
            canonical(next(iter(self.state['instances'].values()))).decode())
        self.enabled = True
        self.actor = SimpleNamespace(plan=self.plan, intents=self.intents,
            wake=lambda: None, wait=self.metadata_done, may_serve_requests=lambda: self.enabled)
        self.end_entered, self.end_release = asyncio.Event(), asyncio.Event()
        self.end_release.set()
        self.posts = 0
        self.requests = []
        self.mode, self.bad_proof = 'success', None
        self.key = Path(self.temp.name) / 'exo-key'
        self.key.write_text('a' * 64)
        self.key.chmod(0o600)
        self.private = PrivateEXO(52515, self.key, transport=httpx.MockTransport(self.transport))
        self.backend = EXOBackend(self.actor, self.private, self.bindings, self.contract,
                                  self.launches, self.verify_runtime)

    async def asyncTearDown(self):
        self.end_release.set()
        await asyncio.gather(*self.supervisor.workers.values(), return_exceptions=True)
        await self.private.close()
        self.supervisor.close()
        self.temp.cleanup()

    async def metadata_done(self):
        return None

    async def verify_runtime(self):
        # CPU injection, not an attestation of running hosts.
        return self.runtime

    async def execute(self, grant, call):
        return await self.backend.execute(grant, call)

    def command(self, post=None):
        return str(UUID(int=post or self.posts, version=4))

    def frame(self, command, *, finish=None, content='hello', model=None):
        return ('data: ' + json.dumps({'id': command, 'model': model or self.plan.model_id,
            'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': finish}]}) + '\n\n').encode()

    async def transport(self, request):
        self.requests.append((request.method, request.url.path))
        self.assertEqual(request.headers['authorization'], 'Bearer ' + 'a' * 64)
        if request.url.path == '/state':
            state = copy.deepcopy(self.state)
            state['lastEventAppliedIdx'] = 100 + 10 * self.posts
            return httpx.Response(200, json=state)
        if request.url.path.startswith('/leased/commands/'):
            self.end_entered.set()
            await self.end_release.wait()
            proof = {'command_id': request.url.path.split('/')[-2],
                'instance_id': self.contract.instance_id, 'task_id': 'task',
                'runner_ids': sorted(self.contract._instance['shardAssignments']['runnerToShard']),
                'end_event_idx': 99 + 10 * self.posts, 'group_id': self.contract.group_id,
                'generation': 1, 'rank_manifest_sha256': self.plan.rank_manifest_sha256}
            if self.bad_proof:
                proof.update(self.bad_proof)
            return httpx.Response(200, json=proof)
        self.assertEqual((request.method, request.url.path), ('POST', '/v1/chat/completions'))
        self.posts += 1
        body = json.loads(request.content)
        self.assertEqual(body['model'], self.plan.model_id)
        command = self.command()
        headers = {'X-EXO-Command-ID': command}
        if not body.get('stream'):
            return httpx.Response(200, headers=headers, json={'id': command,
                'model': self.plan.model_id, 'choices': [{'index': 0,
                'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}]})
        headers['content-type'] = 'text/event-stream'
        mode = self.mode
        if mode == 'error':
            self.mode = 'success'
            parts = [b'data: {"error":{"message":"private diagnostic"}}\n\n', b'data: [DONE]\n\n']
        elif mode == 'empty':
            parts = []
        elif mode == 'done_only':
            parts = [b'data: [DONE]\n\n']
        elif mode == 'wrong_model':
            parts = [self.frame(command, finish='stop', model='foreign'), b'data: [DONE]\n\n']
        elif mode == 'wrong_command':
            parts = [self.frame(self.command(99), finish='stop'), b'data: [DONE]\n\n']
        elif mode == 'no_done':
            parts = [self.frame(command, finish='stop')]
        else:
            data = self.frame(command) + self.frame(command, finish='stop', content='') + b'data: [DONE]\n\n'
            parts = [data[:13], data[13:77], data[77:]]
        return httpx.Response(200, headers=headers, stream=Parts(parts, broken=mode == 'reset'))

    def enqueue(self, key='call', *, stream=True):
        grant = self.journal.submit('caller', key, HOST_RESOURCES['slice'], self.plan.model_id,
                                    lifetime='execution')
        body = {'model': self.plan.model_id, 'messages': [{'role': 'user', 'content': 'hello'}],
                'stream': stream}
        call = self.journal.enqueue(grant['id'], 'caller', key, {'operation': 'chat', 'body': body})
        self.intents.prepare(call['id'], self.plan, self.manifest)
        self.intents.record_group(self.plan.owner_id, self.plan.request_key, self.group)
        self.intents.bind_launches(self.plan.owner_id, self.plan.request_key, self.launches)
        self.bindings.bind_instance(self.plan.owner_id, self.plan.request_key, self.contract)
        return call

    async def run_call(self, call):
        self.assertEqual(self.supervisor.pump(HOST_RESOURCES['slice']), call['id'])
        await self.supervisor.workers[call['id']]
        return self.journal.execution(call['id'], 'caller')

    async def test_terminal_output_waits_for_all_rank_evidence_then_completes(self):
        self.end_release.clear()
        call = self.enqueue()
        self.supervisor.pump(HOST_RESOURCES['slice'])
        await self.end_entered.wait()
        self.assertEqual(self.journal.execution(call['id'], 'caller')['state'], 'running')
        page = self.journal.events(call['id'], 'caller')
        self.assertEqual(len(page['events']), 1)
        self.assertIsNone(self.bindings.snapshot(call['id'])['end'])
        self.end_release.set()
        await self.supervisor.workers[call['id']]
        self.assertEqual(self.journal.execution(call['id'], 'caller')['state'], 'complete')
        self.assertIsNotNone(self.bindings.snapshot(call['id'])['end'])
        self.assertEqual(len(self.journal.events(call['id'], 'caller')['events']), 3)
        self.assertEqual(self.posts, 1)

    async def test_nonstream_result_preserves_saved_mode_and_returns_json(self):
        call = self.enqueue(stream=False)
        row = await self.run_call(call)
        self.assertEqual(row['state'], 'complete')
        self.assertEqual(json.loads(row['result'])['body']['choices'][0]['message']['content'], 'hello')
        self.assertEqual(self.posts, 1)

    async def test_confirmed_error_frees_next_queued_call_without_replaying_first(self):
        self.mode = 'error'
        first = self.enqueue('first')
        second = self.enqueue('second')
        await self.run_call(first)
        while self.supervisor.workers:
            await asyncio.gather(*self.supervisor.workers.values())
            await asyncio.sleep(0)
        self.assertEqual(self.journal.execution(first['id'], 'caller')['state'], 'failed')
        self.assertEqual(self.journal.execution(second['id'], 'caller')['state'], 'complete')
        self.assertEqual(self.posts, 2)
        self.assertNotIn('private diagnostic', json.dumps(self.journal.events(first['id'], 'caller')))

    async def test_ambiguous_streams_keep_claim_and_never_repost(self):
        for mode in ('empty', 'reset', 'done_only', 'no_done', 'wrong_model', 'wrong_command'):
            with self.subTest(mode=mode):
                self.mode = mode
                call = self.enqueue(mode)
                row = await self.run_call(call)
                self.assertEqual(row['state'], 'uncertain')
                posts = self.posts
                self.assertIsNone(self.supervisor.pump(HOST_RESOURCES['slice']))
                self.assertEqual(self.posts, posts)
                # Test isolation only: prove no live backend exists in this CPU
                # injection before retiring the uncertain fixture for next case.
                self.journal.finish(call['id'], {'kind': 'error', 'status': 502}, failed=True)

    async def test_bad_or_stale_end_receipt_keeps_terminal_frames_hidden(self):
        for bad in ({'generation': True}, {'generation': 2}, {'runner_ids': []},
                    {'instance_id': 'foreign'}, {'end_event_idx': 0}):
            with self.subTest(bad=bad):
                self.bad_proof = bad
                call = self.enqueue(str(bad))
                row = await self.run_call(call)
                self.assertEqual(row['state'], 'uncertain')
                self.assertEqual(len(self.journal.events(call['id'], 'caller')['events']), 1)
                self.assertIsNone(self.bindings.snapshot(call['id'])['end'])
                self.journal.finish(call['id'], {'kind': 'error', 'status': 502}, failed=True)

    async def test_event_cursor_loss_before_post_defers_without_a_new_submission(self):
        self.enabled = False
        call = self.enqueue()
        row = await self.run_call(call)
        self.assertEqual(row['state'], 'queued')
        self.assertEqual(self.posts, 0)


if __name__ == '__main__':
    unittest.main()
