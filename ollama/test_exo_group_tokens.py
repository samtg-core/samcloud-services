"""Real private files and ASGI, real M5 oracle; isolated run/end fixtures.

No native model, lease or distributed admission is performed by these tests.
"""

import base64
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase, TestCase

from . import test_exo_group_deployment as deployed
from . import test_exo_group_host_control as host_controls
from . import test_exo_group_instance as instance_controls
from .exo_groups.deployment import HostDeployment
from .exo_groups.host_control import run_identity
from .exo_groups.member import ExpectedMember
from .exo_groups.prepare import canonical
from .exo_groups.qualification import ChatReference, ComparisonUnavailable, compare_raw_tokens
from .exo_groups.rank_process import RunBinding
from .exo_groups.receipt_files import ReceiptUnavailable
from .exo_groups.staging import runtime_file_policy
from .exo_groups.token_receipts import collect_tokens, parse_tokens


def enable_trace(fixture):
    old = fixture.config
    directory = old.runtime_root
    directory.chmod(0o700)
    for path in directory.rglob('*'):
        if path.is_dir():
            path.chmod(0o700)
    pins = list(old.files.runtime['files'])
    for name in ('source/src/exo/utils/qualification_tokens.py',
                 'source/src/exo/worker/runner/llm_inference/batch_generator.py'):
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b'qualification capability fixture; native execution forbidden\n'
        path.write_bytes(raw)
        pins.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'link': None})
    for path in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    directory.chmod(0o500)
    files = replace(old.files, runtime=runtime_file_policy(str(directory), pins))
    fixture.config = HostDeployment(fixture.plan, files, **fixture.options, qualification_token_limit=16)
    return fixture.config


def oracle(plan):
    raw = (Path(__file__).parent.parent/'qualification/fixtures/exo-chat-reference-v1.json').read_bytes()
    return ChatReference(raw, hashlib.sha256(raw).hexdigest(), plan)


def expected(binding, command='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', task='actual-task'):
    member = binding.expected.binding
    return {'group_id': binding.group_id, 'generation': binding.generation,
        'rank': member.rank, 'device_rank': member.pipeline_rank,
        'rank_manifest_sha256': binding.rank_manifest_sha256, 'command_id': command,
        'task_id': task, 'model': 'mlx-community/Qwen3-0.6B-4bit'}


def receipt(case, identity):
    rows = [{'kind': 'exo_raw_tokens_v1', **identity,
        'task_params_json': canonical(case['task']).decode(), 'prompt': case['rendered_prompt'],
        'prompt_token_ids': case['encoded_prompt_ids'], 'token_limit': 16,
        'collection': 'before_output_parsers'}]
    rows.extend({'kind': 'token', 'index': index, 'token_id': token,
        'finish_reason': 'length' if index == 15 else None,
        'usage_json': canonical(case['usage']).decode() if index == 15 else None}
        for index, token in enumerate(case['raw_token_ids']))
    rows.append({'kind': 'complete', 'tokens': 16, 'finish_reason': 'length'})
    return b''.join(canonical(row)+b'\n' for row in rows)


def report(binding, raw):
    return {'identity': run_identity(binding), 'raw_base64': base64.b64encode(raw).decode(),
            'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest()}


class RawTokenControls(TestCase):
    def setUp(self):
        fixture = instance_controls.PrivateInstanceControls()
        fixture.setUp()
        self.plan, self.contract = fixture.plan, fixture.contract
        self.bindings = tuple(RunBinding.from_expected(ExpectedMember.from_launch(self.plan, member),
            self.contract.runtime_sha256) for member in fixture.launches)
        self.reference = oracle(self.plan)
        self.case = json.loads(self.reference.raw)['cases'][0]
        self.snapshot = {'execution_id': 'execution-fixture', 'request_sha256': self.case['body_sha256'],
            'command_id': 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'state_event_idx': 10,
            'instance_id': self.contract.instance_id, 'contract_sha256': self.contract.sha256,
            'contract': json.loads(canonical(fixture.contract.__dict__)),
            'end': {'command_id': 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'task_id': 'actual-task',
                'group_id': self.contract.group_id, 'generation': self.contract.generation,
                'rank_manifest_sha256': self.plan.rank_manifest_sha256, 'end_event_idx': 100,
                'instance_id': self.contract.instance_id,
                'runner_ids': sorted(self.contract._instance['shardAssignments']['runnerToShard'])}}

    def reports(self, case=None):
        case = self.case if case is None else case
        return tuple(report(binding, receipt(case, expected(binding))) for binding in self.bindings)

    def test_actual_oracle_preserves_all_48_raw_ids_while_frontend_has_only_45(self):
        cases = json.loads(self.reference.raw)['cases']
        self.assertEqual([len(case['raw_token_ids']) for case in cases], [16, 16, 16])
        self.assertEqual([len(case['emitted_token_ids']) for case in cases], [15, 15, 15])
        for case in cases:
            self.snapshot['request_sha256'] = case['body_sha256']
            result = compare_raw_tokens(self.reference, self.snapshot, self.bindings, self.reports(case))
            self.assertTrue(result['matched'])
            self.assertEqual(result['expected_token_ids'][0], 151667)
            self.assertFalse(result['physical_admission_or_cleanup_proof'])

    def test_discarding_or_changing_the_hidden_first_id_cannot_pass(self):
        changed = copy.deepcopy(self.case)
        changed['raw_token_ids'][0] = 123
        reports = (self.reports()[0], self.reports()[1], report(self.bindings[2],
            receipt(changed, expected(self.bindings[2]))))
        result = compare_raw_tokens(self.reference, self.snapshot, self.bindings, reports)
        self.assertFalse(result['matched'])
        self.assertEqual(result['members'][2]['different_positions'], [0])
        changed['raw_token_ids'] = self.case['emitted_token_ids']
        with self.assertRaises(ReceiptUnavailable):
            parse_tokens(receipt(changed, expected(self.bindings[0])), expected(self.bindings[0]))

    def test_reference_hash_or_shortened_raw_sequence_refuses(self):
        value = json.loads(self.reference.raw)
        value['cases'][0]['raw_token_ids'] = value['cases'][0]['emitted_token_ids']
        raw = canonical(value)
        with self.assertRaises(ValueError):
            ChatReference(raw, hashlib.sha256(raw).hexdigest(), self.plan)
        with self.assertRaises(ValueError):
            ChatReference(self.reference.raw, '0'*64, self.plan)

    def test_partial_duplicate_boolean_and_out_of_order_records_are_unknown(self):
        raw = receipt(self.case, expected(self.bindings[0]))
        rows = [json.loads(line) for line in raw.splitlines()]
        for damaged in (raw[:-1], b'\n'.join(raw.splitlines()[:-1])+b'\n', raw+raw,
                        raw.replace(b'"index":0', b'"index":false', 1),
                        raw.replace(b'"token_id":151667', b'"token_id":true', 1),
                        raw.replace(b'"token_limit":16', b'"token_limit":16,"token_limit":16', 1),
                        canonical(rows[0])+b'\n'+canonical(rows[2])+b'\n'+canonical(rows[1])+b'\n'+
                        b''.join(canonical(r)+b'\n' for r in rows[3:])):
            with self.subTest(length=len(damaged)), self.assertRaises(ReceiptUnavailable):
                parse_tokens(damaged, expected(self.bindings[0]))

    def test_input_options_cache_or_cross_command_changes_do_not_compare(self):
        for field in ('rendered_prompt', 'encoded_prompt_ids', 'task', 'usage'):
            changed = copy.deepcopy(self.case)
            if field == 'rendered_prompt':
                changed[field] += 'different'
            elif field == 'encoded_prompt_ids':
                changed[field][0] += 1
            elif field == 'task':
                changed[field]['top_p'] = 0.9
            else:
                changed[field]['prompt_tokens_details']['cached_tokens'] += 1
            reports = (report(self.bindings[0], receipt(changed, expected(self.bindings[0]))), *self.reports()[1:])
            with self.subTest(field=field), self.assertRaises(ComparisonUnavailable):
                compare_raw_tokens(self.reference, self.snapshot, self.bindings, reports)
        foreign = dict(expected(self.bindings[0]), command_id='bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee')
        with self.assertRaises(ComparisonUnavailable):
            compare_raw_tokens(self.reference, self.snapshot, self.bindings,
                (report(self.bindings[0], receipt(self.case, foreign)), *self.reports()[1:]))

    def test_terminal_http_or_two_members_cannot_replace_all_rank_end(self):
        for end in (None, dict(self.snapshot['end'], runner_ids=['one']),
                    dict(self.snapshot['end'], end_event_idx=10)):
            with self.subTest(end=end), self.assertRaises(ComparisonUnavailable):
                compare_raw_tokens(self.reference, dict(self.snapshot, end=end), self.bindings, self.reports())
        with self.assertRaises(ComparisonUnavailable):
            compare_raw_tokens(self.reference, self.snapshot, self.bindings, self.reports()[:2])
        with self.assertRaises(ComparisonUnavailable):
            compare_raw_tokens(self.reference, self.snapshot, self.bindings, tuple(reversed(self.reports())))

    def test_collector_requires_current_boot_private_unredirected_single_link_file(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder).resolve()
            directory = home/'qualification-tokens'
            directory.mkdir(mode=0o700)
            identity = expected(self.bindings[0])
            name = identity['command_id']+'-'+hashlib.sha256(identity['task_id'].encode()).hexdigest()+'.jsonl'
            path = directory/name
            raw = receipt(self.case, identity)
            path.write_bytes(raw)
            path.chmod(0o600)
            result = collect_tokens(str(home), os.getuid(), 'boot', identity, boot_reader=lambda: 'boot')
            self.assertEqual(base64.b64decode(result['raw_base64']), raw)
            path.chmod(0o640)
            with self.assertRaises(ReceiptUnavailable):
                collect_tokens(str(home), os.getuid(), 'boot', identity, boot_reader=lambda: 'boot')
            path.chmod(0o600)
            with self.assertRaises(ReceiptUnavailable):
                collect_tokens(str(home), os.getuid(), 'boot', identity, boot_reader=lambda: 'other')
            other = home/'other'
            path.rename(other)
            path.symlink_to(other)
            with self.assertRaises((ReceiptUnavailable, OSError)):
                collect_tokens(str(home), os.getuid(), 'boot', identity, boot_reader=lambda: 'boot')


class TokenHostControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Build the fixed option before installing any ASGI route or source pin.
        original = deployed.DeploymentControls.asyncSetUp

        async def configured(fixture):
            await original(fixture)
            enable_trace(fixture)

        from unittest.mock import patch
        with patch.object(deployed.DeploymentControls, 'asyncSetUp', configured):
            self.controls = host_controls.HostControlControls()
            self.controls.addAsyncCleanup = self.addAsyncCleanup
            await self.controls.asyncSetUp()

    async def test_tokens_use_the_registered_home_and_reject_caller_paths(self):
        control = self.controls
        owner = await control.bind()
        binding = owner.actor.spec.binding
        identity = expected(binding)
        # This route fixture's tiny model is synthetic. Only its metadata
        # identity changes; numerical acceptance uses the real oracle above.
        case = json.loads((Path(__file__).parent.parent/
            'qualification/fixtures/exo-chat-reference-v1.json').read_bytes())['cases'][0]
        identity['model'] = control.config.plan.model_id
        case['task']['model'] = control.config.plan.model_id
        directory = control.config.compile(binding.expected.binding).native_home/'qualification-tokens'
        directory.mkdir(mode=0o700)
        path = directory/(identity['command_id']+'-'+hashlib.sha256(identity['task_id'].encode()).hexdigest()+'.jsonl')
        raw = receipt(case, identity)
        path.write_bytes(raw)
        path.chmod(0o600)
        arguments = {'identity': run_identity(binding), 'command_id': identity['command_id'], 'task_id': identity['task_id']}
        from unittest.mock import patch
        from .exo_groups import host_control

        def physical_fixture(home, uid, boot, bound):
            return collect_tokens(home, uid, boot, bound, boot_reader=lambda: owner.boot_id)

        with patch.object(host_control, 'collect_tokens', physical_fixture):
            response = await control.post('tokens', **arguments)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(base64.b64decode(response.json()['report']['raw_base64']), raw)
        self.assertEqual((await control.post('tokens', **arguments, path='/foreign')).status_code, 422)
        control.caller = {'valid': True, 'owner_id': 'foreign'}
        self.assertEqual((await control.post('tokens', **arguments)).status_code, 403)
        control.caller = {'valid': True, 'owner_id': control.config.plan.owner_id}
        path.unlink()
        self.assertEqual((await control.post('tokens', **arguments)).status_code, 503)


class TokenConfigurationControls(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = deployed.DeploymentControls()
        self.fixture.addAsyncCleanup = self.addAsyncCleanup
        await self.fixture.asyncSetUp()

    async def test_only_explicit_qualified_source_enables_trace(self):
        f = self.fixture
        self.assertNotIn('EXO_LEASED_TOKEN_TRACE', f.config.compile(f.binding).environment)
        with self.assertRaises(ValueError):
            HostDeployment(f.plan, f.config.files, **f.options, qualification_token_limit=16)
        for value in (True, 0, 15, 17, '16'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HostDeployment(f.plan, f.config.files, **f.options, qualification_token_limit=value)
        config = enable_trace(f)
        self.assertEqual(config.compile(f.binding).environment['EXO_LEASED_TOKEN_TRACE'], '16')
