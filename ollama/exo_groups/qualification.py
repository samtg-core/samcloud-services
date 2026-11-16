"""Strict numerical observation of already-complete queued EXO calls.

No sampling, native command, authority or cleanup action. Expected raw IDs come
from the pinned whole-model chat oracle, not text or post-parser token events.
"""

import base64
import hashlib

from ..inference_arbiter import Conflict, NotFound
from .host_control import run_identity
from .http import decode_json
from .prepare import canonical, digest
from .token_receipts import parse_tokens


PROMPTS = ('The capital of France is', 'Count from one to five:',
           'Write one short sentence about a river.')


class ComparisonUnavailable(ValueError):
    pass


class ChatReference:
    def __init__(self, raw, sha256, plan):
        if (not isinstance(raw, bytes) or len(raw) > 128 * 1024
                or hashlib.sha256(raw).hexdigest() != sha256):
            raise ValueError('pinned bounded whole-model chat reference required')
        value = decode_json(raw)
        cases = value.get('cases')
        if (value.get('status') != 'whole_model_chat_reference_completed'
                or value.get('model_id') != plan.model_id
                or value.get('manifest_sha256') != plan.rank_manifest_sha256
                or value.get('tokens_per_case_required') != 16
                or value.get('original_prompts_unchanged') is not True
                or not isinstance(cases, list) or len(cases) != 3
                or tuple(case.get('prompt') for case in cases) != PROMPTS):
            raise ValueError('original three-prompt raw oracle required')
        for case in cases:
            if (not isinstance(case.get('body'), dict) or case.get('body_sha256') != digest(case['body'])
                    or case['body'].get('messages') != [{'role': 'user', 'content': case['prompt']}]
                    or case['body'].get('model') != plan.model_id
                    or case['body'].get('temperature') != 0 or case['body'].get('max_tokens') != 16
                    or case.get('finish_reason') != 'length' or not isinstance(case.get('task'), dict)
                    or not isinstance(case.get('rendered_prompt'), str)
                    or not isinstance(case.get('encoded_prompt_ids'), list) or not case['encoded_prompt_ids']
                    or not isinstance(case.get('raw_token_ids'), list) or len(case['raw_token_ids']) != 16
                    or any(type(token) is not int or token < 0 for token in (
                        *case['raw_token_ids'], *case['encoded_prompt_ids']))
                    or not isinstance(case.get('usage'), dict) or case['usage'].get('completion_tokens') != 16):
                raise ValueError('full raw sequence and exact request/template required')
        self.raw, self.sha256, self.plan = raw, sha256, plan

    def case(self, request_sha256):
        cases = [case for case in decode_json(self.raw)['cases'] if case['body_sha256'] == request_sha256]
        if len(cases) != 1:
            raise ComparisonUnavailable('queued request does not match the pinned reference')
        return cases[0]


def compare_raw_tokens(reference, snapshot, launches, reports):
    """Numerical component only; caller owns actual journal and host reads."""
    case = reference.case(snapshot.get('request_sha256'))
    contract = snapshot.get('contract')
    end = snapshot.get('end')
    if (not isinstance(contract, dict) or snapshot.get('contract_sha256') != digest(contract)
            or not isinstance(end, dict) or end.get('command_id') != snapshot.get('command_id')
            or end.get('group_id') != contract.get('group_id')
            or type(end.get('generation')) is not int or end['generation'] != contract.get('generation')
            or end.get('rank_manifest_sha256') != reference.plan.rank_manifest_sha256
            or not isinstance(end.get('task_id'), str) or not end['task_id']
            or type(end.get('end_event_idx')) is not int or type(snapshot.get('state_event_idx')) is not int
            or end['end_event_idx'] <= snapshot['state_event_idx']
            or not isinstance(launches, tuple) or tuple(b.expected.binding.rank for b in launches)
                != ('slice', 'wafer', 'obelisk')
            or not isinstance(reports, tuple) or len(reports) != 3):
        raise ComparisonUnavailable('same completed command and three exact member observations required')
    try:
        instance = decode_json(contract['state_instance_json'])['MlxRingInstance']
        if (end.get('instance_id') != snapshot.get('instance_id')
                or end['instance_id'] != instance['instanceId']
                or len(instance['shardAssignments']['runnerToShard']) != 3
                or end.get('runner_ids') != sorted(instance['shardAssignments']['runnerToShard'])):
            raise ValueError('end')
        rows = []
        for binding, report in zip(launches, reports):
            launch = binding.expected.binding
            if (binding.group_id != contract['group_id'] or binding.generation != contract['generation']
                    or binding.runtime_sha256 != contract['runtime_sha256']
                    or canonical(report.get('identity')) != canonical(run_identity(binding))
                    or set(report) != {'identity', 'raw_base64', 'raw_bytes', 'raw_sha256'}):
                raise ValueError('member')
            raw = base64.b64decode(report['raw_base64'], validate=True)
            if (type(report['raw_bytes']) is not int or len(raw) != report['raw_bytes']
                    or len(raw) > 128 * 1024 or hashlib.sha256(raw).hexdigest() != report['raw_sha256']):
                raise ValueError('raw')
            expected = {'group_id': binding.group_id, 'generation': binding.generation,
                'rank': launch.rank, 'device_rank': launch.pipeline_rank,
                'rank_manifest_sha256': binding.rank_manifest_sha256, 'command_id': end['command_id'],
                'task_id': end['task_id'], 'model': reference.plan.model_id}
            observed = parse_tokens(raw, expected)
            header = observed['header']
            if (header['prompt'] != case['rendered_prompt']
                    or header['prompt_token_ids'] != case['encoded_prompt_ids']
                    or canonical(observed['task']) != canonical(case['task'])
                    or canonical(observed['usage']) != canonical(case['usage'])):
                raise ValueError('different input, options or cache path')
            differences = [index for index, (wanted, actual) in enumerate(zip(
                case['raw_token_ids'], observed['token_ids'], strict=True)) if wanted != actual]
            rows.append({'rank': launch.rank, 'lease_id': binding.lease_id,
                'raw_sha256': report['raw_sha256'], 'token_ids': observed['token_ids'],
                'matched': not differences, 'different_positions': differences})
        return {'kind': 'exo_raw_token_comparison_v1', 'execution_id': snapshot['execution_id'],
            'command_id': end['command_id'], 'task_id': end['task_id'],
            'contract_sha256': snapshot['contract_sha256'], 'reference_sha256': reference.sha256,
            'request_sha256': snapshot['request_sha256'], 'expected_token_ids': case['raw_token_ids'],
            'members': rows, 'matched': all(row['matched'] for row in rows),
            'physical_admission_or_cleanup_proof': False}
    except (KeyError, TypeError, ValueError):
        raise ComparisonUnavailable('raw receipt or exact generation input unconfirmed') from None


class TokenComparison:
    """Read the provider's own saved end and exact configured host receipts."""
    def __init__(self, provider, fleet, raw_reference, reference_sha256):
        from .factory import HostFleet
        from .install import GroupProvider
        if (not isinstance(provider, GroupProvider) or not isinstance(fleet, HostFleet)
                or provider.actor.plane is not fleet.plane or provider.actor.plan != fleet.plan
                or provider.actor.verify_staging != fleet.verify
                or any(host.qualification_token_limit != 16 for host in fleet.hosts)):
            raise ValueError('exact raw-token qualification provider and fleet required')
        self.provider, self.fleet = provider, fleet
        self.reference = ChatReference(raw_reference, reference_sha256, fleet.plan)

    async def observe(self, execution_id):
        provider = self.provider
        backend = provider.backend
        if backend is None or provider.assembly is None or provider._closed:
            raise ComparisonUnavailable('original installed assembly required')
        try:
            snapshot = backend.bindings.completed_snapshot(execution_id,
                group_owner=self.fleet.plan.owner_id, request_key=self.fleet.plan.request_key,
                contract=backend.contract)
        except (Conflict, NotFound, KeyError, TypeError, ValueError):
            raise ComparisonUnavailable('complete original group call unconfirmed') from None
        if (snapshot['contract_sha256'] != backend.contract.sha256
                or snapshot['owner'] != self.fleet.plan.owner_id
                or snapshot['request_key'] != self.fleet.plan.request_key or snapshot['end'] is None):
            raise ComparisonUnavailable('complete owned call with observed all-rank end required')
        self.reference.case(snapshot['request_sha256'])  # Refuse foreign input before host reads.
        end = snapshot['end']
        bindings = tuple(command.binding for command in provider.assembly.commands)
        reports = tuple([await self.fleet.clients[b.expected.binding.rank].tokens(
            b, end['command_id'], end['task_id']) for b in bindings])
        return compare_raw_tokens(self.reference, snapshot, bindings, reports)
