"""Exact staged artifact policies and authenticated host observations.

The host reader is an explicit trusted transport, not a caller-supplied report.
No credential, file copy, lease request or compute is performed by this module.
"""

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import re
from uuid import UUID
from pathlib import PurePosixPath

from .plan import BindingRefused, check_rank_manifest
from .prepare import canonical, digest
from .staging_files import validate_policy


class StagingUnavailable(BindingRefused):
    pass


@dataclass(frozen=True)
class HostFiles:
    rank: str
    resource_id: str
    uid: int
    scratch_bytes: int
    model: dict
    runtime: dict


@dataclass(frozen=True)
class VerifiedStaging:
    plan_sha256: str
    manifest_sha256: str
    runtime_sha256: str
    policy_sha256: str
    hosts: tuple
    observation_sha256: str
    observations_json: str

    @property
    def boots(self):
        return tuple((host['rank'], host['resource_id'], host['boot_id'], host['uid'])
                     for host in json.loads(self.observations_json))

    def check(self, plan):
        if (self.plan_sha256 != digest(asdict(plan))
                or self.manifest_sha256 != plan.rank_manifest_sha256
                or any(not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value)
                       for value in (self.runtime_sha256, self.policy_sha256, self.observation_sha256))
                or self.hosts != tuple((m.rank, m.resource_id) for m in sorted(
                    plan.members, key=lambda item: item.pipeline_rank))):
            raise BindingRefused('staging proof differs from the exact provider plan')
        try:
            boots = self.boots
            if (hashlib.sha256(self.observations_json.encode()).hexdigest() != self.observation_sha256
                    or tuple(item[:2] for item in boots) != self.hosts
                    or any(type(item[3]) is not int or item[3] < 0
                           or not isinstance(item[2], str) or str(UUID(item[2])) != item[2]
                           or UUID(item[2]).int == 0 for item in boots)):
                raise ValueError('host observation')
        except (ValueError, TypeError, KeyError):
            raise BindingRefused('exact host staging observation evidence required') from None


def model_file_policy(plan, manifest, parent_bytes, rank_name, root):
    check_rank_manifest(plan, manifest)
    bundle, parent = json.loads(manifest), json.loads(parent_bytes)
    config = parent.get('config', {})
    if (bundle.get('format') != 'samcloud-qwen3-rank-staging-v1'
            or digest(parent) != bundle.get('parent_manifest_sha256')
            or digest(config) != bundle.get('config_sha256')
            or parent.get('model_id') != plan.model_id or parent.get('revision') != plan.revision
            or config.get('model_type') not in ('qwen3', 'qwen3_moe')
            or config.get('num_hidden_layers') != max(member.layers[1] for member in plan.members)):
        raise BindingRefused('exact pinned qualified model bundle required')
    allowed = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
               'added_tokens.json', 'vocab.json', 'merges.txt', 'generation_config.json', 'chat_template.jinja'}
    sidecars = parent.get('sidecar_files')
    if (not isinstance(sidecars, list) or any(not isinstance(item, dict) for item in sidecars)
            or not {'config.json', 'tokenizer.json', 'tokenizer_config.json'} <= {item.get('name') for item in sidecars}
            or not {item.get('name') for item in sidecars} <= allowed
            or len({item.get('name') for item in sidecars}) != len(sidecars)):
        raise BindingRefused('complete pinned config/tokenizer sidecars required')
    ranks = [rank for rank in bundle['ranks'] if rank['name'] == rank_name]
    if len(ranks) != 1:
        raise BindingRefused('exact assigned rank required')
    rank = ranks[0]
    if (rank.get('file') != rank_name + '/rank.safetensors'
            or type(rank.get('expected_bytes')) is not int or rank['expected_bytes'] <= 0
            or not isinstance(rank.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', rank['sha256'])):
        raise BindingRefused('exact rank byte identities required')
    def pin(path, data):
        return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'link': None}
    files = [pin('rank-bundle.json', manifest), pin('parent-manifest.json', parent_bytes),
             pin('VERIFIED', (plan.rank_manifest_sha256 + '\n').encode()),
             {'path': rank['file'], 'bytes': rank['expected_bytes'], 'sha256': rank['sha256'], 'link': None},
             {'path': rank_name + '/model-rank.safetensors', 'bytes': rank['expected_bytes'],
              'sha256': rank['sha256'], 'link': 'rank.safetensors'}]
    for item in sidecars:
        files.append({'path': rank_name + '/' + item['name'], 'bytes': item['bytes'],
                      'sha256': item['sha256'], 'link': None})
    names = ['rank.safetensors', 'model-rank.safetensors'] + [item['name'] for item in sidecars]
    policy = {'root': root, 'files': files, 'views': {rank_name: names}}
    validate_policy(policy)
    return policy


class StagingVerifier:
    def __init__(self, plan, manifest, parent_bytes, runtime_sha256, runtime_files, hosts, reader):
        check_rank_manifest(plan, manifest)
        if (not re.fullmatch('[a-f0-9]{64}', runtime_sha256)
                or not isinstance(runtime_files, dict) or not callable(reader)):
            raise BindingRefused('pinned runtime policy and authenticated host reader required')
        ranks = sorted(plan.members, key=lambda member: member.pipeline_rank)
        if set(hosts) != {member.rank for member in ranks} or set(runtime_files) != set(hosts):
            raise BindingRefused('exact three host staging configurations required')
        self.plan, self.manifest, self.runtime_sha256 = plan, manifest, runtime_sha256
        self.reader, self.previous = reader, {}
        policies = []
        for member in ranks:
            setting = hosts[member.rank]
            if (set(setting) != {'model_root', 'runtime_root', 'uid', 'scratch_bytes'}
                    or type(setting['uid']) is not int or setting['uid'] < 0
                    or type(setting['scratch_bytes']) is not int or setting['scratch_bytes'] < 0):
                raise BindingRefused('explicit host uid and scratch reserve required')
            model = model_file_policy(plan, manifest, parent_bytes, member.rank, setting['model_root'])
            # Qualified native artifacts can differ per host build; they must
            # each be pinned, never assumed equal from a version string alone.
            supplied = runtime_files[member.rank]
            if isinstance(supplied, dict):
                if set(supplied) != {'files', 'directories'}:
                    raise BindingRefused('exact runtime files and directories required')
                runtime = runtime_file_policy(setting['runtime_root'], supplied['files'],
                                              directories=supplied['directories'])
            else:
                runtime = runtime_file_policy(setting['runtime_root'], supplied)
            policies.append(HostFiles(member.rank, member.resource_id, setting['uid'],
                                      setting['scratch_bytes'], model, runtime))
        # Canonical serialization freezes mutable input dictionaries/lists.
        self.policy_json = canonical([asdict(host) for host in policies]).decode()
        self.policy_sha256 = hashlib.sha256(self.policy_json.encode()).hexdigest()

    @property
    def policies(self):
        return tuple(HostFiles(**item) for item in json.loads(self.policy_json))

    @staticmethod
    def _root(expected, actual, scratch):
        if (not isinstance(actual, dict) or actual.get('policy_sha256') != digest(expected)
                or actual.get('root') != expected['root'] or actual.get('readonly') is not True
                or actual.get('filesystem', {}).get('local') is not True
                or actual.get('filesystem', {}).get('filesystem') not in ('apfs', 'hfs')
                or type(actual.get('filesystem', {}).get('free_bytes')) is not int
                or actual['filesystem']['free_bytes'] < scratch):
            raise BindingRefused('fresh local readonly files and disk headroom unconfirmed')
        if not StagingVerifier._identity_valid(actual.get('root_identity')):
            raise BindingRefused('staging root readonly identity unconfirmed')
        required = set(expected['views']) - {'.'}  # Root has its own identity.
        for pin in expected['files']:
            parts = PurePosixPath(pin['path']).parts[:-1]
            required.update('/'.join(parts[:count]) for count in range(1, len(parts) + 1))
        directories = actual.get('directories')
        if (not isinstance(directories, dict) or set(directories) != required
                or any(not StagingVerifier._identity_valid(item) for item in directories.values())):
            raise BindingRefused('staging directory readonly identities unconfirmed')
        files = actual.get('files')
        if not isinstance(files, list) or len(files) != len(expected['files']):
            raise BindingRefused('staged file observation set differs')
        by_name = {item.get('path'): item for item in files if isinstance(item, dict)}
        if set(by_name) != {item['path'] for item in expected['files']}:
            raise BindingRefused('staged file observation set differs')
        for pin in expected['files']:
            item = by_name[pin['path']]
            identity = item.get('identity', {})
            if (item.get('sha256') != pin['sha256'] or item.get('link') != pin['link']
                    or type(identity.get('bytes')) is not int or identity['bytes'] != pin['bytes']
                    or not StagingVerifier._identity_valid(identity)
                    or identity.get('device') != actual['root_identity']['device']):
                raise BindingRefused('staged file bytes or readonly identity differs')

    @staticmethod
    def _identity_valid(value):
        return (isinstance(value, dict) and all(type(value.get(name)) is int and value[name] >= 0
            for name in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns', 'mode', 'uid'))
            and value['inode'] > 0 and not value['mode'] & 0o222)

    async def verify(self, plan, manifest):
        if plan != self.plan or manifest != self.manifest:
            raise BindingRefused('staging policy belongs to another exact plan')
        policies = self.policies
        reports = await asyncio.gather(*(self.reader(host, self.previous.get(host.rank))
                                         for host in policies), return_exceptions=True)
        for host, report in zip(policies, reports, strict=True):
            phase = "host_report"
            try:
                if (not isinstance(report, dict) or report.get('rank') != host.rank
                        or report.get('resource_id') != host.resource_id
                        or type(report.get('uid')) is not int or report['uid'] != host.uid):
                    raise BindingRefused('authenticated host staging observation unavailable')
                phase = "boot"
                try:
                    if str(UUID(report.get('boot_id'))) != report['boot_id']:
                        raise ValueError('boot id')
                except (ValueError, TypeError, AttributeError):
                    raise BindingRefused('host boot identity unavailable') from None
                previous = self.previous.get(host.rank)
                if previous is not None and report['boot_id'] != previous['boot_id']:
                    raise BindingRefused('host boot changed under the staging intent')
                phase = "model_files"
                self._root(host.model, report.get('model'), host.scratch_bytes)
                phase = "runtime_files"
                self._root(host.runtime, report.get('runtime'), host.scratch_bytes)
                phase = "reader_identity"
                if report['model'].get('uid') != host.uid or report['runtime'].get('uid') != host.uid:
                    raise BindingRefused('host reader identity differs between file observations')
            except Exception:
                logging.getLogger(__name__).warning(
                    "EXO staging verification failed rank=%s phase=%s", host.rank, phase)
                raise
        proof = VerifiedStaging(digest(asdict(plan)), plan.rank_manifest_sha256,
            self.runtime_sha256, self.policy_sha256,
            tuple((host.rank, host.resource_id) for host in policies), digest(reports),
            canonical(reports).decode())
        proof.check(plan)
        self.previous = json.loads(canonical({host.rank: report for host, report in zip(
            policies, reports, strict=True)}))
        return proof


def runtime_file_policy(root, files, *, directories=()):
    """Every runtime directory, including root, has an exact entry set.

    A manifest of selected pinned files cannot exclude an extra importable
    module. Deployment must supply its complete qualified runtime file set,
    including interpreter/dependencies/config, not just selected source pins.
    Empty resource directories are preserved and pinned explicitly too.
    """
    if (not isinstance(directories, (tuple, list))
            or any(not isinstance(path, str) or not path or path in ('.', '/')
                   or str(PurePosixPath(path)) != path or PurePosixPath(path).is_absolute()
                   or any(part in ('.', '..') for part in PurePosixPath(path).parts)
                   for path in directories) or len(set(directories)) != len(directories)):
        raise BindingRefused('exact relative runtime directory names required')
    views = {'.': set()}
    for directory in directories:
        parts = PurePosixPath(directory).parts
        for index, name in enumerate(parts):
            parent = '/'.join(parts[:index]) if index else '.'
            views.setdefault(parent, set()).add(name)
        views.setdefault(directory, set())
    for pin in files:
        parts = PurePosixPath(pin['path']).parts
        for index, name in enumerate(parts):
            directory = '/'.join(parts[:index]) if index else '.'
            views.setdefault(directory, set()).add(name)
    policy = {'root': root, 'files': files,
              'views': {directory: sorted(names) for directory, names in views.items()}}
    validate_policy(policy)
    return policy
