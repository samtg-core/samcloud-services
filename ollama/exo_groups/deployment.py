"""Construct one exact host role from the frozen three-host runtime policy.

Configuration is not authority. Binding prepares private files and the existing
host journal, but starts no registry stream, rank, model or lease operation.
The host reader always uses the manager's own already-verified credential.
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
from ipaddress import IPv4Address, IPv4Network
import json
import os
from pathlib import Path
import re
import secrets
import stat
from uuid import UUID

from .evidence import write_new
from .clock import authority_clock
from .host_authority import HostAuthorityAdapter, HostPlaneReader
from .member import ExpectedMember, check_projection
from .plan import GroupPlan, HOST_RESOURCES
from .prepare import canonical, digest
from .rank_lifecycle import RankLifecycle, RankSpec
from .rank_process import Listener, RunBinding
from .rank_runtime import HostRankRuntime, RuntimeObservation, launch_sha256
from .staging import HostFiles, StagingVerifier, model_file_policy
from .staging_files import collect_host


_RANKS = ('slice', 'wafer', 'obelisk')


def _path(value):
    path = Path(value)
    if (not isinstance(value, str) or not path.is_absolute() or str(path) != value
            or '\x00' in value or any(p in ('.', '..') for p in path.parts)):
        raise ValueError('exact absolute unredirected deployment path required')
    return path


def _owned_directory(path, uid, *, private):
    # Do not let resolving a symlink silently bless a different state location.
    for component in reversed((path, *path.parents)):
        info = component.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('unredirected deployment directory required')
    info = path.lstat()
    if (info.st_uid != uid or private and (
            stat.S_IMODE(info.st_mode) & 0o077 or stat.S_IMODE(info.st_mode) & 0o700 != 0o700
            or not os.access(path, os.W_OK | os.X_OK))):
        raise ValueError('owned private deployment directory required')


@dataclass(frozen=True)
class CompiledHost:
    expected: ExpectedMember
    spec: RankSpec
    environment_json: str
    session: Path
    native_home: Path
    node_id: str

    @property
    def environment(self):
        return json.loads(self.environment_json)


class HostDeployment:
    """Trusted per-host configuration, with no transport-verifier fallback.

    verify_transport(spec, environment) must freshly verify the installed
    confinement for those exact endpoints and return True; a saved report or
    peer-IP list is insufficient. No implementation is selected implicitly.
    collector is the local read-only file collector, not a coordinator report.
    """

    def __init__(self, plan, files, *, manifest, parent, runtime_sha256, boot_id, home, state_base,
                 peer_ips, api_port, peer_port, discovery_port, ring_port,
                 cleanup_tolerance_bytes, verify_transport, collector=collect_host,
                 qualification_token_limit=None):
        if (not isinstance(plan, GroupPlan) or not isinstance(files, HostFiles) or files.rank not in _RANKS
                or files.resource_id != HOST_RESOURCES[files.rank]
                or type(files.uid) is not int or files.uid < 0
                or type(files.scratch_bytes) is not int or files.scratch_bytes < 0
                or not isinstance(runtime_sha256, str)
                or not re.fullmatch('[a-f0-9]{64}', runtime_sha256)
                or str(UUID(boot_id)) != boot_id or UUID(boot_id).int == 0
                or not callable(verify_transport) or not callable(collector)
                or type(cleanup_tolerance_bytes) is not int or cleanup_tolerance_bytes < 0
                or qualification_token_limit is not None and (
                    type(qualification_token_limit) is not int or qualification_token_limit != 16)):
            raise ValueError('exact host files, boot, runtime and live transport verifier required')
        self.plan = plan
        # Freeze mutable policy dictionaries before producing any spec or file.
        self.files_json = canonical(asdict(files)).decode()
        self.runtime_sha256, self.boot_id = runtime_sha256, boot_id
        self.home, self.state_base = _path(home), _path(state_base)
        self.runtime_root, self.model_root = _path(files.runtime['root']), _path(files.model['root'])
        if canonical(files.model) != canonical(model_file_policy(
                plan, manifest, parent, files.rank, str(self.model_root))):
            raise ValueError('host model policy differs from the exact pinned plan and bundle')
        names = {pin['path'] for pin in files.runtime['files']}
        if not {'run_text.py', 'python/bin/python3.13', 'source/src/exo/main.py'} <= names:
            raise ValueError('pinned frozen interpreter, launcher and node source required')
        if qualification_token_limit is not None and not {
                'source/src/exo/utils/qualification_tokens.py',
                'source/src/exo/worker/runner/llm_inference/batch_generator.py'} <= names:
            raise ValueError('pinned raw-token observation source required')
        if (self.state_base == self.home / '.exo'
                or any(self.state_base.is_relative_to(p) for p in (self.runtime_root, self.model_root))):
            raise ValueError('private attempt state must be outside immutable artifacts and standing EXO home')
        private = tuple(IPv4Network(v) for v in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
        if (not isinstance(peer_ips, tuple) or len(peer_ips) != len(plan.pipeline_ranks) or len(set(peer_ips)) != len(plan.members)
                or any(str(IPv4Address(v)) != v or not any(IPv4Address(v) in network for network in private)
                       for v in peer_ips)):
            raise ValueError('three exact private LAN peer addresses required')
        ports = (api_port, peer_port, discovery_port, ring_port)
        if (any(type(p) is not int or not 1024 <= p <= 65535 for p in ports)
                or len(set(ports)) != 4):
            raise ValueError('four distinct explicit unprivileged ports required')
        self.peer_ips, self.ports = peer_ips, ports
        self.cleanup_tolerance_bytes = cleanup_tolerance_bytes
        self.qualification_token_limit = qualification_token_limit
        self.verify_transport, self.collector = verify_transport, collector
        self.transport_sha256 = getattr(verify_transport, 'configuration_sha256', None)
        if self.transport_sha256 is not None and (
                not isinstance(self.transport_sha256, str)
                or not re.fullmatch('[a-f0-9]{64}', self.transport_sha256)):
            raise ValueError('exact explicit transport configuration fingerprint required')
        self.previous = None

    @property
    def files(self):
        return HostFiles(**json.loads(self.files_json))

    def compile(self, binding):
        """No file, registry or process effects; saved bindings are not permission."""
        expected = ExpectedMember.from_launch(self.plan, binding)
        files = self.files
        if (binding.rank != files.rank or type(binding.generation) is not int or binding.generation < 1
                or not re.fullmatch('group_[a-f0-9]{24}', binding.group_id)
                or not re.fullmatch('lease_[a-f0-9]{16,64}', binding.lease_id)):
            raise ValueError('exact production-shaped group/member identity required')
        session = self.state_base / f'{binding.group_id}-g{binding.generation}-{binding.lease_id}'
        native_home = session / 'native-home'
        api, peer, discovery, ring = self.ports
        node_id = digest([binding.group_id, binding.generation,
                          binding.rank_manifest_sha256, binding.rank, 'zenoh-node'])[:32].lstrip('0') or '1'
        namespace = f'samcloud:{binding.group_id}:{binding.generation}:{binding.rank_manifest_sha256}'
        environment = {
            'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(self.home),
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'EXO_OFFLINE': 'true',
            'ENABLE_DISAGGREGATION': 'false', 'EXO_ENABLE_IMAGE_MODELS': 'false',
            'EXO_HOME': str(native_home), 'EXO_FORCE_ZID': node_id,
            'EXO_MODELS_READ_ONLY_DIRS': str(self.model_root),
            'EXO_RANK_BUNDLE': str(self.model_root),
            'EXO_RANK_MANIFEST_SHA256': binding.rank_manifest_sha256,
            'EXO_RANK_NAME': binding.rank, 'EXO_LEASED_GROUP_ID': binding.group_id,
            'EXO_LEASED_GENERATION': str(binding.generation),
            'EXO_LEASED_PEER_IPS': ','.join(self.peer_ips),
            'EXO_LEASED_API_KEY_FILE': str(native_home / 'private-api.key'),
            **{f'EXO_LEASED_{name}_PORT': str(port)
               for name, port in zip(('API', 'PEER', 'DISCOVERY', 'RING'), self.ports)},
        }
        if self.qualification_token_limit is not None:
            environment['EXO_LEASED_TOKEN_TRACE'] = str(self.qualification_token_limit)
        argv = (str(self.runtime_root / 'python/bin/python3.13'), '-I', '-S', '-B',
                str(self.runtime_root / 'run_text.py'), '--namespace', namespace,
                '--offline', '--no-downloads', '--no-batch', '--api-port', str(api),
                '--zenoh-port', str(peer), '--discovery-port', str(discovery),
                *(['--force-master'] if binding.rank == 'slice' else ['--no-api']))
        local_ip = self.peer_ips[binding.pipeline_rank]
        listeners = (Listener('127.0.0.1', api), Listener(local_ip, peer),
                     Listener('0.0.0.0', discovery, 'udp'), Listener(local_ip, ring))
        spec = RankSpec(RunBinding.from_expected(expected, self.runtime_sha256), argv,
                        native_home, listeners, self.cleanup_tolerance_bytes)
        return CompiledHost(expected, spec, canonical(environment).decode(), session, native_home, node_id)

    async def observe_files(self, compiled):
        files = self.files
        if (not isinstance(compiled, CompiledHost)
                or compiled != self.compile(compiled.expected.binding)
                or os.getuid() != files.uid):
            raise ValueError('exact own-host compiled deployment required')
        _owned_directory(self.home, files.uid, private=False)
        _owned_directory(self.state_base, files.uid, private=True)
        observed = await asyncio.to_thread(self.collector, asdict(files), self.previous)
        if observed.get('uid') != files.uid or observed.get('boot_id') != self.boot_id:
            raise ValueError('host uid or boot changed')
        StagingVerifier._root(files.model, observed.get('model'), files.scratch_bytes)
        StagingVerifier._root(files.runtime, observed.get('runtime'), files.scratch_bytes)
        self.previous = observed  # Verified hash cache only, never authority.
        return observed

    async def observe(self, compiled):
        observed = await self.observe_files(compiled)
        if await self.verify_transport(compiled.spec, compiled.environment) is not True:
            raise ValueError('live exact transport enforcement unconfirmed')
        return observed

    def host_reader(self, runtime):
        if (runtime.manager.initial_reservations.owner == self.plan.owner_id
                or not isinstance(runtime.manager.sc.token, str) or not runtime.manager.sc.token
                or set(runtime.settings.models.values()) != {self.files.resource_id}):
            raise ValueError('distinct verified own-host service identity required')
        return HostPlaneReader(lambda: 'Bearer ' + runtime.manager.sc.token,
                               owner_id=runtime.manager.initial_reservations.owner)

    async def admit_preparation(self, runtime, binding):
        """Fresh own-member read only; configuration/binding is not permission.

        The coordinator checks its saved full group intent before sending this
        binding. The host's scoped projection independently checks ownership,
        admission and the exact declared envelope, without borrowing that token.
        This observation never replaces the rank's launch-time authority read.
        """
        compiled = self.compile(binding)
        reader = self.host_reader(runtime)
        try:
            started = authority_clock()
            value = await reader.projection(compiled.expected)
            received = authority_clock()
            decision = check_projection(compiled.expected, value,
                request_started_monotonic=started, received_monotonic=received,
                now_monotonic=authority_clock())
            if not decision.allows_start(authority_clock()):
                raise ValueError('exact own member is not freshly admitted')
            lease = value['member']['lease']
            # The saved launch expiry may predate a renewal, but cannot invent
            # a later authority envelope. Actual start still reads it afresh.
            if not (datetime.fromisoformat(lease['granted_at'])
                    < datetime.fromisoformat(binding.expires_at)
                    <= datetime.fromisoformat(lease['expires_at'])):
                raise ValueError('launch expiry is outside the observed lease envelope')
        finally:
            await reader.close()

    async def bind(self, runtime, binding, *, emit):
        """Prepare and journal one host attempt, without starting its rank/feed."""
        files = self.files
        previous = getattr(runtime, 'host_rank', None)
        if (runtime.manager.initial_reservations.owner == binding.owner_id
                or not isinstance(runtime.manager.sc.token, str) or not runtime.manager.sc.token
                or not callable(emit) or runtime.manager.models
                or runtime.capacity_source is None or previous is not None and not previous.closed
                or set(runtime.settings.models.values()) != {files.resource_id}
                or runtime.queue.supervisor.journal.db.execute(
                    "SELECT 1 FROM executions WHERE state IN ('running','uncertain') LIMIT 1").fetchone()):
            raise ValueError('distinct verified host-service identity and event sink required')
        compiled = self.compile(binding)
        await self.observe(compiled)
        _owned_directory(self.home, files.uid, private=False)
        _owned_directory(self.state_base, files.uid, private=True)
        compiled.session.mkdir(mode=0o700)  # Existing/unknown attempt is never reused.
        write_new(compiled.session / 'deployment.json', {
            'kind': 'prepared_host_configuration', 'binding': asdict(compiled.expected),
            'runtime_sha256': self.runtime_sha256, 'host_boot': self.boot_id,
            'argv': compiled.spec.argv, 'environment_sha256': digest(compiled.environment),
            'node_id': compiled.node_id, 'lease_mutations': 0})
        private_pins = None

        async def verify_prepared():
            # Only the locked negative-CAS path calls this. It proves source,
            # spec, uid and boot even if private setup or transport failed.
            # Ports/memory and the permanent fence still come from that path.
            await self.observe_files(compiled)
            return RuntimeObservation(self.runtime_sha256, expected_launch_sha, self.boot_id, files.uid)

        async def verify():
            await self.observe(compiled)
            if private_pins is None:
                raise ValueError('private runtime configuration incomplete')
            _owned_directory(compiled.native_home, files.uid, private=True)
            for name, expected in private_pins.items():
                path = compiled.native_home / name
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, 'rb') as source:
                    info = os.fstat(source.fileno())
                    raw = source.read(4097)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != files.uid
                        or info.st_nlink != 1 or info.st_mode & 0o077 or len(raw) > 4096):
                    raise ValueError('private runtime configuration changed')
                actual = digest(json.loads(raw)) if name.endswith('.json') else digest(raw.strip().decode())
                if actual != expected:
                    raise ValueError('private runtime configuration changed')
            return RuntimeObservation(self.runtime_sha256, expected_launch_sha, self.boot_id, files.uid)

        actor = RankLifecycle(compiled.expected, compiled.spec, compiled.session / 'attempt', emit=emit)
        expected_launch_sha = launch_sha256(actor, compiled.environment)
        # Callback stays local to the already-verified manager. There is no
        # coordinator header argument, token file copy or credential in env.
        reader = self.host_reader(runtime)
        adapter = HostAuthorityAdapter(actor, reader)
        owner = HostRankRuntime(actor, adapter, verify_runtime=verify, environment=compiled.environment,
                                boot_id=self.boot_id, verify_unstarted=verify_prepared)
        runtime.bind_exo_rank(owner)
        # Journal the guarded preparation before fallible key/home setup. A
        # setup error keeps runtime.host_rank and its negative-CAS cleanup path;
        # it cannot leave an apparently fresh attempt for an automatic retry.
        compiled.native_home.mkdir(mode=0o700)
        marker = {'group_id': binding.group_id, 'generation': str(binding.generation),
                  'rank_manifest_sha256': binding.rank_manifest_sha256, 'rank': binding.rank}
        write_new(compiled.native_home / '.exo-leased-profile.json', marker)
        key_path = compiled.native_home / 'private-api.key'
        key = secrets.token_hex(32).encode()
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as output:
            output.write(key + b'\n')
            output.flush()
            os.fsync(output.fileno())
        parent = os.open(compiled.native_home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        private_pins = {'.exo-leased-profile.json': digest(marker),
                        'private-api.key': digest(key.decode())}
        return owner
