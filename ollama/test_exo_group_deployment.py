"""Real sealed files and host journal; no model, fleet HTTP or native child.

The existing tiny staging fixture supplies actual local files, not fabricated
hash reports. Volume/counters/transport are isolated test observations. Native
spawn is forbidden; no test treats this as network enforcement qualification.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI

from . import server
from . import test_exo_group_staging as staged
from .exo_groups import deployment
from .exo_groups.deployment import HostDeployment
from .exo_groups.host_control import configuration_sha256
from .exo_groups.plan import LaunchBinding
from .exo_groups.prepare import canonical
from .exo_groups.rank_process import ObservationUnknown
from .exo_groups.staging import HostFiles, runtime_file_policy
from .exo_groups.staging_files import collect_host
from .inference_bootstrap import QueueRuntime, QueueSettings


class DeploymentControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stage = staged.StagingControls()
        await self.stage.asyncSetUp()
        self.addAsyncCleanup(self.stage.asyncTearDown)
        self.root = self.stage.root
        self.plan = self.stage.plan
        setting = self.stage.settings['slice']
        runtime = Path(setting['runtime_root'])
        runtime.chmod(0o700)
        additions = ('run_text.py', 'python/bin/python3.13', 'source/src/exo/main.py')
        pins = list(self.stage.runtime_files['slice'])
        for name in additions:
            p = runtime / name
            p.parent.mkdir(parents=True, exist_ok=True)
            raw = b'fixture must never execute\n'
            p.write_bytes(raw)
            pins.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'link': None})
        for p in sorted(runtime.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if not p.is_symlink():
                p.chmod(0o500 if p.is_dir() else 0o400)
        runtime.chmod(0o500)
        old = self.stage.verifier.policies[0]
        self.files = HostFiles(old.rank, old.resource_id, old.uid, old.scratch_bytes,
                               old.model, runtime_file_policy(str(runtime), pins))
        self.state_base = self.root / 'attempts'
        self.state_base.mkdir(mode=0o700)
        self.network = True
        self.transport_observations = []
        self.file_observations = []

        async def transport(spec, environment):
            self.transport_observations.append((spec, environment))
            return self.network

        def collector(spec, previous):
            value = collect_host(spec, previous,
                filesystem=lambda _: {'local': True, 'filesystem': 'apfs',
                    'filesystem_id': [1, 2], 'free_bytes': 1000}, boot_reader=lambda: self.stage.boot)
            self.file_observations.append(value)
            return value

        self.options = dict(manifest=self.stage.manifest, parent=canonical(self.stage.parent),
            runtime_sha256='f' * 64, boot_id=self.stage.boot, home=str(self.root),
            state_base=str(self.state_base), peer_ips=('192.168.1.3', '192.168.1.19', '192.168.1.84'),
            api_port=52515, peer_port=52514, discovery_port=52513, ring_port=52518,
            cleanup_tolerance_bytes=128, verify_transport=transport, collector=collector)
        self.config = HostDeployment(self.plan, self.files, **self.options)
        member = next(m for m in self.plan.members if m.rank == 'slice')
        self.binding = LaunchBinding('group_' + 'a' * 24, 1, self.plan.owner_id, 'slice', 0,
            member.resource_id, 'lease_' + '1' * 16, member.layers,
            '2040-01-01T00:00:00+00:00', self.plan.rank_manifest_sha256)
        self.manager = SimpleNamespace(models={}, residency_renewal=None,
            sc=SimpleNamespace(token='synthetic-host-only'))
        service = SimpleNamespace(app=FastAPI(), mgr=self.manager, ChatRequest=server.ChatRequest,
            config=SimpleNamespace(WHISPER_MAX_UPLOAD_MB=1, SC_RESOURCE_ID=member.resource_id,
                                   SC_SERVICE_ID='host-fixture'))
        models = {'local': member.resource_id}
        if getattr(self, 'include_group_model', False):
            models[self.plan.model_id] = member.resource_id
        self.runtime = QueueRuntime.install(service, QueueSettings(self.root / 'host.sqlite',
            models, reservation_mb={model: 100 for model in models}), registry_owner='user:9999')
        self.runtime.bind_capacity_source(work_permitted=False)
        self.owner = None
        self.spawner = Mock(side_effect=AssertionError('native child forbidden in deployment controls'))
        original = deployment.RankLifecycle

        def actor(*args, **kwargs):
            return original(*args, **kwargs, memory_reader=lambda: 100,
                port_reader=lambda _: True, boot_reader=lambda: self.stage.boot, spawner=self.spawner)

        self.actor_patch = patch.object(deployment, 'RankLifecycle', actor)
        self.actor_patch.start()
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        if self.owner is not None:
            if self.owner.actor.receipt is None:
                await self.owner.attest_never_started()
            await self.runtime.close_exo_rank()
        self.spawner.assert_not_called()
        self.runtime.close()
        self.actor_patch.stop()

    async def bind(self):
        self.owner = await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        return self.owner

    async def test_compilation_has_no_file_or_authority_effect_and_pins_exact_runtime(self):
        compiled = self.config.compile(self.binding)
        self.assertEqual(compiled, self.config.compile(self.binding))
        self.assertEqual(list(self.state_base.iterdir()), [])
        self.assertFalse(self.file_observations)
        self.assertFalse(self.transport_observations)
        self.assertEqual(compiled.spec.argv[:5], (str(self.config.runtime_root / 'python/bin/python3.13'),
            '-I', '-S', '-B', str(self.config.runtime_root / 'run_text.py')))
        self.assertIn('--force-master', compiled.spec.argv)
        self.assertNotIn('--legacy-daemon', compiled.spec.argv)
        self.assertEqual(compiled.environment['EXO_MODELS_READ_ONLY_DIRS'], self.files.model['root'])
        self.assertEqual(compiled.environment['EXO_RANK_BUNDLE'], self.files.model['root'])
        self.assertEqual(compiled.environment['ENABLE_DISAGGREGATION'], 'false')
        self.assertEqual(compiled.environment['EXO_LEASED_RING_PORT'], '52518')
        self.assertEqual(compiled.spec.binding.expected.member.layers, (0, 7))
        self.assertFalse(any('TOKEN' in k or 'BEARER' in k for k in compiled.environment))
        self.assertNotIn('EXO_PREFILL_STEP_SIZE', compiled.environment)
        changed = self.config.compile(replace(self.binding, generation=2))
        self.assertNotEqual(compiled.node_id, changed.node_id)
        self.assertNotEqual(compiled.session, changed.session)

    async def test_explicit_prefill_step_requires_qualified_native_source(self):
        with self.assertRaisesRegex(ValueError, 'native prefill-profile'):
            HostDeployment(self.plan, self.files, **(self.options | {'prefill_step_size': 256}))
        for invalid in (True, 0, 3, '256'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                HostDeployment(self.plan, self.files, **(self.options | {'prefill_step_size': invalid}))
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_native_memory_envelopes_follow_pipeline_not_acquisition_order(self):
        changed = replace(self.plan, members=tuple(reversed(self.plan.members)))
        configured = HostDeployment(changed, self.files, **self.options)
        compiled = configured.compile(self.binding)
        expected = ','.join(str(m.memory_mb) for m in
                            sorted(self.plan.members, key=lambda m: m.pipeline_rank))
        self.assertEqual(compiled.environment['EXO_LEASED_MEMORY_MB'], expected)
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_prefill_step_is_bound_to_host_configuration_and_launch_environment(self):
        runtime = Path(self.files.runtime['root'])
        runtime.chmod(0o700)
        path = runtime / 'source/src/exo/utils/inference_profile.py'
        path.parent.parent.chmod(0o700)
        path.parent.mkdir(mode=0o700)
        raw = b'qualified control fixture; never executed\n'
        path.write_bytes(raw)
        path.chmod(0o400)
        path.parent.chmod(0o500)
        path.parent.parent.chmod(0o500)
        runtime.chmod(0o500)
        pins = list(self.files.runtime['files']) + [{'path': str(path.relative_to(runtime)),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'link': None}]
        files = replace(self.files, runtime=runtime_file_policy(str(runtime), pins))
        config = HostDeployment(self.plan, files, **(self.options | {'prefill_step_size': 256}))
        other = HostDeployment(self.plan, files, **(self.options | {'prefill_step_size': 512}))
        compiled = config.compile(self.binding)
        self.assertEqual(compiled.environment['EXO_PREFILL_STEP_SIZE'], '256')
        self.assertNotEqual(configuration_sha256(config), configuration_sha256(other))
        self.assertNotEqual(compiled.environment_json, other.compile(self.binding).environment_json)
        self.assertEqual(compiled.expected, self.config.compile(self.binding).expected)
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_served_context_requires_native_enforcement_and_cannot_exceed_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'token-budget enforcement'):
            HostDeployment(self.plan, self.files, **(self.options | {'served_context_length': 1024}))
        runtime = Path(self.files.runtime['root'])
        pins = list(self.files.runtime['files'])
        for name in ('source/src/exo/utils/served_context.py',
                     'source/src/exo/worker/engines/mlx/generator/generate.py'):
            path = runtime / name
            for parent in reversed((path.parent, *path.parent.parents)):
                if parent == runtime or parent.is_relative_to(runtime):
                    parent.mkdir(exist_ok=True)
                    parent.chmod(0o700)
            raw = b'qualified context control fixture; never executed\n'
            path.write_bytes(raw)
            pins.append({'path': name, 'bytes': len(raw),
                         'sha256': hashlib.sha256(raw).hexdigest(), 'link': None})
        for path in sorted(runtime.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if not path.is_symlink():
                path.chmod(0o500 if path.is_dir() else 0o400)
        runtime.chmod(0o500)
        files = replace(self.files, runtime=runtime_file_policy(str(runtime), pins))
        for value in (True, 0, 32769):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HostDeployment(self.plan, files, **(self.options | {'served_context_length': value}))
        configured = HostDeployment(self.plan, files, **(self.options | {'served_context_length': 1024}))
        changed = HostDeployment(self.plan, files, **(self.options | {'served_context_length': 2048}))
        self.assertEqual(configured.compile(self.binding).environment['EXO_SERVED_CONTEXT_LENGTH'], '1024')
        self.assertNotEqual(configuration_sha256(configured), configuration_sha256(changed))
        self.assertEqual(configured.compile(self.binding).expected, self.config.compile(self.binding).expected)
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_wrong_rank_resource_or_lease_never_gets_a_private_attempt(self):
        for binding in (replace(self.binding, rank='wafer'),
                        replace(self.binding, resource_id='foreign/gpu'),
                        replace(self.binding, lease_id='saved-name')):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                self.config.compile(binding)
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_bad_peers_ports_or_unbound_transport_fail_before_files(self):
        for values in ({'peer_ips': ('127.0.0.1', '192.168.1.19', '192.168.1.84')},
                       {'peer_ips': ('169.254.0.1', '192.168.1.19', '192.168.1.84')},
                       {'ring_port': 52514}, {'ring_port': True}, {'verify_transport': None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                HostDeployment(self.plan, self.files, **(self.options | values))
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_missing_or_redirected_private_state_is_rejected_during_preflight(self):
        self.state_base.rmdir()
        with self.assertRaises(FileNotFoundError):
            await self.config.observe(self.config.compile(self.binding))
        self.state_base.symlink_to(self.root, target_is_directory=True)
        try:
            with self.assertRaises(ValueError):
                await self.config.observe(self.config.compile(self.binding))
        finally:
            self.state_base.unlink()
            self.state_base.mkdir(mode=0o700)
        self.state_base.chmod(0o500)
        try:
            with self.assertRaises(ValueError):
                await self.config.observe(self.config.compile(self.binding))
        finally:
            self.state_base.chmod(0o700)
        self.assertFalse(self.transport_observations)

    async def test_real_sealed_files_are_rechecked_and_hash_cache_is_not_permission(self):
        compiled = self.config.compile(self.binding)
        await self.config.observe(compiled)
        self.assertGreater(self.file_observations[-1]['model']['hashed_bytes'], 0)
        await self.config.observe(compiled)
        self.assertEqual(self.file_observations[-1]['model']['hashed_bytes'], 0)
        self.assertEqual(self.file_observations[-1]['runtime']['hashed_bytes'], 0)
        self.network = False
        with self.assertLogs('ollama.exo_groups.deployment', 'WARNING') as captured:
            with self.assertRaisesRegex(ValueError, 'transport'):
                await self.config.observe(compiled)
        self.assertEqual([record.getMessage() for record in captured.records], [
            f'EXO host staging observation failed rank={self.config.files.rank} phase=transport'])
        self.assertEqual(len(self.transport_observations), 3)
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_failed_or_nonboolean_transport_never_binds_or_prepares(self):
        for value in (False, None, 'available', 1):
            self.network = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'transport'):
                await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        self.assertFalse(list(self.state_base.iterdir()))
        self.assertIsNone(getattr(self.runtime, 'host_rank', None))

    async def test_binding_writes_private_profile_and_uses_own_host_credential_only(self):
        owner = await self.bind()
        compiled = self.config.compile(self.binding)
        profile = json.loads((compiled.native_home / '.exo-leased-profile.json').read_text())
        self.assertEqual(profile, {'group_id': self.binding.group_id, 'generation': '1',
            'rank_manifest_sha256': self.binding.rank_manifest_sha256, 'rank': 'slice'})
        key = compiled.native_home / 'private-api.key'
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertRegex(key.read_text().strip(), '^[a-f0-9]{64}$')
        self.assertNotEqual(key.read_text().strip(), self.manager.sc.token)
        self.assertEqual(owner.adapter.reader.owner_id, 'user:9999')
        self.assertEqual(owner.adapter.reader._headers('application/json')['Authorization'],
                         'Bearer synthetic-host-only')
        self.manager.sc.token = 'synthetic-rotated-host'
        self.assertEqual(owner.adapter.reader._headers('application/json')['Authorization'],
                         'Bearer synthetic-rotated-host')
        self.assertEqual(owner._attempt()['phase'], 'prepared')
        self.assertFalse(owner._events_ready)
        with self.assertRaises(ObservationUnknown):
            await owner.start()
        self.assertFalse((owner.actor.state / 'launch-intent.json').exists())

    async def test_network_loss_blocks_launch_but_does_not_prevent_never_started_proof(self):
        owner = await self.bind()
        self.network = False
        owner._events_ready = True  # A connected feed alone cannot allow launch.
        with self.assertRaisesRegex(ValueError, 'transport'):
            await owner.start()
        self.assertEqual(owner._attempt()['phase'], 'prepared')
        await owner.attest_never_started()
        self.assertEqual(owner._attempt()['phase'], 'launch_fenced')
        self.assertEqual(owner.actor.phase, 'awaiting_settlement')
        self.assertFalse(owner.actor.launch_attempted)
        self.assertTrue((owner.actor.state / 'cleanup-evidence.json').is_file())

    async def test_private_setup_failure_keeps_journaled_owner_and_fence_cleanup(self):
        with patch.object(deployment.secrets, 'token_hex', side_effect=OSError('fixture key setup')):
            with self.assertRaises(OSError):
                await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        self.owner = self.runtime.host_rank
        self.assertEqual(self.owner._attempt()['phase'], 'prepared')
        self.assertFalse(self.owner._events_ready)
        self.network = False
        await self.owner.attest_never_started()
        self.assertEqual(self.owner._attempt()['phase'], 'launch_fenced')
        self.assertEqual(self.owner.actor.phase, 'awaiting_settlement')
        self.assertFalse(self.owner.actor.launch_attempted)

    async def test_private_key_change_is_detected_before_runtime_observation(self):
        owner = await self.bind()
        path = self.config.compile(self.binding).native_home / 'private-api.key'
        before = path.read_bytes()
        path.write_text('0' * 64 + '\n')
        try:
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                await owner.verify_runtime()
        finally:
            path.write_bytes(before)

    async def test_changed_frozen_bytes_or_boot_block_before_private_preparation(self):
        launcher = self.config.runtime_root / 'run_text.py'
        before = launcher.read_bytes()
        launcher.chmod(0o600)
        launcher.write_bytes(b'changed runtime, never execute\n')
        launcher.chmod(0o400)
        with self.assertRaises(ValueError):
            await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        launcher.chmod(0o600)
        launcher.write_bytes(before)
        launcher.chmod(0o400)
        self.stage.boot = '00000000-0000-0000-0000-000000000124'
        with self.assertRaisesRegex(ValueError, 'boot'):
            await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_borrowed_coordinator_identity_or_busy_host_is_not_a_preparation(self):
        self.manager.initial_reservations.owner = self.plan.owner_id
        with self.assertRaises(ValueError):
            await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        self.manager.initial_reservations.owner = 'user:9999'
        self.manager.models['busy'] = object()
        with self.assertRaises(ValueError):
            await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        self.manager.models.clear()
        self.assertFalse(list(self.state_base.iterdir()))

    async def test_own_state_alias_and_reused_attempt_are_not_new_owners(self):
        await self.bind()
        compiled = self.config.compile(self.binding)
        before = (compiled.native_home / 'private-api.key').read_bytes()
        await self.owner.attest_never_started()
        await self.runtime.close_exo_rank()
        with self.assertRaises(FileExistsError):
            await self.config.bind(self.runtime, self.binding, emit=lambda _: None)
        self.assertEqual((compiled.native_home / 'private-api.key').read_bytes(), before)
        self.assertEqual(len(list(self.state_base.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
