"""Actual sealed files feed the staging gate; no model or physical lease."""

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from uuid import UUID

from . import test_exo_group_plan as fixtures
import httpx

from .exo_groups.host_http import HostControlClient, HostEndpoint
from .exo_groups.staging import StagingVerifier, HostFileGatewayUnavailable
from .exo_groups.staging_files import collect_host
from .exo_groups.plan import BindingRefused, GroupPlan
from .exo_groups.prepare import canonical, digest


class StagingControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        fixture = fixtures.GroupBindingControls()
        fixture.setUp()
        config = {'model_type': 'qwen3', 'num_hidden_layers': 28, 'max_position_embeddings': 32768}
        self.sidecars = {'config.json': canonical(config), 'tokenizer.json': b'{}',
                         'tokenizer_config.json': b'{}'}
        self.parent = {'model_id': fixture.plan.model_id, 'revision': fixture.plan.revision,
            'config': config, 'sidecar_files': [{'name': name, 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()} for name, data in self.sidecars.items()]}
        self.ranks = [('slice', [0,7]), ('wafer', [7,12]), ('obelisk', [12,28])]
        self.bundle = {'format': 'samcloud-qwen3-rank-staging-v1',
            'model_id': fixture.plan.model_id, 'revision': fixture.plan.revision,
            'parent_manifest_sha256': digest(self.parent), 'config_sha256': digest(config),
            'ranks': [{'name': name, 'layers': layers, 'file': name + '/rank.safetensors',
                'expected_bytes': len(name.encode()), 'sha256': hashlib.sha256(name.encode()).hexdigest()}
                for name, layers in self.ranks]}
        self.manifest = canonical(self.bundle)
        self.plan = GroupPlan.parse(fixture.raw | {'rank_manifest_sha256': hashlib.sha256(self.manifest).hexdigest()})
        self.settings = {}
        for name, _ in self.ranks:
            host = self.root / name
            model, runtime = host / 'models', host / 'runtime'
            view = model / name
            view.mkdir(parents=True)
            runtime.mkdir()
            (runtime / 'native.dylib').write_bytes(name.encode())
            (model / 'rank-bundle.json').write_bytes(self.manifest)
            (model / 'parent-manifest.json').write_bytes(canonical(self.parent))
            (model / 'VERIFIED').write_text(self.plan.rank_manifest_sha256 + '\n')
            (view / 'rank.safetensors').write_bytes(name.encode())
            (view / 'model-rank.safetensors').symlink_to('rank.safetensors')
            for file, data in self.sidecars.items():
                (view / file).write_bytes(data)
            (runtime / 'main.py').write_bytes(b'# fixture source\n')
            for root in (model, runtime):
                for path in root.rglob('*'):
                    if not path.is_symlink():
                        path.chmod(0o500 if path.is_dir() else 0o400)
                root.chmod(0o500)
            self.settings[name] = {'model_root': str(model), 'runtime_root': str(runtime),
                                   'uid': os.getuid(), 'scratch_bytes': 100}
        self.runtime_files = {name: [
            {'path': 'main.py', 'bytes': 17,
             'sha256': hashlib.sha256(b'# fixture source\n').hexdigest(), 'link': None},
            {'path': 'native.dylib', 'bytes': len(name.encode()),
             'sha256': hashlib.sha256(name.encode()).hexdigest(), 'link': None}]
            for name, _ in self.ranks}
        self.reports = []
        self.boot = str(UUID(int=123))
        self.local, self.free = True, 1000
        async def reader(host, previous):
            report = collect_host(asdict(host), previous,
                filesystem=lambda _: {'local': self.local, 'filesystem': 'apfs',
                    'filesystem_id': [1,2], 'free_bytes': self.free}, boot_reader=lambda: self.boot)
            self.reports.append(report)
            return report
        self.verifier = StagingVerifier(self.plan, self.manifest, canonical(self.parent),
            'f' * 64, self.runtime_files, self.settings, reader)

    async def asyncTearDown(self):
        for path in self.root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    async def test_all_three_sealed_copies_hash_then_recheck_metadata_without_weight_rereads(self):
        proof = await self.verifier.verify(self.plan, self.manifest)
        proof.check(self.plan)
        self.assertEqual(proof.runtime_sha256, 'f' * 64)
        self.assertEqual(len(proof.boots), 3)
        self.assertTrue(all(item['model']['hashed_bytes'] > 0 for item in self.reports))
        self.reports.clear()
        await self.verifier.verify(self.plan, self.manifest)
        self.assertTrue(all(item['model']['hashed_bytes'] == item['runtime']['hashed_bytes'] == 0
                            for item in self.reports))

    async def test_files_gateway_errors_defer_only_when_other_reports_validate(self):
        original = self.verifier.reader
        missing = self.verifier.policies[0]
        endpoint = HostEndpoint(missing.rank, missing.resource_id, missing.uid,
                                'https://host.example', 'a' * 64, 'b' * 64)
        status = 502
        def report(request):
            if status is None:
                raise httpx.ConnectError('connection unavailable', request=request)
            return httpx.Response(status)

        client = HostControlClient(endpoint, lambda: 'Bearer fixture',
            transport=httpx.MockTransport(report))
        self.addAsyncCleanup(client.close)
        async def reader(host, previous):
            if host == missing:
                return await client.files(host, previous)
            return await original(host, previous)
        self.verifier.reader = reader
        for status in (None, 502, 504, 403, 200):
            with self.subTest(status=status), self.assertRaises(BindingRefused) as failure:
                await self.verifier.verify(self.plan, self.manifest)
            self.assertEqual(isinstance(failure.exception, HostFileGatewayUnavailable),
                             status in (None, 502, 504))
        status = 502
        self.local = False
        with self.assertRaises(BindingRefused) as failure:
            await self.verifier.verify(self.plan, self.manifest)
        self.assertNotIsInstance(failure.exception, HostFileGatewayUnavailable)
        self.assertFalse(self.verifier.previous)

    async def test_saved_proof_and_hash_cache_detach_from_reader_owned_reports(self):
        proof = await self.verifier.verify(self.plan, self.manifest)
        by_rank = {report['rank']: report for report in self.reports}
        ordered = [by_rank[rank] for rank, _ in proof.hosts]
        self.assertEqual(proof.observations_json.encode(), canonical(ordered))
        by_rank['wafer']['model']['files'].clear()
        self.assertTrue(self.verifier.previous['wafer']['model']['files'])
        proof.check(self.plan)

    async def test_rejected_runtime_report_identifies_rank_and_keeps_previous_proof(self):
        await self.verifier.verify(self.plan, self.manifest)
        previous = self.verifier.previous
        reader = self.verifier.reader

        async def incorrect_runtime(host, prior):
            report = await reader(host, prior)
            if host.rank == 'wafer':
                report['runtime']['filesystem']['free_bytes'] = -1
            return report

        self.verifier.reader = incorrect_runtime
        with self.assertLogs('ollama.exo_groups.staging', 'WARNING') as captured:
            with self.assertRaises(BindingRefused):
                await self.verifier.verify(self.plan, self.manifest)
        self.assertEqual([record.getMessage() for record in captured.records],
            ['EXO staging verification failed rank=wafer phase=runtime_files'])
        self.assertIs(self.verifier.previous, previous)

    async def test_model_or_runtime_byte_change_withdraws_verified_staging(self):
        await self.verifier.verify(self.plan, self.manifest)
        path = self.root / 'wafer/runtime/main.py'
        path.chmod(0o600)
        path.write_bytes(b'# FIXTURE source\n')
        path.chmod(0o400)
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)

    async def test_unlisted_runtime_import_at_root_is_not_qualified(self):
        await self.verifier.verify(self.plan, self.manifest)
        runtime = self.root / 'slice/runtime'
        runtime.chmod(0o700)
        (runtime / 'sitecustomize.py').write_text('raise RuntimeError("unlisted import")\n')
        (runtime / 'sitecustomize.py').chmod(0o400)
        runtime.chmod(0o500)
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)

    async def test_writable_tree_remote_filesystem_or_no_disk_headroom_is_not_ready(self):
        self.local = False
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)
        self.local, self.free = True, 1
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)
        self.free = 1000
        (self.root / 'obelisk/models/obelisk/config.json').chmod(0o600)
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)

    async def test_host_reboot_does_not_rebind_staging_under_existing_intent(self):
        await self.verifier.verify(self.plan, self.manifest)
        self.boot = str(UUID(int=124))
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)

    async def test_policy_is_detached_from_callers_mutable_dictionaries(self):
        self.settings['slice']['model_root'] = '/foreign'
        self.runtime_files['slice'][0]['sha256'] = 'a' * 64
        proof = await self.verifier.verify(self.plan, self.manifest)
        proof.check(self.plan)

    async def test_missing_host_or_changed_plan_does_not_produce_proof(self):
        settings = dict(self.settings)
        settings.pop('wafer')
        with self.assertRaises(BindingRefused):
            StagingVerifier(self.plan, self.manifest, canonical(self.parent),
                'f' * 64, self.runtime_files, settings, self.verifier.reader)
        from dataclasses import replace
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(replace(self.plan, request_key='different'), self.manifest)

    async def test_each_host_native_artifact_is_pinned_and_not_inferred_from_other_hosts(self):
        await self.verifier.verify(self.plan, self.manifest)
        native = self.root / 'wafer/runtime/native.dylib'
        native.chmod(0o600)
        native.write_bytes(b'slice')  # A different host's valid artifact is not this host's pin.
        native.chmod(0o400)
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan, self.manifest)
        files = dict(self.runtime_files)
        files.pop('obelisk')
        with self.assertRaises(BindingRefused):
            StagingVerifier(self.plan, self.manifest, canonical(self.parent),
                'f' * 64, files, self.settings, self.verifier.reader)

    async def test_explicit_empty_resource_directory_is_preserved_and_checked(self):
        runtime=self.root/'slice/runtime'
        runtime.chmod(0o700)
        (runtime/'assets').mkdir()
        (runtime/'assets').chmod(0o500)
        runtime.chmod(0o500)
        with self.assertRaises(BindingRefused):
            await self.verifier.verify(self.plan,self.manifest)
        self.runtime_files['slice']={'files':self.runtime_files['slice'],'directories':['assets']}
        verifier=StagingVerifier(self.plan,self.manifest,canonical(self.parent),'f'*64,
            self.runtime_files,self.settings,self.verifier.reader)
        (await verifier.verify(self.plan,self.manifest)).check(self.plan)
        (runtime/'assets').chmod(0o700)
        (runtime/'assets/new_module.py').write_text('unlisted')
        (runtime/'assets/new_module.py').chmod(0o400)
        (runtime/'assets').chmod(0o500)
        with self.assertRaises(BindingRefused):
            await verifier.verify(self.plan,self.manifest)

    def test_runtime_directory_names_are_exact_relative_and_unique(self):
        from .exo_groups.staging import runtime_file_policy
        for directories in ('assets',['../escape'],['/absolute'],['.'],['a//b'],
                            ['assets','assets'],['a/../b'],['bad\x00name']):
            with self.subTest(directories=directories),self.assertRaises(ValueError):
                runtime_file_policy('/runtime',self.runtime_files['slice'],directories=directories)

    def test_runtime_directory_declaration_is_detached_and_complete(self):
        from .exo_groups.staging import runtime_file_policy
        directories=['a/b']
        policy=runtime_file_policy('/runtime',self.runtime_files['slice'],directories=directories)
        directories.append('foreign')
        self.assertEqual(policy['views']['a/b'],[])
        self.assertEqual(policy['views']['a'],['b'])
        self.assertNotIn('foreign',policy['views']['.'])
        supplied=dict(self.runtime_files)
        supplied['slice']={'files':supplied['slice'],'directories':[],'unknown':True}
        with self.assertRaises(BindingRefused):
            StagingVerifier(self.plan,self.manifest,canonical(self.parent),'f'*64,
                supplied,self.settings,self.verifier.reader)


if __name__ == '__main__':
    unittest.main()
