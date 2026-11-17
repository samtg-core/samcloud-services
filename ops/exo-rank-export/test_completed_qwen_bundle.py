import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from build_completed_qwen_bundle import validate_source, validate_sidecars, write_adapter_view
from checkpoint_metadata import Rank
from rank_bundle import build_bundle, digest
from test_rank_bundle import create_parent


class CompletionControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'checkpoint'
        self.root.mkdir()
        self.addCleanup(self.unseal)
        files = []
        for name, raw in [('config.json', b'{}'), ('model.safetensors', b'weight fixture')]:
            (self.root / name).write_bytes(raw)
            files.append({'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        self.plan = {'uid': os.getuid(), 'source': str(self.root), 'checkpoint': {
            'model_id': 'fixture/qwen3', 'revision': 'a' * 40,
            'plan_sha256': 'b' * 64, 'weights_bytes': len(b'weight fixture'), 'files': files}}
        self.receipt = dict(self.plan['checkpoint'])
        self.receipt['load_or_fit_claim'] = False
        self.write_receipt()
        (self.root / 'STAGING-INTENT.json').write_text(json.dumps({'plan_sha256': 'b' * 64}))
        self.seal()

    def seal(self):
        for p in self.root.iterdir():
            p.chmod(0o444)
        self.root.chmod(0o555)

    def unseal(self):
        self.root.chmod(0o700)
        for p in self.root.iterdir():
            if not p.is_symlink():
                p.chmod(0o600)

    def write_receipt(self):
        (self.root / 'COMPLETE.json').write_text(json.dumps(self.receipt))

    def test_sealed_complete_source_accepts_without_claiming_weight_rehash(self):
        result = validate_source(self.plan)
        self.assertTrue(result['metadata_verified'])
        self.assertIn('by build_bundle', result['weight_hashes'])

    def test_partial_download_refuses_before_output_is_created(self):
        self.unseal()
        (self.root / 'COMPLETE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'not sealed'):
            validate_source(self.plan)
        self.assertEqual(set(self.root.parent.iterdir()), {self.root})

    def test_receipt_cannot_omit_a_weight_or_change_its_digest(self):
        self.unseal()
        self.receipt['files'] = [dict(r) for r in self.receipt['files']]
        self.receipt['files'][1]['sha256'] = 'c' * 64
        self.write_receipt()
        self.seal()
        with self.assertRaisesRegex(ValueError, 'every pinned file'):
            validate_source(self.plan)

    def test_changed_config_refuses_even_with_matching_completion_receipt(self):
        self.unseal()
        (self.root / 'config.json').write_bytes(b'[]')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'metadata hash differs'):
            validate_source(self.plan)

    def test_symlink_weight_refuses(self):
        self.unseal()
        p = self.root / 'model.safetensors'
        outside = self.root.parent / 'outside'
        outside.write_bytes(p.read_bytes())
        outside.chmod(0o444)
        p.unlink()
        p.symlink_to(outside)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'regular file'):
            validate_source(self.plan)

    def test_extra_partial_file_refuses(self):
        self.unseal()
        (self.root / 'stray.part').write_bytes(b'partial')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'extra entries'):
            validate_source(self.plan)


class AdapterViewControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.parent, self.index, self.catalogue, _ = create_parent(self.source)
        rows = []
        for name, value in [('config.json', self.parent['config']),
                            ('tokenizer.json', {}), ('tokenizer_config.json', {})]:
            raw = json.dumps(value).encode()
            (self.source / name).write_bytes(raw)
            rows.append({'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        self.parent['sidecar_files'] = rows
        self.rows = [dict(row) for row in rows]

    def test_missing_or_changed_sidecar_pin_refuses(self):
        self.parent['sidecar_files'] = []
        with self.assertRaisesRegex(ValueError, 'config/tokenizer'):
            validate_sidecars(self.parent, self.rows, self.source)
        self.parent['sidecar_files'] = [dict(row) for row in self.rows]
        self.parent['sidecar_files'][0]['sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'inventory'):
            validate_sidecars(self.parent, self.rows, self.source)

    def test_adapter_view_contains_only_assigned_weights_and_pinned_sidecars(self):
        validate_sidecars(self.parent, self.rows, self.source)
        ranks = [Rank('first', 0, 1, 100000, 100000), Rank('second', 1, 4, 100000, 100000)]
        dest = self.root / 'bundle'
        manifest = build_bundle(self.source, dest, self.parent, self.index,
                                self.catalogue, ranks, scratch_reserve_bytes=0)
        raw = json.dumps(self.parent).encode()
        write_adapter_view(self.source, dest, self.parent, raw)
        self.assertEqual((dest / 'parent-manifest.json').read_bytes(), raw)
        self.assertEqual(digest(self.parent), manifest['parent_manifest_sha256'])
        for rank in ranks:
            directory = dest / rank.name
            self.assertEqual(set(p.name for p in directory.iterdir()), {
                'rank.safetensors', 'model-rank.safetensors', *(r['name'] for r in self.rows)})
            self.assertEqual((directory / 'model-rank.safetensors').readlink(), Path('rank.safetensors'))
            for row in self.rows:
                self.assertEqual(hashlib.sha256((directory / row['name']).read_bytes()).hexdigest(), row['sha256'])


if __name__ == '__main__':
    unittest.main()
