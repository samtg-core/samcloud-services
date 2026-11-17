import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from build_completed_qwen_bundle import validate_source


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


if __name__ == '__main__':
    unittest.main()
