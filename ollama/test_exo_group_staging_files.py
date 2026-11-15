"""Real file descriptors/hashes and Darwin local-volume observation, no compute."""

import copy
import hashlib
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .exo_groups.staging_files import collect_files, darwin_filesystem, StagingFilesUnavailable


class StagingFileControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        model = self.root / 'rank'
        model.mkdir()
        (model / 'rank.safetensors').write_bytes(b'fixture weights')
        (model / 'model-rank.safetensors').symlink_to('rank.safetensors')
        (model / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        self.policy = {'root': str(self.root), 'files': [], 'views': {
            'rank': ['rank.safetensors', 'model-rank.safetensors', 'config.json']}}
        for name in self.policy['views']['rank']:
            data = (model / name).read_bytes()
            self.policy['files'].append({'path': 'rank/' + name, 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(),
                'link': 'rank.safetensors' if name.startswith('model-') else None})
        self.volume = lambda _: {'local': True, 'filesystem': 'apfs',
            'filesystem_id': [1,2], 'free_bytes': 10**9}

    def tearDown(self):
        for directory in (self.root, self.root / 'rank'):
            if directory.is_dir():
                directory.chmod(0o700)
        self.temp.cleanup()

    def collect(self):
        return collect_files(self.policy, filesystem=self.volume)

    def test_exact_hashes_alias_and_writable_status_are_observed_without_mutation(self):
        before = [(pin['path'], (self.root / pin['path']).stat().st_mtime_ns)
                  for pin in self.policy['files']]
        report = self.collect()
        self.assertFalse(report['readonly'])
        self.assertTrue(report['filesystem']['local'])
        self.assertEqual(report['uid'], os.getuid())
        self.assertEqual(len(report['files']), 3)
        self.assertEqual(before, [(pin['path'], (self.root / pin['path']).stat().st_mtime_ns)
                                 for pin in self.policy['files']])
        for path in (self.root / 'rank').iterdir():
            if not path.is_symlink():
                path.chmod(0o400)
        (self.root / 'rank').chmod(0o500)
        self.root.chmod(0o500)
        self.assertTrue(self.collect()['readonly'])

    def test_changed_weights_refuse_even_with_same_file_size(self):
        (self.root / 'rank/rank.safetensors').write_bytes(b'fixture WEIGHTS')
        with self.assertRaises(StagingFilesUnavailable):
            self.collect()

    def test_extra_loading_file_or_missing_sidecar_refuses(self):
        extra = self.root / 'rank/extra.safetensors'
        extra.write_bytes(b'extra')
        with self.assertRaises(StagingFilesUnavailable):
            self.collect()
        extra.unlink()
        (self.root / 'rank/config.json').unlink()
        with self.assertRaises(OSError):
            self.collect()

    def test_unexpected_symlink_and_redirected_parent_refuse(self):
        target = self.root / 'rank/config.json'
        target.unlink()
        target.symlink_to('rank.safetensors')
        with self.assertRaises(StagingFilesUnavailable):
            self.collect()
        target.unlink()
        target.write_bytes(b'{"model_type":"qwen3"}')
        (self.root / 'rank').rename(self.root / 'real-rank')
        (self.root / 'rank').symlink_to('real-rank')
        with self.assertRaises(OSError):
            self.collect()
        (self.root / 'rank').unlink()
        (self.root / 'real-rank').rename(self.root / 'rank')

    def test_root_alias_is_refused(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        self.policy['root'] = str(alias)
        with self.assertRaises(OSError):
            self.collect()

    def test_invalid_policy_and_duplicate_files_refuse(self):
        for name in ('../escape', '/absolute', 'rank/../escape', 'rank//config.json'):
            policy = copy.deepcopy(self.policy)
            policy['files'][0]['path'] = name
            with self.subTest(name=name), self.assertRaises(StagingFilesUnavailable):
                collect_files(policy, filesystem=self.volume)
        self.policy['files'].append(copy.deepcopy(self.policy['files'][0]))
        with self.assertRaises(StagingFilesUnavailable):
            self.collect()

    def test_change_during_streamed_hash_refuses(self):
        original = os.read
        changed = False
        def read(descriptor, size):
            nonlocal changed
            result = original(descriptor, size)
            if result and not changed:
                changed = True
                (self.root / 'rank/rank.safetensors').write_bytes(b'fixture weights')
            return result
        with patch('os.read', read), self.assertRaises(StagingFilesUnavailable):
            self.collect()

    @unittest.skipUnless(sys.platform == 'darwin', 'Darwin host filesystem ABI')
    def test_actual_darwin_filesystem_reports_this_local_test_volume(self):
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            result = darwin_filesystem(descriptor)
        finally:
            os.close(descriptor)
        self.assertTrue(result['local'])
        self.assertIn(result['filesystem'], ('apfs', 'hfs'))
        self.assertGreater(result['free_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
