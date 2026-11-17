import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

p=Path(__file__).with_name('package.py');spec=importlib.util.spec_from_file_location('pf_package_tests',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PackageTests(unittest.TestCase):
    def test_manifest_binds_all_installed_code_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'new';manifest=m.package('slice','/Users/claude-services/var/queue.sqlite',target)
            self.assertEqual(len(list(target.iterdir())),8)
            for name,pin in manifest['files'].items():
                raw=(target/name).read_bytes();self.assertEqual(len(raw),pin['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),pin['sha256'])
            digest=hashlib.sha256((target/'install-manifest.json').read_bytes()).hexdigest()
            self.assertIn(digest,(target/'install.py').read_text())
            with self.assertRaises(FileExistsError):m.package('slice','/Users/claude-services/var/queue.sqlite',target)

    def test_other_home_and_parent_traversal_cannot_become_a_capability(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ('/private/etc/sudoers','/Users/claude-services/../sam/file','relative'):
                with self.subTest(path=path),self.assertRaises(ValueError):m.package('slice',path,Path(d)/'never')
            self.assertFalse((Path(d)/'never').exists())

if __name__=='__main__':unittest.main()
