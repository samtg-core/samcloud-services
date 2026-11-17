import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import shlex
import subprocess
import sys
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

    def test_admin_command_pins_executed_bytes_and_refuses_changed_installer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);installer=root/'install.py';marker=root/'executed'
            raw=('from pathlib import Path;Path('+repr(str(marker))+').write_text("yes")').encode()
            installer.write_bytes(raw)
            argv=shlex.split(m.install_command(installer,hashlib.sha256(raw).hexdigest()))
            self.assertEqual(argv[:6],['/usr/bin/sudo','/usr/bin/python3','-I','-S','-B','-c'])
            r=subprocess.run([sys.executable,'-I','-S','-B','-c',argv[6],'--install'],capture_output=True)
            self.assertEqual(r.returncode,0);self.assertEqual(marker.read_text(),'yes');marker.unlink()
            installer.write_text('raise AssertionError("modified code must never execute")')
            r=subprocess.run([sys.executable,'-I','-S','-B','-c',argv[6],'--install'],capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertIn(b'installer changed',r.stderr)
            self.assertNotIn(b'modified code must never execute',r.stderr)
            self.assertFalse(marker.exists())

    def test_pinned_bootstrap_does_not_follow_a_redirected_installer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'target';target.write_bytes(b'pass');link=root/'install.py';link.symlink_to(target)
            argv=shlex.split(m.install_command(link,hashlib.sha256(b'pass').hexdigest()))
            r=subprocess.run([sys.executable,'-I','-S','-B','-c',argv[6],'--install'],capture_output=True)
            self.assertNotEqual(r.returncode,0)

    def test_other_home_and_parent_traversal_cannot_become_a_capability(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ('/private/etc/sudoers','/Users/claude-services/../sam/file','relative'):
                with self.subTest(path=path),self.assertRaises(ValueError):m.package('slice',path,Path(d)/'never')
            self.assertFalse((Path(d)/'never').exists())

if __name__=='__main__':unittest.main()
