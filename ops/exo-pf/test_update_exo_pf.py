"""Offline fault controls for the attended anchor replacement; no real PF calls."""
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.u = load('update_pf_test', BASE / 'update-exo-pf.py')
        self.m = load('transport_pf_test', BASE.parents[1] / 'ollama/exo_groups/pf_transport.py')
        self.u.m = self.m
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.u.ROOT = Path(self.temp.name)
        self.u.SUDOERS = self.u.ROOT / 'sudoers.fixture'
        self.old = self.m.PFPolicy('wafer', os.getuid(), '9e69070a-6fdb-4812-8faa-c2ddbbaa0881',
            'en0', ('192.168.1.3', '192.168.1.80', '192.168.1.84'), (52515, 52514, 52513, 52518))
        self.new = replace(self.old, interface='en7', peers=('192.168.1.3', '192.168.1.19', '192.168.1.84'))
        self.receipt = {'schema': 1, 'phase': 'active', 'policy': json.loads(self.m.canonical(asdict(self.old))),
            'policy_sha256': self.old.policy_sha256, 'rules_sha256': self.u.sha(self.old.rules().encode()),
            'canonical_rules': 'old canonical rules\n', 'enable_token': '123456789',
            'anchors': ['com.apple', self.m.ANCHOR], 'pfctl_sha256': 'a' * 64,
            'sudoers_sha256': self.u.sha(b'read-only fixture')}
        self.before = {'info': 'Status: Enabled\n', 'root': self.m.STOCK_ROOT,
            'anchors': '\n'.join(self.receipt['anchors']) + '\n', 'interfaces': 'ALL\nen0\nen7\n',
            'states': '', 'references': 'pid 123 token 123456789\n', 'anchor:com.apple': '',
            'anchor:' + self.m.ANCHOR: 'old canonical rules\n'}
        self.after = {**self.before, 'anchor:' + self.m.ANCHOR: 'new canonical rules\n'}
        self.values = {'receipt.json': self.m.canonical(self.receipt),
            'policy.json': self.m.canonical(asdict(self.old)), 'rules.conf': self.old.rules().encode()}
        for name, raw in self.values.items():
            (self.u.ROOT / name).write_bytes(raw)
        self.u.SUDOERS.write_bytes(b'read-only fixture')
        self.plan = {'before_policy': asdict(self.old), 'after_policy': asdict(self.new)}

    def test_transition_cannot_change_host_boot_uid_or_ports(self):
        self.u.validate_transition(self.old, self.new)
        for changed in (self.old, replace(self.new, uid=os.getuid() + 1),
                        replace(self.new, rank='slice'), replace(self.new, ports=(52515, 52514, 52513, 52519)),
                        replace(self.new, boot_id='00000000-0000-0000-0000-000000000001')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.u.validate_transition(self.old, changed)

    def test_old_kernel_is_checked_without_inventing_an_old_live_address(self):
        self.u.validate_old_kernel(self.old, self.receipt, self.before)
        for key, value in (('root', 'pass all\n'), ('info', 'Status: Disabled\n'),
                           ('references', 'No pf starter references held\n'),
                           ('anchor:' + self.m.ANCHOR, 'pass all\n'),
                           ('anchor:com.apple', 'block all\n')):
            changed = {**self.before, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.u.validate_old_kernel(self.old, self.receipt, changed)

    def test_pending_receipt_is_not_previous_permission(self):
        with self.assertRaises(ValueError):
            self.u.validate_old_kernel(self.old, {**self.receipt, 'phase': 'updating'}, self.before)

    def test_only_exact_frozen_gateway_closed_loopback_api_client_is_allowed(self):
        fence = {'pid': 777, 'identity': str(os.getuid()) + ' exact birth /gateway'}
        header = 'COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n'
        row = 'Python 777 fixture 10u IPv4 0x123 0t0 TCP 127.0.0.1:60007->127.0.0.1:52515 (CLOSED)\n'
        def read(argv):
            if argv[-1] == 'stat=':
                return 'T'
            return fence['identity']
        with patch.object(self.m, 'read_command', side_effect=read):
            self.assertTrue(self.u.closed_gateway_clients(self.new,
                SimpleNamespace(returncode=0, stdout=header + row, stderr=''), fence))
            for bad in [row.replace('(CLOSED)', '(' + s + ')') for s in ('ESTABLISHED', 'CLOSE_WAIT', 'LISTEN', 'TIME_WAIT')
                        ] + [row.replace('Python 777', 'Python 778'),
                             row.replace('52515', '52514'), row.replace('127.0.0.1:52515', '192.168.1.19:52515'),
                             row.replace('IPv4', 'IPv6'), row.replace('TCP', 'UDP')]:
                with self.subTest(row=bad):
                    self.assertFalse(self.u.closed_gateway_clients(self.new,
                        SimpleNamespace(returncode=0, stdout=header + bad, stderr=''), fence))
            self.assertFalse(self.u.closed_gateway_clients(self.new,
                SimpleNamespace(returncode=0, stdout=header + row, stderr=''), None))

    def test_actual_macos_closed_descriptor_status_one_is_not_a_live_socket(self):
        fence = {'pid': 25652, 'identity': str(os.getuid()) + ' exact birth /gateway'}
        output = ('COMMAND   PID            USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME\n'
                  'Python  25652 claude-services   10u  IPv4 0xdd3b91b2faa0c041      0t0  TCP '
                  '127.0.0.1:60007->127.0.0.1:52515 (CLOSED)\n')
        with patch.object(self.m, 'read_command', side_effect=[fence['identity'], 'T']):
            self.assertTrue(self.u.closed_gateway_clients(self.new,
                SimpleNamespace(returncode=1, stdout=output, stderr=''), fence))
        for rc, stdout, stderr in [(2, output, ''), (1, '', ''), (1, output, 'permission denied'),
                                   (1, output.replace('(CLOSED)', '(ESTABLISHED)'), '')]:
            with self.subTest(rc=rc, stdout=stdout, stderr=stderr), patch.object(self.m, 'read_command',
                    side_effect=[fence['identity'], 'T']):
                self.assertFalse(self.u.closed_gateway_clients(self.new,
                    SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr), fence))

    def test_running_or_changed_gateway_never_gets_closed_client_exception(self):
        fence = {'pid': 777, 'identity': str(os.getuid()) + ' exact birth /gateway'}
        reply = SimpleNamespace(returncode=0, stdout='ignored', stderr='')
        for identity, status in ((fence['identity'], 'S'), (fence['identity'] + 'changed', 'T')):
            with self.subTest(identity=identity, status=status), patch.object(self.m, 'read_command',
                    side_effect=[identity, status]):
                self.assertFalse(self.u.closed_gateway_clients(self.new, reply, fence))

    def test_closed_descriptor_does_not_bypass_the_independent_pf_state_check(self):
        fence = {'pid': 777, 'identity': str(os.getuid()) + ' exact birth /gateway'}
        reply = SimpleNamespace(returncode=0, stderr='', stdout=
            'COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n'
            'Python 777 fixture 10u IPv4 0x123 0t0 TCP 127.0.0.1:60007->127.0.0.1:52515 (CLOSED)\n')
        for states in ('', 'all tcp 192.168.1.3:52518 <- 192.168.1.19:60000 ESTABLISHED:ESTABLISHED'):
            def read(argv):
                if argv[-1] == 'stat=':
                    return 'T'
                if argv[-1] == '-ss':
                    return states
                return fence['identity']
            with patch.object(self.m, 'read_command', side_effect=read), \
                    patch.object(self.u.subprocess, 'run', return_value=reply):
                if states:
                    with self.assertRaisesRegex(ValueError, 'EXO state remains'):
                        self.u.no_exo_sockets(self.new, frozen_gateway=fence)
                else:
                    self.u.no_exo_sockets(self.new, frozen_gateway=fence)

    def test_changed_receipt_binary_or_delegation_refuses_before_writes(self):
        binary = self.u.ROOT / 'pfctl.fixture'
        binary.write_bytes(b'qualified binary')
        self.receipt['pfctl_sha256'] = self.u.sha(binary.read_bytes())
        self.values['receipt.json'] = self.m.canonical(self.receipt)
        (self.u.ROOT / 'receipt.json').write_bytes(self.values['receipt.json'])
        self.plan.update(before_sha256={k: self.u.sha(v) for k, v in self.values.items()},
                         pfctl_sha256=self.receipt['pfctl_sha256'])
        with patch.object(self.m, 'PFCTL', str(binary)), \
             patch.object(self.m, 'protected_file', side_effect=lambda path, **kwargs: Path(path).read_bytes()):
            self.assertEqual(self.u.read_installation(self.plan), (self.values, self.receipt))
            for path in (self.u.ROOT / 'receipt.json', binary, self.u.SUDOERS):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.u.read_installation(self.plan)
                path.write_bytes(original)

    def test_another_owner_reference_or_rule_change_refuses(self):
        for key in ('root', 'anchors', 'references', 'anchor:com.apple'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.u.unchanged_other_rules(self.before, {**self.after, key: 'changed\n'}, self.receipt['anchors'])

    def run_fenced(self, failure=None):
        out = self.u.ROOT / 'operation'
        out.mkdir()
        snapshots = [copy.deepcopy(self.before), copy.deepcopy(self.before),
                     copy.deepcopy(self.after), copy.deepcopy(self.after)]
        commands = []

        def action(directory, name, argv):
            commands.append(argv)
            if name == failure:
                raise RuntimeError('simulated lost/failed load response')
            return SimpleNamespace(stdout='new canonical rules\n')

        with patch.object(self.u, 'read_installation', return_value=(self.values, self.receipt)), \
             patch.object(self.u, 'snapshot', side_effect=snapshots), \
             patch.object(self.u, 'live_network', return_value=(self.new.boot_id, {'en7': [self.new.local_ip]})), \
             patch.object(self.u, 'no_exo_sockets'), patch.object(self.u, 'action', side_effect=action), \
             patch.object(self.m, 'protected_file', side_effect=lambda path, **kwargs: Path(path).read_bytes()):
            if failure:
                with self.assertRaises(RuntimeError):
                    self.u.apply_fenced(self.plan, out)
            else:
                self.u.apply_fenced(self.plan, out)
        return out, commands

    def test_success_preserves_enable_and_delegation_and_changes_only_own_anchor(self):
        out, commands = self.run_fenced()
        receipt = json.loads((self.u.ROOT / 'receipt.json').read_bytes())
        self.assertEqual(receipt['enable_token'], self.receipt['enable_token'])
        self.assertEqual(receipt['sudoers_sha256'], self.receipt['sudoers_sha256'])
        self.assertEqual(self.u.SUDOERS.read_bytes(), b'read-only fixture')
        self.assertEqual(receipt['phase'], 'active')
        self.assertEqual(receipt['policy_sha256'], self.new.policy_sha256)
        self.assertEqual((self.u.ROOT / 'policy.json').read_bytes(), self.m.canonical(asdict(self.new)))
        self.assertEqual((self.u.ROOT / 'rules.conf').read_bytes(), self.new.rules().encode())
        self.assertTrue((out / 'verified.json').is_file())
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(argv[:1] == [self.m.PFCTL] and argv[argv.index('-a') + 1] == self.m.ANCHOR for argv in commands))
        self.assertFalse(any(set(argv) & {'-E', '-e', '-d', '-X', '-F', '-k', '-K'} for argv in commands))

    def test_unknown_load_leaves_pending_and_never_retries_or_releases_enable(self):
        out, commands = self.run_fenced('load-own-rules')
        receipt = json.loads((self.u.ROOT / 'receipt.json').read_bytes())
        self.assertEqual(receipt['phase'], 'updating')
        self.assertEqual(receipt['enable_token'], self.receipt['enable_token'])
        self.assertEqual((self.u.ROOT / 'rules.conf').read_bytes(), self.old.rules().encode())
        self.assertFalse((out / 'verified.json').exists())
        self.assertEqual(len(commands), 2)

    def test_parse_failure_never_marks_receipt_pending(self):
        self.run_fenced('parse-own-rules')
        self.assertEqual((self.u.ROOT / 'receipt.json').read_bytes(), self.values['receipt.json'])

    def test_atomic_receipt_replacement_does_not_follow_existing_temporary_link(self):
        victim = self.u.ROOT / 'victim'
        victim.write_text('keep')
        (self.u.ROOT / '.receipt.json.test.new').symlink_to(victim)
        with self.assertRaises(FileExistsError):
            self.u.replace_protected(self.u.ROOT / 'receipt.json', b'change', 'test')
        self.assertEqual(victim.read_text(), 'keep')
        self.assertEqual((self.u.ROOT / 'receipt.json').read_bytes(), self.values['receipt.json'])

    def journal(self, state=None):
        path = self.u.ROOT / 'queue.sqlite'
        c = sqlite3.connect(path)
        c.executescript('CREATE TABLE executions(id TEXT,state TEXT); CREATE TABLE grants(state TEXT,active_execution TEXT,idle_since REAL);')
        if state:
            c.execute('INSERT INTO executions VALUES (?,?)', ('preserve', state))
        c.commit()
        c.close()
        info = path.stat()
        self.plan['gateway_fence'] = {'pid': 777, 'identity': str(os.getuid()) + ' pinned birth /fixture',
            'journal': str(path), 'journal_inode': [info.st_dev, info.st_ino]}
        return path

    def test_queued_running_and_uncertain_work_refuse_before_any_signal(self):
        for state in ('queued', 'running', 'uncertain'):
            with self.subTest(state=state):
                path = self.journal(state)
                with patch.object(self.u.os, 'kill') as kill, self.assertRaisesRegex(ValueError, 'not drained'):
                    with self.u.quiet_gateway(self.plan, self.new, self.u.ROOT):
                        self.fail('must not enter')
                kill.assert_not_called()
                reader = sqlite3.connect(path)
                try:
                    self.assertEqual(reader.execute('SELECT * FROM executions').fetchall(), [('preserve', state)])
                finally:
                    reader.close()
                path.unlink()

    def test_real_sqlite_barrier_and_stop_span_rule_work_resume_on_exception(self):
        path = self.journal()
        out = self.u.ROOT / 'fence'
        out.mkdir()
        seen = []

        def kill(pid, sig):
            self.assertTrue((out / ('gateway-stop-intent.json' if sig == signal.SIGSTOP else 'gateway-resume-intent.json')).is_file())
            seen.append((pid, sig))

        def read(argv):
            return 'T' if argv[-1] == 'stat=' else self.plan['gateway_fence']['identity']

        listener = SimpleNamespace(returncode=0, stdout='777\n', stderr='')
        with patch.object(self.u.os, 'kill', side_effect=kill), patch.object(self.m, 'read_command', side_effect=read), \
             patch.object(self.u.subprocess, 'run', return_value=listener):
            with self.assertRaisesRegex(RuntimeError, 'simulated rule failure'):
                with self.u.quiet_gateway(self.plan, self.new, out):
                    rival = sqlite3.connect(path, timeout=0)
                    with self.assertRaisesRegex(sqlite3.OperationalError, 'locked'):
                        rival.execute('BEGIN IMMEDIATE')
                    rival.close()
                    raise RuntimeError('simulated rule failure')
        self.assertEqual(seen, [(777, signal.SIGSTOP), (777, signal.SIGCONT)])
        rival = sqlite3.connect(path, timeout=0)
        rival.execute('BEGIN IMMEDIATE')
        rival.rollback()
        rival.close()
        self.assertTrue((out / 'gateway-resumed.json').is_file())

    def test_full_audit_disk_cannot_leave_the_pinned_gateway_stopped(self):
        path = self.journal()
        out = self.u.ROOT / 'fence'
        out.mkdir()
        write = self.u.write_new
        seen = []

        def failing_write(target, raw, mode=0o400):
            if target.name == 'gateway-resume-intent.json':
                raise OSError('simulated full audit disk')
            write(target, raw, mode)

        def read(argv):
            return 'T' if argv[-1] == 'stat=' else self.plan['gateway_fence']['identity']

        listener = SimpleNamespace(returncode=0, stdout='777\n', stderr='')
        with patch.object(self.u.os, 'kill', side_effect=lambda pid, sig: seen.append((pid, sig))), \
             patch.object(self.m, 'read_command', side_effect=read), \
             patch.object(self.u.subprocess, 'run', return_value=listener), \
             patch.object(self.u, 'write_new', side_effect=failing_write):
            with self.assertRaisesRegex(OSError, 'full audit disk'):
                with self.u.quiet_gateway(self.plan, self.new, out):
                    pass
        self.assertEqual(seen, [(777, signal.SIGSTOP), (777, signal.SIGCONT)])
        rival = sqlite3.connect(path, timeout=0)
        rival.execute('BEGIN IMMEDIATE')
        rival.rollback()
        rival.close()

    def test_unprivileged_entrypoint_refuses_without_loading_or_mutating(self):
        if os.geteuid() == 0:
            self.skipTest('requires an unprivileged runner')
        reply = subprocess.run([sys.executable, '-I', '-B', str(BASE / 'update-exo-pf.py'), '--apply'], capture_output=True, text=True)
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn('Human administrator only', reply.stderr)


if __name__ == '__main__':
    unittest.main()
