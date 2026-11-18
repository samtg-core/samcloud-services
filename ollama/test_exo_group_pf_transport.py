"""PF guards use synthetic kernel observations; actual PF admission is separate."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .exo_groups.pf_transport import (
    ANCHOR, PFCTL, STOCK_ROOT, LivePFVerifier, PFPolicy, PFUnavailable,
    canonical, clean, interface_addresses, protected_file,
    read_command, scaffold_only, show_commands, validate_snapshot, validate_states,
)


class PFTransportTests(unittest.TestCase):
    def setUp(self):
        self.policy = PFPolicy('wafer', 504, '9e69070a-6fdb-4812-8faa-c2ddbbaa0881', 'en0',
                               ('192.168.1.3', '192.168.1.80', '192.168.1.84'),
                               (52515, 52514, 52513, 52518))
        self.receipt = {
            'schema': 1, 'phase': 'active', 'policy': json.loads(canonical(asdict(self.policy))),
            'policy_sha256': self.policy.policy_sha256,
            'rules_sha256': hashlib.sha256(self.policy.rules().encode()).hexdigest(),
            'canonical_rules': 'block drop in quick inet proto udp from any to self port = 52513\n',
            'enable_token': '1467890231234567890',
            'anchors': sorted(('com.apple', 'com.apple/250.ApplicationFirewall', ANCHOR)),
            'pfctl_sha256': hashlib.sha256(b'pfctl-fixture').hexdigest(),
        }
        self.snapshot = {
            'info': 'Status: Enabled for 0 days 00:00:02 Debug: Urgent\n',
            'root': STOCK_ROOT,
            'anchors': '\n'.join(self.receipt['anchors']) + '\n',
            'interfaces': 'ALL\nen0\nlo0 (skip)\n', 'states': '',
            'references': 'pid 123 (pfctl) token 1467890231234567890\n',
            'anchor:com.apple': 'anchor "250.ApplicationFirewall/*" all\n',
            'anchor:com.apple/250.ApplicationFirewall': '',
            'anchor:' + ANCHOR: self.receipt['canonical_rules'],
        }
        self.interfaces = {'en0': ['192.168.1.80']}

    def validate(self):
        return validate_snapshot(self.policy, self.receipt, self.snapshot,
                                 boot_id=self.policy.boot_id, interfaces=self.interfaces)

    def test_expected_observation_passes_but_does_not_claim_actual_enforcement(self):
        self.assertTrue(self.validate())

    def test_disabled_kernel_from_actual_inspection_is_not_permission(self):
        self.snapshot['info'] = 'Status: Disabled\n'
        with self.assertRaisesRegex(PFUnavailable, 'not enabled'):
            self.validate()

    def test_zero_exit_diagnostic_from_actual_wildcard_query_is_rejected(self):
        with self.assertRaisesRegex(PFUnavailable, 'diagnostic'):
            clean('No ALTQ support in kernel\npfctl: DIOCGETRULES: Invalid argument\n')

    def test_altq_notices_are_the_only_ignored_diagnostics(self):
        self.assertEqual(clean('No ALTQ support in kernel\nALTQ related functions disabled\n' + STOCK_ROOT), STOCK_ROOT)
        for message in ('pfctl: Permission denied', 'unexpected error from kernel'):
            with self.subTest(message=message), self.assertRaises(PFUnavailable):
                clean(message)

    def test_changed_kernel_evaluation_paths_fail_closed(self):
        changes = (
            ('root', 'pass quick all\n' + STOCK_ROOT),
            ('root', 'anchor "other/*" all\n'),
            ('anchors', self.snapshot['anchors'] + 'com.apple/foreign\n'),
            ('anchor:' + ANCHOR, 'pass all\n'),
            ('anchor:com.apple/250.ApplicationFirewall', 'pass quick all\n'),
            ('interfaces', 'ALL (skip)\nen0\n'),
            ('interfaces', 'ALL\nen0 (skip)\n'),
            ('references', 'No pf starter references held\n'),
        )
        original = copy.deepcopy(self.snapshot)
        for name, value in changes:
            with self.subTest(name=name, value=value):
                self.snapshot = {**original, name: value}
                with self.assertRaises(PFUnavailable):
                    self.validate()

    def test_reboot_or_address_change_refuses(self):
        with self.assertRaises(PFUnavailable):
            validate_snapshot(self.policy, self.receipt, self.snapshot,
                              boot_id='00000000-0000-0000-0000-000000000001', interfaces=self.interfaces)
        self.interfaces = {'en0': ['192.168.1.81']}
        with self.assertRaisesRegex(PFUnavailable, 'address changed'):
            self.validate()

    def test_pending_and_removing_receipts_are_not_permission(self):
        for phase in ('pending_enable', 'removing', 'removed', None):
            with self.subTest(phase=phase):
                self.receipt['phase'] = phase
                with self.assertRaises(PFUnavailable):
                    self.validate()

    def test_receipt_must_match_host_policy(self):
        for key, value in (('policy_sha256', '0' * 64), ('rules_sha256', '0' * 64), ('enable_token', '')):
            with self.subTest(key=key):
                altered = {**self.receipt, key: value}
                with self.assertRaises(PFUnavailable):
                    validate_snapshot(self.policy, altered, self.snapshot,
                                      boot_id=self.policy.boot_id, interfaces=self.interfaces)

    def test_anchor_command_injection_and_wildcards_refused(self):
        for name in ('com.apple/*', 'foo -F all', 'foo;id', 'foo\nfoo', '../foo', 'foo//bar'):
            with self.subTest(name=name), self.assertRaises(PFUnavailable):
                show_commands((name,))

    def test_exact_show_commands_cannot_enable_load_flush_or_release(self):
        commands = show_commands(tuple(self.receipt['anchors']))
        self.assertEqual(commands['anchor:' + ANCHOR], ('-a', ANCHOR, '-sr'))
        for arguments in commands.values():
            self.assertFalse({'-f', '-F', '-E', '-e', '-d', '-X', '-k', '-K'} & set(arguments))
            self.assertFalse(any('*' in arg for arg in arguments))

    def test_other_anchor_scaffolding_cannot_hide_filter_rules(self):
        self.assertTrue(scaffold_only('anchor "250.ApplicationFirewall/*" all\n'))
        for rule in ('pass quick all', 'anchor "foo" all quick', 'block in all'):
            with self.subTest(rule=rule):
                self.assertFalse(scaffold_only(rule))

    def test_foreign_reserved_port_states_refuse_but_unrelated_states_survive(self):
        validate_states('all tcp 192.168.1.80:22 <- 10.0.0.8:12345 ESTABLISHED:ESTABLISHED\n', self.policy)
        validate_states('all tcp 192.168.1.80:52514 <- 192.168.1.3:12345 ESTABLISHED:ESTABLISHED\n', self.policy)
        for line in (
            'all tcp 192.168.1.80:52514 <- 192.168.1.45:12345 ESTABLISHED:ESTABLISHED',
            'all udp 192.168.1.80:52513 <- 192.168.1.3:12345 SINGLE:NO_TRAFFIC',
            'all tcp [fe80::1]:52514 <- [fe80::2]:12345 ESTABLISHED:ESTABLISHED',
        ):
            with self.subTest(line=line), self.assertRaises(PFUnavailable):
                validate_states(line, self.policy)

    def test_policy_has_no_network_or_interface_defaults(self):
        for kwargs in ({'interface': 'en0;id'}, {'peers': ('127.0.0.1', '192.168.1.80', '192.168.1.84')},
                       {'ports': (52515, 52514, 52513, True)}, {'uid': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.policy, **kwargs)

    def test_configuration_fingerprint_pins_the_interface_and_apple_binary(self):
        first = LivePFVerifier(self.policy, 'a' * 64).configuration_sha256
        self.assertNotEqual(first, LivePFVerifier(self.policy, 'b' * 64).configuration_sha256)
        self.assertNotEqual(first, LivePFVerifier(replace(self.policy, interface='en1'), 'a' * 64).configuration_sha256)

    def test_unprivileged_or_redirected_receipt_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            path.write_text('{}')
            with self.assertRaises(PFUnavailable):
                protected_file(path, maximum=1024)
            alias = Path(directory) / 'alias'
            alias.symlink_to(path)
            with self.assertRaises(PFUnavailable):
                protected_file(alias, maximum=1024)

    def test_interface_observation_uses_actual_address(self):
        self.assertEqual(interface_addresses('en0: flags=8863\n\tinet 192.168.1.80 netmask 0xffffff00\nlo0: flags=1\n\tinet 127.0.0.1 netmask 0xff000000\n'),
                         {'en0': ['192.168.1.80'], 'lo0': ['127.0.0.1']})

    def test_read_child_failure_and_size_bound_refuse(self):
        with self.assertRaises(PFUnavailable):
            read_command(['/usr/bin/false'])
        self.assertEqual(read_command(['/usr/bin/printf', 'control read']), 'control read')


class LiveObserverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        PFTransportTests.setUp(self)

    def endpoints(self):
        return SimpleNamespace(listeners=tuple(SimpleNamespace(address=ip, port=port, transport=proto)
            for ip, port, proto in (('127.0.0.1', 52515, 'tcp'), ('192.168.1.80', 52514, 'tcp'),
                                   ('0.0.0.0', 52513, 'udp'), ('192.168.1.80', 52518, 'tcp'))))

    def environment(self):
        return {'EXO_RANK_NAME': 'wafer', 'EXO_LEASED_PEER_IPS': ','.join(self.policy.peers),
                **{'EXO_LEASED_' + key + '_PORT': str(value)
                   for key, value in zip(('API', 'PEER', 'DISCOVERY', 'RING'), self.policy.ports)}}

    async def test_wrong_host_or_endpoint_refuses_before_privileged_read(self):
        verifier = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        with patch.object(verifier, '_observe') as observe, patch('os.getuid', return_value=504):
            environment = self.environment()
            environment['EXO_RANK_NAME'] = 'slice'
            with self.assertRaises(PFUnavailable):
                await verifier(self.endpoints(), environment)
            self.assertFalse(observe.called)

    async def test_two_host_runtime_uses_only_exact_slice_obelisk_subset(self):
        policy = replace(self.policy, rank='slice', uid=502)
        verifier = LivePFVerifier(policy, self.receipt['pfctl_sha256'])
        endpoints = self.endpoints()
        for listener in endpoints.listeners:
            if listener.address == self.policy.local_ip:
                listener.address = policy.local_ip
        environment = self.environment() | {
            'EXO_RANK_NAME': 'slice', 'EXO_LEASED_PEER_IPS': '192.168.1.3,192.168.1.84'}
        with patch.object(verifier, '_observe', return_value=True) as observe, patch('os.getuid', return_value=502):
            self.assertTrue(await verifier(endpoints, environment))
            for peers in ('192.168.1.3', '192.168.1.3,192.168.1.80',
                          '192.168.1.84,192.168.1.3', '192.168.1.3,192.168.1.99'):
                with self.subTest(peers=peers), self.assertRaises(PFUnavailable):
                    await verifier(endpoints, environment | {'EXO_LEASED_PEER_IPS': peers})
            self.assertEqual(observe.call_count, 1)
        wafer = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        with patch.object(wafer, '_observe') as observe, patch('os.getuid', return_value=504):
            with self.assertRaises(PFUnavailable):
                await wafer(self.endpoints(), self.environment() | {
                    'EXO_LEASED_PEER_IPS': '192.168.1.3,192.168.1.84'})
            self.assertFalse(observe.called)

    async def test_live_reader_always_queries_kernel_and_rechecks_rules(self):
        verifier = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        calls = []
        commands = {arguments: name for name, arguments in show_commands(tuple(self.receipt['anchors'])).items()}

        def read(args):
            calls.append(args)
            if args == ['/sbin/route', '-n', 'get', '192.168.1.84']:
                return 'route to: 192.168.1.84\n  interface: en0\n'
            if args == ['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid']:
                return self.policy.boot_id + '\n'
            if args == ['/sbin/ifconfig', '-a']:
                return 'en0: flags=1\n\tinet 192.168.1.80 netmask 0xffffff00\n'
            self.assertEqual(args[:4], ['/usr/bin/sudo', '-n', '--', PFCTL])
            return self.snapshot[commands[tuple(args[4:])]]

        def protected(path, **_):
            return b'pfctl-fixture' if str(path) == PFCTL else canonical(self.receipt)

        module = 'ollama.exo_groups.pf_transport.'
        with patch(module + 'read_command', side_effect=read), patch(module + 'protected_file', side_effect=protected), patch('os.getuid', return_value=504), patch('socket.socket') as sockets:
            sockets.return_value.__enter__.return_value.getsockname.return_value = ('192.168.1.80', 49152)
            self.assertTrue(await verifier(self.endpoints(), self.environment()))
            first = len(calls)
            self.assertTrue(await verifier(self.endpoints(), self.environment()))
        self.assertEqual(len(calls), first * 2)  # No saved-snapshot acceptance.
        self.assertEqual(calls.count(['/usr/bin/sudo', '-n', '--', PFCTL, '-sr']), 4)
        self.assertEqual(sockets.call_args.args, (socket.AF_INET, socket.SOCK_DGRAM))
        probe = sockets.return_value.__enter__.return_value
        self.assertEqual(probe.connect.call_count, 4)
        probe.connect.assert_called_with(('192.168.1.84', 52518))
        probe.send.assert_not_called()
        probe.sendto.assert_not_called()
        probe.sendall.assert_not_called()

    async def test_wifi_route_refuses_before_any_privileged_read(self):
        verifier = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        module = 'ollama.exo_groups.pf_transport.'
        with patch(module + 'read_command', return_value='  interface: en1\n') as read, patch(module + 'protected_file') as protected, patch('os.getuid', return_value=504), patch('socket.socket') as sockets:
            with self.assertRaisesRegex(PFUnavailable, 'qualified interface'):
                await verifier(self.endpoints(), self.environment())
            read.assert_called_once_with(['/sbin/route', '-n', 'get', '192.168.1.84'])
            protected.assert_not_called()
            sockets.assert_not_called()

    async def test_wifi_source_refuses_even_when_interface_address_exists(self):
        verifier = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        module = 'ollama.exo_groups.pf_transport.'
        with patch(module + 'read_command', return_value='  interface: en0\n'), patch(module + 'protected_file') as protected, patch('os.getuid', return_value=504), patch('socket.socket') as sockets:
            sockets.return_value.__enter__.return_value.getsockname.return_value = ('192.168.1.104', 49152)
            with self.assertRaisesRegex(PFUnavailable, 'source address'):
                await verifier(self.endpoints(), self.environment())
            protected.assert_not_called()

    async def test_route_probe_failure_is_unconfirmed_not_permission(self):
        verifier = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        module = 'ollama.exo_groups.pf_transport.'
        with patch(module + 'read_command', return_value='  interface: en0\n'), patch('os.getuid', return_value=504), patch('socket.socket', side_effect=OSError('route unavailable')):
            with self.assertRaisesRegex(PFUnavailable, 'unconfirmed'):
                await verifier(self.endpoints(), self.environment())

    async def test_valid_kernel_firewall_cannot_approve_wrong_ring_route(self):
        """Old checks accepted this exact valid-PF/multihomed-route shape."""
        verifier = LivePFVerifier(self.policy, self.receipt['pfctl_sha256'])
        commands = {arguments: name for name, arguments in show_commands(tuple(self.receipt['anchors'])).items()}

        def read(args):
            if args == ['/sbin/route', '-n', 'get', '192.168.1.84']:
                return 'route to: 192.168.1.84\n  interface: en1\n'
            if args == ['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid']:
                return self.policy.boot_id + '\n'
            if args == ['/sbin/ifconfig', '-a']:
                return 'en0: flags=1\n\tinet 192.168.1.80 netmask 0xffffff00\nen1: flags=1\n\tinet 192.168.1.104 netmask 0xffffff00\n'
            self.assertEqual(args[:4], ['/usr/bin/sudo', '-n', '--', PFCTL])
            return self.snapshot[commands[tuple(args[4:])]]

        def protected(path, **_):
            return b'pfctl-fixture' if str(path) == PFCTL else canonical(self.receipt)

        module = 'ollama.exo_groups.pf_transport.'
        with patch(module + 'read_command', side_effect=read), patch(module + 'protected_file', side_effect=protected), patch('os.getuid', return_value=504):
            with self.assertRaisesRegex(PFUnavailable, 'qualified interface'):
                await verifier(self.endpoints(), self.environment())


if __name__ == '__main__':
    unittest.main()
