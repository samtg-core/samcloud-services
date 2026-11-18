"""Explicit Darwin PF qualification; read-only delegation, no resident helper.

The human-installed receipt is configuration, never live permission. Every
verification invokes only exact, sudoers-whitelisted pfctl show commands and
checks current kernel rules, interfaces and states. This module cannot install,
enable, flush or release PF. No caller paths or command selectors are accepted.
"""

import asyncio
from dataclasses import asdict, dataclass
import hashlib
from ipaddress import IPv4Address, IPv4Network
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from uuid import UUID


ANCHOR = 'com.apple/990.SamcloudEXO'
PFCTL = '/sbin/pfctl'
STOCK_ROOT = 'scrub-anchor "com.apple/*" all fragment reassemble\nanchor "com.apple/*" all\n'
RECEIPT = Path('/Library/Preferences/xyz.samtg.exo-pf/receipt.json')
WARNINGS = {'No ALTQ support in kernel', 'ALTQ related functions disabled'}
RANKS = ('slice', 'wafer', 'obelisk')


class PFUnavailable(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def clean(text):
    """Ignore only the two stock ALTQ notices, never diagnostic errors."""
    lines = [line.strip() for line in text.splitlines() if line.strip() not in WARNINGS]
    if any('pfctl:' in line or 'error' in line.lower() for line in lines):
        raise PFUnavailable('PF diagnostic is not a successful observation')
    return '\n'.join(line for line in lines if line) + ('\n' if any(lines) else '')


@dataclass(frozen=True)
class PFPolicy:
    rank: str
    uid: int
    boot_id: str
    interface: str
    peers: tuple
    ports: tuple  # API, peer, discovery, ring; explicit, no defaults.

    def __post_init__(self):
        private = tuple(IPv4Network(n) for n in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
        if (self.rank not in RANKS or type(self.uid) is not int or self.uid <= 0
                or str(UUID(self.boot_id)) != self.boot_id or UUID(self.boot_id).int == 0
                or not re.fullmatch(r'en\d+', self.interface)
                or not isinstance(self.peers, tuple) or len(self.peers) != 3
                or len(set(self.peers)) != 3
                or any(str(IPv4Address(ip)) != ip or not any(IPv4Address(ip) in n for n in private) for ip in self.peers)
                or not isinstance(self.ports, tuple) or len(self.ports) != 4
                or len(set(self.ports)) != 4
                or any(type(p) is not int or not 1024 <= p <= 65535 for p in self.ports)):
            raise PFUnavailable('exact private host identity, interface, peers and ports required')

    @property
    def local_ip(self):
        return self.peers[RANKS.index(self.rank)]

    @property
    def policy_sha256(self):
        return hashlib.sha256(canonical(asdict(self))).hexdigest()

    def rules(self):
        api, peer, discovery, ring = self.ports
        others = ', '.join(ip for ip in self.peers if ip != self.local_ip)
        local = f'{{ 127.0.0.1, {self.local_ip} }}'
        local_ports = f'{{ {api}, {peer}, {ring} }}'
        peer_ports = f'{{ {peer}, {ring} }}'
        all_ports = '{ ' + ', '.join(map(str, self.ports)) + ' }'
        # Only this reserved port set is filtered. Loopback remains local and
        # the HTTP API has its independently generated private API key.
        return (
            f'pass in quick on lo0 inet proto tcp from {local} to {local} port {local_ports} flags S/SA keep state\n'
            f'pass out quick on lo0 inet proto tcp from {local} to {local} port {local_ports} flags S/SA keep state\n'
            f'pass in quick on {self.interface} inet proto tcp from {{ {others} }} to {self.local_ip} port {peer_ports} flags S/SA keep state\n'
            f'pass out quick on {self.interface} inet proto tcp from {self.local_ip} to {{ {others} }} port {peer_ports} flags S/SA keep state\n'
            # Dynamic self also covers addresses added while a rank is alive.
            # UDP is blocked for multicast destinations as well as local ones.
            f'block in quick inet proto tcp from any to (self) port {all_ports}\n'
            f'block in quick inet6 proto tcp from any to (self) port {all_ports}\n'
            f'block in quick proto udp from any to any port {all_ports}\n'
            f'block out quick proto {{ tcp, udp }} from any to any port {all_ports}\n'
        )


def anchor_names(text):
    names = clean(text).splitlines()
    if (len(names) > 128 or len(set(names)) != len(names)
            or any(not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name)
                   or any(part in ('.', '..') for part in name.split('/')) for name in names)):
        raise PFUnavailable('bounded literal anchor names required')
    return tuple(sorted(names))


def show_commands(anchors):
    if anchor_names('\n'.join(anchors)) != tuple(sorted(anchors)):
        raise PFUnavailable('literal existing anchors required')
    return {
        'info': ('-s', 'info'), 'root': ('-sr',),
        'anchors': ('-s', 'Anchors', '-v'), 'interfaces': ('-v', '-s', 'Interfaces'),
        'states': ('-ss',), 'references': ('-s', 'References'),
        **{'anchor:' + name: ('-a', name, '-sr') for name in anchors},
    }


def scaffold_only(text):
    return all(re.fullmatch(r'(?:scrub-)?anchor "[A-Za-z0-9_./*:-]+" all(?: fragment reassemble)?', line)
               for line in clean(text).splitlines())


def validate_states(text, policy):
    reserved = re.compile(r':(?:' + '|'.join(map(str, policy.ports)) + r')(?=\D|$)')
    for line in clean(text).splitlines():
        if not reserved.search(line):
            continue
        endpoints = re.findall(r'(?<![\d.])((?:\d{1,3}\.){3}\d{1,3}):(\d+)', line)
        if (' tcp ' not in ' ' + line + ' ' or len(endpoints) != 2
                or any(address not in (*policy.peers, '127.0.0.1') for address, _ in endpoints)):
            raise PFUnavailable('a reserved-port state is outside the qualified peers')


def validate_snapshot(policy, receipt, snapshot, *, boot_id, interfaces):
    """Pure validation; only LivePFVerifier supplies an actual live snapshot."""
    if (receipt.get('schema') != 1 or receipt.get('phase') != 'active'
            or receipt.get('policy_sha256') != policy.policy_sha256
            or receipt.get('policy') != json.loads(canonical(asdict(policy)))
            or boot_id.lower() != policy.boot_id or not receipt.get('enable_token', '').isdigit()
            or receipt.get('rules_sha256') != hashlib.sha256(policy.rules().encode()).hexdigest()):
        raise PFUnavailable('installed receipt differs from the exact qualified host')
    if not re.search(r'^Status: Enabled(?:\s|$)', clean(snapshot['info']), re.M):
        raise PFUnavailable('PF is not enabled')
    if clean(snapshot['root']) != STOCK_ROOT:
        raise PFUnavailable('main filter evaluation path changed')
    names = anchor_names(snapshot['anchors'])
    if names != tuple(receipt['anchors']) or ANCHOR not in names:
        raise PFUnavailable('anchor inventory changed')
    expected = clean(receipt['canonical_rules'])
    if not expected or clean(snapshot['anchor:' + ANCHOR]) != expected:
        raise PFUnavailable('live EXO rules differ from the installed rule set')
    for name in names:
        if name != ANCHOR and not scaffold_only(snapshot['anchor:' + name]):
            raise PFUnavailable('another filter anchor can change the verdict')
    skipped = {line.split()[0] for line in clean(snapshot['interfaces']).splitlines() if '(skip)' in line}
    if {'ALL', policy.interface} & skipped:
        raise PFUnavailable('qualified LAN interface bypasses PF')
    current = interfaces.get(policy.interface, ())
    if policy.local_ip not in current:
        raise PFUnavailable('qualified interface address changed')
    token = receipt['enable_token']
    if not re.search(r'(?<!\d)' + re.escape(token) + r'(?!\d)', clean(snapshot['references'])):
        raise PFUnavailable('owned PF reference is absent')
    validate_states(snapshot['states'], policy)
    return True


def protected_file(path, *, maximum):
    path = Path(path)
    for parent in (*reversed(path.parents), path):
        info = parent.lstat()
        if (info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode)
                or os.geteuid() != 0 and os.access(parent, os.W_OK)
                or parent != path and not stat.S_ISDIR(info.st_mode)):
            raise PFUnavailable('root-owned unredirected configuration required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise PFUnavailable('bounded root-owned regular file required')
        raw = os.read(fd, maximum + 1)
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_mode')
        if len(raw) != before.st_size or any(getattr(before, n) != getattr(after, n) for n in fields):
            raise PFUnavailable('configuration changed during read')
        return raw
    finally:
        os.close(fd)


def read_command(arguments):
    """One owned control subprocess, no shell, timeout, retry or compute."""
    process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C'})
    try:
        raw = process.stdout.read(1048577)
        if len(raw) > 1048576:
            process.kill()  # Only this oversized read-only observation process.
            raise PFUnavailable('PF observation exceeds the explicit byte bound')
        if process.wait() != 0:
            raise PFUnavailable('live read-only PF command failed')
        return raw.decode('utf-8', errors='strict')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()


def interface_addresses(text):
    result, name = {}, None
    for line in text.splitlines():
        match = re.match(r'^([A-Za-z0-9]+):', line)
        if match:
            name = match[1]
            result[name] = []
        match = re.match(r'\s+inet (\d+\.\d+\.\d+\.\d+)\b', line)
        if match and name is not None:
            result[name].append(match[1])
    return result


def observe_ring_egress(policy, peers):
    """Check the next ring hop's actual unbound source without sending data.

    MLX binds its listener to the hostfile address, but its outgoing connection
    follows the OS route. Merely finding the intended Ethernet address on an
    interface does not prove that a multihomed host will use it to connect.
    A UDP connect performs the kernel's route/source selection without sending
    a datagram or touching the live ring's TCP listener.
    """
    if (peers != policy.peers and not (policy.rank in ('slice', 'obelisk')
            and peers == (policy.peers[0], policy.peers[2]))):
        raise PFUnavailable('ring route requires the exact configured membership')
    peer = peers[(peers.index(policy.local_ip) + 1) % len(peers)]
    route = read_command(['/sbin/route', '-n', 'get', peer])
    interfaces = re.findall(r'^\s*interface:\s*(\S+)\s*$', route, re.M)
    if interfaces != [policy.interface]:
        raise PFUnavailable('ring peer route does not use the qualified interface')
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((peer, policy.ports[3]))
            source = probe.getsockname()[0]
    except OSError:
        raise PFUnavailable('ring peer source selection is unconfirmed') from None
    if source != policy.local_ip:
        raise PFUnavailable('ring peer route selects an unqualified source address')


class LivePFVerifier:
    """Fixed local receipt, explicit host policy; no fake/saved-report fallback."""

    def __init__(self, policy, pfctl_sha256):
        if not isinstance(policy, PFPolicy) or not re.fullmatch('[a-f0-9]{64}', pfctl_sha256):
            raise PFUnavailable('explicit qualified policy and Apple pfctl pin required')
        self.policy, self.pfctl_sha256 = policy, pfctl_sha256

    @property
    def configuration_sha256(self):
        return hashlib.sha256(canonical({'kind': 'darwin-pf-v1',
            'policy_sha256': self.policy.policy_sha256,
            'pfctl_sha256': self.pfctl_sha256})).hexdigest()

    async def __call__(self, spec, environment):
        p = self.policy
        api, peer, discovery, ring = p.ports
        expected = {('127.0.0.1', api, 'tcp'), (p.local_ip, peer, 'tcp'),
                    ('0.0.0.0', discovery, 'udp'), (p.local_ip, ring, 'tcp')}
        actual = {(v.address, v.port, v.transport) for v in spec.listeners}
        if (os.getuid() != p.uid or actual != expected or len(spec.listeners) != 4
                or environment.get('EXO_RANK_NAME') != p.rank
                or environment.get('EXO_LEASED_PEER_IPS') not in (
                    (','.join(p.peers), ','.join((p.peers[0], p.peers[2])))
                    if p.rank in ('slice', 'obelisk') else (','.join(p.peers),))
                or any(environment.get('EXO_LEASED_' + name + '_PORT') != str(port)
                       for name, port in zip(('API', 'PEER', 'DISCOVERY', 'RING'), p.ports))):
            raise PFUnavailable('live confinement requested for another rank or endpoint')
        peers = tuple(environment['EXO_LEASED_PEER_IPS'].split(','))
        return await asyncio.to_thread(self._observe, peers)

    def _observe(self, peers):
        p = self.policy
        observe_ring_egress(p, peers)
        receipt = json.loads(protected_file(RECEIPT, maximum=262144))
        if hashlib.sha256(protected_file(PFCTL, maximum=4194304)).hexdigest() != self.pfctl_sha256:
            raise PFUnavailable('Apple PF binary changed')
        if receipt.get('pfctl_sha256') != self.pfctl_sha256:
            raise PFUnavailable('installed PF binary pin differs')
        boot = read_command(['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid']).strip()
        snapshot = {name: read_command(['/usr/bin/sudo', '-n', '--', PFCTL, *arguments])
                    for name, arguments in show_commands(tuple(receipt['anchors'])).items()}
        interfaces = interface_addresses(read_command(['/sbin/ifconfig', '-a']))
        validate_snapshot(p, receipt, snapshot, boot_id=boot, interfaces=interfaces)
        # Re-read the evaluation path and owned rules after the other reads.
        for name in ('info', 'root', 'anchors', 'anchor:' + ANCHOR):
            current = read_command(['/usr/bin/sudo', '-n', '--', PFCTL,
                                    *show_commands(tuple(receipt['anchors']))[name]])
            if name == 'info':
                if not re.search(r'^Status: Enabled(?:\s|$)', clean(current), re.M):
                    raise PFUnavailable('PF stopped during observation')
            elif clean(current) != clean(snapshot[name]):
                raise PFUnavailable('PF evaluation path changed during observation')
        if read_command(['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid']).strip() != boot:
            raise PFUnavailable('host rebooted during observation')
        observe_ring_egress(p, peers)
        return True
