"""Human-run replacement of one installed EXO anchor policy, after ranks drain.

Retains the existing enable reference, read-only delegation and main ruleset.
An incomplete operation leaves an updating receipt: native admission refuses.
No automatic retry, rollback, global flush, enable or disable operation exists.
"""
from dataclasses import asdict
from contextlib import contextmanager
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.parse

PLAN_SHA256 = 'REPLACE_WITH_EXACT_PLAN_SHA256'
MODULE_SHA256 = 'REPLACE_WITH_EXACT_MODULE_SHA256'
ROOT = Path('/Library/Preferences/xyz.samtg.exo-pf')
ATTEMPT_ID = None
DIAGNOSTIC_DIRECTORY = None
FENCE_ALIVE = None
SUDOERS = Path('/private/etc/sudoers.d/90-samcloud-exo-read')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def payload(name, pin):
    path = Path(__file__).parent / name
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 262144:
            raise ValueError('bounded unredirected payload required')
        raw = os.read(fd, 262145)
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if sha(raw) != pin or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError('pinned payload changed')
        return raw
    finally:
        os.close(fd)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, raw, mode=0o400):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)
    fsync_dir(path.parent)


def replace_protected(path, raw, suffix):
    temporary = path.with_name('.' + path.name + '.' + suffix + '.new')
    write_new(temporary, raw, 0o444)
    os.replace(temporary, path)
    fsync_dir(path.parent)


def policy(value):
    value = dict(value)
    value['peers'], value['ports'] = tuple(value['peers']), tuple(value['ports'])
    return m.PFPolicy(**value)


def validate_transition(old, new):
    if (old == new or any(getattr(old, k) != getattr(new, k)
                          for k in ('rank', 'uid', 'boot_id', 'ports'))):
        raise ValueError('only an explicit peer/interface policy change on this host and boot is permitted')


def show(args):
    return m.read_command([m.PFCTL, *args])


def snapshot(anchors):
    return {name: show(args) for name, args in m.show_commands(tuple(anchors)).items()}


def closed_gateway_clients(p, reply, fence):
    """A frozen gateway's CLOSED loopback API descriptors are not listeners.

    No live TCP state, native peer socket, other process, or unpinned gateway
    receives this exception. The queue writer barrier is held by the caller.
    """
    if not isinstance(fence, dict) or type(fence.get('pid')) is not int:
        return False
    pid = fence['pid']
    identity = m.read_command(['/bin/ps', '-p', str(pid), '-o', 'uid=,lstart=,comm=']).strip()
    status = m.read_command(['/bin/ps', '-p', str(pid), '-o', 'stat=']).strip()
    if identity != fence.get('identity') or identity.split()[0] != str(p.uid) or 'T' not in status:
        return False
    lines = reply.stdout.strip().splitlines()
    if (reply.returncode not in (0, 1) or reply.stderr.strip() or len(lines) < 2
            or lines[0].split() != 'COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME'.split()):
        return False
    for line in lines[1:]:
        fields = line.split(None, 8)
        if (len(fields) != 9 or fields[1] != str(pid) or fields[4] != 'IPv4' or fields[7] != 'TCP'):
            return False
        match = re.fullmatch(r'127\.0\.0\.1:(\d+)->127\.0\.0\.1:' + str(p.ports[0]) + r' \(CLOSED\)', fields[8])
        if match is None or not 1024 <= int(match[1]) <= 65535 or int(match[1]) in p.ports:
            return False
    return True


def no_exo_sockets(p, *, frozen_gateway=None):
    if FENCE_ALIVE is not None and not FENCE_ALIVE():
        raise ValueError('owned service writer barrier exited; no PF mutation permitted')
    argv = ['/usr/sbin/lsof', '-nP']
    for port in p.ports:
        argv.extend(('-iTCP:' + str(port), '-iUDP:' + str(port)))
    reply = subprocess.run(argv, capture_output=True, text=True)
    if DIAGNOSTIC_DIRECTORY is not None:
        record = {'rc': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr}
        with (DIAGNOSTIC_DIRECTORY / 'socket-observations.ndjson').open('a') as stream:
            stream.write(json.dumps(record) + '\n'); stream.flush(); os.fsync(stream.fileno())
    absent = reply.returncode == 1 and not reply.stdout.strip() and not reply.stderr.strip()
    if not absent and not closed_gateway_clients(p, reply, frozen_gateway):
        raise ValueError('EXO port occupied or absence observation failed')
    states = m.clean(show(('-ss',)))
    if re.search(r':(?:' + '|'.join(map(str, p.ports)) + r')(?=\D|$)', states):
        raise ValueError('EXO state remains; drain before changing policy')


def live_network(p):
    boot = m.read_command(['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid']).strip().lower()
    interfaces = m.interface_addresses(m.read_command(['/sbin/ifconfig', '-a']))
    if boot != p.boot_id or p.local_ip not in interfaces.get(p.interface, ()):
        raise ValueError('new policy does not match the live host/interface')
    for peer in p.peers:
        if peer == p.local_ip:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((peer, p.ports[2]))  # Kernel route lookup, no send.
            if route.getsockname()[0] != p.local_ip:
                raise ValueError('outgoing route source differs from new policy')
    return boot, interfaces


def validate_old_kernel(old, receipt, current):
    """Validate existing enforcement without pretending its old IP is present.

    The replacement exists precisely because the old address/interface changed.
    New addresses and routes are independently checked against live observations.
    """
    if (receipt.get('schema') != 1 or receipt.get('phase') != 'active'
            or receipt.get('policy') != json.loads(m.canonical(asdict(old)))
            or receipt.get('policy_sha256') != old.policy_sha256
            or receipt.get('rules_sha256') != sha(old.rules().encode())
            or not re.fullmatch(r'\d+', receipt.get('enable_token', ''))):
        raise ValueError('old active receipt is not the exact installed policy')
    if (not re.search(r'^Status: Enabled(?:\s|$)', m.clean(current['info']), re.M)
            or m.clean(current['root']) != m.STOCK_ROOT
            or m.anchor_names(current['anchors']) != tuple(receipt['anchors'])
            or m.ANCHOR not in receipt['anchors']
            or not m.clean(receipt['canonical_rules'])
            or m.clean(current['anchor:' + m.ANCHOR]) != m.clean(receipt['canonical_rules'])
            or not re.search(r'(?<!\d)' + re.escape(receipt['enable_token']) + r'(?!\d)',
                             m.clean(current['references']))):
        raise ValueError('old kernel rules, evaluation path or enable reference changed')
    for name in receipt['anchors']:
        if name != m.ANCHOR and not m.scaffold_only(current['anchor:' + name]):
            raise ValueError('another anchor has effective filter rules')


def unchanged_other_rules(before, after, anchors):
    keys = ('root', 'anchors', 'references', *('anchor:' + a for a in anchors if a != m.ANCHOR))
    if any(m.clean(before[k]) != m.clean(after[k]) for k in keys):
        raise ValueError('another PF owner changed the evaluation path or enable references')


def read_installation(plan):
    values = {}
    for name in ('receipt.json', 'policy.json', 'rules.conf'):
        raw = m.protected_file(ROOT / name, maximum=262144)
        if sha(raw) != plan['before_sha256'][name]:
            raise ValueError('installed ' + name + ' changed')
        values[name] = raw
    receipt = json.loads(values['receipt.json'])
    binary = sha(m.protected_file(m.PFCTL, maximum=4194304))
    if binary != plan['pfctl_sha256'] or binary != receipt['pfctl_sha256']:
        raise ValueError('Apple PF binary changed')
    if sha(m.protected_file(SUDOERS, maximum=262144)) != receipt['sudoers_sha256']:
        raise ValueError('existing read-only delegation changed')
    return values, receipt


@contextmanager
def quiet_gateway(plan, new, out):
    """Exclude a launch racing the empty-port check; never stop a live turn.

    The SQLite writer lock is acquired before SIGSTOP. One connection stays
    open throughout; opening/closing a second FD would drop POSIX locks.
    No queue rows, grant state or model process are changed.
    """
    fence = plan['gateway_fence']
    pid = fence['pid']
    if type(pid) is not int or pid <= 1:
        raise ValueError('one explicitly pinned gateway process required')

    def identity():
        value = m.read_command(['/bin/ps', '-p', str(pid), '-o', 'uid=,lstart=,comm=']).strip()
        if value != fence['identity'] or value.split()[0] != str(new.uid):
            raise ValueError('gateway process identity changed')
        listener = subprocess.run(['/usr/sbin/lsof', '-nP', '-t', '-iTCP:8800', '-sTCP:LISTEN'],
                                  capture_output=True, text=True)
        if listener.returncode != 0 or listener.stderr.strip() or set(listener.stdout.split()) != {str(pid)}:
            raise ValueError('gateway listener changed')

    journal = Path(fence['journal'])
    before = journal.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid != new.uid or [before.st_dev, before.st_ino] != fence['journal_inode']):
        raise ValueError('gateway journal identity changed')
    connection = sqlite3.connect('file:' + urllib.parse.quote(str(journal), safe='/') + '?mode=rw',
                                 uri=True, timeout=10)
    stopped = False
    try:
        if connection.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
            raise ValueError('qualified rollback journal required')
        connection.execute('BEGIN IMMEDIATE')
        if connection.execute("SELECT 1 FROM executions WHERE state NOT IN ('complete','failed','cancelled') LIMIT 1").fetchone():
            raise ValueError('ordinary work has not drained')
        held = connection.execute("SELECT state,active_execution,idle_since FROM grants WHERE state IN ('granted','uncertain')").fetchall()
        if any(state != 'granted' or active is not None or idle is None or idle > time.time() - 120
               for state, active, idle in held):
            raise ValueError('seat turn is still active')
        identity()
        write_new(out / 'gateway-stop-intent.json', m.canonical({'pid': pid,
            'identity': fence['identity'], 'queue_writer_barrier': True}))
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        identity()
        # SIGSTOP delivery is asynchronous; require a kernel-observed stopped
        # process before any receipt/rule mutation. Refusal still resumes it.
        status = m.read_command(['/bin/ps', '-p', str(pid), '-o', 'stat=']).strip()
        if 'T' not in status:
            raise ValueError('gateway stop not observed')
        current = journal.lstat()
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('journal replaced during fence')
        write_new(out / 'gateway-stopped.json', m.canonical({'pid': pid, 'status': status}))
        yield
    finally:
        try:
            if stopped:
                # Identity is pinned again before signalling, never a reused PID.
                value = m.read_command(['/bin/ps', '-p', str(pid), '-o', 'uid=,lstart=,comm=']).strip()
                if value != fence['identity']:
                    raise RuntimeError('gateway identity changed while stopped; inspect without signalling')
                try:
                    write_new(out / 'gateway-resume-intent.json', m.canonical({'pid': pid, 'identity': value}))
                finally:
                    # A full audit disk must not strand an otherwise unchanged
                    # stopped gateway. Resuming this pinned process is cleanup.
                    os.kill(pid, signal.SIGCONT)
                write_new(out / 'gateway-resumed.json', m.canonical({'pid': pid}))
        finally:
            connection.rollback()
            connection.close()


def action(out, name, argv):
    write_new(out / (name + '-intent.json'), m.canonical({'argv': argv,
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    reply = subprocess.run(argv, capture_output=True, text=True,
        env={'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C'})
    write_new(out / (name + '-reply.json'), m.canonical({
        'rc': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr}))
    if reply.returncode:
        raise RuntimeError(name + ' unconfirmed: inspect saved reply; do not repeat')
    return reply


def create_output(new):
    updates = ROOT / 'policy-updates'
    if updates.exists():
        if not stat.S_ISDIR(updates.lstat().st_mode):
            raise ValueError('unredirected updates directory required')
        info = updates.lstat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('protected root-owned update directory required')
    else:
        updates.mkdir(mode=0o700)
        fsync_dir(ROOT)
    if ATTEMPT_ID is not None and not re.fullmatch('[a-f0-9]{64}', ATTEMPT_ID):
        raise ValueError('exact unique operation id required')
    suffix = '' if ATTEMPT_ID is None else '-' + ATTEMPT_ID
    out = updates / (new.policy_sha256 + suffix)
    out.mkdir(mode=0o700)  # Existing/partial operation is never reused.
    fsync_dir(updates)
    return out


def apply_fenced(plan, out):
    old, new = policy(plan['before_policy']), policy(plan['after_policy'])
    validate_transition(old, new)
    values, receipt = read_installation(plan)
    if values['policy.json'] != m.canonical(asdict(old)) or values['rules.conf'] != old.rules().encode():
        raise ValueError('installed policy/rules bytes do not match the old policy')
    before = snapshot(receipt['anchors'])
    validate_old_kernel(old, receipt, before)
    live_network(new)
    skipped = {v.split()[0] for v in m.clean(before['interfaces']).splitlines() if '(skip)' in v}
    if {'ALL', new.interface} & skipped:
        raise ValueError('new interface skips PF')
    no_exo_sockets(new, frozen_gateway=plan.get('gateway_fence'))
    for name, raw in values.items():
        write_new(out / (name + '.before'), raw)
    write_new(out / 'kernel.before.json', m.canonical(before))
    rules = new.rules().encode()
    write_new(out / 'rules.after', rules, 0o444)
    parsed = action(out, 'parse-own-rules', [m.PFCTL, '-n', '-a', m.ANCHOR,
        '-v', '-f', str(out / 'rules.after')])
    canonical = m.clean(parsed.stdout)
    if not canonical:
        raise ValueError('syntax check returned no canonical rules')
    # Recheck original bytes and physical absence after parsing, before mutation.
    if read_installation(plan) != (values, receipt):
        raise ValueError('installation changed during preparation')
    current = snapshot(receipt['anchors'])
    validate_old_kernel(old, receipt, current)
    unchanged_other_rules(before, current, receipt['anchors'])
    live_network(new)
    no_exo_sockets(new, frozen_gateway=plan.get('gateway_fence'))
    pending = {**receipt, 'phase': 'updating', 'pending_policy_sha256': new.policy_sha256,
               'policy_update': str(out)}
    replace_protected(ROOT / 'receipt.json', m.canonical(pending), 'pending-' + new.policy_sha256)
    # Nothing can use a pending receipt as launch authority. Recheck absence.
    no_exo_sockets(new, frozen_gateway=plan.get('gateway_fence'))
    action(out, 'load-own-rules', [m.PFCTL, '-a', m.ANCHOR, '-f', str(out / 'rules.after')])
    after = snapshot(receipt['anchors'])
    unchanged_other_rules(before, after, receipt['anchors'])
    boot, interfaces = live_network(new)
    active = {**receipt, 'phase': 'active', 'policy': json.loads(m.canonical(asdict(new))),
              'policy_sha256': new.policy_sha256, 'rules_sha256': sha(rules),
              'canonical_rules': canonical, 'policy_update': str(out)}
    m.validate_snapshot(new, active, after, boot_id=boot, interfaces=interfaces)
    no_exo_sockets(new, frozen_gateway=plan.get('gateway_fence'))
    if sha(m.protected_file(SUDOERS, maximum=262144)) != receipt['sudoers_sha256']:
        raise ValueError('read-only delegation changed during update')
    replace_protected(ROOT / 'policy.json', m.canonical(asdict(new)), new.policy_sha256)
    replace_protected(ROOT / 'rules.conf', rules, new.policy_sha256)
    write_new(out / 'kernel.after.json', m.canonical(after))
    replace_protected(ROOT / 'receipt.json', m.canonical(active), 'active-' + new.policy_sha256)
    final = snapshot(receipt['anchors'])
    unchanged_other_rules(before, final, receipt['anchors'])
    boot, interfaces = live_network(new)
    m.validate_snapshot(new, json.loads(m.protected_file(ROOT / 'receipt.json', maximum=262144)),
                        final, boot_id=boot, interfaces=interfaces)
    write_new(out / 'verified.json', m.canonical({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'before_policy_sha256': old.policy_sha256, 'after_policy_sha256': new.policy_sha256,
        'enable_token_unchanged': True, 'delegation_unchanged': True, 'kernel': final}))
    print('Verified: only the EXO anchor policy changed; enable reference and read delegation retained.')


def apply(plan):
    old = policy(plan['before_policy'])
    new = policy(plan['after_policy'])
    validate_transition(old, new)
    read_installation(plan)
    live_network(new)
    out = create_output(new)
    write_new(out / 'plan.json', m.canonical(plan))
    # The caller's concrete plan also pins the exact drained local gateway.
    # Its writer barrier and stop span every absence read and the anchor load.
    with quiet_gateway(plan, new, out):
        apply_fenced(plan, out)


if __name__ == '__main__':
    if os.geteuid() != 0 or sys.argv[1:] != ['--apply']:
        raise SystemExit('Human administrator only: --apply. No default action or automatic retry.')
    raw = payload('pf_transport.py', MODULE_SHA256)
    spec = importlib.util.spec_from_loader('qualified_pf_update', loader=None)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    exec(compile(raw, '<pinned-pf-transport>', 'exec'), m.__dict__)
    apply(json.loads(payload('plan.json', PLAN_SHA256)))
