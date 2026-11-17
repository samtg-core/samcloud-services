"""Root-owned, delegated EXO anchor updates from bounded data only.

No user-supplied code, paths, shell, credentials, PF enable/disable or global
ruleset changes. An administrator installs these fixed modules once per host.
"""
import datetime
from contextlib import redirect_stdout, contextmanager
import io
import fcntl
import hashlib
import importlib.util
from ipaddress import IPv4Address, IPv4Network
import json
import os
import pwd
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path('/Library/Preferences/xyz.samtg.exo-pf')
CONTROL = ROOT / 'control'
MAXIMUM = 262144


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate input key')
        value[key] = item
    return value


def read_object(raw):
    if len(raw) > MAXIMUM:
        raise ValueError('bounded data required')
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def protected(path, maximum=MAXIMUM):
    path = Path(path)
    for parent in path.parents:
        value = parent.lstat()
        if not stat.S_ISDIR(value.st_mode) or value.st_uid != 0 or value.st_mode & 0o022:
            raise ValueError('protected root-owned parents required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0
                or before.st_nlink != 1 or before.st_mode & 0o022 or before.st_size > maximum):
            raise ValueError('protected bounded root-owned file required')
        raw = b''
        while len(raw) <= maximum:
            block = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not block:
                break
            raw += block
        after = os.fstat(fd)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in keys):
            raise ValueError('protected file changed')
        return raw
    finally:
        os.close(fd)


def load_module(name, filename, digest):
    raw = protected(CONTROL / filename)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('installed helper module changed')
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    sys.modules[name] = module
    exec(compile(raw, '<root-owned-' + filename + '>', 'exec'), module.__dict__)
    return module


def validate_request(value, cap, updater):
    if set(value) != {'request_id', 'plan'} or not re.fullmatch('[a-f0-9]{64}', value.get('request_id', '')):
        raise ValueError('exact request id and plan required')
    plan = value['plan']
    if not isinstance(plan, dict) or set(plan) != {'before_policy', 'after_policy', 'before_sha256', 'pfctl_sha256', 'gateway_fence'}:
        raise ValueError('fixed update plan schema required')
    old, new = updater.policy(plan['before_policy']), updater.policy(plan['after_policy'])
    updater.validate_transition(old, new)
    for policy in (old, new):
        if (policy.rank, policy.uid, list(policy.ports)) != (cap['rank'], cap['uid'], cap['ports']):
            raise ValueError('capability does not cover another host, uid or port')
        if any(not any(IPv4Address(peer) in IPv4Network(network) for network in cap['private_networks'])
               for peer in policy.peers):
            raise ValueError('only explicitly admitted private peer ranges allowed')
    fence = plan['gateway_fence']
    if (not isinstance(fence, dict) or set(fence) != {'pid', 'identity', 'journal', 'journal_inode'}
            or type(fence['pid']) is not int or fence['pid'] <= 1
            or not isinstance(fence['identity'], str) or len(fence['identity']) > 4096
            or fence['journal'] != cap['journal']
            or not isinstance(fence['journal_inode'], list) or len(fence['journal_inode']) != 2
            or any(type(v) is not int or v < 0 for v in fence['journal_inode'])):
        raise ValueError('only the configured owned gateway journal may be fenced')
    hashes = plan['before_sha256']
    if (not isinstance(hashes, dict) or set(hashes) != {'receipt.json', 'policy.json', 'rules.conf'}
            or any(not isinstance(v, str) or not re.fullmatch('[a-f0-9]{64}', v) for v in hashes.values())
            or not isinstance(plan['pfctl_sha256'], str) or not re.fullmatch('[a-f0-9]{64}', plan['pfctl_sha256'])):
        raise ValueError('exact existing protected installation pins required')
    return plan


def write(path, value):
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def gateway_resumed(plan):
    fence = plan['gateway_fence']
    identity = subprocess.run(['/bin/ps', '-p', str(fence['pid']), '-o', 'uid=,lstart=,comm='], capture_output=True, text=True)
    state = subprocess.run(['/bin/ps', '-p', str(fence['pid']), '-o', 'stat='], capture_output=True, text=True)
    return (identity.returncode == state.returncode == 0 and not identity.stderr and not state.stderr
            and identity.stdout.strip() == fence['identity'] and 'T' not in state.stdout.strip())


@contextmanager
def service_fence(plan, policy, out, cap, updater):
    raw = protected(CONTROL / 'service-fence.py')
    if hashlib.sha256(raw).hexdigest() != cap['fence_sha256']:
        raise ValueError('installed service fence changed')
    account = pwd.getpwuid(cap['uid'])
    if account.pw_name != cap['username']:
        raise ValueError('service identity changed')
    payload = canonical({'uid':cap['uid'], 'fence':plan['gateway_fence']}) + b'\n'
    updater.write_new(out / 'service-fence-intent.json', canonical({'uid':cap['uid'],
        'pid':plan['gateway_fence']['pid'], 'fence_sha256':cap['fence_sha256']}))
    child = subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(CONTROL/'service-fence.py')],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        user=cap['uid'],group=account.pw_gid,extra_groups=os.getgrouplist(account.pw_name,account.pw_gid),
        env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','LC_ALL':'C'},cwd='/')
    try:
        child.stdin.write(payload);child.stdin.flush()
        line = child.stdout.readline(16385)
        observed = read_object(line)
        fence = plan['gateway_fence']
        if observed != {'state':'frozen','pid':fence['pid'],'identity':fence['identity'],
                        'uid':cap['uid'],'queue_writer_barrier':True,'journal_inode':fence['journal_inode']}:
            raise ValueError('unprivileged writer barrier and exact gateway freeze unconfirmed')
        updater.write_new(out/'gateway-stopped.json', canonical(observed))
        updater.FENCE_ALIVE = lambda: child.poll() is None
        yield
    finally:
        updater.FENCE_ALIVE = None
        if child.stdin is not None:
            child.stdin.close();child.stdin=None
        stdout,stderr=child.communicate()
        updater.write_new(out/'service-fence-exit.json',canonical({'rc':child.returncode,
            'stdout':stdout.decode(errors='replace')[-16384:],'stderr':stderr.decode(errors='replace')[-16384:]}))
        if child.returncode!=0:
            raise ValueError('service fence refused or cleanup unconfirmed: '+stderr.decode(errors='replace')[-2048:])
        lines=stdout.splitlines()
        if not lines or read_object(lines[-1])!={'state':'resumed','pid':plan['gateway_fence']['pid'],
            'identity':plan['gateway_fence']['identity'],'writer_barrier_released':True}:
            raise ValueError('unprivileged exact gateway resume unconfirmed')
        updater.write_new(out/'gateway-resumed.json',canonical({'pid':plan['gateway_fence']['pid']}))


def apply_request(value, cap, updater):
    plan = validate_request(value, cap, updater)
    request_id = value['request_id']
    directory = CONTROL / 'requests' / request_id
    digest = hashlib.sha256(canonical(value)).hexdigest()
    if directory.exists():
        original = read_object(protected(directory / 'intent.json'))
        if original['request_sha256'] != digest:
            raise ValueError('request id already binds different data')
        if not (directory / 'result.json').exists():
            return {'state': 'unconfirmed', 'request_id': request_id, 'repeat_mutation': False}
        return read_object(protected(directory / 'result.json'))
    for previous in (CONTROL / 'requests').iterdir():
        outcome = previous / 'result.json'
        if not outcome.exists() or read_object(protected(outcome)).get('state') == 'unconfirmed':
            raise ValueError('an earlier operation is unconfirmed; observe it, do not create another')
    directory.mkdir(mode=0o700)
    write(directory / 'intent.json', {'request_id': request_id, 'request_sha256': digest,
          'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'plan': plan})
    updater.ATTEMPT_ID = request_id
    out = ROOT / 'policy-updates' / (updater.policy(plan['after_policy']).policy_sha256 + '-' + request_id)
    updater.DIAGNOSTIC_DIRECTORY = directory
    try:
        # All mutation remains in the existing, qualified writer-barrier/fence.
        with redirect_stdout(io.StringIO()):
            updater.apply(plan)
        receipt = read_object(protected(ROOT / 'receipt.json'))
        if receipt['phase'] != 'active' or receipt['policy_sha256'] != updater.policy(plan['after_policy']).policy_sha256:
            raise RuntimeError('terminal root receipt unconfirmed')
        result = {'state': 'verified', 'request_id': request_id, 'policy_sha256': receipt['policy_sha256'],
                  'gateway_resumed': gateway_resumed(plan), 'operation': str(out)}
        if not result['gateway_resumed']:
            raise RuntimeError('gateway resume unconfirmed')
    except Exception as error:
        # A fresh attempt is permitted only after a provably pre-mutation refusal.
        unchanged = all(hashlib.sha256(protected(ROOT / name)).hexdigest() == digest
                        for name, digest in plan['before_sha256'].items())
        before_load = not (out / 'load-own-rules-intent.json').exists()
        resumed = gateway_resumed(plan)
        result = {'state': 'refused_before_policy_change' if unchanged and before_load and resumed else 'unconfirmed',
                  'request_id': request_id, 'error_type': type(error).__name__, 'detail': str(error)[:2048],
                  'protected_installation_unchanged': unchanged, 'gateway_resumed': resumed,
                  'repeat_mutation': False, 'operation': str(out)}
    write(directory / 'result.json', result)
    return result


def main():
    if os.geteuid() != 0 or sys.argv[1:] not in (['--apply'], ['--status']):
        raise SystemExit('fixed --apply or --status only')
    cap = read_object(protected(CONTROL / 'capability.json'))
    if int(os.environ.get('SUDO_UID', '-1')) not in (0, cap['uid']):
        raise SystemExit('this capability belongs to the named service account')
    source_pins = {name: hashlib.sha256(protected(CONTROL / name)).hexdigest() for name in
        ('control.py','update.py','pf_transport.py','service-fence.py','run')}
    if any(source_pins[name] != cap[key] for name,key in (
        ('control.py','control_sha256'),('update.py','update_sha256'),('pf_transport.py','transport_sha256'),
        ('service-fence.py','fence_sha256'),('run','wrapper_sha256'))):
        raise ValueError('installed capability code changed')
    if sys.argv[1:] == ['--status']:
        receipt = read_object(protected(ROOT / 'receipt.json'))
        results = []
        for directory in sorted((CONTROL / 'requests').iterdir())[-50:]:
            if not re.fullmatch('[a-f0-9]{64}', directory.name):
                raise ValueError('unexpected request directory')
            value = read_object(protected(directory / 'result.json')) if (directory / 'result.json').exists() else {'state': 'unconfirmed'}
            result = {'request_id': directory.name, **value}
            diagnostic = directory / 'socket-observations.ndjson'
            if diagnostic.exists():
                result['socket_observations'] = protected(diagnostic, MAXIMUM).decode()
            results.append(result)
        print(json.dumps({'rank': cap['rank'], 'policy': receipt['policy'], 'phase': receipt['phase'], 'source_sha256':source_pins, 'operations': results}))
        return
    raw = sys.stdin.buffer.read(MAXIMUM + 1)
    value = read_object(raw)
    updater = load_module('installed_exo_pf_update', 'update.py', cap['update_sha256'])
    updater.m = load_module('installed_exo_pf_transport', 'pf_transport.py', cap['transport_sha256'])
    updater.quiet_gateway = lambda plan, policy, out: service_fence(plan, policy, out, cap, updater)
    fd = os.open(CONTROL / 'operation.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_nlink != 1 or info.st_mode & 0o077:
            raise ValueError('protected operation lock required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = apply_request(value, cap, updater)
        print(json.dumps(result))
        if result['state'] != 'verified':
            raise SystemExit(1)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
