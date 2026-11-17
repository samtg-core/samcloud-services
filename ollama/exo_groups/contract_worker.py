"""Fixed frozen-Python metadata compiler. Standard library until isolation.

Invoked only by the configured model-service compiler with -I -S -B. It has
no model/registry client, compute import, network, child or filesystem write.
"""

import contextlib
from dataclasses import asdict
import hashlib
import importlib.abc
import io
import json
import math
import os
from pathlib import Path
import runpy
import sys


MAX_INPUT = 2 * 1024 * 1024
MAX_OUTPUT = 512 * 1024
SOURCE_FILES = ('ollama/__init__.py', 'ollama/exo_groups/__init__.py',
    'ollama/exo_groups/compiler.py', 'ollama/exo_groups/contract_worker.py',
    'ollama/exo_groups/instance.py', 'ollama/exo_groups/prepare.py',
    'ollama/exo_groups/plan.py', 'ollama/exo_groups/member.py')
COMPUTE = {'mlx', 'mlx_lm', 'exo_rs', 'mflux', 'torch'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def decode(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('ambiguous compiler input')
            value[key] = item
        return value

    def bad(_):
        raise ValueError('nonfinite compiler input')

    def number(raw):
        value = float(raw)
        if not math.isfinite(value):
            bad(None)
        return value

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=bad, parse_float=number)
    if not isinstance(value, dict):
        raise ValueError('compiler input object required')
    return value


class NoCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in COMPUTE:
            raise RuntimeError('compute import forbidden in metadata compilation')


def audit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty',
                 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('network and child processes forbidden in metadata compilation')
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink',
                 'os.chmod', 'os.chown', 'os.truncate', 'os.utime', 'os.setxattr', 'os.removexattr'}:
        raise RuntimeError('filesystem mutations forbidden in metadata compilation')
    if event == 'open':
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise RuntimeError('filesystem writes forbidden in metadata compilation')


def main():
    if (not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode)
            or sys.version_info[:2] != (3, 13)
            or any(any(part in key.upper() for part in ('TOKEN', 'BEARER', 'AUTH', 'SECRET'))
                   for key in os.environ)):
        raise ValueError('isolated credential-free frozen interpreter required')
    sys.meta_path.insert(0, NoCompute())
    sys.addaudithook(audit)
    raw = sys.stdin.buffer.read(MAX_INPUT + 1)
    if len(raw) > MAX_INPUT:
        raise ValueError('compiler input exceeds bound')
    request = decode(raw)
    fields = {'nonce', 'bootstrap_sha256', 'source_pins', 'plan', 'manifest',
              'parent', 'runtime_sha256', 'group_id', 'generation', 'ring_port', 'endpoints'}
    if set(request) not in (fields, fields | {'served_context_length'}):
        raise ValueError('exact metadata compiler request required')
    provider = Path(__file__).resolve().parents[2]
    pins = {name: hashlib.sha256((provider/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if canonical(pins) != canonical(request['source_pins']):
        raise ValueError('qualified provider source changed')
    runtime = Path(sys.base_prefix).resolve().parent
    bootstrap = runtime/'run_text.py'
    if (Path(sys.base_prefix).resolve().name != 'python'
            or hashlib.sha256(bootstrap.read_bytes()).hexdigest() != request['bootstrap_sha256']):
        raise ValueError('qualified frozen bootstrap changed')
    sys.argv = [str(bootstrap), '--inspect-runtime']
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(bootstrap), run_name='exo_contract_frozen_inspection')
    sys.path.insert(0, str(provider))
    from ollama.exo_groups.plan import GroupPlan
    from ollama.exo_groups.prepare import RankEndpoint
    from ollama.exo_groups.instance import InstanceContract
    contract = InstanceContract.prepare(GroupPlan.parse(request['plan']),
        request['manifest'].encode(), request['parent'].encode(),
        tuple(RankEndpoint(**value) for value in request['endpoints']),
        group_id=request['group_id'], generation=request['generation'],
        ring_port=request['ring_port'], runtime_sha256=request['runtime_sha256'],
        served_context_length=request.get('served_context_length'))
    if any(name.split('.')[0] in COMPUTE for name in sys.modules):
        raise RuntimeError('compute module imported during metadata compilation')
    output = canonical({'nonce': request['nonce'], 'request_sha256': hashlib.sha256(raw).hexdigest(),
        'source_sha256': hashlib.sha256(canonical(pins)).hexdigest(),
        'contract': asdict(contract), 'contract_sha256': contract.sha256,
        'native_compute_imported': False})
    if len(output) > MAX_OUTPUT:
        raise ValueError('compiled contract exceeds bound')
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
