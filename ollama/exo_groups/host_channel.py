"""Read-only staging probes over explicitly configured authenticated host routes.

The service/attended operator supplies its existing SSH route, interpreter and
environment. Neither a queued caller nor an inference request supplies commands.
The coordinator's plane bearer is never serialized into the probe. This channel
cannot launch a rank, request a lease, or declare runtime qualification.
"""

import asyncio
from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import json
from pathlib import PurePosixPath
import re
import secrets
import shlex

from .prepare import canonical
from .staging import HostFiles, StagingUnavailable


# A metadata bound, not an observation or compute deadline. One frame contains
# file identities for a complete runtime; weight bytes never cross this channel.
MAX_FRAME = 8 * 1024 * 1024
MAX_DIAGNOSTICS = 32 * 1024
PREFIX = b'EXO_STAGING_V1:'


@dataclass(frozen=True)
class HostRoute:
    rank: str
    resource_id: str
    uid: int
    command: tuple[str, ...]
    interpreter: str

    def __post_init__(self):
        if (not isinstance(self.rank, str) or not re.fullmatch('[a-z][a-z0-9_-]*', self.rank)
                or not isinstance(self.resource_id, str) or not self.resource_id
                or type(self.uid) is not int or self.uid < 0
                or not isinstance(self.command, tuple) or not self.command
                or any(not isinstance(value, str) or not value or '\x00' in value
                       for value in self.command)
                or not PurePosixPath(self.command[0]).is_absolute()
                or not isinstance(self.interpreter, str)
                or not PurePosixPath(self.interpreter).is_absolute()
                or str(PurePosixPath(self.interpreter)) != self.interpreter
                or any(part in ('.', '..') for part in PurePosixPath(self.interpreter).parts)
                or '\x00' in self.interpreter):
            raise ValueError('explicit frozen host route and interpreter required')


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate host field')
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError('nonfinite host field')

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _program(source):
    # Static code only; host policies travel as stdin JSON, never shell text.
    return source + '''
if __name__ == '__main__':
    import sys
    raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('staging request exceeds bound')
    request = json.loads(raw)
    if set(request) != {'request_id', 'collector_sha256', 'spec', 'previous'}:
        raise ValueError('exact staging request required')
    report = collect_host(request['spec'], request['previous'])
    value = {'request_id': request['request_id'],
             'collector_sha256': request['collector_sha256'], 'report': report}
    print('EXO_STAGING_V1:' + json.dumps(value, sort_keys=True, separators=(',', ':'),
          ensure_ascii=False, allow_nan=False), flush=True)
'''


async def _bounded(stream, limit):
    value = bytearray()
    while chunk := await stream.read(min(64 * 1024, limit + 1 - len(value))):
        value.extend(chunk)
        if len(value) > limit:
            raise StagingUnavailable('host observation exceeds metadata bound')
    return bytes(value)


async def run_probe(argv, raw, *, environment, spawn=asyncio.create_subprocess_exec):
    """One finite read-only probe. Cancellation stops this observer, not compute.

    No timeout, automatic retry, shell, background service or cached output. The
    supplied executable/route is trusted startup configuration, not remote data.
    """
    process = await spawn(*argv, env=environment, stdin=asyncio.subprocess.PIPE,
                          stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    async def send():
        process.stdin.write(raw)
        await process.stdin.drain()
        process.stdin.close()
        # asyncio's subprocess protocol owns this shared close future. Do not
        # cancel it when the observer is cancelled (Python3.12 otherwise raises
        # InvalidStateError in its later pipe_connection_lost callback).
        await asyncio.shield(process.stdin.wait_closed())

    tasks = [asyncio.create_task(send()),
             asyncio.create_task(_bounded(process.stdout, MAX_FRAME)),
             asyncio.create_task(_bounded(process.stderr, MAX_DIAGNOSTICS)),
             asyncio.create_task(process.wait())]
    try:
        _, output, _, status = await asyncio.gather(*tasks)
        if status != 0:
            raise StagingUnavailable('authenticated host probe unavailable')
        return output
    finally:
        if process.returncode is None:
            try:
                process.terminate()  # Only the fixed read-only SSH/probe process.
            except ProcessLookupError:
                pass
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await process.wait()


class AuthenticatedHostReader:
    """StagingVerifier reader, with an explicit trusted route per host.

    Route commands must accept one final remote-command argument (e.g. the
    existing SSH broker). Their configured SSH identity and host-key trust stay
    local. This class does not select credentials, alter trust, or inherit the
    coordinator's process environment. The operator supplies a minimal explicit
    environment, including only what its route actually needs.
    """
    def __init__(self, routes, *, collector_sha256, environment, runner=run_probe):
        routes = tuple(routes)
        if (not routes or any(not isinstance(route, HostRoute) for route in routes)
                or len({route.rank for route in routes}) != len(routes)
                or len({route.resource_id for route in routes}) != len(routes)
                or not isinstance(environment, dict)
                or any(not isinstance(key, str) or not key or not isinstance(value, str)
                       or '=' in key or '\x00' in key or '\x00' in value
                       for key, value in environment.items())
                or not callable(runner)):
            raise ValueError('exact host routes and explicit local environment required')
        # Do not permit the plane/EXO bearer or a credential forwarding knob in
        # this environment. SSH key/config paths belong in the trusted route.
        if any(any(part in key.upper() for part in ('TOKEN', 'AUTH', 'BEARER', 'SECRET',
                                                   'SSH_AUTH_SOCK'))
               for key in environment):
            raise ValueError('coordinator credentials must stay outside the host channel')
        source = files(__package__).joinpath('staging_files.py').read_text()
        if (not isinstance(collector_sha256, str)
                or hashlib.sha256(source.encode()).hexdigest() != collector_sha256):
            raise ValueError('host collector differs from the configured source pin')
        self._routes = {route.rank: route for route in routes}
        self._source, self.collector_sha256 = source, collector_sha256
        self._environment, self._runner = dict(environment), runner

    async def __call__(self, host, previous):
        if not isinstance(host, HostFiles):
            raise StagingUnavailable('exact frozen host file policy required')
        route = self._routes.get(host.rank)
        if route is None or (route.resource_id, route.uid) != (host.resource_id, host.uid):
            raise StagingUnavailable('staging policy differs from authenticated host route')
        request_id = secrets.token_hex(32)
        raw = canonical({'request_id': request_id, 'collector_sha256': self.collector_sha256,
                         'spec': asdict(host), 'previous': previous})
        if len(raw) > MAX_FRAME:
            raise StagingUnavailable('host staging policy exceeds metadata bound')
        remote = shlex.join([route.interpreter, '-I', '-S', '-B', '-c', _program(self._source)])
        try:
            output = await self._runner((*route.command, remote), raw,
                                        environment=dict(self._environment))
            if not isinstance(output, bytes) or len(output) > MAX_FRAME:
                raise ValueError('host output')
            lines = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
            if len(lines) != 1:
                raise ValueError('exact host frame required')
            result = _json(lines[0])
            if (not isinstance(result, dict) or set(result) != {'request_id', 'collector_sha256', 'report'}
                    or result['request_id'] != request_id
                    or result['collector_sha256'] != self.collector_sha256):
                raise ValueError('host frame differs from current probe')
            report = result['report']
            if (not isinstance(report, dict) or type(report.get('uid')) is not int
                    or (report.get('rank'), report.get('resource_id'), report.get('uid')) != (
                        route.rank, route.resource_id, route.uid)):
                raise ValueError('host principal differs')
            return report
        except Exception:
            # Never expose host stderr, shell text, file contents or credentials.
            raise StagingUnavailable('authenticated host staging unavailable') from None
