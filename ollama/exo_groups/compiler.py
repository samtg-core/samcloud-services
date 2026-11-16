"""Owned CPU contract preparation using the exact frozen EXO type environment.

This does not run Node, import native compute, request a lease or grant launch
permission. The existing supervisor owns the helper across waiter disconnect.
"""

import asyncio
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import re
import secrets

from ..inference_arbiter import Conflict
from ..inference_supervisor import Supervisor
from .contract_worker import MAX_INPUT, MAX_OUTPUT, SOURCE_FILES
from .deployment import HostDeployment
from .http import decode_json
from .instance import InstanceContract
from .member import ExpectedMember
from .prepare import RankEndpoint, canonical, digest


class CompilerUnavailable(RuntimeError):
    pass


def source_pins():
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


async def bounded(stream, limit):
    value = bytearray()
    while chunk := await stream.read(65536):
        if len(value) + len(chunk) > limit:
            raise CompilerUnavailable('compiler metadata exceeds bound')
        value.extend(chunk)
    return bytes(value)


class ContractCompiler:
    def __init__(self, deployment, supervisor, *, spawn=asyncio.create_subprocess_exec):
        if (not isinstance(deployment, HostDeployment) or not callable(spawn)
                or not isinstance(supervisor, Supervisor)):
            raise ValueError('fixed host deployment and existing supervisor required')
        self.deployment, self.supervisor, self.spawn = deployment, supervisor, spawn
        self.pins, self._task, self._identity = source_pins(), None, None
        self.process = None

    async def compile(self, launches):
        ordered = sorted(self.deployment.plan.members, key=lambda m: m.pipeline_rank)
        if (not isinstance(launches, tuple) or len(launches) != 3
                or tuple(binding.rank for binding in launches) != tuple(m.rank for m in ordered)):
            raise ValueError('exact pipeline-ordered launches required')
        for binding in launches:
            ExpectedMember.from_launch(self.deployment.plan, binding)
            if (binding.group_id != launches[0].group_id or type(binding.generation) is not int
                    or binding.generation != launches[0].generation
                    or type(binding.pipeline_rank) is not int
                    or any(type(value) is not int for value in binding.layers)
                    or not isinstance(binding.lease_id, str)
                    or not re.fullmatch('lease_[a-f0-9]{16,64}', binding.lease_id)):
                raise ValueError('one exact group generation required')
        if len({binding.lease_id for binding in launches}) != 3:
            raise ValueError('three exact unique member leases required')
        local = next(binding for binding in launches if binding.rank == self.deployment.files.rank)
        compiled = self.deployment.compile(local)
        identity = digest([asdict(binding) for binding in launches])
        if self._task is not None:
            if identity != self._identity:
                raise Conflict('metadata compiler already owns another exact group')
        else:
            self._identity = identity
            operation = self._run(compiled, launches)
            try:
                self._task = self.supervisor.prepare(operation)
            except BaseException:
                operation.close()
                raise
        return await asyncio.shield(self._task)

    async def wait(self):
        if self._task is None:
            raise CompilerUnavailable('no owned compiler operation')
        return await asyncio.shield(self._task)

    def _metadata(self, name):
        pins = [p for p in self.deployment.files.model['files'] if p['path'] == name and p['link'] is None]
        if len(pins) != 1:
            raise CompilerUnavailable('exact model metadata pin required')
        descriptor = os.open(self.deployment.model_root/name, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as source:
            raw = source.read(MAX_INPUT+1)
        if len(raw) > MAX_INPUT or len(raw) != pins[0]['bytes'] or hashlib.sha256(raw).hexdigest() != pins[0]['sha256']:
            raise CompilerUnavailable('model metadata differs from the staged policy')
        return raw.decode()

    async def _run(self, compiled, launches):
        streams = ()
        try:
            await self.deployment.observe_files(compiled)
            if source_pins() != self.pins:
                raise ValueError('provider compiler source changed')
            bootstrap = next(pin for pin in self.deployment.files.runtime['files'] if pin['path'] == 'run_text.py')
            local = launches[0]
            endpoints = tuple(RankEndpoint(binding.rank, digest([local.group_id, local.generation,
                binding.rank_manifest_sha256, binding.rank, 'zenoh-node'])[:32].lstrip('0') or '1',
                self.deployment.peer_ips[binding.pipeline_rank]) for binding in launches)
            request = {'nonce': secrets.token_hex(32), 'bootstrap_sha256': bootstrap['sha256'],
                'source_pins': self.pins, 'plan': asdict(self.deployment.plan),
                'manifest': self._metadata('rank-bundle.json'), 'parent': self._metadata('parent-manifest.json'),
                'runtime_sha256': self.deployment.runtime_sha256, 'group_id': local.group_id,
                'generation': local.generation, 'ring_port': self.deployment.ports[3],
                'endpoints': [asdict(endpoint) for endpoint in endpoints]}
            raw = canonical(request)
            if len(raw) > MAX_INPUT:
                raise ValueError('compiler input exceeds bound')
            environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(self.deployment.home),
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'EXO_OFFLINE': 'true',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'EXO_HOME': str(self.deployment.state_base/'contract-metadata-home')}
            self.process = await self.spawn(str(self.deployment.runtime_root/'python/bin/python3.13'),
                '-I', '-S', '-B', str(Path(__file__).with_name('contract_worker.py')),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=environment, cwd=str(self.deployment.state_base),
                start_new_session=True)

            async def write():
                self.process.stdin.write(raw)
                await self.process.stdin.drain()
                self.process.stdin.close()
                await self.process.stdin.wait_closed()

            streams = tuple(asyncio.create_task(task) for task in (
                write(), bounded(self.process.stdout, MAX_OUTPUT), bounded(self.process.stderr, MAX_OUTPUT)))
            _, output, _error = await asyncio.gather(*streams)
            if await self.process.wait() != 0:
                raise ValueError('fixed metadata worker failed')
            value = decode_json(output)
            if (set(value) != {'nonce', 'request_sha256', 'source_sha256', 'contract',
                              'contract_sha256', 'native_compute_imported'}
                    or value['nonce'] != request['nonce']
                    or value['request_sha256'] != hashlib.sha256(raw).hexdigest()
                    or value['source_sha256'] != digest(self.pins)
                    or value['native_compute_imported'] is not False
                    or not isinstance(value['contract'], dict)):
                raise ValueError('compiler response identity differs')
            contract = InstanceContract(**value['contract'])
            if contract.sha256 != value['contract_sha256']:
                raise ValueError('compiled contract digest differs')
            for name in ('request_json', 'state_instance_json'):
                if canonical(decode_json(getattr(contract, name))).decode() != getattr(contract, name):
                    raise ValueError('noncanonical compiled contract')
            contract._check_binding(self.deployment.plan, launches, runtime_sha256=self.deployment.runtime_sha256)
            await self.deployment.observe_files(compiled)
            if source_pins() != self.pins:
                raise ValueError('provider compiler source changed')
            return contract
        except Exception:
            raise CompilerUnavailable('qualified metadata compilation unavailable; inspect the same operation') from None
        finally:
            # This child is a write/network/native-compute-forbidden CPU
            # compiler, never an inference rank. Bound violations terminate
            # only this retained child, not unknown model work or another PID.
            if self.process is not None and self.process.returncode is None:
                self.process.kill()
                await self.process.wait()
            for task in streams:
                if not task.done():
                    task.cancel()
            if streams:
                await asyncio.gather(*streams, return_exceptions=True)
