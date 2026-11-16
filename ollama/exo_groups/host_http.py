"""Exact authenticated model-service host routes; no shell or compute retry.

The deployment supplies HTTPS endpoints and source/configuration pins. Reader
methods fit the existing staging and receipt contracts. Native control POSTs
require a durable once-only decision in the coordinator's existing journal.
"""

import asyncio
from dataclasses import asdict, dataclass
import re
import secrets
import time
from urllib.parse import urlsplit

import httpx

from ..inference_arbiter import Conflict
from .host_control import run_identity
from .host_receipts import AuthenticatedReceiptReader
from .http import credential_headers, decode_json
from .member import ExpectedMember
from .prepare import canonical, digest
from .rank_process import RunBinding
from .staging import HostFiles


class HostControlUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class HostEndpoint:
    rank: str
    resource_id: str
    uid: int
    url: str
    source_sha256: str
    configuration_sha256: str

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if (not isinstance(self.rank, str) or not self.rank or not self.resource_id
                or type(self.uid) is not int or self.uid < 0
                or parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or any(not isinstance(v, str) or not re.fullmatch('[a-f0-9]{64}', v)
                       for v in (self.source_sha256, self.configuration_sha256))):
            raise ValueError('exact trusted HTTPS host route and source/configuration pins required')


class HostControlClient:
    def __init__(self, endpoint, header, *, transport=None):
        if not isinstance(endpoint, HostEndpoint) or not callable(header):
            raise ValueError('configured host endpoint and own coordinator header required')
        self.endpoint, self.header = endpoint, header
        self._client = httpx.AsyncClient(base_url=endpoint.url, timeout=None,
            follow_redirects=False, trust_env=False, transport=transport)

    async def _request(self, action, payload, *, request_id=None):
        if (action not in ('files', 'status', 'receipts', 'tokens', 'prepare', 'start', 'stop', 'fence-unstarted')
                or not isinstance(payload, dict) or 'request_id' in payload):
            raise ValueError('fixed host operation and payload required')
        request_id = secrets.token_hex(32) if request_id is None else request_id
        if not isinstance(request_id, str) or not re.fullmatch('[a-f0-9]{64}', request_id):
            raise ValueError('exact host control request id required')
        try:
            headers = credential_headers(self.header)
            async with self._client.stream('POST', '/v1/exo/host/' + action,
                    headers=headers, json={'request_id': request_id, **payload}) as response:
                if (response.status_code != 200 or response.headers.get('content-type', '').split(';')[0]
                        != 'application/json' or response.headers.get('cache-control') != 'no-store'):
                    raise ValueError('host metadata unavailable')
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > 32 * 1024 * 1024:
                        raise ValueError('host metadata exceeds bound')
                    raw.extend(chunk)
            value = decode_json(raw)
            if (set(value) != {'request_id', 'source_sha256', 'configuration_sha256', 'report'}
                    or value['request_id'] != request_id
                    or value['source_sha256'] != self.endpoint.source_sha256
                    or value['configuration_sha256'] != self.endpoint.configuration_sha256
                    or not isinstance(value['report'], dict)):
                raise ValueError('host source/request/configuration differs')
            return value['report']
        except Exception:
            raise HostControlUnavailable('exact host observation or control outcome unavailable') from None

    async def files(self, policy, _previous=None):
        if (not isinstance(policy, HostFiles) or (policy.rank, policy.resource_id, policy.uid) != (
                self.endpoint.rank, self.endpoint.resource_id, self.endpoint.uid)):
            raise ValueError('exact configured host staging policy required')
        # The server owns its verified file cache; a caller cannot inject one.
        return await self._request('files', {'policy_sha256': digest(asdict(policy))})

    async def status(self, binding):
        return await self._request('status', {'identity': run_identity(binding)})

    async def receipts(self, binding):
        return await self._request('receipts', {'identity': run_identity(binding)})

    async def tokens(self, binding, command_id, task_id):
        return await self._request('tokens', {'identity': run_identity(binding),
                                             'command_id': command_id, 'task_id': task_id})

    async def close(self):
        await self._client.aclose()


class HTTPReceiptReader(AuthenticatedReceiptReader):
    """Same exact preparation/child/cleanup validator as the SSH reader."""
    def __init__(self, client):
        if not isinstance(client, HostControlClient):
            raise ValueError('pinned authenticated host HTTP client required')
        self.client, self.route = client, client.endpoint

    async def read(self, plan, *, binding, journal_path, state_path, boot_id,
                   launch_sha256, runtime_sha256):
        expected = ExpectedMember.from_launch(plan, binding)
        if (binding.rank, binding.resource_id) != (self.route.rank, self.route.resource_id):
            raise ValueError('receipt route differs from exact member')
        run = RunBinding.from_expected(expected, runtime_sha256)
        spec = {'resource_id': binding.resource_id, 'group_id': binding.group_id,
            'generation': binding.generation, 'lease_id': binding.lease_id,
            'runtime_sha256': runtime_sha256, 'launch_sha256': launch_sha256,
            'boot_id': boot_id, 'uid': self.route.uid,
            'state_path': state_path, 'journal_path': journal_path}
        return self._checked(plan, binding, spec, await self.client.receipts(run))


class HostCommands:
    """Durable coordinator POST ownership, independent of its HTTP observer."""
    def __init__(self, actor, client, launch, runtime_sha256, *, service):
        if not isinstance(client, HostControlClient) or not re.fullmatch('[a-f0-9]{64}', runtime_sha256):
            raise ValueError('exact host client and qualified runtime pin required')
        self.actor, self.client = actor, client
        expected = ExpectedMember.from_launch(actor.plan, launch)
        if (launch.rank, launch.resource_id) != (client.endpoint.rank, client.endpoint.resource_id):
            raise ValueError('host command route differs from member')
        self.binding = RunBinding.from_expected(expected, runtime_sha256)
        self.launch = {k: v for k, v in asdict(launch).items() if k != 'expires_at'}
        self.tasks = {}
        self.accepting = True
        self.service = service
        self.journal = actor.intents.journal
        # The gateway owns finite control operations and drains them before
        # closing its coordinator/host observers. No independent HTTP driver.
        service.own_host_commands(self)
        with self.journal.transaction():
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS exo_host_submissions (
                group_id TEXT NOT NULL,generation INTEGER NOT NULL,lease_id TEXT NOT NULL,
                action TEXT NOT NULL,request_id TEXT NOT NULL UNIQUE,
                identity_json TEXT NOT NULL,endpoint_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,created REAL NOT NULL,
                PRIMARY KEY(group_id,generation,lease_id,action))''')

    async def submit(self, action, *, verify_runtime):
        if action not in ('prepare', 'start', 'stop', 'fence-unstarted') or not callable(verify_runtime):
            raise ValueError('explicit host action and runtime verifier required')
        if (not self.service.started or self.service.closed
                or action in ('prepare', 'start') and not self.accepting):
            raise Conflict('configured service host control is not accepting starts')
        if action in ('prepare', 'start'):
            self.actor.wake()
            await self.actor.wait()
            epoch = self.actor._epoch
            if not self.actor.may_start_ranks():
                raise Conflict('fresh all-member authority required for host start')
            if await verify_runtime() != self.binding.runtime_sha256:
                raise Conflict('host runtime differs')
            if epoch != self.actor._epoch or not self.actor.may_start_ranks():
                raise Conflict('host authority changed during runtime read')
            if not self.accepting or self.service.draining or self.service.closed:
                raise Conflict('service began draining during runtime read')
        binding = self.binding
        saved = self.actor.intents.snapshot(self.actor.plan.owner_id, self.actor.plan.request_key)
        staged = self.actor.intents.staging(self.actor.plan.owner_id, self.actor.plan.request_key)
        if (saved['group_id'] != binding.group_id or saved['generation'] != binding.generation
                or saved['launches'] is None
                or not any(canonical(v) == canonical(self.launch) for v in saved['launches'])
                or staged is None or staged['runtime_sha256'] != binding.runtime_sha256):
            raise Conflict('host action differs from the frozen group/runtime')
        key = binding.group_id, binding.generation, binding.lease_id, action
        request_id = secrets.token_hex(32)
        with self.journal.transaction():
            if self.journal.db.execute('SELECT 1 FROM exo_host_submissions '
                    'WHERE group_id=? AND generation=? AND lease_id=? AND action=?', key).fetchone():
                raise Conflict('host action already submitted; observe the same attempt')
            self.journal.db.execute('INSERT INTO exo_host_submissions VALUES(?,?,?,?,?,?,?,?,?)',
                (*key, request_id, canonical(run_identity(binding)).decode(),
                 digest(asdict(self.client.endpoint)), 'owned', time.time()))
        # Register with the existing supervisor task factory immediately after
        # commit, before yielding to a caller. Disconnect cancels only its wait.
        # A process death retains the row: recovery observes, never repeats POST.
        operation = self._send(action, key, request_id)
        try:
            task = self.tasks[action] = self.actor.task_factory(operation)
        except BaseException:
            operation.close()
            raise
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(task)

    async def wait(self, action):
        if action not in self.tasks:
            raise Conflict('no live owned host request; inspect status and receipts')
        return await asyncio.shield(self.tasks[action])

    def pause(self):
        self.accepting = False

    async def drain(self):
        tasks = tuple(task for task in self.tasks.values() if not task.done())
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    async def _send(self, action, key, request_id):
        binding = self.binding
        payload = ({'launch': asdict(binding.expected.binding), 'plan_sha256': digest(asdict(self.actor.plan))}
            if action == 'prepare' else {'identity': run_identity(binding)})
        result = await self.client._request(action, payload, request_id=request_id)
        operations = result.get('operations')
        if (canonical(result.get('identity')) != canonical(run_identity(binding))
                or not isinstance(operations, list)
                or sum(isinstance(v, dict) and v.get('request_id') == request_id
                       and v.get('action') == action
                       and v.get('state') in ('owned', 'returned', 'unconfirmed')
                       for v in operations) != 1):
            raise HostControlUnavailable('host operation acknowledgement differs from the submitted attempt')
        with self.journal.transaction():
            self.journal.db.execute('UPDATE exo_host_submissions SET state=? '
                'WHERE group_id=? AND generation=? AND lease_id=? AND action=?', ('returned', *key))
        return result
