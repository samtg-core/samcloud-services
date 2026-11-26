"""Private EXO transport; inference must consume a journal submission decision."""

import os
from pathlib import Path
import re
import stat
from contextlib import asynccontextmanager

import httpx

from .http import bounded_json


class PrivateEXOUnavailable(RuntimeError):
    """No private observation established; preserve ownership and uncertainty."""


def private_key(path: Path) -> str:
    """Read the same owned, single-link 0600 key shape required by EXO."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise ValueError("private EXO key file is not owned and private")
        data = os.read(fd, 259)
        if len(data) > 258:
            raise ValueError("private EXO key exceeds bound")
        key = data.decode('ascii').strip()
        if not re.fullmatch(r'[a-f0-9]{64}', key):
            raise ValueError("private EXO key has invalid format")
        return key
    finally:
        os.close(fd)


class PrivateEXO:
    def __init__(self, port: int, key_file: Path, *, transport=None):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("explicit private EXO loopback port required")
        self.key_file = Path(key_file)
        self._client = httpx.AsyncClient(
            base_url=f'http://127.0.0.1:{port}', timeout=None,
            follow_redirects=False, trust_env=False, transport=transport,
        )

    async def state(self) -> dict:
        try:
            headers = {'Authorization': 'Bearer ' + private_key(self.key_file),
                       'Accept': 'application/json', 'Cache-Control': 'no-cache'}
            return await bounded_json(self._client, 'GET', '/leased/state', headers)
        except Exception:
            # No token, local file path, response body or prompt in diagnostics.
            raise PrivateEXOUnavailable('private EXO state unavailable') from None

    def _headers(self):
        return {'Authorization': 'Bearer ' + private_key(self.key_file),
                'Accept': 'application/json', 'Cache-Control': 'no-cache'}

    async def create_instance(self, starts, plan, contract, *, state_event_idx):
        headers = self._headers()
        starts.begin(plan, contract, state_event_idx=state_event_idx)
        try:
            response = await bounded_json(self._client, 'POST', '/instance', headers,
                payload=contract.create_request())
            return starts.accept(plan, contract, response)
        except Exception:
            raise PrivateEXOUnavailable('private instance creation outcome unconfirmed; do not replay') from None

    @asynccontextmanager
    async def chat(self, bindings, execution_id, contract, body, *, state_event_idx):
        """Commit once-only ownership immediately before the sole inference POST."""
        headers = self._headers()
        headers['Accept'] = 'text/event-stream'
        bindings.begin(execution_id, contract, body, state_event_idx=state_event_idx)
        async with self._client.stream('POST', '/v1/chat/completions',
                                       headers=headers, json=body) as response:
            yield response

    async def command_end(self, command_id):
        if (not isinstance(command_id, str)
                or not re.fullmatch(r'[a-f0-9-]{36}', command_id)):
            raise PrivateEXOUnavailable('private EXO command identity unavailable')
        try:
            return await bounded_json(self._client, 'GET',
                '/leased/commands/' + command_id + '/end', self._headers())
        except Exception:
            raise PrivateEXOUnavailable('private EXO command end unavailable') from None

    async def close(self):
        await self._client.aclose()
