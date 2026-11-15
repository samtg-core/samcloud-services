"""Authenticated loopback EXO metadata, separate from the plane credential.

No inference submission, placement or retry is exposed by this reader.
"""

import os
from pathlib import Path
import re
import stat

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
        if not re.fullmatch(r'[a-fA-F0-9]{64}', key):
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
            return await bounded_json(self._client, 'GET', '/state', headers)
        except Exception:
            # No token, local file path, response body or prompt in diagnostics.
            raise PrivateEXOUnavailable('private EXO state unavailable') from None

    async def close(self):
        await self._client.aclose()
