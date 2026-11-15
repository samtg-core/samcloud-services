"""Coordinator's exact metadata operations, using its own owner credential."""

import hashlib
import re
from urllib.parse import quote

import httpx

from .http import bounded_json, bounded_response, credential_headers


class PlaneUnavailable(RuntimeError):
    """Metadata outcome unknown; reconcile its saved identity, never replace it."""


class CoordinatorPlane:
    def __init__(self, owner_id, header, *, transport=None):
        self.owner_id, self.header = owner_id, header
        self._verified_digest = None
        self._file_owner = None
        self._client = httpx.AsyncClient(base_url='https://cloud.samtg.xyz', timeout=None,
            follow_redirects=False, trust_env=False, transport=transport)

    async def verify_owner(self):
        try:
            headers = credential_headers(self.header)
            digest = hashlib.sha256(headers['Authorization'].encode()).digest()
            if digest != self._verified_digest:
                identity = await bounded_json(self._client, 'GET', '/api/v1/auth/verify', headers)
                if identity.get('valid') is not True or identity.get('owner_id') != self.owner_id:
                    raise ValueError('credential differs from coordinator owner')
                name = identity.get('username') or self.owner_id
                if not isinstance(name, str) or not name:
                    raise ValueError('file owner identity unavailable')
                self._file_owner = name
                self._verified_digest = digest
            return headers
        except Exception:
            raise PlaneUnavailable('coordinator owner unavailable') from None

    async def _request(self, method, path, payload=None):
        try:
            headers = await self.verify_owner()
            return await bounded_json(self._client, method, path, headers, payload=payload)
        except Exception:
            # No credentials, remote response body or arbitrary exception text.
            raise PlaneUnavailable('coordinator metadata unavailable') from None

    @staticmethod
    def _group(group_id):
        if not isinstance(group_id, str) or not re.fullmatch(r'group_[a-f0-9]{24}', group_id):
            raise ValueError('exact group id required')
        return '/api/v1/lease-groups/' + group_id

    async def create(self, saved_request):
        # This body must already be committed to GroupIntents by the actor.
        return await self._request('POST', '/api/v1/lease-groups', saved_request)

    async def group(self, group_id):
        return await self._request('GET', self._group(group_id))

    async def advance(self, group_id, generation):
        if type(generation) is not int or generation < 1:
            raise ValueError('explicit generation required')
        return await self._request('POST', self._group(group_id) + '/advance',
                                   {'generation': generation})

    async def begin_release(self, group_id, generation):
        if type(generation) is not int or generation < 1:
            raise ValueError('explicit generation required')
        return await self._request('POST', self._group(group_id) + '/release',
                                   {'generation': generation})

    async def release_member(self, group_id, lease_id, generation, proof):
        if (not isinstance(lease_id, str) or not re.fullmatch(r'lease_[a-f0-9]{16,64}', lease_id)
                or type(generation) is not int or generation < 1
                or not isinstance(proof, dict) or set(proof) != {'evidence_ref', 'evidence_sha256'}
                or not isinstance(proof['evidence_ref'], str)
                or not re.fullmatch(r'https://cloud\.samtg\.xyz/api/v1/files/[1-9][0-9]*/raw',
                                    proof['evidence_ref'])
                or not isinstance(proof['evidence_sha256'], str)
                or not re.fullmatch(r'[a-f0-9]{64}', proof['evidence_sha256'])):
            raise ValueError('exact member, generation and pinned plane evidence required')
        return await self._request('POST', self._group(group_id) + '/members/' + lease_id + '/release',
                                   {'generation': generation, **proof})

    @staticmethod
    def _file(file_id):
        if type(file_id) is not int or file_id < 1:
            raise ValueError('exact published plane file id required')
        return '/api/v1/files/' + str(file_id)

    async def artifact(self, file_id):
        return await self._request('GET', self._file(file_id))

    async def file_owner(self):
        await self.verify_owner()
        return self._file_owner

    async def artifacts_named(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'exo-cleanup-[a-f0-9]{64}\.json', name):
            raise ValueError('exact cleanup artifact name required')
        return await self._request('GET', '/api/v1/files?mine=true&q=' + quote(name, safe=''))

    async def upload_artifact(self, *, name, group_scope, raw):
        if (not isinstance(name, str) or not re.fullmatch(r'exo-cleanup-[a-f0-9]{64}\.json', name)
                or not isinstance(group_scope, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', group_scope)
                or not isinstance(raw, bytes) or not 0 < len(raw) <= 128 * 1024):
            raise ValueError('explicit cleanup bytes/name/audience required')
        try:
            headers = await self.verify_owner()
            # No retry/idempotency claim: the journal commits the sole upload
            # decision first; reply loss is reconciled by exact file metadata.
            async with self._client.stream('POST', '/api/v1/files', headers=headers,
                    data={'group_scope': group_scope},
                    files={'file': (name, raw, 'application/octet-stream')}) as response:
                return await bounded_response(response)
        except Exception:
            raise PlaneUnavailable('cleanup publication outcome requires reconciliation') from None

    async def artifact_bytes(self, file_id):
        path = self._file(file_id) + '/raw'
        try:
            headers = await self.verify_owner()
            async with self._client.stream('GET', path, headers=headers) as response:
                if response.status_code != 200:
                    raise ValueError('evidence unavailable')
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > 128 * 1024:
                        raise ValueError('evidence exceeds bound')
                    raw.extend(chunk)
            return bytes(raw)
        except Exception:
            raise PlaneUnavailable('coordinator evidence unavailable') from None

    async def lease(self, lease_id):
        if not isinstance(lease_id, str) or not re.fullmatch(r'lease_[a-f0-9]{16,64}', lease_id):
            raise ValueError('exact member lease id required')
        return await self._request('GET', '/api/v1/leases/' + quote(lease_id, safe=''))

    async def member(self, group_id, lease_id):
        if not isinstance(lease_id, str) or not re.fullmatch(r'lease_[a-f0-9]{16,64}', lease_id):
            raise ValueError('exact member lease id required')
        return await self._request('GET', self._group(group_id) + '/members/' + lease_id + '/authority')

    async def renew(self, lease_id, ttl_seconds):
        if (not isinstance(lease_id, str) or not re.fullmatch(r'lease_[a-f0-9]{16,64}', lease_id)
                or type(ttl_seconds) is not int or ttl_seconds < 1):
            raise ValueError('exact lease and explicit positive renewal TTL required')
        return await self._request('POST', '/api/v1/leases/' + lease_id + '/renew',
                                   {'ttl_seconds': ttl_seconds})

    async def close(self):
        await self._client.aclose()
