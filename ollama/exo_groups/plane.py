"""Coordinator's exact metadata operations, using its own owner credential."""

import hashlib
import re
from urllib.parse import quote

import httpx

from .http import bounded_json, credential_headers


class PlaneUnavailable(RuntimeError):
    """Metadata outcome unknown; reconcile its saved identity, never replace it."""


class CoordinatorPlane:
    def __init__(self, owner_id, header, *, transport=None):
        self.owner_id, self.header = owner_id, header
        self._verified_digest = None
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

    async def lease(self, lease_id):
        if not isinstance(lease_id, str) or not re.fullmatch(r'lease_[a-f0-9]{16,64}', lease_id):
            raise ValueError('exact member lease id required')
        return await self._request('GET', '/api/v1/leases/' + quote(lease_id, safe=''))

    async def close(self):
        await self._client.aclose()
