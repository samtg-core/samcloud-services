"""Read exact EXO group authority for a model-service provider.

The coordinator supplies its own credential callback and the frozen launch
bindings. This component only reads the plane; it never acquires/releases a
resource, starts EXO or authorizes a caller-supplied placement. Queue integration
must additionally confirm the exact instance and runner readiness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from typing import Callable
from urllib.parse import quote

import httpx

from .clock import authority_clock
from .http import bounded_json, credential_headers
from .member import AuthorityDecision, ExpectedMember, check_projection, event_requires_read
from .plan import BindingRefused, GroupPlan, LaunchBinding


class AuthorityReadUnavailable(RuntimeError):
    """No new authority established; preserve existing physical ownership."""


@dataclass(frozen=True)
class GroupObservation:
    expected: tuple[ExpectedMember, ...]
    decisions: tuple[AuthorityDecision, ...]
    checked_at: float
    phases: tuple[str, ...]

    def permits(self, now: float) -> bool:
        return bool(
            len(self.expected) == len(self.decisions) == 3
            and type(now) in (int, float) and math.isfinite(now)
            and self.checked_at <= now
            and all(decision.expected == expected and decision.allows_start(now)
                    for expected, decision in zip(self.expected, self.decisions))
        )

    def permits_requests(self, now: float) -> bool:
        return self.permits(now) and self.phases == ('serving', 'serving', 'serving')


class GroupAuthorityReader:
    """One pinned group/generation; an event invalidates and requests a read.

    There is no polling/reconnect loop or default observation/compute timeout.
    Dispatch permission checks the continuous clock, including host suspend.
    A slow read cannot extend any member's authority, and a group event received
    while a read is in flight prevents that old observation being republished.
    """

    def __init__(self, plan: GroupPlan, launches: tuple[LaunchBinding, ...],
                 header: Callable[[], str], *, clock=authority_clock, transport=None):
        if not isinstance(plan, GroupPlan):
            raise TypeError('provider-owned GroupPlan required')
        expected = tuple(ExpectedMember.from_launch(plan, binding) for binding in launches)
        if (len(expected) != 3
                or tuple(item.binding.pipeline_rank for item in expected) != (0, 1, 2)
                or len({item.binding.group_id for item in expected}) != 1
                or len({item.binding.generation for item in expected}) != 1
                or len({item.binding.lease_id for item in expected}) != 3
                or any(type(item.binding.generation) is not int or item.binding.generation < 1
                       or type(item.binding.pipeline_rank) is not int for item in expected)
                or {item.binding.rank for item in expected} != {member.rank for member in plan.members}):
            raise BindingRefused('all three exact pipeline-ordered member bindings required')
        self.expected = expected
        self.owner_id = plan.owner_id
        self._header, self._clock = header, clock
        self._verified_digest = None
        self._current = None
        self._epoch = 0
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(base_url='https://cloud.samtg.xyz', timeout=None,
            follow_redirects=False, trust_env=False, transport=transport)

    @property
    def current(self) -> GroupObservation | None:
        return self._current

    def invalidate(self) -> None:
        self._epoch += 1
        self._current = None

    def on_event(self, event: Mapping) -> bool:
        """Withdraw on any exact member/resource hint; caller owns fresh read.

        No event payload establishes authority. The event consumer must also
        call invalidate on contact loss or a gap in its durable event cursor.
        """
        if isinstance(event, Mapping) and any(
                event_requires_read(expected, event) for expected in self.expected):
            self.invalidate()
            return True
        return False

    def _permitted(self, requests: bool) -> bool:
        try:
            if self._current is None:
                return False
            now = self._clock()
            return self._current.permits_requests(now) if requests else self._current.permits(now)
        except Exception:
            self.invalidate()
            return False

    def may_start_ranks(self) -> bool:
        return self._permitted(False)

    def may_serve_requests(self) -> bool:
        """Lease prerequisite only; exact EXO instance/readiness is additional."""
        return self._permitted(True)

    def _headers(self) -> dict[str, str]:
        return credential_headers(self._header)

    async def _json(self, path: str, headers: dict[str, str]) -> dict:
        return await bounded_json(self._client, 'GET', path, headers)

    async def refresh(self) -> GroupObservation:
        async with self._lock:
            self.invalidate()
            epoch = self._epoch
            try:
                headers = self._headers()
                digest = hashlib.sha256(headers['Authorization'].encode()).digest()
                if digest != self._verified_digest:
                    identity = await self._json('/api/v1/auth/verify', headers)
                    if identity.get('valid') is not True or identity.get('owner_id') != self.owner_id:
                        raise AuthorityReadUnavailable('credential differs from group owner')
                    self._verified_digest = digest
                decisions, phases = [], []
                for expected in self.expected:
                    binding = expected.binding
                    path = ('/api/v1/lease-groups/' + quote(binding.group_id, safe='')
                            + '/members/' + quote(binding.lease_id, safe='') + '/authority')
                    started = self._clock()
                    projection = await self._json(path, headers)
                    received = self._clock()
                    phases.append(projection.get('phase'))
                    decisions.append(check_projection(expected, projection,
                        request_started_monotonic=started, received_monotonic=received,
                        now_monotonic=self._clock()))
                now = self._clock()
                if type(now) not in (int, float) or not math.isfinite(now):
                    raise AuthorityReadUnavailable('continuous authority clock unavailable')
                observation = GroupObservation(self.expected, tuple(decisions), now, tuple(phases))
                if epoch != self._epoch:
                    raise AuthorityReadUnavailable('authority changed during observation')
                self._current = observation
                return observation
            except Exception:
                self.invalidate()
                # No HTTP body, credentials, prompts or parser details in diagnostics.
                raise AuthorityReadUnavailable('fresh group authority unavailable') from None

    async def close(self) -> None:
        self.invalidate()
        await self._client.aclose()
