"""Validate fresh, authenticated host-local group observations.

No HTTP, credential, process, renewal or release effects. An event only requests
a fresh read. A decision is process-local and must never be persisted as launch
authority. Unknown/lost authority grants no replacement or cleanup permission.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping

from .plan import BindingRefused, GroupPlan, LaunchBinding, MemberPlan


@dataclass(frozen=True)
class ExpectedMember:
    binding: LaunchBinding
    member: MemberPlan

    @classmethod
    def from_launch(cls, plan: GroupPlan, binding: LaunchBinding):
        matching = [member for member in plan.members if member.rank == binding.rank]
        if len(matching) != 1:
            raise BindingRefused("launch rank is not in the pinned group plan")
        member = matching[0]
        if (
            plan.owner_id != binding.owner_id
            or plan.rank_manifest_sha256 != binding.rank_manifest_sha256
            or member.resource_id != binding.resource_id
            or member.layers != binding.layers
            or member.pipeline_rank != binding.pipeline_rank
        ):
            raise BindingRefused("host launch binding differs from its pinned plan")
        return cls(binding, member)


@dataclass(frozen=True)
class AuthorityDecision:
    state: Literal["authorized", "stop_required", "unknown"]
    reason: str
    checked_at_monotonic: float
    expires_at_monotonic: float | None = None
    expected: ExpectedMember | None = None

    def allows_start(self, now_monotonic: float) -> bool:
        return bool(
            self.state == "authorized"
            and self.expires_at_monotonic is not None
            and _finite(now_monotonic)
            and self.checked_at_monotonic <= now_monotonic < self.expires_at_monotonic
        )


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _instant(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("lease time must be an aware ISO timestamp")
    instant = datetime.fromisoformat(value)
    if instant.utcoffset() is None:
        raise ValueError("lease time has no offset")
    return instant


def check_projection(
    expected: ExpectedMember,
    projection: Mapping[str, Any] | None,
    *,
    request_started_monotonic: float,
    received_monotonic: float,
    now_monotonic: float,
) -> AuthorityDecision:
    """Consume a fresh HTTPS result; never use an event or saved JSON as one.

    Conservative monotonic authority is derived from the server's explicit
    expiry minus its observation time, subtracting the entire request duration.
    This needs no synchronized host wall clock or default observation/work cap.
    A slow observation can exhaust authority; it cannot extend that authority.
    """

    def decision(state, reason, expiry=None):
        return AuthorityDecision(state, reason, now_monotonic, expiry, expected)

    if (
        not all(
            _finite(value)
            for value in (request_started_monotonic, received_monotonic, now_monotonic)
        )
        or not request_started_monotonic <= received_monotonic <= now_monotonic
    ):
        return decision("unknown", "invalid observation clock")
    if not isinstance(projection, Mapping):
        return decision("unknown", "member observation unavailable")
    binding, member = expected.binding, expected.member
    if (
        projection.get("group_id") != binding.group_id
        or projection.get("owner_id") != binding.owner_id
    ):
        return decision("stop_required", "group binding changed")
    generation = projection.get("generation")
    if type(generation) is not int:
        return decision("unknown", "invalid generation")
    if generation != binding.generation:
        return decision("stop_required", "coordinator generation changed")
    if (
        type(projection.get("authoritative")) is not bool
        or type(projection.get("release_requested")) is not bool
    ):
        return decision("unknown", "invalid authority observation")
    if (
        projection["authoritative"] is False
        or projection["release_requested"]
        or projection.get("phase") not in ("admitted", "serving")
    ):
        return decision("stop_required", "group is not authoritative")
    own = projection.get("member")
    if not isinstance(own, Mapping):
        return decision("unknown", "member projection missing")
    if (
        own.get("resource_id") != binding.resource_id
        or own.get("lease_id") != binding.lease_id
    ):
        return decision("stop_required", "member binding changed")
    if (
        type(own.get("cleanup_attested")) is not bool
        or type(own.get("settlement_recorded")) is not bool
    ):
        return decision("unknown", "invalid cleanup observation")
    if own["cleanup_attested"] or own["settlement_recorded"]:
        return decision("stop_required", "member cleanup already recorded")
    lease = own.get("lease")
    if not isinstance(lease, Mapping):
        return decision("unknown", "exact lease unavailable")
    if not {"service_id", "released_at", "settled_at"}.issubset(lease):
        return decision("unknown", "incomplete exact lease observation")
    if (
        lease.get("id") != binding.lease_id
        or lease.get("resource_id") != binding.resource_id
        or lease.get("agent_id") != binding.owner_id
        or lease.get("exclusive") is not True
        or lease.get("priority") != "normal"
        or lease.get("service_id") is not None
    ):
        return decision("stop_required", "lease binding changed")
    for name, value in (
        ("memory_mb", member.memory_mb),
        ("ttl_s", member.ttl_seconds),
        ("max_total_s", member.max_total_s),
    ):
        if type(lease.get(name)) is not int or lease[name] != value:
            return decision("stop_required", "declared lease envelope changed")
    if (
        lease.get("status") != "active"
        or lease.get("released_at") is not None
        or lease.get("settled_at") is not None
    ):
        return decision("stop_required", "exact lease is not active")
    try:
        observed = _instant(projection.get("observed_at"))
        expires = _instant(lease.get("expires_at"))
        authority_expires = _instant(projection.get("authority_expires_at"))
        granted = _instant(lease.get("granted_at"))
    except (ValueError, TypeError):
        return decision("unknown", "lease clock unavailable")
    if not granted <= observed < authority_expires <= expires:
        return decision("stop_required", "lease is outside its granted authority")
    remaining = (authority_expires - observed).total_seconds()
    if (
        remaining > member.ttl_seconds
        or (expires - granted).total_seconds() > member.max_total_s
    ):
        return decision("unknown", "lease authority exceeds its declared envelope")
    deadline = request_started_monotonic + remaining
    if now_monotonic >= deadline:
        return decision("stop_required", "lease authority elapsed during observation")
    return decision("authorized", "exact admitted member observed", deadline)


def settlement_confirmed(expected, projection, *, require_fence=False, minimum_generation=None):
    """True only for the exact released group and settled member allocation."""
    binding = expected.binding
    if not isinstance(projection, Mapping):
        return False
    observed = projection.get('member')
    if not isinstance(observed, Mapping) or not isinstance(observed.get('lease'), Mapping):
        return False
    lease = observed['lease']
    generation = projection.get('generation')
    # A successor can settle the original member after fencing its coordinator.
    return (
        projection.get('group_id') == binding.group_id
        and projection.get('owner_id') == binding.owner_id
        and type(generation) is int
        and generation >= binding.generation + int(require_fence)
        and (minimum_generation is None or generation >= minimum_generation)
        and projection.get('phase') == 'released'
        and projection.get('release_requested') is True
        and observed.get('resource_id') == binding.resource_id
        and observed.get('lease_id') == binding.lease_id
        and observed.get('cleanup_attested') is True
        and observed.get('settlement_recorded') is True
        and lease.get('id') == binding.lease_id
        and lease.get('resource_id') == binding.resource_id
        and lease.get('agent_id') == binding.owner_id
        and lease.get('exclusive') is True
        and lease.get('memory_mb') == expected.member.memory_mb
        and (lease.get('status') in ('released', 'cancelled')
             or (lease.get('status') == 'expired' and bool(lease.get('settled_at'))))
    )


def event_requires_read(expected: ExpectedMember, event: Mapping[str, Any]) -> bool:
    """A scoped event is a hint to reconcile, never permission or proof of exit."""
    data = event.get("data")
    if not isinstance(data, Mapping):
        return False
    binding = expected.binding
    kind = event.get("type")
    if not isinstance(kind, str) or data.get("resource_id") != binding.resource_id:
        return False
    if kind.startswith("resource."):
        return event.get("target") == binding.resource_id
    if not kind.startswith("lease.") or event.get("target") != binding.lease_id:
        return False
    if data.get("lease_id") != binding.lease_id:
        return False
    return kind != "lease.group_changed" or data.get("group_id") == binding.group_id
