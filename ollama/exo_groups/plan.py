"""Bind an EXO placement to exact, freshly read ordinary group leases.

This module performs no HTTP, process launch, renewal, release or inference.
The coordinator supplies the group and every exact lease read. Successful
validation is a launch prerequisite, never evidence that a rank is running.
Acquisition order and pipeline rank order are deliberately independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping


HOST_RESOURCES = {
    "slice": "claude-services-slice/gpu-0",
    "wafer": "wafer-services/gpu-metal",
    "obelisk": "obelisk-services/gpu-0",
}
PIPELINES = (("slice", "obelisk"), ("slice", "wafer", "obelisk"))
_SHA = re.compile(r"[a-f0-9]{64}")


class BindingRefused(ValueError):
    """No authority was established; do not start or substitute a rank."""


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise BindingRefused(f"{name} must be an explicit integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BindingRefused(f"{name} must be a nonempty exact string")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise BindingRefused(f"{name} must be a SHA256 digest")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    try:
        instant = datetime.fromisoformat(_text(value, name))
    except ValueError as error:
        raise BindingRefused(f"{name} must be an aware ISO timestamp") from error
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise BindingRefused(f"{name} must be an aware ISO timestamp")
    return instant


@dataclass(frozen=True)
class MemberPlan:
    rank: str
    resource_id: str
    pipeline_rank: int
    layers: tuple[int, int]
    memory_mb: int
    ttl_seconds: int
    max_total_s: int

    def request(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "memory_mb": self.memory_mb,
            "ttl_seconds": self.ttl_seconds,
            "max_total_s": self.max_total_s,
            "service_id": None,
        }


@dataclass(frozen=True)
class GroupPlan:
    request_key: str
    owner_id: str
    model_id: str
    revision: str
    rank_manifest_sha256: str
    members: tuple[MemberPlan, ...]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> GroupPlan:
        if set(raw) != {
            "request_key",
            "owner_id",
            "model_id",
            "revision",
            "rank_manifest_sha256",
            "members",
        }:
            raise BindingRefused("qualification plan has missing or unknown fields")
        request_key = _text(raw["request_key"], "request_key")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_key):
            raise BindingRefused("request_key is not a stable explicit identifier")
        owner = _text(raw["owner_id"], "owner_id")
        revision = _text(raw["revision"], "revision")
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise BindingRefused("model revision must be pinned")
        entries = raw["members"]
        if not isinstance(entries, list) or len(entries) not in (2, 3):
            raise BindingRefused("qualification requires an explicit two- or three-host pipeline")
        members = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "rank",
                "resource_id",
                "pipeline_rank",
                "layers",
                "memory_mb",
                "ttl_seconds",
                "max_total_s",
            }:
                raise BindingRefused("member has missing or unknown fields")
            rank = _text(entry["rank"], "rank")
            if (
                rank not in HOST_RESOURCES
                or entry["resource_id"] != HOST_RESOURCES[rank]
            ):
                raise BindingRefused("rank must use its canonical host HOLD resource")
            layers = entry["layers"]
            if not isinstance(layers, list) or len(layers) != 2:
                raise BindingRefused("member needs its explicit layer range")
            start = _integer(layers[0], "start layer", minimum=0)
            end = _integer(layers[1], "end layer")
            if end <= start:
                raise BindingRefused("member layer range is empty")
            ttl = _integer(entry["ttl_seconds"], "ttl_seconds")
            total = _integer(entry["max_total_s"], "max_total_s")
            if ttl > total:
                raise BindingRefused(
                    "lease TTL exceeds the explicit qualification envelope"
                )
            # Current ordinary exclusive leases have a registry hard ceiling.
            # Refuse an unsupported plan before acquisition; never let the
            # registry silently clamp an intended longer-arc qualification.
            if total > 21600:
                raise BindingRefused(
                    "qualification exceeds the current registry lease ceiling"
                )
            members.append(
                MemberPlan(
                    rank,
                    entry["resource_id"],
                    _integer(entry["pipeline_rank"], "pipeline_rank", minimum=0),
                    (start, end),
                    _integer(entry["memory_mb"], "memory_mb"),
                    ttl,
                    total,
                )
            )
        if len({member.rank for member in members}) != len(members):
            raise BindingRefused("a host is duplicated or missing")
        ordered = sorted(members, key=lambda member: member.pipeline_rank)
        if (tuple(member.rank for member in ordered) not in PIPELINES
                or [member.pipeline_rank for member in ordered] != list(range(len(members)))):
            raise BindingRefused("pipeline ranks must be unique and contiguous")
        end = 0
        for member in ordered:
            if member.layers[0] != end:
                raise BindingRefused("pipeline has a layer gap or overlap")
            end = member.layers[1]
        return cls(
            request_key,
            owner,
            _text(raw["model_id"], "model_id"),
            revision,
            _digest(raw["rank_manifest_sha256"], "rank manifest"),
            tuple(members),
        )

    @property
    def pipeline_ranks(self) -> tuple[str, ...]:
        ordered = sorted(self.members, key=lambda member: member.pipeline_rank)
        names = tuple(member.rank for member in ordered)
        if names not in PIPELINES or tuple(member.pipeline_rank for member in ordered) != tuple(range(len(names))):
            raise BindingRefused('exact two- or three-host pipeline required')
        return names

    def request(self) -> dict[str, Any]:
        # No priority boost, wait deadline, assembly deadline or idle grace.
        return {
            "request_key": self.request_key,
            "members": [member.request() for member in self.members],
        }

    @property
    def intent_sha256(self) -> str:
        encoded = json.dumps(
            self.request(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LaunchBinding:
    group_id: str
    generation: int
    owner_id: str
    rank: str
    pipeline_rank: int
    resource_id: str
    lease_id: str
    layers: tuple[int, int]
    expires_at: str
    rank_manifest_sha256: str


def bind_admitted_group(
    plan: GroupPlan,
    group: Mapping[str, Any],
    leases: Mapping[str, Mapping[str, Any]],
    *,
    expected_generation: int,
    now: datetime,
) -> tuple[LaunchBinding, ...]:
    """Validate fresh API reads; never infer permission from local state alone.

    An observation failure is handled by the caller as unknown, not replaced
    with an older saved lease. TTL is lease authority, not an inference timer.
    The returned bindings are in pipeline order, not acquisition order.
    """
    _integer(expected_generation, "expected_generation")
    if now.tzinfo is None or now.utcoffset() is None:
        raise BindingRefused("the observation time must be timezone-aware")
    if (
        type(group.get("generation")) is not int
        or group.get("generation") != expected_generation
    ):
        raise BindingRefused(
            "coordinator generation changed; reconcile, never rebind silently"
        )
    if (
        group.get("phase") not in ("admitted", "serving")
        or group.get("release_requested") is not False
    ):
        raise BindingRefused("all members must be admitted and not releasing")
    if (
        group.get("owner_id") != plan.owner_id
        or group.get("request_key") != plan.request_key
        or group.get("intent_sha256") != plan.intent_sha256
    ):
        raise BindingRefused("group does not match the pinned owner and exact intent")
    group_id = _text(group.get("id"), "group id")
    members = group.get("members")
    if not isinstance(members, list) or len(members) != len(plan.members):
        raise BindingRefused("group member set differs from the placement")
    result = []
    seen = set()
    for ordinal, (expected, member) in enumerate(zip(plan.members, members)):
        if (
            not isinstance(member, dict)
            or type(member.get("ordinal")) is not int
            or member.get("ordinal") != ordinal
            or member.get("resource_id") != expected.resource_id
            or member.get("status") != "active"
            or member.get("cleanup_evidence") is not None
            or member.get("release_receipt") is not None
        ):
            raise BindingRefused("member is changed, incomplete or already cleaned up")
        lease_id = _text(member.get("lease_id"), "member lease id")
        if lease_id in seen:
            raise BindingRefused("a lease cannot stand in for two members")
        seen.add(lease_id)
        lease = leases.get(lease_id)
        if not isinstance(lease, Mapping):
            raise BindingRefused("every exact member lease needs a fresh read")
        if (
            lease.get("id") != lease_id
            or lease.get("resource_id") != expected.resource_id
            or lease.get("agent_id") != plan.owner_id
            or lease.get("status") != "active"
            or lease.get("exclusive") is not True
            or lease.get("priority") != "normal"
            or lease.get("memory_mb") != expected.memory_mb
            or lease.get("max_total_s") != expected.max_total_s
            or lease.get("ttl_s") != expected.ttl_seconds
            or lease.get("service_id") is not None
            or lease.get("released_at") is not None
            or lease.get("settled_at") is not None
        ):
            raise BindingRefused(
                "exact member lease does not match its declared authority"
            )
        expiry = _timestamp(lease.get("expires_at"), "lease expires_at")
        granted = _timestamp(lease.get("granted_at"), "lease granted_at")
        if expiry <= now or granted > now or expiry <= granted:
            raise BindingRefused("member lease is not currently authoritative")
        if member.get("expires_at") != lease["expires_at"]:
            raise BindingRefused("group and lease reads raced; read both again")
        result.append(
            LaunchBinding(
                group_id,
                expected_generation,
                plan.owner_id,
                expected.rank,
                expected.pipeline_rank,
                expected.resource_id,
                lease_id,
                expected.layers,
                lease["expires_at"],
                plan.rank_manifest_sha256,
            )
        )
    if set(leases) != seen:
        raise BindingRefused("lease observation includes unrelated or substituted work")
    return tuple(sorted(result, key=lambda item: item.pipeline_rank))


def check_rank_manifest(plan: GroupPlan, manifest_bytes: bytes) -> None:
    """Join the immutable placement to actual pinned artifact metadata."""
    if hashlib.sha256(manifest_bytes).hexdigest() != plan.rank_manifest_sha256:
        raise BindingRefused("rank manifest differs from the plan")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("model_id") != plan.model_id
        or manifest.get("revision") != plan.revision
    ):
        raise BindingRefused("rank manifest is a different model or revision")
    actual = manifest.get("ranks")
    if not isinstance(actual, list) or len(actual) != len(plan.pipeline_ranks):
        raise BindingRefused("rank manifest must cover every planned host")
    ordered = sorted(plan.members, key=lambda item: item.pipeline_rank)
    for rank, member in zip(actual, ordered):
        if rank.get("name") != member.rank or rank.get("layers") != list(member.layers):
            raise BindingRefused("rank metadata differs from the declared pipeline")
