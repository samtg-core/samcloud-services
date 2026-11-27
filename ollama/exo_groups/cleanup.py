"""Exact host cleanup evidence, pinned to the actual launch before settlement.

Pure validation only. No credential, process, lease write or elapsed-time proof.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import Any, Mapping
from uuid import UUID
import re

from .http import decode_json
from .plan import BindingRefused
from .prepare import canonical
from .rank_process import CleanupObservation, Listener, ProcessIdentity, RunBinding

def _aware(value: Any) -> str:
    if not isinstance(value, str):
        raise BindingRefused("cleanup time is missing")
    try:
        instant = datetime.fromisoformat(value)
    except ValueError as error:
        raise BindingRefused("cleanup time is not an ISO timestamp") from error
    if instant.utcoffset() is None:
        raise BindingRefused("cleanup time needs an explicit offset")
    return value


@dataclass(frozen=True)
class CleanupExpectation:
    """Pin BEFORE cleanup from the actual host launch journal/spec, not its report."""

    binding: RunBinding
    child: ProcessIdentity
    listeners: tuple[Listener, ...]
    baseline_in_use_bytes: int
    tolerance_bytes: int

    def __post_init__(self):
        if (
            not isinstance(self.binding, RunBinding)
            or not isinstance(self.child, ProcessIdentity)
            or type(self.listeners) is not tuple
            or not self.listeners
            or len(set(self.listeners)) != len(self.listeners)
            or any(not isinstance(item, Listener) for item in self.listeners)
            or type(self.baseline_in_use_bytes) is not int
            or self.baseline_in_use_bytes < 0
            or type(self.tolerance_bytes) is not int
            or self.tolerance_bytes < 0
        ):
            raise BindingRefused(
                "frozen launch baseline, tolerance and complete listeners required"
            )


@dataclass(frozen=True)
class NoStartExpectation:
    """Actual prepared host attempt, pinned before drain; no fictional child."""

    binding: RunBinding
    host_boot: str
    launch_sha256: str
    attempt_sha256: str
    listeners: tuple[Listener, ...]
    baseline_in_use_bytes: int
    tolerance_bytes: int
    prepared_at: str

    def __post_init__(self):
        try:
            boot = UUID(self.host_boot)
        except (ValueError, TypeError, AttributeError):
            raise BindingRefused('exact host boot required for a prepared attempt') from None
        if (not isinstance(self.binding, RunBinding) or str(boot) != self.host_boot or boot.int == 0
                or any(not isinstance(v, str) or not re.fullmatch(r'[a-f0-9]{64}', v)
                       for v in (self.launch_sha256, self.attempt_sha256))
                or type(self.listeners) is not tuple or not self.listeners
                or any(not isinstance(v, Listener) for v in self.listeners)
                or len(set(self.listeners)) != len(self.listeners)
                or type(self.baseline_in_use_bytes) is not int or self.baseline_in_use_bytes < 0
                or type(self.tolerance_bytes) is not int or self.tolerance_bytes < 0):
            raise BindingRefused('exact prepared attempt, listener set and memory baseline required')
        _aware(self.prepared_at)


def validate_cleanup(raw: bytes, pinned: CleanupExpectation | NoStartExpectation) -> dict[str, Any]:
    """Validate the real host actor's shape against its frozen launch binding."""
    if not isinstance(raw, bytes) or len(raw) > 128 * 1024:
        raise BindingRefused("cleanup artifact exceeds the evidence bound")
    try:
        value = decode_json(raw)
    except (ValueError, UnicodeError) as error:
        raise BindingRefused("cleanup artifact is not JSON") from error
    expected = pinned.binding
    binding = json.loads(canonical(asdict(expected)))
    if (
        not isinstance(value, dict)
        or value.get("kind") != ('observed_rank_never_started'
                                if isinstance(pinned, NoStartExpectation) else 'observed_rank_cleanup')
        or canonical({key: value.get(key) for key in binding}) != canonical(binding)
        or value.get("lease_released") is not False
        or value.get("settlement_recorded") is not False
    ):
        raise BindingRefused("cleanup artifact differs from the frozen launch")
    observation = value.get("observation")
    if not isinstance(observation, dict) or canonical(
        observation.get("binding")
    ) != canonical(binding):
        raise BindingRefused("cleanup observation names another member or runtime")
    if isinstance(pinned, NoStartExpectation):
        _validate_no_start(observation, pinned)
        return value
    child = observation.get("child")
    fields = tuple(ProcessIdentity.__dataclass_fields__)
    if (
        not isinstance(child, dict)
        or set(child) != set(fields)
        or any(type(child.get(name)) is not int for name in fields)
        or any(child[name] < 0 for name in fields)
        or any(child[name] == 0 for name in ("pid", "group_id", "started_seconds"))
        or child.get("started_microseconds", 1000000) >= 1000000
    ):
        raise BindingRefused("cleanup has no exact child birth identity")
    if canonical(child) != canonical(asdict(pinned.child)):
        raise BindingRefused(
            "cleanup child differs from the actual frozen launch identity"
        )
    if not isinstance(observation.get("group_members"), list):
        raise BindingRefused("cleanup process-group observation is missing")
    ports = observation.get("listeners_closed")
    if not isinstance(ports, list) or len(ports) != len(pinned.listeners):
        raise BindingRefused("cleanup listener set differs from the frozen launch")
    if canonical(observation.get("baseline_in_use_bytes")) != canonical(
        pinned.baseline_in_use_bytes
    ) or canonical(observation.get("tolerance_bytes")) != canonical(
        pinned.tolerance_bytes
    ):
        raise BindingRefused(
            "cleanup baseline/tolerance differs from the frozen launch"
        )
    checked = CleanupObservation(
        binding=expected,
        child=ProcessIdentity(**child),
        exit_code=observation.get("exit_code"),
        group_members=tuple(observation["group_members"]),
        listeners_closed=tuple(ports),
        baseline_in_use_bytes=observation.get("baseline_in_use_bytes"),
        observed_in_use_bytes=observation.get("observed_in_use_bytes"),
        tolerance_bytes=observation.get("tolerance_bytes"),
        observed_at=_aware(observation.get("observed_at")),
    )
    if not checked.clean:
        raise BindingRefused("physical cleanup report is incomplete; do not settle")
    return value


def _validate_no_start(observation, pinned):
    exact = {k: v for k, v in asdict(pinned).items() if k not in ('binding', 'listeners')}
    if (any(type(observation.get(k)) is not type(v) or observation.get(k) != v
            for k, v in exact.items())
            or observation.get('launch_fenced') is not True
            or observation.get('launch_intent_absent') is not True
            or observation.get('launch_attempted') is not False
            or type(observation.get('processes_started')) is not int or observation['processes_started'] != 0
            or observation.get('attempt_phase') != 'launch_fenced'
            or 'child' in observation or 'exit_code' in observation):
        raise BindingRefused('never-started evidence differs from the actual prepared/fenced attempt')
    ports, memory = observation.get('listeners_closed'), observation.get('observed_in_use_bytes')
    if (not isinstance(ports, list) or len(ports) != len(pinned.listeners)
            or any(v is not True for v in ports) or type(memory) is not int or memory < 0
            or memory > pinned.baseline_in_use_bytes + pinned.tolerance_bytes):
        raise BindingRefused('never-started listener or device observation is unconfirmed')
    _aware(observation.get('observed_at'))


def _ended(lease: Mapping[str, Any]) -> bool:
    return bool(
        lease.get("status") == "released"
        and lease.get("released_at") is not None
        or lease.get("status") == "expired"
        and lease.get("settled_at") is not None
    )
