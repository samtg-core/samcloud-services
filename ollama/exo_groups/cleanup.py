"""Exact host cleanup evidence, pinned to the actual launch before settlement.

Pure validation only. No credential, process, lease write or elapsed-time proof.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import Any, Mapping

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


def validate_cleanup(raw: bytes, pinned: CleanupExpectation) -> dict[str, Any]:
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
        or value.get("kind") != "observed_rank_cleanup"
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


def _ended(lease: Mapping[str, Any]) -> bool:
    return bool(
        lease.get("status") == "released"
        and lease.get("released_at") is not None
        or lease.get("status") == "expired"
        and lease.get("settled_at") is not None
    )
