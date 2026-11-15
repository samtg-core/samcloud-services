"""Event-driven host rank lifecycle; no polling loop or lease mutations.

The service adapter supplies authenticated fresh observations, existing stream
wakeups/heartbeats, child-exit events and the explicit authority-expiry alarm.
No group/idle/job deadline, automatic restart, coordinator credential or
physical cleanup inferred from an elapsed clock is provided here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .clock import authority_clock
from .evidence import write_new
from .member import (
    AuthorityDecision,
    ExpectedMember,
    check_projection,
    event_requires_read,
)
from .rank_process import (
    Listener,
    ObservationUnknown,
    OwnedRankProcess,
    RunBinding,
    cleanup_evidence,
    darwin_gpu_in_use,
    listener_closed,
    observe_cleanup,
)


@dataclass(frozen=True)
class RankSpec:
    binding: RunBinding
    argv: tuple[str, ...]
    cwd: Path
    listeners: tuple[Listener, ...]
    cleanup_tolerance_bytes: int

    def __post_init__(self):
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or any(not isinstance(value, str) or not value for value in self.argv)
            or not Path(self.argv[0]).is_absolute()
            or "--legacy-daemon" in self.argv
            or not self.cwd.is_absolute()
            or not isinstance(self.listeners, tuple)
            or not self.listeners
            or any(not isinstance(value, Listener) for value in self.listeners)
            or type(self.cleanup_tolerance_bytes) is not int
            or self.cleanup_tolerance_bytes < 0
        ):
            raise ValueError(
                "explicit foreground runtime, listeners and cleanup tolerance required"
            )


def darwin_boot_id() -> str:
    import subprocess
    import uuid

    raw = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"], text=True
    ).strip()
    return str(uuid.UUID(raw))


class RankLifecycle:
    """One exact member, one launch, one physical cleanup record.

    Calls are serialized by the owning service's event loop. Existing state is
    never reopened as a fresh run: an uncertain spawn or controller death needs
    recovery/inspection of that same attempt, not another Popen.
    """

    def __init__(
        self,
        expected: ExpectedMember,
        spec: RankSpec,
        state: Path,
        *,
        memory_reader: Callable[[], int] = darwin_gpu_in_use,
        port_reader: Callable[[Listener], bool] = listener_closed,
        boot_reader: Callable[[], str] = darwin_boot_id,
        monotonic: Callable[[], float] = authority_clock,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        spawner=OwnedRankProcess.spawn,
        emit: Callable[[dict[str, Any]], None],
    ):
        if spec.binding.expected != expected:
            raise ValueError("rank spec differs from its exact authority binding")
        state.mkdir(mode=0o700)  # A durable run directory is never replayed.
        parent = os.open(state.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        self.expected, self.spec, self.state = expected, spec, state
        self.memory_reader, self.port_reader, self.boot_reader = (
            memory_reader,
            port_reader,
            boot_reader,
        )
        self.monotonic, self.clock, self.spawner, self.emit = (
            monotonic,
            clock,
            spawner,
            emit,
        )
        self.process = None
        self.phase = "prepared"
        self.authority: AuthorityDecision | None = None
        self.last_good: AuthorityDecision | None = None
        self.serving_observed = False
        self.draining = False
        self.launch_attempted = False
        self.baseline_memory = None
        self.host_boot = None
        self.receipt = None
        self.audit_write_failures = []
        self._sequence = 0
        write_new(
            state / "prepared.json",
            {
                "state": "prepared_not_launched",
                "binding": asdict(spec.binding),
                "listeners": [asdict(value) for value in spec.listeners],
                "cleanup_tolerance_bytes": spec.cleanup_tolerance_bytes,
                "argv_sha256": hashlib.sha256(
                    json.dumps(spec.argv).encode()
                ).hexdigest(),
                "lease_mutations": 0,
            },
        )

    def _record(self, kind: str, **details) -> dict[str, Any]:
        self._sequence += 1
        value = {
            "type": kind,
            "sequence": self._sequence,
            "at": self.clock().isoformat(),
            "phase": self.phase,
            "group_id": self.spec.binding.group_id,
            "generation": self.spec.binding.generation,
            "lease_id": self.spec.binding.lease_id,
            "resource_id": self.spec.binding.resource_id,
            **details,
        }
        write_new(self.state / f"event-{self._sequence:06d}.json", value)
        # Delivery loss does not erase evidence or crash an owned rank. The
        # adapter can reconcile this durable event after a reconnect.
        try:
            self.emit(value)
        except Exception as error:
            write_new(
                self.state / f"delivery-failed-{self._sequence:06d}.json",
                {"sequence": self._sequence, "error_type": type(error).__name__},
            )
        return value

    def observe_authority(
        self,
        projection: Mapping[str, Any] | None,
        *,
        request_started_monotonic: float,
        received_monotonic: float,
    ) -> AuthorityDecision:
        decision = check_projection(
            self.expected,
            projection,
            request_started_monotonic=request_started_monotonic,
            received_monotonic=received_monotonic,
            now_monotonic=self.monotonic(),
        )
        self.authority = decision
        self.serving_observed = bool(
            decision.state == "authorized" and projection.get("phase") == "serving"
        )
        if decision.state == "authorized":
            self.last_good = decision
        elif decision.state == "stop_required":
            waiting = (
                not self.launch_attempted
                and self.last_good is None
                and decision.reason == "group is not authoritative"
                and isinstance(projection, Mapping)
                and projection.get("phase") in ("pending", "acquiring")
                and projection.get("release_requested") is False
            )
            if not waiting:
                self.request_stop(decision.reason)
        # Unknown transport revokes new admission. A running child is never
        # replayed or declared dead; the last explicit lease alarm still applies.
        self._record("authority_observed", state=decision.state, reason=decision.reason)
        self.authority_alarm()
        return decision

    def event_requires_observation(self, event: Mapping[str, Any]) -> bool:
        wake = event_requires_read(self.expected, event)
        if wake:
            self.require_observation("scoped authority event")
        return wake

    def require_observation(self, reason: str) -> None:
        """Revoke admission immediately, retaining only the explicit stop alarm."""
        self.authority = None
        self.serving_observed = False
        self.authority_alarm()
        self._record("authority_read_required", reason=reason)

    def _record_stop(self, kind: str, **details):
        """Audit I/O failure cannot prevent a required process stop.

        Retain these unconfirmed writes in memory. A later successful durable
        record must account for them before cleanup evidence can be published.
        """
        try:
            return self._record(kind, **details)
        except OSError as error:
            self.audit_write_failures.append(
                {
                    "sequence": self._sequence,
                    "type": kind,
                    "error_type": type(error).__name__,
                }
            )
            return None

    @property
    def authority_deadline(self) -> float | None:
        return self.last_good.expires_at_monotonic if self.last_good else None

    def authority_alarm(self) -> None:
        deadline = self.authority_deadline
        if deadline is not None and self.monotonic() >= deadline:
            self.request_stop("explicit member authority expired")

    def start(self, *, environment: Mapping[str, str]) -> None:
        if (
            self.launch_attempted
            or self.draining
            or self.phase != "prepared"
            or self.authority is None
            or not self.authority.allows_start(self.monotonic())
        ):
            raise ObservationUnknown(
                "rank has no fresh launch authority or launch was already attempted"
            )
        before = self.memory_reader()
        if type(before) is not int or before < 0:
            raise ObservationUnknown("physical baseline memory unknown")
        if not all(self.port_reader(value) is True for value in self.spec.listeners):
            raise ObservationUnknown("declared listener not closed before launch")
        boot = self.boot_reader()
        # Recheck after potentially slow physical reads. No observation delay
        # may convert an expired grant into permission to start a process.
        if not self.authority.allows_start(self.monotonic()):
            raise ObservationUnknown("authority elapsed during physical preflight")
        write_new(
            self.state / "launch-intent.json",
            {
                "binding": asdict(self.spec.binding),
                "host_boot": boot,
                "baseline_in_use_bytes": before,
                "at": self.clock().isoformat(),
                "processes_started": "unknown_until_observed",
            },
        )
        self.baseline_memory, self.host_boot = before, boot
        self.launch_attempted = True
        self.phase = "launching"
        try:
            with (
                (self.state / "rank.stdout").open("xb") as stdout,
                (self.state / "rank.stderr").open("xb") as stderr,
            ):
                self.process = self.spawner(
                    self.spec.argv,
                    binding=self.spec.binding,
                    authority=self.authority,
                    cwd=self.spec.cwd,
                    environment=environment,
                    stdout=stdout,
                    stderr=stderr,
                )
        except Exception:
            self.phase = "uncertain"
            self.draining = True
            self._record(
                "cleanup_required",
                reason="spawn outcome unknown; inspect this exact attempt",
            )
            raise
        self.phase = "running"
        self._record(
            "rank_started", child=asdict(self.process.identity), host_boot=boot
        )
        self.authority_alarm()

    def may_admit(self) -> bool:
        self.authority_alarm()
        return bool(
            not self.draining
            and self.phase == "running"
            and self.serving_observed
            and self.authority
            and self.authority.allows_start(self.monotonic())
            and self.process
            and self.process.exit_code() is None
        )

    def request_stop(self, reason: str) -> None:
        already_draining = self.draining
        self.draining = True
        self.serving_observed = False
        if already_draining:
            return
        self._record_stop("admission_stopped", reason=reason)
        if self.process is not None and self.process.exit_code() is None:
            try:
                if self.boot_reader() != self.host_boot:
                    raise ObservationUnknown("host boot changed")
                self.process.stop()
                self.phase = "stopping"
                self._record_stop("stop_requested", reason=reason)
            except (ObservationUnknown, OSError):
                self.phase = "uncertain"
                self._record_stop(
                    "cleanup_required",
                    reason="stop unconfirmed; inspect same process/member",
                )

    def process_exited(self) -> dict[str, Any] | None:
        """Called by the owning service's actual wait/child-exit event."""
        if self.process is not None and self.process.exit_code() is None:
            self._record("child_exit_unconfirmed", reason="owned child still running")
            return None
        self.request_stop("rank exited")
        return self.refresh_cleanup()

    def refresh_cleanup(self) -> dict[str, Any] | None:
        """Read physical evidence; never release the lease or infer settlement."""
        if self.receipt is not None:
            return self.receipt
        if self.audit_write_failures:
            if (
                self._record_stop(
                    "audit_write_failures", writes=list(self.audit_write_failures)
                )
                is None
            ):
                return None
            self.audit_write_failures.clear()
        if self.process is None or self.baseline_memory is None:
            self.phase = "uncertain"
            self._record(
                "cleanup_required",
                reason="no owned child handle; inspect launch journal",
            )
            return None
        if not self.draining and self.process.exit_code() is None:
            return None
        # Never allow a normal child exit to leave admission enabled.
        if self.process.exit_code() is not None:
            self.draining = True
            self.serving_observed = False
        try:
            if self.boot_reader() != self.host_boot:
                raise ObservationUnknown("host boot changed")
            observation = observe_cleanup(
                self.process,
                listeners=self.spec.listeners,
                baseline_in_use_bytes=self.baseline_memory,
                tolerance_bytes=self.spec.cleanup_tolerance_bytes,
                observed_at=self.clock().isoformat(),
                memory_reader=self.memory_reader,
                port_reader=self.port_reader,
            )
            if not observation.clean:
                raise ObservationUnknown("process/listener/memory cleanup incomplete")
            raw, digest = cleanup_evidence(observation)
        except (ObservationUnknown, OSError):
            self._record(
                "cleanup_required", reason="physical cleanup incomplete or unobservable"
            )
            return None
        path = self.state / "cleanup-evidence.json"
        write_new(path, json.loads(raw))
        # Hash the exact file that the coordinator will fetch, not an earlier
        # serialization with different whitespace/newline bytes.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.phase = "awaiting_settlement"
        self.draining = True
        self.receipt = self._record(
            "cleanup_required",
            reason="physical cleanup observed; exact owner settlement required",
            evidence_path=str(path),
            evidence_sha256=digest,
            lease_released=False,
            settlement_recorded=False,
        )
        return self.receipt
