"""Owned, non-daemonizing rank process and physical cleanup observations.

No HTTP, credentials, lease writes, automatic restart or job deadline. The host
supervisor supplies authority separately. This containment assumes the pinned
runtime does not detach children (EXO's legacy-daemon mode is forbidden). Process
exit alone never establishes cleanup: group, listeners and memory must agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import plistlib
import signal
import socket
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from .clock import authority_clock
from .member import AuthorityDecision, ExpectedMember
from .rank_lifeline import GuardLaunch, prepare_guard


class ObservationUnknown(RuntimeError):
    """No absence or cleanup established; retain the exact claim."""


@dataclass(frozen=True)
class RunBinding:
    group_id: str
    generation: int
    lease_id: str
    resource_id: str
    rank_manifest_sha256: str
    runtime_sha256: str
    expected: ExpectedMember

    def __post_init__(self):
        import re

        binding = self.expected.binding
        if (
            self.group_id,
            self.generation,
            self.lease_id,
            self.resource_id,
            self.rank_manifest_sha256,
        ) != (
            binding.group_id,
            binding.generation,
            binding.lease_id,
            binding.resource_id,
            binding.rank_manifest_sha256,
        ) or not re.fullmatch(r"[a-f0-9]{64}", self.runtime_sha256):
            raise ValueError("rank process binding differs from the pinned member")

    @classmethod
    def from_expected(cls, expected: ExpectedMember, runtime_sha256: str):
        import re

        if not re.fullmatch(r"[a-f0-9]{64}", runtime_sha256):
            raise ValueError("pinned runtime digest required")
        binding = expected.binding
        return cls(
            binding.group_id,
            binding.generation,
            binding.lease_id,
            binding.resource_id,
            binding.rank_manifest_sha256,
            runtime_sha256,
            expected,
        )


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    group_id: int
    uid: int
    started_seconds: int
    started_microseconds: int

    @property
    def birth(self) -> tuple[int, int]:
        return self.started_seconds, self.started_microseconds


class _BSDInfo(ctypes.Structure):
    # Darwin SDK sys/proc_info.h, proc_bsdinfo / PROC_PIDTBSDINFO (3).
    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in (
                "flags",
                "status",
                "xstatus",
                "pid",
                "ppid",
                "uid",
                "gid",
                "ruid",
                "rgid",
                "svuid",
                "svgid",
                "reserved",
            )
        ]
        + [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
        + [
            (name, ctypes.c_uint32)
            for name in (
                "nfiles",
                "pgid",
                "jobc",
                "tdev",
                "tpgid",
            )
        ]
        + [
            ("nice", ctypes.c_int32),
            ("start_seconds", ctypes.c_uint64),
            ("start_microseconds", ctypes.c_uint64),
        ]
    )


def darwin_process(pid: int) -> ProcessIdentity | None:
    """Kernel birth identity; permission/short reads never mean absence."""
    if sys.platform != "darwin" or type(pid) is not int or pid <= 0:
        raise ObservationUnknown("Darwin process identity unavailable")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    read = library.proc_pidinfo
    read.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    read.restype = ctypes.c_int
    value = _BSDInfo()
    ctypes.set_errno(0)
    size = read(pid, 3, 0, ctypes.byref(value), ctypes.sizeof(value))
    if size == 0 and ctypes.get_errno() == errno.ESRCH:
        return None
    if size != ctypes.sizeof(value) or value.pid != pid:
        raise ObservationUnknown("kernel process identity not readable")
    return ProcessIdentity(
        value.pid,
        value.ppid,
        value.pgid,
        value.uid,
        value.start_seconds,
        value.start_microseconds,
    )


def darwin_group(group_id: int) -> tuple[ProcessIdentity, ...]:
    """Inspect the dedicated process group; incomplete observations refuse."""
    if type(group_id) is not int or group_id <= 1:
        raise ObservationUnknown("invalid owned process group")
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid="], capture_output=True, text=True, check=False
    )
    if result.returncode or result.stderr.strip():
        raise ObservationUnknown("process group inventory unavailable")
    members = []
    for line in result.stdout.splitlines():
        try:
            pid, group = map(int, line.split())
        except ValueError as error:
            raise ObservationUnknown("process inventory malformed") from error
        if group == group_id:
            member = darwin_process(pid)
            if member is not None:
                if member.group_id != group_id:
                    raise ObservationUnknown("process changed group during inventory")
                members.append(member)
    return tuple(members)


def gpu_in_use_from_plist(raw: bytes) -> int:
    """Read the observed AGX device counter; missing/ambiguous is never zero."""
    try:
        entries = plistlib.loads(raw)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError("one accelerator required")
        value = entries[0]["PerformanceStatistics"]["In use system memory"]
        if type(value) is not int or value < 0:
            raise ValueError("invalid memory counter")
        return value
    except (ValueError, TypeError, KeyError, plistlib.InvalidFileException) as error:
        raise ObservationUnknown("device memory observation unavailable") from error


def darwin_gpu_in_use() -> int:
    result = subprocess.run(
        ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"],
        capture_output=True,
        check=False,
    )
    if result.returncode or result.stderr.strip():
        raise ObservationUnknown("device memory inventory unavailable")
    return gpu_in_use_from_plist(result.stdout)


@dataclass(frozen=True)
class Listener:
    address: str
    port: int
    transport: str = "tcp"

    def __post_init__(self):
        try:
            socket.inet_pton(socket.AF_INET6 if ':' in self.address else socket.AF_INET, self.address)
        except (OSError, TypeError) as error:
            raise ValueError(
                "listener must name an explicit local IP address"
            ) from error
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("invalid listener port")
        if self.transport not in ("tcp", "udp"):
            raise ValueError("listener transport must be tcp or udp")


def listener_closed(listener: Listener) -> bool:
    """Kernel bind, not a possibly firewalled connect failure, proves closure.

    No reuse flags are set. The serving runtime must bind this same explicit
    address; callers cannot use this probe to attest an unlisted listener.
    """
    kind = socket.SOCK_STREAM if listener.transport == "tcp" else socket.SOCK_DGRAM
    family=socket.AF_INET6 if ':' in listener.address else socket.AF_INET
    with socket.socket(family, kind) as probe:
        if family==socket.AF_INET6:
            probe.setsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY,1)
        try:
            probe.bind((listener.address, listener.port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise ObservationUnknown("listener closure unobservable") from error
    return True


class OwnedRankProcess:
    """A single launch; retain this handle through physical cleanup.

    This class never adopts a pid from disk. A lost handle requires operator
    recovery; neither constructor nor stop is a crash replay path.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        identity: ProcessIdentity,
        binding: RunBinding,
        identity_reader: Callable[[int], ProcessIdentity | None] = darwin_process,
        group_reader: Callable[[int], tuple[ProcessIdentity, ...]] = darwin_group,
        lifeline: GuardLaunch | None = None,
    ):
        if (
            identity.pid != process.pid
            or identity.group_id != process.pid
            or identity.uid != os.getuid()
        ):
            raise ObservationUnknown("new child lacks its own owned process group")
        self.process = process
        self.identity = identity
        self.binding = binding
        self._identity_reader = identity_reader
        self._group_reader = group_reader
        self._lifeline = lifeline
        self.stop_requested = False

    @classmethod
    def spawn(
        cls,
        argv: Sequence[str],
        *,
        binding: RunBinding,
        authority: AuthorityDecision,
        cwd: Path,
        environment: Mapping[str, str],
        stdout,
        stderr,
    ):
        if (
            not argv
            or any(not isinstance(arg, str) or not arg for arg in argv)
            or "--legacy-daemon" in argv
            or not Path(argv[0]).is_absolute()
            or not cwd.is_absolute()
            or not cwd.is_dir()
        ):
            raise ValueError("explicit foreground runtime and cwd required")
        if authority.expected != binding.expected or not authority.allows_start(
            authority_clock()
        ):
            raise ObservationUnknown("fresh member authority required before spawn")
        lifeline = prepare_guard(
            argv,
            cwd=cwd,
            environment=environment,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            identity = darwin_process(lifeline.process.pid)
            if identity is None:
                raise ObservationUnknown(
                    "guard exited before birth identity was captured"
                )
            owned = cls(lifeline.process, identity, binding, lifeline=lifeline)
            # The wrapper cannot launch the runtime before this final check.
            if not authority.allows_start(authority_clock()):
                raise ObservationUnknown("member authority elapsed before activation")
            lifeline.activate()
            return owned
        except BaseException:
            # EOF prevents a not-yet-started runtime, or drains the same unknown
            # attempt. It is neither a replay nor evidence of physical cleanup.
            lifeline.close()
            raise

    def stop(self) -> bool:
        """Request TERM once, only while the pinned group leader still matches.

        No default escalation or forced-kill clock. A dead leader with surviving
        children remains charged for operator inspection, never signalled by a
        reused group id. Caller loss and TERM do not attest cleanup.
        """
        if self.stop_requested:
            return False
        current = self._identity_reader(self.identity.pid)
        if current is None or current != self.identity:
            raise ObservationUnknown("owned group leader absent or identity changed")
        self.stop_requested = True
        if self._lifeline is None:
            os.killpg(self.identity.group_id, signal.SIGTERM)
        else:
            # The still-pinned guard requests group TERM once, then reaps its
            # child. Signalling the guard avoids duplicate TERM delivery.
            os.kill(self.identity.pid, signal.SIGTERM)
        return True

    def group_members(self) -> tuple[ProcessIdentity, ...]:
        return self._group_reader(self.identity.group_id)

    def exit_code(self) -> int | None:
        code = self.process.poll()
        if code is not None and self._lifeline is not None:
            self._lifeline.close()
        return code


@dataclass(frozen=True)
class CleanupObservation:
    binding: RunBinding
    child: ProcessIdentity
    exit_code: int | None
    group_members: tuple[ProcessIdentity, ...]
    listeners_closed: tuple[bool, ...]
    baseline_in_use_bytes: int
    observed_in_use_bytes: int
    tolerance_bytes: int
    observed_at: str

    @property
    def clean(self) -> bool:
        counters = (
            self.baseline_in_use_bytes,
            self.observed_in_use_bytes,
            self.tolerance_bytes,
        )
        return bool(
            type(self.exit_code) is int
            and not self.group_members
            and self.listeners_closed
            and all(type(value) is bool and value for value in self.listeners_closed)
            and all(type(value) is int and value >= 0 for value in counters)
            and self.observed_in_use_bytes
            <= self.baseline_in_use_bytes + self.tolerance_bytes
        )


def observe_cleanup(
    process: OwnedRankProcess,
    *,
    listeners: Sequence[Listener],
    baseline_in_use_bytes: int,
    tolerance_bytes: int,
    observed_at: str,
    memory_reader: Callable[[], int] = darwin_gpu_in_use,
    port_reader: Callable[[Listener], bool] = listener_closed,
) -> CleanupObservation:
    """A fresh read, never an elapsed-time inference or lease release."""
    if not listeners:
        raise ObservationUnknown("the complete explicit listener set is required")
    code = process.exit_code()  # Reap only this Popen child, then inspect its group.
    members = process.group_members()
    ports = tuple(port_reader(listener) for listener in listeners)
    memory = memory_reader()
    # A second group read brackets memory/port sampling. Any process remaining
    # in either inventory blocks cleanup, including descendants of a dead root.
    fresh_members = process.group_members()
    merged = tuple(
        {member.pid: member for member in (*members, *fresh_members)}.values()
    )
    return CleanupObservation(
        process.binding,
        process.identity,
        code,
        merged,
        ports,
        baseline_in_use_bytes,
        memory,
        tolerance_bytes,
        observed_at,
    )


def cleanup_evidence(observation: CleanupObservation) -> tuple[bytes, str]:
    """Prepare evidence only; owner must still attest and read exact settlement."""
    if not observation.clean:
        raise ObservationUnknown("physical cleanup not established; retain member")
    payload = {
        "kind": "observed_rank_cleanup",
        **asdict(observation.binding),
        "observation": asdict(observation),
        "lease_released": False,
        "settlement_recorded": False,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return raw, hashlib.sha256(raw).hexdigest()
