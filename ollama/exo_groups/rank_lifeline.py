"""Controller-death containment for a foreground, non-daemonizing rank.

The controller alone holds a pipe writer. Its loss closes the pipe and asks
the owned process group to terminate. This neither proves physical cleanup nor
releases a lease. No watchdog daemon, inference deadline, restart or forced-kill
timer is used. The pinned runtime must not escape its process group.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
from typing import Mapping, Sequence


_writers: set[int] = set()


def _after_fork() -> None:
    # A different forked controller must not keep another rank's lifeline open.
    for descriptor in tuple(_writers):
        os.close(descriptor)
    _writers.clear()


os.register_at_fork(after_in_child=_after_fork)


@dataclass
class GuardLaunch:
    """Retain for the whole rank lifetime; activation is a one-time operation.

    Birth identity and current member authority must be checked by the caller
    before activate(). Before activation, the wrapper cannot start the rank.
    A lost/ambiguous wrapper is never replaced by this object.
    """

    process: subprocess.Popen
    _writer: int | None
    activated: bool = False

    def activate(self) -> None:
        if self.activated or self._writer is None or self._writer not in _writers:
            raise RuntimeError("lifeline activation cannot be replayed")
        self.activated = True  # Unknown write outcome must never be retried.
        if os.write(self._writer, b"S") != 1:
            raise RuntimeError("lifeline activation outcome unknown")

    def close(self) -> None:
        if self._writer is not None:
            descriptor, self._writer = self._writer, None
            if descriptor in _writers:
                _writers.remove(descriptor)
                os.close(descriptor)


def prepare_guard(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout,
    stderr,
) -> GuardLaunch:
    if (
        not argv
        or not Path(argv[0]).is_absolute()
        or any(not isinstance(value, str) or not value for value in argv)
        or "--legacy-daemon" in argv
        or not cwd.is_absolute()
        or not cwd.is_dir()
    ):
        raise ValueError("explicit foreground runtime and cwd required")
    reader, writer = os.pipe()  # Non-inheritable, including the sole writer.
    _writers.add(writer)
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).resolve()),
                str(reader),
                *argv,
            ],
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            pass_fds=(reader,),
            start_new_session=True,
        )
    except BaseException:
        _writers.remove(writer)
        os.close(writer)
        raise
    finally:
        os.close(reader)
    return GuardLaunch(child, writer)


def _guard(reader: int, argv: list[str]) -> int:
    """One group leader; all wakeups come from pipe, signal or child exit."""
    if (
        os.getpgrp() != os.getpid()
        or not stat.S_ISFIFO(os.fstat(reader).st_mode)
        or not argv
    ):
        raise RuntimeError("guard must own its session and a private pipe")
    os.set_blocking(reader, False)
    wake_reader, wake_writer = os.pipe()
    os.set_blocking(wake_reader, False)
    os.set_blocking(wake_writer, False)
    signal.set_wakeup_fd(wake_writer)
    requested = False
    terminated = False
    child = None

    def handle(number, _frame):
        nonlocal requested
        if number != signal.SIGCHLD:
            requested = True

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGCHLD):
        signal.signal(number, handle)

    def terminate() -> None:
        nonlocal terminated
        if not terminated:
            terminated = True
            # This process is the still-live leader, so its group cannot have
            # been recycled. Our handler survives to reap the direct child.
            os.killpg(os.getpgrp(), signal.SIGTERM)

    try:
        with selectors.DefaultSelector() as selector:
            selector.register(reader, selectors.EVENT_READ)
            selector.register(wake_reader, selectors.EVENT_READ)
            while True:
                for key, _mask in selector.select():  # No default clock.
                    if key.fd == wake_reader:
                        while True:
                            try:
                                if not os.read(wake_reader, 4096):
                                    break
                            except BlockingIOError:
                                break
                    else:
                        data = os.read(reader, 4096)
                        if data == b"S" and child is None and not requested:
                            child = subprocess.Popen(
                                argv,
                                stdin=subprocess.DEVNULL,
                                close_fds=True,
                            )
                        else:
                            # EOF or any second/malformed activation drains;
                            # no input can start another child.
                            requested = True
                            selector.unregister(reader)
                if requested:
                    if child is None:
                        return 0
                    terminate()
                if child is not None:
                    code = child.poll()
                    if code is not None:
                        # Root exit must not silently leave helper children.
                        # This is a TERM request, not evidence they are absent.
                        terminate()
                        return code if code >= 0 else 128 - code
    finally:
        signal.set_wakeup_fd(-1)
        for descriptor in (reader, wake_reader, wake_writer):
            os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(_guard(int(sys.argv[1]), sys.argv[2:]))
