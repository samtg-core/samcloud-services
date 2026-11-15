"""One native expiry event, using the same sleep-inclusive lease clock.

This arms only an explicitly observed authority deadline, not a job/idle cap.
The descriptor joins the owning service's existing asyncio loop. No poller,
lease mutation, process signal, replacement or cleanup inference lives here.
"""

from __future__ import annotations

import asyncio
import errno
import math
import select
import sys
from typing import Callable

from .clock import ClockUnavailable, _darwin_clock


# Installed Darwin SDK sys/event.h. Combined, these flags use the mach
# continuous epoch and keep ticking during host sleep, independently of wall
# clock corrections. Python's KQ_NOTE_* exports omit these timer flags.
NOTE_ABSOLUTE = 0x00000008
NOTE_MACH_CONTINUOUS_TIME = 0x00000080
NOTE_MACHTIME = 0x00000100


class ContinuousAuthorityAlarm:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        notify: Callable[[], None],
        failed: Callable[[], None],
    ):
        if sys.platform != "darwin":
            raise ClockUnavailable("native Darwin continuous timer required")
        self.loop, self.notify, self.failed = loop, notify, failed
        self._seconds_per_tick = _darwin_clock()[1]
        self._queue = select.kqueue()
        self._sequence = 0
        self._active = None
        self.deadline = None
        self._closed = False
        try:
            loop.add_reader(self._queue.fileno(), self._ready)
        except BaseException:
            self._queue.close()
            raise

    def _delete(self, ident: int) -> None:
        try:
            self._queue.control(
                [
                    select.kevent(
                        ident, filter=select.KQ_FILTER_TIMER, flags=select.KQ_EV_DELETE
                    )
                ],
                0,
                0,
            )
        except OSError as error:
            if error.errno != errno.ENOENT:  # Fired one-shot is already absent.
                raise

    def arm(self, deadline: float) -> None:
        """Install before accepting a renewed projection; failure keeps old alarm."""
        if (
            self._closed
            or type(deadline) not in (int, float)
            or not math.isfinite(deadline)
            or deadline <= 0
        ):
            raise ClockUnavailable("invalid or closed authority alarm")
        ticks = math.ceil(deadline / self._seconds_per_tick)
        if ticks >= 2**63:
            raise ClockUnavailable("authority deadline exceeds native timer range")
        self._sequence += 1
        ident = self._sequence
        event = select.kevent(
            ident,
            filter=select.KQ_FILTER_TIMER,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=NOTE_ABSOLUTE | NOTE_MACHTIME | NOTE_MACH_CONTINUOUS_TIME,
            data=ticks,
        )
        # Add first: a failed replacement never silently drops the old expiry.
        self._queue.control([event], 0, 0)
        previous = self._active
        self._active, self.deadline = ident, deadline
        if previous is not None:
            # Failure to remove an older one-shot is safe: its id is obsolete.
            try:
                self._delete(previous)
            except OSError:
                pass

    def _ready(self) -> None:
        if self._closed:
            return
        try:
            for event in self._queue.control(None, 32, 0):
                if event.ident != self._active:
                    continue
                self._active = None
                if event.flags & select.KQ_EV_ERROR:
                    self.failed()
                else:
                    self.notify()  # Owner rechecks the exact continuous deadline.
        except OSError:
            self.failed()

    def disarm(self) -> None:
        """Withdraw a metadata wakeup; this never cancels compute or authority."""
        previous, self._active = self._active, None
        self.deadline = None
        if previous is not None and not self._closed:
            try:
                self._delete(previous)
            except OSError:
                pass  # An obsolete one-shot is ignored even if deletion failed.

    def close(self) -> None:
        """Only after the owner has stopped/cleaned its attempt or is exiting."""
        if not self._closed:
            self._closed = True
            self.loop.remove_reader(self._queue.fileno())
            self._queue.close()
            self._active = None
