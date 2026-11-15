"""Native CPU-only expiry events; no host sleep, model or production effects."""

import asyncio
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest import mock

from .exo_groups.alarm import ContinuousAuthorityAlarm
from .exo_groups import alarm as alarms
from .exo_groups.clock import ClockUnavailable, authority_clock


@unittest.skipUnless(sys.platform == "darwin", "native Darwin kqueue timer")
class ContinuousAlarmControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.called = asyncio.Event()
        self.failed = []

        def notify():
            self.events.append(authority_clock())
            self.called.set()

        self.alarm = ContinuousAuthorityAlarm(
            asyncio.get_running_loop(), notify, lambda: self.failed.append(True)
        )

    async def asyncTearDown(self):
        self.alarm.close()

    async def test_actual_event_uses_explicit_continuous_deadline(self):
        deadline = authority_clock() + 0.05
        self.alarm.arm(deadline)
        await asyncio.wait_for(self.called.wait(), 3)  # CPU test bound only.
        self.assertGreaterEqual(self.events[0], deadline)
        self.assertEqual(self.failed, [])

    async def test_new_exact_deadline_replaces_old_alarm(self):
        self.alarm.arm(authority_clock() + 0.03)
        deadline = authority_clock() + 0.1
        self.alarm.arm(deadline)
        await asyncio.wait_for(self.called.wait(), 3)
        self.assertEqual(len(self.events), 1)
        self.assertGreaterEqual(self.events[0], deadline)

    async def test_invalid_replacement_keeps_previous_deadline(self):
        deadline = authority_clock() + 0.05
        self.alarm.arm(deadline)
        for value in (None, True, -1, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ClockUnavailable):
                    self.alarm.arm(value)
        self.assertEqual(self.alarm.deadline, deadline)
        await asyncio.wait_for(self.called.wait(), 3)
        self.assertGreaterEqual(self.events[0], deadline)

    async def test_native_registration_failure_keeps_previous_alarm(self):
        deadline = authority_clock() + 0.05
        self.alarm.arm(deadline)
        with mock.patch.object(
            alarms.select, "kevent", side_effect=OSError("fixture registration failed")
        ):
            with self.assertRaises(OSError):
                self.alarm.arm(deadline + 1)
        self.assertEqual(self.alarm.deadline, deadline)
        await asyncio.wait_for(self.called.wait(), 3)

    async def test_close_is_final_and_never_rearms(self):
        self.alarm.arm(authority_clock() + 60)
        self.alarm.close()
        self.alarm.close()
        with self.assertRaises(ClockUnavailable):
            self.alarm.arm(authority_clock() + 1)

    async def test_disarmed_wakeup_is_ignored_and_can_be_rearmed(self):
        self.alarm.arm(authority_clock() + 0.02)
        self.alarm.disarm()
        await asyncio.sleep(0.04)
        self.assertEqual(self.events, [])
        self.assertIsNone(self.alarm.deadline)
        self.alarm.arm(authority_clock() + 0.02)
        await asyncio.wait_for(self.called.wait(), 3)
        self.assertEqual(len(self.events), 1)

    async def test_flags_match_installed_sdk_sleep_inclusive_epoch(self):
        sdk = Path(
            subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        )
        text = (sdk / "usr/include/sys/event.h").read_text()
        for name in ("NOTE_ABSOLUTE", "NOTE_MACH_CONTINUOUS_TIME", "NOTE_MACHTIME"):
            found = re.search(
                r"^#define\s+" + name + r"\s+(0x[0-9a-fA-F]+)", text, re.M
            )
            self.assertIsNotNone(found)
            self.assertEqual(int(found.group(1), 16), getattr(alarms, name))
        self.assertIn(
            "with NOTE_MACHTIME and NOTE_ABSOLUTE: uses mach continuous time epoch",
            text,
        )


if __name__ == "__main__":
    unittest.main()
