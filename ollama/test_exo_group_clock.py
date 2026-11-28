"""Authority clock controls; no sleep/wake action on a host or GPU call."""

import ctypes
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .exo_groups import clock as clocks


class AuthorityClockControls(unittest.TestCase):
    def test_darwin_uses_continuous_counter_including_simulated_sleep(self):
        ticks = [100]
        with (
            patch.object(clocks.sys, "platform", "darwin"),
            patch.object(clocks, "_darwin_clock", return_value=(lambda: ticks[0], 0.5)),
        ):
            self.assertEqual(clocks.authority_clock(), 50)
            # The continuous counter advances even if an uptime clock froze.
            ticks[0] += 100
            self.assertEqual(clocks.authority_clock(), 100)

    def test_unavailable_platform_never_silently_uses_uptime_clock(self):
        with (
            patch.object(clocks.sys, "platform", "unsupported"),
            patch.object(clocks, "time", SimpleNamespace()),
        ):
            with self.assertRaises(clocks.ClockUnavailable):
                clocks.authority_clock()

    @unittest.skipUnless(sys.platform == "darwin", "native Darwin continuous clock")
    def test_real_native_clock_and_conversion_are_available(self):
        function, factor = clocks._darwin_clock()
        self.assertEqual(ctypes.sizeof(clocks._Timebase), 8)
        before = function() * factor
        sample = clocks.authority_clock()
        after = function() * factor
        self.assertLessEqual(before, sample)
        self.assertLessEqual(sample, after)
        self.assertGreater(factor, 0)
        self.assertGreaterEqual(sample, time.monotonic() - 0.1)


if __name__ == "__main__":
    unittest.main()
