"""Lease elapsed time includes host sleep, without trusting wall-clock changes.

Darwin's Python monotonic uses mach_absolute_time, which excludes sleep. The
SDK's mach_continuous_time explicitly advances during sleep. Never fall back
to the former on these hosts: authority must not grow while a Mac sleeps.
All timestamps within a member observation/decision must use this same clock.
"""

import ctypes
from functools import lru_cache
import sys
import time


class ClockUnavailable(RuntimeError):
    pass


class _Timebase(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


@lru_cache(maxsize=1)
def _darwin_clock():
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    continuous = library.mach_continuous_time
    continuous.argtypes = []
    continuous.restype = ctypes.c_uint64
    info = library.mach_timebase_info
    info.argtypes = [ctypes.POINTER(_Timebase)]
    info.restype = ctypes.c_int
    base = _Timebase()
    if info(ctypes.byref(base)) != 0 or not base.numerator or not base.denominator:
        raise ClockUnavailable("continuous authority clock unavailable")
    return continuous, base.numerator / base.denominator / 1_000_000_000


def authority_clock() -> float:
    if sys.platform == "darwin":
        continuous, seconds_per_tick = _darwin_clock()
        return continuous() * seconds_per_tick
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    raise ClockUnavailable("a monotonic clock including suspend is required")
