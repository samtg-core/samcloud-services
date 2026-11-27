"""Validate the queued Metal envelopes before load; claims stay owned.

The group's ordinary host queues own the slots. This gate neither acquires nor
releases them, starts a poller, changes a deadline, nor submits native work.
"""

import asyncio
import ctypes
import sys

from .plan import BindingRefused
from .prepare import canonical


def metal_policy(reading):
    """Read existing kernel policy and non-pageable occupancy, without Metal.

    Current free RAM is telemetry, not a maximum allocatable wired envelope.
    The queue's envelope already leaves host headroom. Do not subtract a plane
    claim (which allocates no pages). Wired and accelerator counters can overlap;
    without an overlap proof the gate conservatively reserves both counters.
    This check does not promise a measured load/prefill peak or eliminate the
    cost of compressing/swapping other pageable host processes.
    """
    from .. import capacity
    from .staging_files import darwin_boot_id
    if sys.platform != 'darwin':
        raise ValueError('explicit Metal host policy required')
    libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    query = libc.sysctlbyname
    query.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                      ctypes.c_void_p, ctypes.c_size_t]
    query.restype = ctypes.c_int
    def integer(name):
        value, size = ctypes.c_uint64(), ctypes.c_size_t(8)
        if query(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0 or size.value not in (4, 8):
            raise ValueError('fresh kernel policy unavailable')
        return value.value
    wired_mb, total = integer('iogpu.wired_limit_mb'), integer('hw.memsize')
    pages, page_size = capacity._vm_stat()
    wired = pages.get('wired down')
    device = reading.get('memory_device_inuse_mb')
    if (type(wired_mb) is not int or wired_mb <= 0
            or type(wired) is not int or wired < 0
            or type(page_size) is not int or page_size <= 0
            or type(device) is not int or device < 0
            or type(total) is not int or total <= 0
            or reading.get('memory_total_mb') * 1024 * 1024 != total
            or wired_mb * 1024 * 1024 > total):
        raise ValueError('explicit wired policy and locked-memory evidence required')
    return {'boot_id': darwin_boot_id(), 'wired_limit_bytes': wired_mb * 1024 * 1024,
            'kernel_wired_bytes': wired * page_size,
            'device_inuse_bytes': device * 1024 * 1024}


class LoadCapacityPending(Exception):
    """Positive capacity observation: retain the queued call and group claims."""

    def __init__(self, ranks):
        self.ranks = tuple(ranks)
        super().__init__('assigned queue envelope is outside current host Metal policy')


class LoadCapacityGate:
    def __init__(self, commands, routes):
        self.commands, self.routes = commands, routes

    async def check(self, current):
        from .host_control import run_identity
        reports = await asyncio.gather(*(c.client.capacity(c.binding) for c in self.commands))
        current()  # A read or a queued claim is never permission to launch.
        pending = []
        for command, report in zip(self.commands, reports, strict=True):
            binding = command.binding
            route = self.routes[binding.lease_id]
            available, total = report.get('available_bytes'), report.get('total_bytes')
            limit, wired, device = (report.get(key) for key in
                ('wired_limit_bytes', 'kernel_wired_bytes', 'device_inuse_bytes'))
            if (canonical(report.get('identity')) != canonical(run_identity(binding))
                    or report.get('boot_id') != route.boot_id
                    or type(report.get('uid')) is not int
                    or report['uid'] != command.client.endpoint.uid
                    or report.get('manager_empty') is not True
                    or type(available) is not int or type(total) is not int
                    or not 0 <= available <= total
                    or any(type(v) is not int for v in (limit, wired, device))
                    or not 0 < limit <= total or not 0 <= wired <= total
                    or not 0 <= device <= total):
                raise BindingRefused('exact fresh host Metal policy unconfirmed')
            # The already pinned member envelope includes weights, context and
            # runtime headroom. Summed spare RAM cannot cover a short rank.
            required = binding.expected.member.memory_mb * 1024 * 1024
            if required > total:
                raise BindingRefused('assigned member envelope exceeds physical RAM')
            # A foreign non-pageable allocation cannot be compressed away. The
            # counters may overlap, but their sum is a conservative upper bound
            # until their actual overlap is proven. Free RAM remains informative,
            # but does not veto the same queue-owned fixed placement a second time.
            if required > limit or required + wired + device > total:
                pending.append(binding.expected.binding.rank)
        if pending:
            raise LoadCapacityPending(pending)
