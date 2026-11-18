"""Fresh per-rank RAM admission before native load; claims stay owned.

The group's ordinary host queues own the slots. This gate neither acquires nor
releases them, starts a poller, changes a deadline, nor submits native work.
"""

import asyncio

from .host_control import run_identity
from .plan import BindingRefused
from .prepare import canonical


class LoadCapacityPending(Exception):
    """Positive capacity observation: retain the queued call and group claims."""

    def __init__(self, ranks):
        self.ranks = tuple(ranks)
        super().__init__('assigned rank memory is not yet available')


class LoadCapacityGate:
    def __init__(self, commands, routes):
        self.commands, self.routes = commands, routes

    async def check(self, current):
        reports = await asyncio.gather(*(c.client.capacity(c.binding) for c in self.commands))
        current()  # A read or a queued claim is never permission to launch.
        pending = []
        for command, report in zip(self.commands, reports, strict=True):
            binding = command.binding
            route = self.routes[binding.lease_id]
            available, total = report.get('available_bytes'), report.get('total_bytes')
            if (canonical(report.get('identity')) != canonical(run_identity(binding))
                    or report.get('boot_id') != route.boot_id
                    or type(report.get('uid')) is not int
                    or report['uid'] != command.client.endpoint.uid
                    or report.get('manager_empty') is not True
                    or type(available) is not int or type(total) is not int
                    or not 0 <= available <= total):
                raise BindingRefused('exact fresh host RAM capacity unconfirmed')
            # The already pinned member envelope includes weights, context and
            # runtime headroom. Summed spare RAM cannot cover a short rank.
            required = binding.expected.member.memory_mb * 1024 * 1024
            if required > total:
                raise BindingRefused('assigned member envelope exceeds physical RAM')
            if available < required:
                pending.append(binding.expected.binding.rank)
        if pending:
            raise LoadCapacityPending(pending)
