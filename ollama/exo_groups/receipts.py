"""Service-owned receipt collection before drain and exact host settlement.

Host routes are trusted deployment configuration, never caller arguments. This
owner reads existing host journals; it cannot launch, stop or replay a rank.
"""

from dataclasses import dataclass
import hashlib

from .cleanup import validate_cleanup
from .host_receipts import AuthenticatedReceiptReader, HostReceiptObservation
from .member import ExpectedMember
from .plan import BindingRefused, LaunchBinding
from .rank_process import RunBinding


@dataclass(frozen=True)
class ReceiptRoute:
    reader: AuthenticatedReceiptReader
    binding: LaunchBinding
    journal_path: str
    state_path: str
    boot_id: str
    launch_sha256: str
    runtime_sha256: str

    def check(self, plan):
        if not isinstance(self.reader, AuthenticatedReceiptReader):
            raise BindingRefused("configured authenticated receipt reader required")
        expected = ExpectedMember.from_launch(plan, self.binding)
        if (expected.binding.rank, expected.binding.resource_id) != (
            self.reader.route.rank,
            self.reader.route.resource_id,
        ):
            raise BindingRefused("receipt route differs from its member")
        RunBinding.from_expected(expected, self.runtime_sha256)
        if (
            any(
                not isinstance(value, str)
                or not value.startswith("/")
                or "\x00" in value
                for value in (self.journal_path, self.state_path)
            )
            or not isinstance(self.boot_id, str)
            or not self.boot_id
            or not isinstance(self.launch_sha256, str)
            or len(self.launch_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.launch_sha256)
        ):
            raise BindingRefused("exact host paths, boot and launch source required")

    async def read(self, plan):
        self.check(plan)
        return await self.reader.read(
            plan,
            binding=self.binding,
            journal_path=self.journal_path,
            state_path=self.state_path,
            boot_id=self.boot_id,
            launch_sha256=self.launch_sha256,
            runtime_sha256=self.runtime_sha256,
        )


class GroupReceipts:
    def __init__(self, completion, publication, routes):
        if (
            publication.completion is not completion
            or not isinstance(routes, tuple)
            or not routes
            or len(routes) > len(completion.plan.members)
            or any(not isinstance(route, ReceiptRoute) for route in routes)
        ):
            raise ValueError(
                "exact completion, publication and configured host routes required"
            )
        for route in routes:
            route.check(completion.plan)
        if (
            len({route.binding.lease_id for route in routes}) != len(routes)
            or len({route.binding.resource_id for route in routes}) != len(routes)
            or len(
                {
                    (
                        route.binding.group_id,
                        route.binding.generation,
                        route.runtime_sha256,
                    )
                    for route in routes
                }
            )
            != 1
        ):
            raise BindingRefused(
                "host receipt routes must name one exact group/runtime"
            )
        self.completion, self.publication = completion, publication
        self.routes = {route.binding.lease_id: route for route in routes}
        self.plan, self.actor = completion.plan, completion.actor

    def _observation(self, route, observed):
        if (
            not isinstance(observed, HostReceiptObservation)
            or observed.prepared.binding
            != RunBinding.from_expected(
                ExpectedMember.from_launch(self.plan, route.binding),
                route.runtime_sha256,
            )
            or observed.prepared.host_boot != route.boot_id
            or observed.prepared.launch_sha256 != route.launch_sha256
            or observed.started is not None
            and observed.started.binding != observed.prepared.binding
        ):
            raise BindingRefused(
                "observed host attempt differs from its configured route"
            )
        return observed

    async def observe(self):
        """Pin independent preparations/children before closure; no lease write."""
        return await self.completion._owned(("host_receipts",), self._observe)

    async def _observe(self):
        reports = {}
        for lease_id, route in self.routes.items():
            observed = self._observation(route, await route.read(self.plan))
            # pin() enforces original launch generation, staged boot and the
            # one-way preparation -> actual-child transition. No cleanup body
            # can synthesize a pin, including after a group is already closing.
            await observed.pin(self.completion, generation=route.binding.generation)
            reports[lease_id] = observed
        return reports

    async def begin_release(self, *, generation):
        async def release():
            group = await self.completion._group(generation)
            allocated = {
                m["lease_id"] for m in group["members"] if m.get("lease_id") is not None
            }
            if allocated != set(self.routes):
                raise BindingRefused(
                    "every allocated host requires its exact configured receipt route"
                )
            await self._observe()
            # Reread the generation after host I/O. Only then freeze closure.
            group = await self.completion._group(generation)
            if {
                m["lease_id"] for m in group["members"] if m.get("lease_id") is not None
            } != allocated:
                raise BindingRefused(
                    "allocated member set changed during receipt collection"
                )
            self.completion.intents.begin_drain(
                self.plan.owner_id, self.plan.request_key, generation
            )
            self.actor.wake()
            return await self.completion._begin(generation)

        return await self.completion._owned(("receipt_release", generation), release)

    async def settle_host(self, lease_id, *, generation):
        route = self.routes.get(lease_id)
        if route is None:
            raise BindingRefused("exact configured host receipt route is missing")

        async def settle():
            observed = self._observation(route, await route.read(self.plan))
            # Settlement cannot learn its independent authority from a later
            # cleanup report. Startup/drain must have saved the same host pin.
            if self.completion._pin(lease_id) != (
                observed.started or observed.prepared
            ):
                raise BindingRefused(
                    "fresh host attempt differs from its pre-drain cleanup pin"
                )
            raw = observed.cleanup_bytes
            if (
                not isinstance(raw, bytes)
                or hashlib.sha256(raw).hexdigest() != observed.cleanup_sha256
            ):
                raise BindingRefused("host cleanup has not been observed")
            validate_cleanup(raw, self.completion._pin(lease_id))
            self.completion.intents.begin_drain(
                self.plan.owner_id, self.plan.request_key, generation
            )
            self.actor.wake()
            publication = await self.publication._publish(lease_id, raw)
            return await self.completion._settle(
                lease_id, generation, publication["file_id"], publication["sha256"]
            )

        return await self.completion._owned(
            ("receipt_settle", lease_id, generation), settle
        )
