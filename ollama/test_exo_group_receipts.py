"""Coordinator host-read/pin/drain/publication ordering; injected host reports."""

import asyncio
from dataclasses import replace
import hashlib
from importlib.resources import files
import sys
import unittest
from uuid import UUID

from . import test_exo_group_publication as publication_controls
from .exo_groups.cleanup import NoStartExpectation
from .exo_groups.host_channel import HostRoute
from .exo_groups.host_receipts import AuthenticatedReceiptReader, HostReceiptObservation
from .exo_groups.install import GroupProvider
from .exo_groups.plan import BindingRefused
from .exo_groups.receipts import GroupReceipts, ReceiptRoute
from .inference_arbiter import Conflict


class ReceiptOrchestrationControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.publication = publication_controls.PublicationControls()
        await self.publication.asyncSetUp()
        self.parts = self.publication.parts
        self.addAsyncCleanup(self.parts.base.asyncTearDown)
        self.calls, self.reports = [], {}
        self.fail = None
        self.entered, self.proceed = asyncio.Event(), asyncio.Event()
        self.proceed.set()
        routes = []
        sha = hashlib.sha256(
            files("ollama.exo_groups").joinpath("receipt_files.py").read_bytes()
        ).hexdigest()
        for i, (lease, pin) in enumerate(self.parts.pins.items()):
            launch = pin.binding.expected.binding
            prepared = NoStartExpectation(
                pin.binding,
                str(UUID(int=i + 1)),
                "a" * 64,
                "b" * 64,
                pin.listeners,
                pin.baseline_in_use_bytes,
                pin.tolerance_bytes,
                "2026-11-14T10:00:00+00:00",
            )
            self.reports[lease] = HostReceiptObservation(
                prepared, pin, "spawning", None, None
            )
            reader = AuthenticatedReceiptReader(
                HostRoute(
                    launch.rank,
                    launch.resource_id,
                    501,
                    ("/unused-ssh",),
                    sys.executable,
                ),
                collector_sha256=sha,
                environment={},
            )

            async def read(_plan, *, binding, **_spec):
                self.calls.append(binding.lease_id)
                self.entered.set()
                await self.proceed.wait()
                if self.fail == binding.lease_id:
                    raise BindingRefused("injected host read unavailable")
                return self.reports[binding.lease_id]

            reader.read = read
            routes.append(
                ReceiptRoute(
                    reader,
                    launch,
                    "/private/host/queue.sqlite",
                    "/private/host/attempt",
                    prepared.host_boot,
                    prepared.launch_sha256,
                    pin.binding.runtime_sha256,
                )
            )
        self.routes = tuple(routes)
        self.owner = GroupReceipts(
            self.parts.completion, self.publication.publisher, self.routes
        )
        self.lease = self.parts.lease_id

    def cleanup(self, lease=None):
        lease = lease or self.lease
        raw = self.parts.files[self.parts.proofs[lease][0]]
        self.reports[lease] = replace(
            self.reports[lease],
            cleanup_bytes=raw,
            cleanup_sha256=hashlib.sha256(raw).hexdigest(),
        )

    async def test_all_independent_host_pins_precede_group_drain(self):
        self.parts.base.journal.db.execute("DELETE FROM exo_group_cleanup_pins")
        self.parts.end_call()
        original = self.parts.plane.begin_release

        async def release(*args):
            self.assertEqual(len(self.calls), 3)
            self.assertEqual(
                self.parts.base.journal.db.execute(
                    "SELECT COUNT(*) FROM exo_group_cleanup_pins"
                ).fetchone()[0],
                3,
            )
            return await original(*args)

        self.parts.plane.begin_release = release
        self.assertEqual(
            (await self.owner.begin_release(generation=1))["state"],
            "host_cleanup_required",
        )
        self.assertEqual(self.publication.uploads, [])
        self.assertEqual(self.parts.releases(), [])

    async def test_missing_route_cannot_withdraw_authority_or_guess_host_absence(self):
        owner = GroupReceipts(
            self.parts.completion, self.publication.publisher, self.routes[:2]
        )
        self.parts.end_call()
        with self.assertRaises(BindingRefused):
            await owner.begin_release(generation=1)
        self.assertEqual(self.calls, [])
        self.assertFalse(
            self.parts.base.intents.closing(
                self.owner.plan.owner_id, self.owner.plan.request_key
            )
        )
        self.assertEqual(self.parts.mutations, [])

    async def test_failed_host_read_keeps_earlier_independent_pin_without_draining(
        self,
    ):
        self.parts.base.journal.db.execute("DELETE FROM exo_group_cleanup_pins")
        self.fail = self.routes[1].binding.lease_id
        self.parts.end_call()
        with self.assertRaises(BindingRefused):
            await self.owner.begin_release(generation=1)
        self.assertEqual(
            self.parts.base.journal.db.execute(
                "SELECT COUNT(*) FROM exo_group_cleanup_pins"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(self.parts.mutations, [])
        self.assertFalse(
            self.parts.base.intents.closing(
                self.owner.plan.owner_id, self.owner.plan.request_key
            )
        )

    async def test_observer_cancel_cannot_cancel_owned_collection_and_drain(self):
        self.parts.end_call()
        self.proceed.clear()
        observer = asyncio.create_task(self.owner.begin_release(generation=1))
        await self.entered.wait()
        task = self.parts.completion._task
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(task.done())
        self.proceed.set()
        self.assertEqual(
            (await self.parts.completion.wait())["state"], "host_cleanup_required"
        )
        self.assertEqual(len(self.calls), 3)

    async def test_fresh_host_cleanup_flows_to_one_publication_and_exact_settlement(
        self,
    ):
        self.parts.end_call()
        await self.owner.begin_release(generation=1)
        self.cleanup()
        result = await self.owner.settle_host(self.lease, generation=1)
        self.assertEqual(result["state"], "exact_member_settlement_confirmed")
        self.assertEqual(self.publication.uploads, [100])
        self.assertEqual(len(self.parts.releases()), 1)
        self.assertEqual(
            sum(lease["status"] == "active" for lease in self.parts.base.leases.values()), 2
        )
        await self.owner.settle_host(self.lease, generation=1)
        self.assertEqual(self.publication.uploads, [100])
        self.assertEqual(len(self.parts.releases()), 1)

    async def test_missing_cleanup_and_changed_bytes_cannot_publish_or_settle(self):
        self.parts.end_call()
        await self.owner.begin_release(generation=1)
        with self.assertRaises(BindingRefused):
            await self.owner.settle_host(self.lease, generation=1)
        self.cleanup()
        self.reports[self.lease] = replace(
            self.reports[self.lease], cleanup_bytes=b"{}"
        )
        with self.assertRaises(BindingRefused):
            await self.owner.settle_host(self.lease, generation=1)
        self.assertEqual(self.publication.uploads, [])
        self.assertEqual(self.parts.releases(), [])

    async def test_cleanup_cannot_create_a_missing_posthoc_pin(self):
        self.parts.end_call()
        await self.owner.begin_release(generation=1)
        self.cleanup()
        self.parts.base.journal.db.execute(
            "DELETE FROM exo_group_cleanup_pins WHERE lease_id=?", (self.lease,)
        )
        with self.assertRaises(BindingRefused):
            await self.owner.settle_host(self.lease, generation=1)
        self.assertEqual(self.publication.uploads, [])
        self.assertEqual(self.parts.releases(), [])

    async def test_changed_child_and_wrong_host_boot_do_not_replace_frozen_pin(self):
        old = self.reports[self.lease]
        self.reports[self.lease] = replace(
            old, prepared=replace(old.prepared, host_boot=str(UUID(int=10)))
        )
        with self.assertRaises(BindingRefused):
            await self.owner.observe()
        self.reports[self.lease] = replace(
            old, started=replace(old.started, child=replace(old.started.child, pid=999))
        )
        with self.assertRaises(Conflict):
            await self.owner.observe()
        self.assertEqual(self.parts.completion._pin(self.lease), old.started)
        self.assertEqual(self.parts.mutations, [])

    async def test_current_fence_generation_still_uses_original_independent_launch_pin(
        self,
    ):
        await self.owner.observe()
        self.parts.end_call()
        self.parts.group.update(generation=2, phase="releasing", release_requested=True)
        self.cleanup()
        result = await self.owner.settle_host(self.lease, generation=2)
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["launch_generation"], 1)

    async def test_pending_compute_prevents_drain_and_publication(self):
        self.cleanup()
        for state in ("queued", "running", "uncertain"):
            self.parts.base.journal.db.execute(
                "UPDATE executions SET state=? WHERE id=?",
                (state, self.parts.base.call["id"]),
            )
            with self.assertRaises(Conflict):
                await self.owner.begin_release(generation=1)
            with self.assertRaises(Conflict):
                await self.owner.settle_host(self.lease, generation=1)
        self.assertFalse(
            self.parts.base.intents.closing(
                self.owner.plan.owner_id, self.owner.plan.request_key
            )
        )
        self.assertEqual(self.publication.uploads, [])
        self.assertEqual(self.parts.mutations, [])

    async def test_duplicate_foreign_or_changed_routes_refuse_before_host_read(self):
        for routes in (
            self.routes[:1] * 2,
            self.routes[:2] + (replace(self.routes[2], runtime_sha256="f" * 64),),
        ):
            with self.assertRaises(BindingRefused):
                GroupReceipts(self.parts.completion, self.publication.publisher, routes)
        self.assertEqual(self.calls, [])

    async def test_provider_owns_one_configured_receipt_route_set(self):
        provider = GroupProvider.__new__(GroupProvider)
        provider.publication, provider.receipts, provider.completion = (
            None,
            None,
            self.parts.completion,
        )
        with self.assertRaises(RuntimeError):
            provider.bind_receipts(self.routes)
        provider.publication = self.publication.publisher
        self.assertIs(
            provider.bind_receipts(self.routes), provider.bind_receipts(self.routes)
        )
        with self.assertRaises(ValueError):
            provider.bind_receipts(self.routes[:2])
