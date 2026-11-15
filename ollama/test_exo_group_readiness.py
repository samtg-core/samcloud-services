"""Fresh private/host readiness before the attributed core serving transition."""

import asyncio
import copy
from dataclasses import replace
import hashlib
from importlib.resources import files
import sys
import unittest
from uuid import UUID

from . import test_exo_group_startup as startup_controls
from .exo_groups.cleanup import CleanupExpectation, NoStartExpectation
from .exo_groups.host_channel import HostRoute
from .exo_groups.host_receipts import AuthenticatedReceiptReader, HostReceiptObservation
from .exo_groups.member import ExpectedMember
from .exo_groups.plan import BindingRefused
from .exo_groups.rank_process import RunBinding, ProcessIdentity, Listener
from .exo_groups.readiness import GroupReadiness
from .exo_groups.receipts import ReceiptRoute
from .inference_arbiter import Conflict


class ReadinessControls(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = startup_controls.StartupControls()
        await self.fixture.asyncSetUp()
        self.addAsyncCleanup(self.fixture.asyncTearDown)
        f = self.fixture
        c = f.control
        c.source.pause()  # A ready test must not dispatch its queued fixture call.
        self.provider, self.parts = c.provider, f.parts
        self.provider.bind_backend(self.parts.backend)
        await f.startup.submit()
        f.initial = copy.deepcopy(self.parts.state)
        f.initial["lastEventAppliedIdx"] = 20
        self.reports = {}
        routes = []
        source = files("ollama.exo_groups").joinpath("receipt_files.py").read_bytes()
        for i, launch in enumerate(self.parts.launches):
            binding = RunBinding.from_expected(
                ExpectedMember.from_launch(self.parts.plan, launch), self.parts.runtime
            )
            pin = CleanupExpectation(
                binding,
                ProcessIdentity(100 + i, 99, 100 + i, 501, 1000 + i, 1),
                (Listener("127.0.0.1", 39999 + i),),
                100,
                0,
            )
            prepared = NoStartExpectation(
                binding,
                str(UUID(int=i + 1)),
                "a" * 64,
                "b" * 64,
                pin.listeners,
                100,
                0,
                "2026-11-14T10:00:00+00:00",
            )
            self.reports[launch.lease_id] = HostReceiptObservation(
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
                collector_sha256=hashlib.sha256(source).hexdigest(),
                environment={},
            )

            async def read(_plan, *, binding, **_):
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
                    self.parts.runtime,
                )
            )
        self.provider.bind_publication(group_scope="projects")
        self.provider.bind_receipts(tuple(routes))
        self.ready = self.provider.bind_readiness(f.startup)
        self.writes = []
        self.lose = self.fence = False
        self.bad_ack = False
        self.entered, self.proceed = asyncio.Event(), asyncio.Event()
        self.proceed.set()

        async def serving(group_id, generation, proof):
            saved = self.ready.saved()
            self.assertEqual({k: saved[k] for k in proof}, proof)
            self.assertFalse(c.journal.db.in_transaction)
            self.writes.append((group_id, generation, copy.deepcopy(proof)))
            self.entered.set()
            await self.proceed.wait()
            if self.fence:
                self.parts.group["generation"] = 2
                raise BindingRefused("injected generation race")
            self.parts.group.update(
                phase="serving",
                serving_evidence={
                    **proof,
                    "generation": generation,
                    "attested_by": self.parts.plan.owner_id,
                },
            )
            if self.bad_ack:
                self.parts.group["serving_evidence"]["evidence_sha256"] = "f" * 64
            if self.lose:
                self.lose = False
                raise OSError("lost after core commit")
            return copy.deepcopy(self.parts.group)

        f.actor.plane.mark_serving = serving

    async def test_exact_ready_evidence_commits_before_serving_and_reuses_same_ack(
        self,
    ):
        result = await self.ready.attest()
        self.assertEqual(result["state"], "exact_group_readiness_confirmed")
        self.assertTrue(self.fixture.actor.may_serve_requests())
        self.assertEqual(self.ready.saved()["confirmed"], 1)
        self.assertEqual(self.fixture.posts, 1)
        self.assertEqual(self.parts.posts, 0)
        self.assertEqual(self.fixture.control.state(self.fixture.call), "queued")
        self.fixture.initial["lastEventAppliedIdx"] += 1
        self.assertEqual(await self.ready.attest(), result)
        self.assertEqual(len(self.writes), 1)

    async def test_prepared_or_ended_host_is_not_ready(self):
        lease = next(iter(self.reports))
        old = self.reports[lease]
        self.reports[lease] = replace(old, started=None, phase="prepared")
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.reports[lease] = replace(old, cleanup_bytes=b"ended")
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.ready.saved())

    async def test_unready_private_runner_and_old_event_index_do_not_mark_serving(self):
        original = copy.deepcopy(self.fixture.initial)
        self.fixture.initial["runners"] = {}
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.fixture.initial = original
        self.fixture.initial["lastEventAppliedIdx"] = 11
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.assertEqual(self.writes, [])

    async def test_runtime_change_or_event_loss_withdraws_before_serving_write(self):
        original = self.fixture.startup.verify_runtime

        async def changed():
            return "f" * 64

        self.fixture.startup.verify_runtime = changed
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.fixture.startup.verify_runtime = original
        self.fixture.mode = "cursor-loss"
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.assertEqual(self.writes, [])

    async def test_lost_serving_reply_reconciles_same_journal_proof_after_new_owner(
        self,
    ):
        self.lose = True
        with self.assertRaises(OSError):
            await self.ready.attest()
        old = self.ready.saved()
        self.ready = GroupReadiness(self.provider, self.fixture.startup)
        result = await self.ready.attest()
        self.assertEqual(result["evidence_sha256"], old["evidence_sha256"])
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.fixture.posts, 1)

    async def test_observer_disconnect_does_not_cancel_owned_serving_transition(self):
        self.proceed.clear()
        observer = asyncio.create_task(self.ready.attest())
        await self.entered.wait()
        task = self.provider.completion._task
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(task.done())
        self.proceed.set()
        self.assertEqual(
            (await self.provider.completion.wait())["state"],
            "exact_group_readiness_confirmed",
        )
        self.assertEqual(len(self.writes), 1)

    async def test_fence_or_wrong_ack_does_not_confirm_readiness(self):
        self.bad_ack = True
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.assertEqual(self.ready.saved()["confirmed"], 0)
        self.parts.group.update(phase="admitted", serving_evidence=None)
        self.bad_ack = False
        self.fence = True
        with self.assertRaises(BindingRefused):
            await self.ready.attest()
        self.assertEqual(self.ready.saved()["confirmed"], 0)

    async def test_running_or_uncertain_compute_is_not_overridden(self):
        for state in ("running", "uncertain"):
            self.fixture.control.journal.db.execute(
                "UPDATE executions SET state=? WHERE id=?", (state, self.fixture.call)
            )
            with self.assertRaises(Conflict):
                await self.ready.attest()
        self.assertEqual(self.writes, [])

    async def test_provider_readiness_owner_is_fixed(self):
        self.assertIs(self.provider.bind_readiness(self.fixture.startup), self.ready)
        with self.assertRaises(ValueError):
            self.provider.bind_readiness(object())
