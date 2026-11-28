"""Host lifecycle controls, including one real CPU child; no fleet/GPU calls."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from . import test_exo_group_member as authority_fixtures
from .exo_groups.clock import authority_clock
from .exo_groups.rank_lifecycle import RankLifecycle, RankSpec
from .exo_groups.rank_process import Listener, ObservationUnknown, ProcessIdentity, RunBinding


class FakeProcess:
    def __init__(self, binding):
        self.binding = binding
        self.identity = ProcessIdentity(100, 90, 100, os.getuid(), 1000, 10)
        self.code = None
        self.stop_count = 0
        self.extra_members = ()

    def stop(self):
        self.stop_count += 1
        return True

    def exit_code(self):
        return self.code

    def group_members(self):
        return ((self.identity,) if self.code is None else ()) + self.extra_members


class RankLifecycleControls(unittest.TestCase):
    def setUp(self):
        fixture = authority_fixtures.MemberAuthorityControls()
        fixture.setUp()
        self.expected, self.projection = fixture.expected, fixture.projection
        self.binding = RunBinding.from_expected(self.expected, "b" * 64)
        self.temporary = tempfile.TemporaryDirectory(prefix="exo-lifecycle-")
        self.root = Path(self.temporary.name)
        self.now, self.memory, self.port_free, self.boot = 101.0, 100, True, "test-boot"
        self.child = FakeProcess(self.binding)
        self.spawn_count = 0
        self.events = []
        self.actor = self.make_actor()

    def tearDown(self):
        self.temporary.cleanup()

    def make_actor(self, **overrides):
        def spawn(*args, **kwargs):
            self.assertTrue((self.root / "run/launch-intent.json").exists())
            self.assertTrue(kwargs["authority"].allows_start(self.now))
            self.spawn_count += 1
            return self.child

        def emit(value):
            self.assertTrue(
                (self.root / f"run/event-{value['sequence']:06d}.json").exists()
            )
            self.events.append(value)

        spec = RankSpec(
            self.binding,
            (sys.executable, "-B", "-c", "pass"),
            self.root,
            (Listener("127.0.0.1", 39999),),
            5,
        )
        parameters = dict(
            memory_reader=lambda: self.memory,
            port_reader=lambda _: self.port_free,
            boot_reader=lambda: self.boot,
            monotonic=lambda: self.now,
            spawner=spawn,
            emit=emit,
        )
        parameters.update(overrides)
        return RankLifecycle(self.expected, spec, self.root / "run", **parameters)

    def observe(self, projection=None, *, absent=False):
        return self.actor.observe_authority(
            None if absent else self.projection if projection is None else projection,
            request_started_monotonic=self.now - 1,
            received_monotonic=self.now,
        )

    def start(self):
        self.observe()
        self.actor.start(environment={"PATH": os.defpath})

    def test_admitted_starts_once_serving_required_for_requests(self):
        self.start()
        self.assertEqual(self.spawn_count, 1)
        self.assertFalse(self.actor.may_admit())
        self.observe({**self.projection, "phase": "serving"})
        self.assertTrue(self.actor.may_admit())
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.spawn_count, 1)

    def test_partial_acquisition_waits_without_poisoning_future_admission(self):
        decision = self.observe(
            {**self.projection, "phase": "acquiring", "authoritative": False}
        )
        self.assertEqual(decision.state, "stop_required")
        self.assertFalse(self.actor.draining)
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.start()
        self.assertEqual(self.spawn_count, 1)

    def test_no_quiet_timer_unknown_read_stops_admission_not_running_work(self):
        self.start()
        self.observe({**self.projection, "phase": "serving"})
        self.assertTrue(self.actor.may_admit())
        self.now = 399.9
        self.assertTrue(self.actor.may_admit())
        self.assertEqual(self.child.stop_count, 0)
        self.observe(absent=True)
        self.assertFalse(self.actor.may_admit())
        self.assertEqual(self.child.stop_count, 0)
        self.assertEqual(self.actor.authority_deadline, 400)
        self.now = 400
        self.actor.authority_alarm()
        self.assertEqual(self.child.stop_count, 1)
        self.assertFalse(self.actor.may_admit())
        self.assertFalse((self.root / "run/cleanup-evidence.json").exists())

    def test_fence_stops_exact_rank_does_not_adopt_or_restart(self):
        self.start()
        self.observe({**self.projection, "generation": 2, "release_requested": True})
        self.assertTrue(self.actor.draining)
        self.assertEqual(self.child.stop_count, 1)
        self.assertEqual(self.actor.spec.binding.generation, 1)
        self.observe({**self.projection, "phase": "serving"})
        self.assertFalse(self.actor.may_admit())
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.spawn_count, 1)

    def test_event_is_a_wakeup_not_new_authority(self):
        self.start()
        self.observe({**self.projection, "phase": "serving"})
        wake = self.actor.event_requires_observation(
            {
                "type": "lease.group_changed",
                "target": self.binding.lease_id,
                "data": {
                    "group_id": self.binding.group_id,
                    "generation": 2,
                    "resource_id": self.binding.resource_id,
                    "lease_id": self.binding.lease_id,
                },
            }
        )
        self.assertTrue(wake)
        self.assertFalse(self.actor.may_admit())
        self.assertEqual(self.child.stop_count, 0)
        self.assertEqual(self.actor.authority_deadline, 400)

    def test_expiry_during_physical_reads_prevents_any_spawn(self):
        self.observe()
        self.actor.memory_reader = lambda: setattr(self, "now", 401) or 100
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.spawn_count, 0)
        self.assertFalse((self.root / "run/launch-intent.json").exists())

    def test_ambiguous_spawn_is_durable_and_never_replayed(self):
        self.observe()

        def unknown(*args, **kwargs):
            self.assertTrue((self.root / "run/launch-intent.json").exists())
            self.spawn_count += 1
            raise ObservationUnknown("lost child observation")

        self.actor.spawner = unknown
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.actor.phase, "uncertain")
        with self.assertRaises(ObservationUnknown):
            self.actor.start(environment={})
        self.assertEqual(self.spawn_count, 1)
        self.assertIsNone(self.actor.refresh_cleanup())
        with self.assertRaises(FileExistsError):
            self.make_actor()

    def test_false_exit_event_never_cancels_running_work(self):
        self.start()
        self.assertIsNone(self.actor.process_exited())
        self.assertEqual(self.child.stop_count, 0)
        self.assertFalse(self.actor.draining)

    def test_memory_and_group_must_return_before_evidence_no_auto_settlement(self):
        self.start()
        self.child.code = 0
        self.memory = 106  # Above the recorded explicit five-byte tolerance.
        self.assertIsNone(self.actor.process_exited())
        self.assertFalse((self.root / "run/cleanup-evidence.json").exists())
        self.memory = 105
        self.child.extra_members = (replace(self.child.identity, pid=101),)
        self.assertIsNone(self.actor.refresh_cleanup())
        self.child.extra_members = ()
        receipt = self.actor.refresh_cleanup()
        self.assertEqual(self.actor.phase, "awaiting_settlement")
        self.assertFalse(receipt["lease_released"])
        self.assertFalse(receipt["settlement_recorded"])
        raw = Path(receipt["evidence_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt["evidence_sha256"])
        self.assertEqual(json.loads(raw)["lease_id"], self.binding.lease_id)
        self.assertEqual(self.spawn_count, 1)

    def test_boot_change_or_unobservable_memory_never_proves_cleanup(self):
        self.start()
        self.child.code = 0
        self.boot = "another-boot"
        self.assertIsNone(self.actor.process_exited())
        self.boot = "test-boot"
        self.actor.memory_reader = lambda: None
        self.assertIsNone(self.actor.refresh_cleanup())
        self.assertFalse((self.root / "run/cleanup-evidence.json").exists())

    def test_event_delivery_error_keeps_durable_evidence_and_owned_child(self):
        self.actor.emit = lambda _: (_ for _ in ()).throw(ConnectionError("offline"))
        self.start()
        self.assertEqual(self.actor.phase, "running")
        self.assertEqual(self.spawn_count, 1)
        self.assertTrue(list((self.root / "run").glob("delivery-failed-*.json")))

    def test_failed_audit_write_never_prevents_required_stop_or_fakes_cleanup(self):
        self.start()
        self.now = self.actor.authority_deadline
        with mock.patch(
            "ollama.exo_groups.rank_lifecycle.write_new", side_effect=OSError("fixture disk full")
        ):
            self.actor.authority_alarm()
            self.assertEqual(self.child.stop_count, 1)
            self.assertTrue(self.actor.draining)
            self.assertTrue(self.actor.audit_write_failures)
            self.child.code = 0
            self.assertIsNone(self.actor.refresh_cleanup())
            self.assertIsNone(self.actor.receipt)
        receipt = self.actor.refresh_cleanup()
        self.assertIsNotNone(receipt)
        self.assertFalse(receipt["lease_released"])
        recovered = [e for e in self.events if e["type"] == "audit_write_failures"]
        self.assertEqual(len(recovered), 1)
        self.assertGreaterEqual(len(recovered[0]["writes"]), 2)
        self.assertEqual(self.actor.audit_write_failures, [])


@unittest.skipUnless(sys.platform == "darwin", "real Darwin child lifecycle")
class ActualLifecycleControls(unittest.TestCase):
    def test_real_child_is_stopped_and_observed_before_cleanup_event(self):
        import socket

        fixture = authority_fixtures.MemberAuthorityControls()
        fixture.setUp()
        with tempfile.TemporaryDirectory(prefix="exo-actual-lifecycle-") as directory:
            root = Path(directory)
            with socket.socket() as temporary:
                temporary.bind(("127.0.0.1", 0))
                port = temporary.getsockname()[1]
            code = f"import socket,pathlib,time;s=socket.socket();s.bind(('127.0.0.1',{port}));s.listen();pathlib.Path('ready').write_text('ready');time.sleep(60)"
            spec = RankSpec(
                RunBinding.from_expected(fixture.expected, "b" * 64),
                (sys.executable, "-B", "-c", code),
                root,
                (Listener("127.0.0.1", port),),
                0,
            )
            events = []
            actor = RankLifecycle(
                fixture.expected,
                spec,
                root / "run",
                memory_reader=lambda: 100,
                emit=events.append,
            )
            before = authority_clock()
            actor.observe_authority(
                fixture.projection,
                request_started_monotonic=before,
                received_monotonic=authority_clock(),
            )
            actor.start(environment={"PATH": os.defpath})
            try:
                end = time.monotonic() + 5
                while not (root / "ready").exists() and time.monotonic() < end:
                    time.sleep(0.01)
                self.assertTrue((root / "ready").exists())
                before = authority_clock()
                actor.observe_authority(
                    {**fixture.projection, "generation": 2},
                    request_started_monotonic=before,
                    received_monotonic=authority_clock(),
                )
                actor.process.process.wait(timeout=5)
                receipt = actor.process_exited()
                self.assertEqual(receipt["lease_id"], spec.binding.lease_id)
                self.assertFalse(receipt["lease_released"])
                self.assertEqual(actor.phase, "awaiting_settlement")
                self.assertIsNotNone(actor.process.exit_code())
                self.assertFalse(actor.process.group_members())
                self.assertTrue((root / "run/launch-intent.json").exists())
            finally:
                if actor.process.exit_code() is None:
                    actor.process.stop()
                    actor.process.process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
