"""CPU-only owned child/listener checks; no model, production API or lease."""

from dataclasses import replace
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from .exo_groups.clock import authority_clock
from .exo_groups.member import AuthorityDecision, ExpectedMember
from .exo_groups import rank_process as process
from . import test_exo_group_plan as fixtures


def binding():
    source = fixtures.GroupBindingControls()
    source.setUp()
    return process.RunBinding.from_expected(
        ExpectedMember.from_launch(source.plan, source.bind()[0]), "b" * 64
    )


def authorized():
    now = authority_clock()
    return AuthorityDecision(
        "authorized", "test scoped authority", now, now + 60, binding().expected
    )


class CleanupPolicyControls(unittest.TestCase):
    def setUp(self):
        self.binding = binding()
        self.identity = process.ProcessIdentity(100, 90, 100, os.getuid(), 1000, 123)
        self.observation = process.CleanupObservation(
            self.binding,
            self.identity,
            0,
            (),
            (True,),
            100,
            101,
            1,
            datetime.now(timezone.utc).isoformat(),
        )

    def test_evidence_is_bound_to_launched_member_and_does_not_settle(self):
        raw, digest = process.cleanup_evidence(self.observation)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        value = json.loads(raw)
        self.assertEqual(value["lease_id"], self.binding.lease_id)
        self.assertEqual(value["generation"], self.binding.generation)
        self.assertFalse(value["lease_released"])
        self.assertFalse(value["settlement_recorded"])
        self.assertEqual(value["observation"]["child"]["started_microseconds"], 123)

    def test_binding_cannot_relabel_cleanup_for_a_different_member(self):
        for change in (
            {"group_id": "group_" + "f" * 24},
            {"generation": 2},
            {"lease_id": "lease_" + "f" * 16},
            {"resource_id": "other/gpu"},
            {"runtime_sha256": "bad"},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(self.binding, **change)

    def test_each_missing_physical_proof_retains_claim(self):
        changes = (
            {"exit_code": None},
            {"exit_code": True},
            {"group_members": (self.identity,)},
            {"listeners_closed": ()},
            {"listeners_closed": (False,)},
            {"listeners_closed": (1,)},
            {"observed_in_use_bytes": 102},
            {"baseline_in_use_bytes": -1},
            {"observed_in_use_bytes": None},
            {"tolerance_bytes": True},
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(process.ObservationUnknown):
                    process.cleanup_evidence(replace(self.observation, **change))

    def test_unknown_memory_is_not_zero(self):
        valid = [{"PerformanceStatistics": {"In use system memory": 123}}]
        self.assertEqual(process.gpu_in_use_from_plist(plistlib.dumps(valid)), 123)
        for data in (
            [],
            [{}],
            valid + valid,
            [{"PerformanceStatistics": {"In use system memory": True}}],
            [{"PerformanceStatistics": {"In use system memory": -1}}],
        ):
            with self.subTest(data=data):
                with self.assertRaises(process.ObservationUnknown):
                    process.gpu_in_use_from_plist(plistlib.dumps(data))
        with self.assertRaises(process.ObservationUnknown):
            process.gpu_in_use_from_plist(b"not a plist")

    def test_listener_checks_bind_conflicts_not_connection_failures(self):
        with socket.socket() as listening:
            listening.bind(("127.0.0.1", 0))
            listening.listen()
            target = process.Listener("127.0.0.1", listening.getsockname()[1])
            self.assertFalse(process.listener_closed(target))
        self.assertTrue(process.listener_closed(target))
        with self.assertRaises(process.ObservationUnknown):
            process.listener_closed(process.Listener("192.0.2.123", 39999))

    def test_udp_discovery_port_is_checked_as_udp(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as discovery:
            discovery.bind(("127.0.0.1", 0))
            target = process.Listener("127.0.0.1", discovery.getsockname()[1], "udp")
            self.assertFalse(process.listener_closed(target))
        self.assertTrue(process.listener_closed(target))
        with self.assertRaises(ValueError):
            process.Listener("127.0.0.1", 9999, "unknown")

    def test_ipv6_tcp_and_udp_ports_use_the_ipv6_family(self):
        for kind, protocol in ((socket.SOCK_STREAM, "tcp"), (socket.SOCK_DGRAM, "udp")):
            with self.subTest(protocol=protocol):
                with socket.socket(socket.AF_INET6, kind) as listening:
                    listening.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    listening.bind(("::1", 0))
                    if kind == socket.SOCK_STREAM:
                        listening.listen()
                    target = process.Listener("::1", listening.getsockname()[1], protocol)
                    self.assertFalse(process.listener_closed(target))
                self.assertTrue(process.listener_closed(target))

    def test_group_read_brackets_memory_and_port_samples(self):
        class Observed:
            identity = self.identity
            binding = self.binding
            reads = 0

            def exit_code(inner):
                return 0

            def group_members(inner):
                inner.reads += 1
                return () if inner.reads == 1 else (inner.identity,)

        target = Observed()
        seen = process.observe_cleanup(
            target,
            listeners=[process.Listener("127.0.0.1", 39999)],
            baseline_in_use_bytes=100,
            tolerance_bytes=5,
            observed_at="test",
            memory_reader=lambda: 100,
            port_reader=lambda _: True,
        )
        self.assertFalse(seen.clean)
        self.assertEqual(target.reads, 2)


@unittest.skipUnless(sys.platform == "darwin", "Darwin kernel identity controls")
class DarwinProcessControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="exo-owned-child-")
        self.root = Path(self.directory.name)
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.exit_code() is None:
                if not child.stop_requested:
                    child.stop()
            child.process.wait(timeout=5)
        self.directory.cleanup()

    def spawn(self, code, authority=None):
        child = process.OwnedRankProcess.spawn(
            [sys.executable, "-B", "-c", code],
            binding=binding(),
            authority=authorized() if authority is None else authority,
            cwd=self.root,
            environment={"PATH": os.defpath},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.children.append(child)
        return child

    def wait_file(self, name):
        end = time.monotonic() + 5  # Test harness only, never an inference limit.
        path = self.root / name
        while not path.exists() and time.monotonic() < end:
            time.sleep(0.01)
        self.assertTrue(path.exists(), "CPU fixture did not become ready")
        return path

    def test_kernel_structure_matches_installed_sdk_and_real_pid(self):
        code = '#include <stdio.h>\n#include <sys/proc_info.h>\nint main(void){printf("%zu",sizeof(struct proc_bsdinfo));}\n'
        (self.root / "size.c").write_text(code)
        subprocess.run(
            [
                "xcrun",
                "clang",
                str(self.root / "size.c"),
                "-o",
                str(self.root / "size"),
            ],
            check=True,
        )
        actual = int(subprocess.check_output([str(self.root / "size")]))
        self.assertEqual(actual, ctypes.sizeof(process._BSDInfo))
        own = process.darwin_process(os.getpid())
        self.assertEqual(own.pid, os.getpid())
        self.assertEqual(own.uid, os.getuid())
        self.assertEqual(own.group_id, os.getpgrp())
        self.assertGreater(own.started_seconds, 0)
        self.assertLess(own.started_microseconds, 1000000)
        self.assertIsNone(process.darwin_process(2147483647))

    def test_no_spawn_without_current_explicit_authority(self):
        end = authority_clock() - 1
        for state in ("authorized", "unknown", "stop_required"):
            with self.subTest(state=state):
                with self.assertRaises(process.ObservationUnknown):
                    self.spawn(
                        "import time;time.sleep(60)",
                        AuthorityDecision(state, "test", end - 1, end),
                    )
        self.assertEqual(self.children, [])

    def test_live_authority_for_another_member_never_spawns(self):
        other_expected = replace(
            binding().expected, member=replace(binding().expected.member, memory_mb=1)
        )
        wrong = replace(authorized(), expected=other_expected)
        with self.assertRaises(process.ObservationUnknown):
            self.spawn("import time;time.sleep(60)", wrong)
        self.assertEqual(self.children, [])

    def test_authority_elapsed_during_guard_start_never_activates_rank(self):
        authority = authorized()
        guards = []
        original = process.prepare_guard

        def prepare(*args, **kwargs):
            guard = original(*args, **kwargs)
            guards.append(guard)
            return guard

        with (
            mock.patch.object(process, "prepare_guard", side_effect=prepare),
            mock.patch.object(
                process,
                "authority_clock",
                side_effect=[
                    authority.checked_at_monotonic,
                    authority.expires_at_monotonic + 1,
                ],
            ),
        ):
            with self.assertRaises(process.ObservationUnknown):
                self.spawn(
                    "import pathlib;pathlib.Path('should-not-start').touch()", authority
                )
        self.assertEqual(len(guards), 1)
        self.assertFalse(guards[0].activated)
        self.assertEqual(guards[0].process.wait(timeout=5), 0)
        self.assertFalse((self.root / "should-not-start").exists())

    def test_stop_then_wait_for_process_listener_and_memory(self):
        child = self.spawn(
            "import socket,pathlib,time; s=socket.socket();s.bind(('127.0.0.1',0));s.listen();pathlib.Path('port').write_text(str(s.getsockname()[1]));time.sleep(60)"
        )
        port = int(self.wait_file("port").read_text())
        listener = process.Listener("127.0.0.1", port)
        self.assertFalse(process.listener_closed(listener))
        self.assertTrue(child.stop())
        self.assertFalse(child.stop())
        child.process.wait(timeout=5)
        observation = process.observe_cleanup(
            child,
            listeners=[listener],
            baseline_in_use_bytes=100,
            tolerance_bytes=10,
            observed_at=datetime.now(timezone.utc).isoformat(),
            memory_reader=lambda: 110,
        )
        self.assertTrue(observation.clean)
        raw, _ = process.cleanup_evidence(observation)
        self.assertEqual(json.loads(raw)["lease_id"], child.binding.lease_id)
        self.assertIsNotNone(child.exit_code())

    def test_changed_birth_refuses_signal_and_retains_same_process(self):
        child = self.spawn("import time;time.sleep(60)")
        actual = child._identity_reader
        child._identity_reader = lambda _: replace(
            child.identity, started_microseconds=child.identity.started_microseconds + 1
        )
        with self.assertRaises(process.ObservationUnknown):
            child.stop()
        self.assertFalse(child.stop_requested)
        self.assertIsNone(child.exit_code())
        child._identity_reader = actual
        self.assertTrue(child.stop())
        child.process.wait(timeout=5)

    def test_dead_root_with_live_child_cannot_attest_cleanup_or_signal_new_group(self):
        code = "import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c',\"import pathlib,time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path('descendant-ready').write_text('ready');time.sleep(60)\"]);pathlib.Path('descendant-pid').write_text(str(p.pid));time.sleep(.1)"
        child = self.spawn(code)
        self.wait_file("descendant-ready")
        descendant_pid = int(self.wait_file("descendant-pid").read_text())
        descendant = process.darwin_process(descendant_pid)
        try:
            child.process.wait(timeout=5)
            self.assertIn(descendant_pid, [x.pid for x in child.group_members()])
            with self.assertRaises(process.ObservationUnknown):
                child.stop()
            observation = process.observe_cleanup(
                child,
                listeners=[process.Listener("127.0.0.1", 39999)],
                baseline_in_use_bytes=100,
                tolerance_bytes=0,
                observed_at=datetime.now(timezone.utc).isoformat(),
                memory_reader=lambda: 100,
            )
            self.assertFalse(observation.clean)
            with self.assertRaises(process.ObservationUnknown):
                process.cleanup_evidence(observation)
        finally:
            current = process.darwin_process(descendant_pid)
            if current is not None and current.birth == descendant.birth:
                os.kill(descendant_pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
