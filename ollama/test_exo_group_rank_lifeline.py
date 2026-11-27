"""Real CPU processes/pipes/listeners; no Metal, model, lease or production API."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from .exo_groups.rank_lifeline import prepare_guard
from .exo_groups.rank_process import darwin_group, darwin_process


@unittest.skipUnless(sys.platform == "darwin", "Darwin owned-group qualification")
class LifelineControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="exo-lifeline-")
        self.root = Path(self.directory.name)
        self.guards = []
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            if controller.poll() is None:
                controller.kill()
            controller.wait(timeout=5)
        for guard in self.guards:
            guard.close()
            # A bounded CPU-fixture cleanup, never the production stop policy.
            try:
                guard.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                identity = darwin_process(guard.process.pid)
                if identity and identity.group_id == guard.process.pid:
                    os.killpg(identity.group_id, signal.SIGKILL)
                guard.process.wait(timeout=5)
        self.directory.cleanup()

    def until(self, predicate):
        end = time.monotonic() + 8  # Test harness, not a runtime deadline.
        while time.monotonic() < end:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        self.fail("CPU process fixture did not produce its expected observation")

    def file(self, name):
        path = self.root / name
        self.until(lambda: path.exists() and path.stat().st_size > 0)
        return path

    def prepare(self, code):
        guard = prepare_guard(
            [sys.executable, "-B", "-c", code],
            cwd=self.root,
            environment={"PATH": os.defpath},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.guards.append(guard)
        return guard

    def test_no_activation_no_rank_and_closed_lifeline_exits(self):
        guard = self.prepare("import pathlib;pathlib.Path('started').touch()")
        self.assertIsNone(guard.process.poll())
        guard.close()
        self.assertEqual(guard.process.wait(timeout=5), 0)
        self.assertFalse((self.root / "started").exists())
        with self.assertRaises(RuntimeError):
            guard.activate()

    def test_exactly_one_activation_and_normal_exit_status(self):
        guard = self.prepare(
            "import pathlib;pathlib.Path('started').touch();raise SystemExit(7)"
        )
        guard.activate()
        with self.assertRaises(RuntimeError):
            guard.activate()
        self.assertEqual(guard.process.wait(timeout=5), 7)
        guard.close()
        guard.close()
        self.assertTrue((self.root / "started").exists())

    def test_malformed_activation_never_starts_rank(self):
        guard = self.prepare("import pathlib;pathlib.Path('started').touch()")
        os.write(guard._writer, b"not-start")
        self.assertEqual(guard.process.wait(timeout=5), 0)
        self.assertFalse((self.root / "started").exists())

    def test_rank_does_not_inherit_lifeline_or_other_controller_descriptors(self):
        code = """
import json,os,pathlib,signal
seen=[]
for fd in range(3,256):
 try:seen.append([fd,os.fstat(fd).st_dev,os.fstat(fd).st_ino])
 except OSError:pass
pathlib.Path('descriptors').write_text(json.dumps(seen))
signal.pause()
"""
        guard = self.prepare(code)
        pipe = os.fstat(guard._writer)
        guard.activate()
        seen = json.loads(self.file("descriptors").read_text())
        self.assertNotIn((pipe.st_dev, pipe.st_ino), [(x[1], x[2]) for x in seen])
        self.assertFalse(os.get_inheritable(guard._writer))
        guard.close()
        self.assertEqual(guard.process.wait(timeout=5), 143)

    def test_still_owned_controller_allows_rank_to_continue(self):
        code = """
import pathlib,signal
def progress(_number,_frame):pathlib.Path('progress').write_text('alive')
signal.signal(signal.SIGUSR1,progress)
pathlib.Path('root-pid').write_text(str(__import__('os').getpid()))
while True:signal.pause()
"""
        guard = self.prepare(code)
        guard.activate()
        pid = int(self.file("root-pid").read_text())
        os.kill(pid, signal.SIGUSR1)
        self.file("progress")
        self.assertIsNone(guard.process.poll())
        guard.close()
        self.assertEqual(guard.process.wait(timeout=5), 143)

    def test_direct_guard_stop_drains_same_group(self):
        guard = self.prepare(
            "import pathlib,signal;pathlib.Path('ready').touch();signal.pause()"
        )
        guard.activate()
        self.until(lambda: (self.root / "ready").exists())
        os.kill(guard.process.pid, signal.SIGTERM)
        self.assertEqual(guard.process.wait(timeout=5), 143)
        self.until(lambda: not darwin_group(guard.process.pid))

    def test_ignored_term_stays_observable_without_escalation_or_restart(self):
        code = """
import pathlib,signal
def term(_number,_frame):pathlib.Path('term').write_text('observed')
def progress(_number,_frame):pathlib.Path('after-term').write_text('still-running')
signal.signal(signal.SIGTERM,term)
signal.signal(signal.SIGUSR1,progress)
pathlib.Path('root-pid').write_text(str(__import__('os').getpid()))
while True:signal.pause()
"""
        guard = self.prepare(code)
        guard.activate()
        pid = int(self.file("root-pid").read_text())
        guard.close()
        self.file("term")
        self.assertIsNone(guard.process.poll())
        os.kill(pid, signal.SIGUSR1)
        self.file("after-term")
        self.assertIsNone(guard.process.poll())
        self.assertIn(pid, [member.pid for member in darwin_group(guard.process.pid)])
        os.kill(pid, signal.SIGKILL)  # Fixture cleanup is explicitly separate.
        self.assertEqual(guard.process.wait(timeout=5), 137)

    def test_controller_death_terminates_real_root_descendant_and_listener(self):
        descendant = "import pathlib,signal;pathlib.Path('descendant-ready').touch();signal.pause()"
        runtime = f"""
import json,os,pathlib,signal,socket,subprocess,sys
s=socket.socket();s.bind(('127.0.0.1',0));s.listen()
p=subprocess.Popen([sys.executable,'-B','-c',{descendant!r}])
pathlib.Path('runtime.json').write_text(json.dumps({{'root':os.getpid(),'descendant':p.pid,'port':s.getsockname()[1]}}))
signal.pause()
"""
        controller_source = f"""
import json,os,pathlib,signal,subprocess,sys
from ollama.exo_groups.rank_lifeline import prepare_guard
g=prepare_guard([sys.executable,'-B','-c',{runtime!r}],cwd=pathlib.Path.cwd(),environment={{'PATH':os.defpath}},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
g.activate()
pathlib.Path('guard-pid').write_text(str(g.process.pid))
signal.pause()
"""
        controller = subprocess.Popen(
            [sys.executable, "-B", "-c", controller_source],
            cwd=self.root,
            env={"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.controllers.append(controller)
        guard_pid = int(self.file("guard-pid").read_text())
        identity = darwin_process(guard_pid)
        try:
            runtime_value = json.loads(self.file("runtime.json").read_text())
            self.until(lambda: (self.root / "descendant-ready").exists())
            members = darwin_group(guard_pid)
            self.assertTrue(
                {
                    guard_pid,
                    runtime_value["root"],
                    runtime_value["descendant"],
                }.issubset({x.pid for x in members})
            )
            self.assertEqual(identity.group_id, identity.pid)
            controller.kill()  # Actual controller death, not a mocked callback.
            controller.wait(timeout=5)
            self.until(lambda: not darwin_group(guard_pid))
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", runtime_value["port"]))
        finally:
            current = darwin_process(guard_pid)
            if current is not None and current.birth == identity.birth:
                os.killpg(guard_pid, signal.SIGKILL)

    def test_forked_other_controller_does_not_retain_writer(self):
        guard = self.prepare(
            "import pathlib,signal;pathlib.Path('ready').touch();signal.pause()"
        )
        guard.activate()
        self.until(lambda: (self.root / "ready").exists())
        child = os.fork()
        if child == 0:
            # at-fork callback must have closed this writer in the child.
            try:
                os.fstat(guard._writer)
            except OSError:
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(child, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        guard.close()
        self.assertEqual(guard.process.wait(timeout=5), 143)

    def test_fork_cannot_activate_through_a_recycled_descriptor(self):
        guard = self.prepare("import pathlib;pathlib.Path('started').touch()")
        fork = os.fork()
        if fork == 0:
            try:
                guard.activate()
            except RuntimeError:
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(fork, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        guard.close()
        self.assertEqual(guard.process.wait(timeout=5), 0)
        self.assertFalse((self.root / "started").exists())


if __name__ == "__main__":
    unittest.main()
