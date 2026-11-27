"""Use the pinned EXO environment for CPU-only provider configuration checks."""

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest


def forbidden(*args, **kwargs):
    raise AssertionError("network and process launch forbidden during preparation controls")


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"mlx", "mlx_lm", "mflux"}:
            raise AssertionError("native compute imports forbidden")


def audit(event, arguments):
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                 "socket.connect", "socket.bind", "socket.getaddrinfo"}:
        forbidden()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exo-source", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = args.exo_source.resolve()
    pin_bytes = (here / "fixtures/exo-source-pins.json").read_bytes()
    pins = json.loads(pin_bytes)
    for name, expected in pins["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("EXO qualification source changed: " + name)
    fixture = json.loads((here / "fixtures/exo-instance-v1.json").read_bytes())
    if fixture["runtime_sha256"] != hashlib.sha256(pin_bytes).hexdigest():
        raise ValueError("fixture source manifest changed")
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    socket.getaddrinfo = forbidden
    sys.meta_path.insert(0, NoNative())
    sys.addaudithook(audit)
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(source / "src"))
    with tempfile.TemporaryDirectory(prefix="exo-provider-controls-") as private:
        os.chmod(private, 0o700)
        os.environ["EXO_HOME"] = private
        os.environ["EXO_DASHBOARD_DIR"] = str(source / "dashboard")
        os.environ["EXO_RESOURCES_DIR"] = str(source / "resources")
        suite = unittest.defaultTestLoader.discover(str(here), pattern="test_exo_instance_types.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or result.testsRun != 15:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
