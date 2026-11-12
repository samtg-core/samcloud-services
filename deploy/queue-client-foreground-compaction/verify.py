#!/usr/bin/env python3
"""Read-only pins for a staged client copy. Never imports or activates it."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(root, expected, *, inventory=False):
    failures = []
    if inventory:
        observed = {str(p.relative_to(root)) for p in (root / "ollama").rglob("*.py")
                    if "__pycache__" not in p.parts}
        failures.extend({"path": p, "reason": "unexpected"} for p in sorted(observed - expected.keys()))
        failures.extend({"path": p, "reason": "missing"} for p in sorted(expected.keys() - observed))
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            failures.append({"path": name, "reason": "missing_or_symlink"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append({"path": name, "reason": "sha256_mismatch"})
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("input", "output"), default="input")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    failures = inspect(args.root, manifest[args.phase], inventory=True)
    native = inspect(args.native_root, manifest["native_seams"])
    print(json.dumps({"phase": args.phase, "client_failures": failures,
                      "native_seam_failures": native, "read_only": True}))
    return int(bool(failures or native))


if __name__ == "__main__":
    raise SystemExit(main())
