# Delegated EXO firewall updates

A service account can inspect the installed EXO filter through its existing
read-only delegation. Changing a peer address previously needed an administrator
for every attempt. This package installs a bounded capability once per host;
the ordinary service account can then submit and inspect its own anchor updates.
It does not change the core registry, daemon imports, Python environment, lease
ownership or inference behavior.

The installer adds root-owned code under
`/Library/Preferences/xyz.samtg.exo-pf/control` and one sudoers file,
`91-samcloud-exo-control`. The rule names the service account and grants exactly
two digest-bound commands: `control/run --apply` and `control/run --status`.
The wrapper starts the system Python with `-I -S -B`. It accepts bounded JSON
update data, never a module, script, shell command or arbitrary file path.

The capability fixes the host's service UID, queue journal and ports
52515/52514/52513/52518. Peer addresses must be private IPv4 addresses. Only the
existing `com.apple/990.SamcloudEXO` anchor may change. The existing PF enable
reference, main rules, other anchors and read-only delegation are preserved.
The current installed policy, binary, receipt, boot, route source and exact
process identities are checked again during each update.

Queue inspection and signals run in a finite child with the service UID, primary
group and supplementary groups. That child holds a real SQLite writer barrier,
checks that all work and the current seat turn have drained, and freezes the
exact owned gateway. Pipe closure resumes that gateway and releases the barrier.
Root never opens the service journal or signals a caller-selected process.

Each request has a persisted id and immutable body. Re-reading a completed id
returns its saved result. An unknown result blocks subsequent updates. A fresh
explicit attempt is possible after a proved pre-mutation refusal. No mutation is
automatically retried, and no daemon, watchdog, lease action or compute deadline
is added. Privileged code upgrades still require an administrator.

## Prepare and install

Use the actual loaded service journal path. Packages are created in a new
local directory and contain source plus an exact per-host manifest.

```sh
python3 ops/exo-pf/package.py --rank slice \
  --journal /Users/claude-services/var/queue.sqlite \
  --destination /path/to/fresh-package
```

Copy the complete package to its named host and verify every manifest entry.
The packager prints `installer_sha256` and a hash-pinned `administrator_command`.
In an administrator terminal on the named host, run that command once. Its small
system-Python bootstrap opens the installer without following symlinks, verifies
the reviewed SHA256, and compiles/executes those same buffered bytes. It does not
re-open a service-owned code path after verification. If the package moves to a
different host path, regenerate the command for that exact path using
`install_command(path, reviewed_sha256)`; retain the reviewed digest.

Installation only adds the scoped capability; it performs no PF update, queue
change or restart. It may be installed while ordinary work runs. The service
account subsequently reads `control/run --status` through `sudo -n`, verifies
all installed source hashes against its package, captures a fresh transition
plan, and sends one request to `control/run --apply` on stdin. The plan schema
is the same as `update-exo-pf.py`, wrapped in
`{"request_id":"<64 lowercase hex characters>","plan":{...}}`.
Observe the existing operation after a lost reply; never mint a second request
merely because an observation failed.

The old human-only updater is retained as a shared implementation and historic
entry point. Its macOS closed-descriptor check now accepts lsof status 1 only
when every reported row is the exact frozen gateway's closed IPv4 loopback API
client. A listener, live connection, foreign PID, UDP row, diagnostic or PF state
still refuses.

## Validation and deployment state

```sh
python3 -m unittest discover -s ops/exo-pf -p 'test_*.py' -v
```

Controls cover the actual macOS lsof output, exact capability boundaries,
request deduplication, unknown-result quarantine, parser and load failures,
unchanged PF enable/delegation, root code pins and a real cross-process SQLite
writer lock. Gateway signals and PF operations are mocked in the offline suite.
A separate live Slice rehearsal ran the barrier as UID 502, held the actual
writer lock, resumed the same gateway, retained a byte-identical PF receipt and
returned healthy. The native visudo parser accepted the staged rule on all three
hosts. The repository packager reproduces all eight staged files byte for byte.

These are qualification results. Root installation and the actual rule update
remain separate observable operations; neither is implied by passing tests.
