# Capability playground v1

A versioned, small useful-agent baseline **inside model-services**. It measures what a
model/client/configuration can accomplish, rather than using host health or raw speed
as a capability score. [Cloud RFC / PRD 545](https://cloud.samtg.xyz/docs/545) records
Sam's direction; [repository design](../docs/Capability Evaluation.md) records scope.

Three tasks, identical across configured clients/compute:

- **CSV rollup:** implement a utility and deliver output; original Obelisk task plus
  four independently held checks (example, empty, quoting, invalid integers).
- **Repository repair:** fix a scheduling-window utility without breaking API or
  mutating inputs; independent normal, edge and invalid-input checks.
- **Evidence audit:** reconcile conflicting source files, classify what is supported,
  cite sources and deliver a handback. Automated classification/source checks plus
  a separate human review of explanation quality.

No composite ranking. Three small tasks cannot establish broad model superiority.
General performance/context/concurrency benchmarks follow this capability baseline.

## Configure and run

Use `config.example.json` as a private deployment template. Pin model digest/context,
compute resource, identity, backend, gateway and client versions. Configure the existing
native/rail client; do not add a new physical claim around a leased model-service.
Preflight records verified identity/digest and explicit deployment pins, not a task pass.

```sh
python -m model_eval run --config /PRIVATE/deployment.json --run-dir /PRIVATE/runs/baseline-001
python -m model_eval compare /PRIVATE/runs/baseline-001 /PRIVATE/runs/baseline-002
python -m unittest discover -s tests -p test_model_eval.py
```

Run on the configured host under its own worker principal. The preflight checks the
actual requester and installed model digest. Hermes profiles also check model/context.
Version labels for hardware/backend/client/gateway are operator supplied; reports
separate those from verified identity/digest/profile observations. An ordinary caller
must have legitimate scope; the runner neither provisions users nor widens access.

Native Hermes needs the **existing deployed native runtime and queue plugins**.
The runner copies only config/SOUL/plugin sources into a fresh per-task profile, not
auth.json, memory, sessions, cron, caches, histories or skills. Runtime paths and tokens
stay outside Git. Tokens come from an explicit private file into named process-env
variables; no token fallback. Do not put secrets in command/environment configuration.
The argv is executed directly, never through an implicit shell.

`native-service` and `rail-service` describe the operator-configured delivery adapter;
the CLI does not commission a rail or silently convert a local run into a rail result.
A rail comparison requires a real configured client/delivery command and its receipts.
Obelisk validates the native-service baseline first; then match tasks/tools/assistance
policy on the earlier 35B and v1 Hermes rail configurations. Preserve old CSV outcomes
as historical evidence, not fresh passes for this expanded suite.

## Records and interruptions

Each run has an immutable config/fixture binding, manifest, per-task workspace,
private profile, agent log, launch intent/PID, exit status, artifact hashes and held-out
check results. Reusing a run directory never launches an existing task again. Changed
configuration/profile/fixtures refuse resume. A failed task stops later dispatch.
An interrupted or launch-intent-only task is unresolved; inspect the exact native and
service execution instead of making a fresh run. The runner is an observer/experiment
controller, not a replacement ownership supervisor. It does not assert process absence
or release physical capacity.

There is no inference wall-clock kill. Only the independent acceptance subprocess
has a ten-second limit so a broken generated utility cannot hang verification.
Client exit zero alone is not a task pass. Missing metrics are not zero: no kernel
peak, token efficiency, decode throughput or cleanup is inferred from an artifact.
Elapsed time includes queue/tool work. Test preparation is separate from agent work,
but this is **not an adversarial sandbox**: the native client runs with its OS account's
permissions. Do not give untrusted agents additional credentials or claim hidden tests
are inaccessible against a malicious same-account process.

Task pass means the stated automated checks passed without operator intervention.
Review agent logs for tool/test discipline and handback quality; source-file citation
and a nonempty handback alone do not prove sound prose. Keep infrastructure/client
failures distinct from wrong task answers. Keep original failures when fixing adapters.
