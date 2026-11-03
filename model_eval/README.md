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

V1 accepts only `native-service`. `rail-service` is refused until a real rail
adapter records delivery receipts. A configuration label cannot create a rail result.
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


## Review hardening and comparison cohorts

Agent processes inherit only PATH/LANG/LC_ALL/TMPDIR plus explicitly configured
values and named bearer variables. Checker subprocesses get a separate minimal
environment and empty temporary HOME, never the observer's credentials.
After each task and checker pass the controller scans task evidence for the exact
bearer; a match fails as `secret_in_evidence` and only redacted bytes are retained.
This is evidence hygiene, not isolation from a malicious same-OS-account process.

Resume binds harness and active checker source hashes as well as config/profile/task
inputs. Comparisons group by fixture hash, harness/checker sources, allowed tools and
assistance policy. The first run is the reference; mismatches or missing cohort fields
are explicitly not comparable. Missing provenance is `unknown`, never `fresh`.

Intervention count and unattended status default to unknown. After inspecting the
trace, record an operator intervention or attest zero; recorded interventions cannot
be erased with a zero attestation. These commands refuse an active observer lock:

```sh
python -m model_eval assistance --run-dir /PRIVATE/run --task csv-rollup --note 'Operator repaired the submitted code'
python -m model_eval assistance --run-dir /PRIVATE/run --task csv-rollup --attest-zero --note 'Reviewed trace; no task assistance'
```

Check results list unchecked requirements explicitly. Source citations may be a valid
subset of the fixture files; requiring every filename in every reason was an overly
strict first checker. Preserve the old run's failed result and checker version rather
than silently replacing it. Compare a fresh repaired-baseline cohort separately.
