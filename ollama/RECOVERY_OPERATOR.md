# Offline dead-attempt recovery

`inference_recovery.OfflineRecovery` is an operator interface for an existing
journal. Its scheduler lock excludes a running supervisor. The operator also
fences automatic service/backend revival and establishes backend termination.
The interface has no consumer route and starts no inference.

Two explicit outcomes are available:

- `retry_dead_attempt`: audit the dead attempt and queue the same execution/input.
- `fail_dead_attempt`: audit the dead attempt and record a terminal provider failure.

Both require the same inspection token and trusted death attestation. The
same-boot `StoppedLocalBackend` proof records the daemon and complete owned
runner inventory, each with a birth identity, plus observations that every
identity is gone, revival is fenced and the listener is closed. These fields
record an operator's measurements; validation checks their consistency. The
legacy `DeadLocalBackend` proof requires different host boot UUIDs.

## Terminal operation

`fail_dead_attempt(expected, *, proof, operator, code)`
accepts only the literal `backend_attempt_dead` code. It requires an owned,
unreleased, output-free `uncertain` execution and its matching `uncertain`
grant. A changed inspection token, existing output/result or missing proof
refuses the operation.

One transaction writes a separate `backend_attempt_terminations` audit row,
the failed execution and the grant transition. The audit binds the prior
lifecycle sequence, request hash, operator and proof. Existing retry audit rows
retain their meaning. An execution-lifetime grant becomes released; a work-session
grant retains idle affinity with no active execution. The job-aware branch also advances any requested parent finish/yield inside
that transaction, matching its own `Arbiter.finish`. It still dispatches no work.
Physical registry claims and residency records have their own exact settlement receipts.

The saved body is a fixed, sanitized `kind:error`, `status:502`,
`type:provider_error`, `code:backend_attempt_dead` result with the audit ID.
`DurableConsumer` raises `ExecutionStopped`; the SDK adapter returns a 502 with
`x-should-retry:false` and the original execution/grant IDs. This closes the
attempt as an infrastructure/provider failure. It supplies no model answer or
model-quality score. The operation does not pump or retry the queue.

## Operational sequence

1. Bind the exact execution, owner, resource, journal and release identities.
   Inspect other work and wait for a safe boundary. Record the caller effect:
   the preserved consumer will receive a terminal provider failure.
2. Fence the owning scheduler and backend revival. Archive the complete owned
   process inventory with birth identities. Stop that generation, observe every
   recorded identity gone and the listener closed, and hash the evidence.
3. Settle exact physical claims and residency through their supported interfaces
   using that evidence. Save the acknowledgement receipts.
4. Open `OfflineRecovery` under its lock, inspect the exact attempt again, and
   apply the terminal operation with the archived proof and fixed code.
5. Verify the termination audit, failed result, original input hash and logical
   grant state. Restore the selected backend first, check health, then restore
   the gateway. The original consumer receives its original attempt's outcome.
6. Record the intervention and preserve the failed attempt in evaluation reports.
   Preserve its terminal receipt across rollback. A later intentional attempt
   gets its own identity after the old ownership is settled.

An empty model list, elapsed time, transport failure or a changed PID cannot
supply the required death proof. Keep uncertain ownership visible while the
measurements remain incomplete.

## Source and qualification

The same-boot proof baseline is the previously reviewed operator module SHA256
`0ea7f4e78974fed823a0e42436b71e43b11473f70ea3a0bd1cea3c4a1a2edc10`.
The terminal operator preserves those proof classes and the retry method.
The branch module keeps its qualified `StintPolicy` journal and job-advance hook.
The earlier pinned-runtime operator is the immutable blob
`82fb1f8:ollama/inference_recovery.py` (SHA256
`c31b04ed29551fe458e58f8d7bddb599615879bd1b0dfb12b155b5145331b115`).
That selected blob is qualified with the pinned pre-requester-job runtime;
it is separate from the branch module.

Use the selected module with the exact reviewed runtime dependency manifest;
source publication of this operator branch is separate from a gateway upgrade.

```sh
python -m unittest ollama.test_terminal_recovery
```

The synthetic tests cover proof refusal, CAS, exact identity/input preservation,
transaction rollback, lock exclusion, grant lifetime, durable audit, replay
refusal and consumer/SDK failure delivery. The reviewed packet also runs the
legacy recovery and same-boot proof suites against the selected runtime.
Their fixtures are not evidence about any live backend.
