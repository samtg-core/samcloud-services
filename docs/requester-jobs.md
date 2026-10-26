# Requester jobs: staged, not activated

A parent job owns its scheduling place across text, compaction and tool children.
An independent request does not inherit that place just because its username is
the same. Existing grants, accepted execution receipts and legacy calls retain
their frozen identities across retries and reconnects.

Source review450/c832 found lifecycle gaps. This candidate must remain disabled
on every real parent until session-ID rotation during native compaction, missing
reservation reconciliation under retained ownership, and an explicit operator
end path are repaired and qualified. Interrupted/failed work must stay recoverable;
neither session reset nor process exit alone is proof that it can be released.
The first repair addresses offline journal recovery and late-intake finish
fencing; it does not close the other review findings or authorize activation.

The native compaction lineage repair uses the context engine's existing
post-commit on_session_start boundary. A verified SessionDB compression child
and live owning Relay turn record a profile-local alias to the logical owner.
New calls under the child session then retain the original job, even while the
Relay lease still names the old transcript session. Tools, compaction, API
completion and later recovery use the same explicit lineage. Existing accepted
call records are not rewritten. An unknown/uncommitted or closed copied context
cannot create an alias; a live foreground with missing lineage refuses intake
rather than sending an unlinked call behind itself. No Hermes core hook was added.
This qualifies the live committed-notification path. Recovery after process exit
between SessionDB publication and alias persistence remains unqualified; the
candidate must not be activated with that reconnect window unresolved.

This is the inference service's requester scheduler. The shared Samcloud registry
remains the only physical allocation authority. Cross-service child delegation is
not implemented; the capabilities response explicitly advertises it as false.
Do not let an image service independently request an exclusive claim while its
text parent holds capacity and waits for it: that is a deadlock, not delegation.

## API

Check GET /v1/inference/capabilities before new linked intake. jobs.version=1,
enabled, same_service_children, explicit_finish and explicit_yield must be true.
An execution body can then contain job_key beside key and request. The service
returns job_id with the accepted execution. Persist the exact grant/execution/job
receipt; a lost observer resumes that receipt, never submits replacement work.

GET /v1/inference/jobs?key=<key> returns only the authenticated owner's job.
POST /v1/inference/jobs/end accepts {"key":"<key>","action":"finish"} or an
explicit "yield". Its 202 response echoes key/action and contains jobs. Finish
seals new intake, including late intake after a lost acknowledgement, and drains
accepted children. Unknown/running children retain ownership. Yield can later
resume the same job. Neither action cancels an execution or proves backend death.

## Native adapter

The explicit owning profile may opt in at a natural maintenance boundary:

```yaml
samcloud_queue:
  job_continuity: true
```

The default is legacy behavior; malformed/non-boolean policy refuses. This option
is read from the supplied profile at startup, with no environment behavior switch.
The same policy covers every Hermes role and both main queued-provider types.
It must not be enabled on a running seat from this source checkpoint.

Parent linkage alone does not override a native tool observer's deadline. A
named tool that observes a durable queued job must use the existing per-call
wait policy, for example timeouts.tools.calls.vision_analyze: 0 for queued vision.
Keep the ordinary generic tool/batch limits; do not disable all tool timeouts or
apply this to arbitrary terminal commands. Interrupted observation still retains
the accepted receipt. The native round-trip test deliberately holds a child
beyond a one-second generic guard and requires its original result, rather than
accepting a timeout message as a successful handback. Both providers pass with
the named policy. No live profile is configured by this test.

Existing native foreground ownership links main calls, active compaction and
originating tools. Concurrent tools get distinct call identities under one parent.
Detached/background work does not borrow that parent. Copied contexts must still
prove a live owning turn. An incomplete/interrupted/error attempt preserves its
parent for recovery. Only authoritative successful on_session_end queues the
durable job finish. The existing per-profile lifecycle outbox retries metadata
without resubmitting inference. All opted-in queued client types run its relay.

An old sc-queue-wait annotation on linked work refuses rather than silently
yielding the whole parent. Native explicit yield and qualified child delegation
still require their final integration; no model-specific prompt workaround is
part of this candidate.

## Physical ownership and recovery

Automatic cooldown and ordinary foreign demand do not release a parent job's
residency between calls. A drained, ended owned vision child can be replaced under
its exact active claim, after reap/absence and fresh registry admission checks.
No replacement lease is acquired. The child handle is stored before health
observation; persisted recovery intent survives a lost observation and resumes
that child rather than spawning a duplicate. Existing renewal covers long startup.
Expired/unknown/wrong-owner/insufficient claims refuse load. Elapsed time is never
proof of completion, death or release.

This recovery has synthetic tests only. Cold process restart without an owned
process handle, expired accounting debt and actual registry/MLX behavior remain
unqualified. Saved teardown/accounting debt must reconcile; it is not undone by
parent ownership.

## Validation and rollback

Run native integration through the matching Hermes checkout's scripts/run_tests.sh,
using its PM-built test interpreter and absolute service-test paths. The ordinary
requester/HTTP/physical tests use temporary journals, fake registry/backend and
localhost HTTP. Native tests run actual AIAgent/plugins/provider/SDK and a synthetic
backend, including sequential/concurrent children, interruption/recovery and stock
compaction. Green tests do not prove model quality or real physical admission.

Before a downgrade, verify no unfinished requester_jobs, no nonterminal linked
service executions, no service_handoffs recovery/teardown debt and no pending
new-kind client lifecycle events. Older code does not know these invariants.
Preserve source, journals and receipts; never delete them to pass a rollback guard.
No live fence, restart, activation or production SQL edit is part of this branch.

Source lineage: docs/staged-inference-source.json records the prior staged
snapshot as a separate baseline commit. The following commit contains the parent
job changes. Neither is a claim that every staged component is already deployed.
