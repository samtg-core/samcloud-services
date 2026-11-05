# Keep an incumbent's queue position through compaction

An accepted auxiliary request with `yield_grant_keys` currently yields its parent
as ordinary FIFO work. A competitor can run before the summary, and again before
the parent submits its next call. Compaction therefore adds two queue waits to
one ongoing conversation.

This candidate transfers the incumbent's scheduling position to its dependency
and returns it atomically when that dependency ends. The parent is granted/idle
before its next POST, so a short handback gap does not lose its position. This
uses the existing authenticated dependency keys; no rail, routing or prompt
change is involved.

## Exact source lineage

This branch is the additive job-aware port on `4ee1356`. It preserves
`_own_call_job`, `_start_queued_call`, `_advance_call_job`, cancellation job
advancement and qualified journal opening. The inherited dependency fix is
`92e08f0` on the pinned Slice lineage described below. This branch is not
the module selected for the installed Slice overlay.

Base commit: `8d977e7`. Its `ollama/inference_arbiter.py` has SHA256
`510efff7a795cecf73367b764f64def3696780663bd77ef19be071cc92dcac52`,
matching the inspected Slice runtime. This is a pinned-source patch, not an
instruction to replace the runtime with this entire historical checkout. Newer
job-aware branches need an additive port preserving their job advancement and
qualified `StintPolicy` opening behavior.

## Durable transitions

`dependency_continuations` is an additive journal table:

1. **pending**: enqueue binds the exact saved auxiliary execution to a same-owner,
   same-resource incumbent work-session parent and saves the incumbent position.
   Only an execution-lifetime child qualifies, with one open link per parent.
2. **active**: at a safe dispatch boundary, park the parent without reordering it;
   run the dependency at that position through normal admission.
3. **returned**: after terminal child completion and release, restore the parent
   as granted/idle in the same transaction. This survives a journal reopen.
4. **closed**: consume the credit on the first resumed parent dispatch, or waive
   it on explicit yield, release or a normal capacity/idle handoff.

The parent's original stint clock is retained. One resumed call is protected
from an already-elapsed fairness boundary; subsequent calls use the existing
fairness policy. Existing idle grace still bounds a return with no ready parent
call. Advisory wakeups exclude the temporarily waived fairness boundary, avoiding
repeated wakeups on an expired deadline.

## Boundaries retained

- A running or uncertain execution blocks admission. Disconnect does not release
  anything; no request is resampled, rekeyed or automatically recovered.
- A failed dependency returns the position for the parent's failure handling;
  this does not turn the failed result into success.
- Explicit parent yield or release waives the return. An ordinary long tool
  yield continues to give up scheduling position.
- UNKNOWN model permission blocks dispatch. Positive model unavailability can
  use the existing capacity handoff. Cancelling a deferred dependency never
  displaces work already admitted during that capacity wait.
- A pre-provider capacity deferral retains the exact child execution and its
  inherited order. No new inference identity is created.
- Replaying an old enqueue or terminal result cannot yield a later parent stint.
- Separately audited terminal recovery can be projected on the next dispatch,
  only when both the execution and its grant are positively terminal. The
  dispatcher does not establish backend death or perform the recovery.
- Foreign owners, other resources, non-incumbent parents and additional concurrent
  dependencies gain no new reserved position.

## Verification and rollout state

The four original offline regressions against the pinned runtime give three
assertion failures (entry, return gap, resumed call) and one passing ordinary
voluntary-yield control. The candidate includes 23 targeted cases covering those
paths plus restart, two journal connections, uncertainty, cancellation, release,
replay, capacity deferral, migration, terminal-recovery projection and fairness.

Run the targeted and existing queue suites with a dependency-equipped Python:

```sh
python -m unittest ollama.test_compaction_continuity ollama.test_inference_stints \
  ollama.test_inference_arbiter ollama.test_capacity_wait \
  ollama.test_inference_supervisor ollama.test_inference_telemetry \
  ollama.test_inference_events ollama.test_inference_api ollama.test_inference_recovery
```

This candidate is not deployed. Rollout must use a pinned overlay on the actual
release, preserve existing saved executions, and add the table under the service's
normal journal ownership. Existing historical dependency calls are not retroactively
relinked. No live queue, process, model, physical claim or profile is changed by
these tests. Rolling back to code that ignores the table would lose continuation
semantics; do not do so with an open continuation without an explicit plan.
