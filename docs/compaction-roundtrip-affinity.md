# Keep an incumbent's queue position through compaction

Before this change, an accepted auxiliary request with `yield_grant_keys` yielded
its parent as ordinary FIFO work. A competitor can run before the summary, and again before
the parent submits its next call. Compaction therefore adds two queue waits to
one ongoing conversation.

This candidate transfers the incumbent's scheduling position to its dependency
and returns it atomically when that dependency ends. The parent is granted/idle
before its next POST, so a short handback gap does not lose its position. This
uses the existing authenticated dependency keys; no rail, routing or prompt
change is involved.

## Installed state (read-only recheck 6 November 2026, 02:10 UTC)

Slice serves the pinned overlay `7da881b`, arbiter SHA256
`631cfbaf8147d534e7075f7ac124a82e2ffcada8688bbc85c125d89304ff2373`, from
`~/.local/share/codex-reviews/compaction-return-7da881b/release`.
Gateway PID33439, born6November12:12:40local, has that cwd and answers health200.
The launcher hash is `7aa1c59103b09a9b200248967ca2188deace28f3b672dda834f55e2b77137c81`;
its fence marker is absent. Original backend59832 was not restarted. All114Python
hashes matched the candidate at cutover; the latest check rechecked the arbiter,
launcher, process generation, cwd, fence absence and health. The separate job-aware
branch is not installed on Slice; its source port is `d5b60c4`.

The exact accepted foreground call survived the gateway switch with its ID,
key, creation time and request hash unchanged. The initial natural compaction
proved entry and atomic return, then exposed the capacity-return gap corrected
by `7da881b`. A future natural round trip must still qualify the complete revised
behavior; installation and offline tests are not that proof. Historical links
were not retrospectively repaired. Receipts are on Samcloud doc551 and under the
host overlay's `cutover/` directory. Completed operator actions must not be replayed.

Guarded qualification:132 checks against pinned source and a copy of the actual
release,71 targeted checks on Slice's Python3.14.7,239 on the separate job-aware
lineage; zero IP attempts in the guarded runs. These checks create no live claim
or model request.

## Exact source lineage

Base commit: `8d977e7`. Its `ollama/inference_arbiter.py` has SHA256
`510efff7a795cecf73367b764f64def3696780663bd77ef19be071cc92dcac52`,
matching the pre-overlay Slice runtime. This is a pinned-source patch, not an
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
   it on explicit yield, release or an expired unused idle return.

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
voluntary-yield control. The candidate includes 30 targeted cases covering those
paths plus restart, two journal connections, uncertainty, cancellation, release,
replay, capacity deferral, migration, terminal-recovery projection and fairness.

Run the targeted and existing queue suites with a dependency-equipped Python:

```sh
python -m unittest ollama.test_compaction_continuity ollama.test_inference_stints \
  ollama.test_inference_arbiter ollama.test_capacity_wait \
  ollama.test_inference_supervisor ollama.test_inference_telemetry \
  ollama.test_inference_events ollama.test_inference_api ollama.test_inference_recovery
```

The pinned source at `7da881b` is deployed as recorded above. A later rollout
must use a pinned overlay on the actual
release, preserve existing saved executions, and add the table under the service's
normal journal ownership. Existing historical dependency calls are not retroactively
relinked. No live queue, process, model, physical claim or profile is changed by
these tests. Rolling back to code that ignores the table would lose continuation
semantics; do not do so with an open continuation without an explicit plan.

## Physical capacity loss during return

The first live natural return restored the parent correctly, then a registry
handoff made its model unavailable before the resumed call ran. The ordinary
capacity handoff requeued it at a new order and closed its return credit.

The additive correction retains the saved queue order, original stint clock and
return credit when physical capacity is unavailable. It leaves the parent queued,
so available work on another model may run. Nothing overrides model permission,
preempts admitted work or retains a physical lease. The first resumed call spends
the credit; normal fairness follows using the original stint clock.

When capacity disappears before the next parent POST, the existing idle grace
still bounds that empty return gap. A call accepted within grace can wait for
physical admission without a deadline. Explicit yield while queued drops the
saved order as well as the credit. Reopen preserves the same behavior.

Seven additional regressions cover ready-call and pre-POST capacity loss,
available other-model work without preemption, expired unused grace, original
fairness accounting, reopen and explicit yield. On the initial overlay the three
order/fairness regressions fail; on the extension all thirty cases pass.
