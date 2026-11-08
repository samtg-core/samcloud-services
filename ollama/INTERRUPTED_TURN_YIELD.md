# Exact lifecycle yield after native finalization

The queue provider's `post_llm_call` callback records turn-end metadata. On the
reviewed native runtime, `agent/turn_finalizer.py` invokes that output hook only
when `final_response` is present and the turn is not interrupted. An interrupted
child can therefore finish without recording the lifecycle event. Closing an
HTTP client is not a turn-end event. The service's configured idle grace remains
the fallback, rather than a signal that the accepted compute is stuck.

The context engine's existing `on_turn_complete` notification is outside that
output-only conditional. The queued compressor now records its exact bound
session/turn through that seam as well, for the explicit stint provider only.
It reads no transcript, performs no network call and changes no compression
algorithm, threshold, model, queue policy or deadline. The existing outbox,
relay and arbiter still own delivery and handoff. Repeating the existing output
hook and this callback produces the same event. Missing or unrelated bindings
cannot yield a parent, and stale events cannot yield a later call.

An event does not cancel or terminalize accepted inference. Running and uncertain
calls retain their state and ownership; the existing arbiter waits for a safe
boundary. This is not a crash observer or image-service recovery implementation.
The native context-engine callback itself is best-effort: abnormal early returns
that bypass finalization and process death still need their existing fallback.

## Evidence and limits

One observed child call completed, its exact grant remained granted/idle, and
the next main call began in the same transition that marked the child grant
`idle_grace`, 300.005 seconds later. The child's identity-store rows contained no
lifecycle outbox event. The reviewed native finalizer and child-run modules match
the installed hashes. The source-level interrupted-finalization gap is reproduced;
this does not establish every detail of that live child's private outcome.

The regression uses the native completion-notification function, real durable
identity store, existing relay with an explicit HTTP double, and real local
arbiter. It proves exact child scope, replay, missing/non-stint binding refusal,
immediate safe handoff after completion, and running/uncertain/stale-call fences.
Real socket connections are blocked. No inference or physical lease is used.

The new checks fail 7/12 against a copy of the installed client and pass 12/12
with only this method inserted. The broader source candidate passes 117 focused
native-client, identity, stint and compaction-owner tests. This branch is based on
the separately staged compaction-affinity lineage: do not substitute its entire
module for the older installed module. The runtime-qualified overlay is the new
method only; the installed compression implementation remains unchanged.

Publication is source review, not activation. Use the normal pinned-client release
and maintenance workflow after accounting for all foreground/background accepted
work and pending inputs. Do not monkeypatch a live holder, replay a call, close a
worker to make room, or change idle grace to qualify this patch. Observe a later
comparable interrupted turn before claiming live latency improvement.
