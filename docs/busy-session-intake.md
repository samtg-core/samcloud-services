# Observe existing work before a new call on the same session

A native controller can disappear while the service continues an accepted call.
If its replacement offers a **different** logical call on that session before
the first ends, the journal must refuse overlapping execution. Previously that
refusal was an untyped 409, which the SDK surfaced as a terminal provider error.

An open work-session grant with exactly one queued, running or uncertain call
now returns an owner-scoped 409:

```json
{"detail":{"code":"grant_busy","grant_id":"<grant>","execution_id":"<predecessor>"}}
```

This is a refusal, not acceptance or a new queue position. The journal still
stores at most one pending/active call per grant. Same-key/same-body replay is
checked first and returns its original execution, including after release.
Same-key/different-body conflicts, released/cancelled grants, release requests,
one-execution grants and inconsistent journal pointers retain ordinary errors.
Other owners get the existing 404 without the predecessor identity.

`DurableConsumer.submit` freezes the new JSON body before its first await. On
this exact typed refusal, it observes that predecessor through authenticated
status reads. Queued, running and uncertain states wait without a queue deadline.
Only a terminal predecessor permits offering the same unaccepted new input
again, with the original grant/call keys. A competing call can win that race;
the journal checks again and the observer waits for the newly named predecessor.
Explicit release during the wait still refuses the new input. No observer
release, yield, cancellation, generation retry or priority change is introduced.

The predecessor's result is **not** returned as the new answer. This protocol
does not repair native conversation history or ingest output lost by a previous
controller. A host that intends to resume the original logical call must use its
persisted original keys or exact saved receipt. A genuinely new request remains
a new request. The intake callback receives only its eventual accepted receipt.

Malformed/foreign/missing receipts refuse. A server naming the same predecessor
as busy after reporting it terminal also refuses, rather than spinning POSTs.
Cancelling observation leaves service-owned work unchanged. A lost intake reply
uses the existing same-key reconciliation; it cannot duplicate accepted work.

## Compatibility and release

The gateway changes are confined to `inference_arbiter.py` and
`inference_api.py`; the client change is `inference_consumer.py`. They require
normal pinned gateway and queue-client releases. An older server's untyped 409
is deliberately not retried. An older client still fails on the new typed 409.
Do not replace an installed runtime with this historical source tree wholesale;
carry the reviewed hunks onto its exact release, preserving unrelated changes.
No running controller, accepted call, lease or image-service deployment is changed
by this source packet. It is not an image-release prerequisite.

## Offline controls

`python -m unittest ollama.test_busy_intake` uses the actual journal, FastAPI
router, consumer and SDK intake path over ASGI, without a provider or sockets.
It covers predecessor states, owner isolation, same-key replay and mismatch,
release, observer cancellation/restart, lost accepted reply, caller mutation,
competing intake, unchanged explicit handoff and invalid/stale receipts.
The initial regression run against the unchanged base fails on the observed 409.
All request bodies in these fixtures are synthetic.
