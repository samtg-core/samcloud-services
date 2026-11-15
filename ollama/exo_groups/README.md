# EXO provider integration

This package belongs to **model-service**. Model-service remains the caller's
inference endpoint and the EXO coordinator. Samcloud's registry owns physical
resource admission, durable lease groups, fencing and settlement. EXO workers
never run inside the registry.

The target request path is:

```text
caller -> existing Slice model-service inference journal
       -> model-service EXO provider/coordinator
       -> durable group -> each host's ordinary physical resource queue
       -> private, explicitly placed EXO instance
```

The inference queue schedules requests; the registry's resource queues admit
physical ownership. Group membership does not create a second physical lease
for Slice, and the existing legacy `exo-pool` route is not an admission bypass.

## Implemented authority component

`plan.py` binds a pinned model revision and rank manifest to exact, ordinary,
normal-priority HOLD leases on Slice, Wafer and Obelisk. Acquisition order and
pipeline order remain separate. Every lease lifetime is explicit in the plan;
there is no default assembly or inference deadline.

`member.py` validates the registry's scoped member-authority projection. The
server checks all required members at its observation instant. Permission ends
at the earliest member expiry, not just the queried member's expiry. HOLD expiry
withdraws execution authority without proving cleanup or freeing ownership.

`authority.py` reads the three exact projections with the coordinator's own
credential. It verifies the credential's owner before using it, verifies again
after credential rotation, rejects redirects and ambiguous JSON, bounds response
bytes and omits credentials/body details from errors. It performs GETs only.

An admitted group can permit rank startup; serving inference additionally needs
the group to be `serving`. Both permissions are prerequisites: the provider must
also verify its exact private EXO instance, shard placement and ready runners.
An authority observation is process-local and cannot be persisted as permission.
Events invalidate it before reconciliation. A slow or failed read cannot retain
the old permission, and an event during a read prevents the read from publishing
an obsolete observation. Elapsed authority uses a continuous clock that includes
host suspend; it does not depend on synchronized host wall clocks.

The plan/member/clock code is migrated from the qualified consumer preparation
in task967 (docs575/630/632). There are no imports from that workspace staging
directory. The HTTP reader and its real-registry tests live here as provider code.

`intents.py` links group-create provenance to queued executions in that same
`Arbiter` database. The exact plan and server-idempotent create body commit before
a coordinator can submit them. Multiple queued calls can share one pinned intent;
changed placement/key reuse conflicts. Reopening the journal recovers the same
request, and a known group ID is reconciled rather than created again. A fence
never silently advances the saved generation. Cancelled, running or uncertain
work alone cannot initiate new acquisition. These rows have no dispatch order
or scheduler of their own, and they perform no plane writes.

## Integration still required

This component is **not enabled by startup and is not a complete EXO provider**.
It does not acquire leases, launch ranks, load a model or run inference. The
remaining provider integration must:

1. Connect the saved intents/demand links to a service-owned acquisition actor
   before dispatch, without requiring the HTTP observer to stay connected.
   Persist the exact group, generation, manifest and execution/instance binding
   before an inference POST. The create-intent store is implemented; the actor
   and execution/instance binding are not.
2. Publish group-covered capacity alongside local-model capacity. Slice's group
   lease must not be mistaken for spare local memory or trigger a second lease.
3. Bind the pinned EXO runtime, its private namespace/API key and exact explicit
   instance. EXO's chat API selects by model, not a caller-supplied instance ID;
   its state must contain only the expected instance and assigned runners.
4. Execute through the existing supervisor. Transport loss, a missing terminal
   frame or unconfirmed backend end retains uncertainty and physical HOLD;
   neither reconnect nor elapsed time retries inference or releases the group.
5. Keep call completion distinct from the residency session. Release only after
   observed rank cleanup, exact evidence publication and acknowledged settlement
   of every member. There is no default idle-release timer.

Production integration remains gated by the small-model native and multi-host
qualification, network confinement and the existing campaign boundary. CPU and
HTTP controls below are not evidence of distributed execution or 200 GB residency.

## Validation

From the model-service repository:

```sh
python -B -m unittest -v ollama.test_exo_group_plan \
  ollama.test_exo_group_member ollama.test_exo_group_clock \
  ollama.test_exo_group_authority ollama.test_exo_group_intents
```

The real registry qualification uses Samcloud PR19's source, currently
`9756b01`, and its test dependencies. Set `REGISTRY_SOURCE` to that checkout,
then run from the model-service repository:

```sh
PYTHONPATH="$REGISTRY_SOURCE/registry:$PWD" python -B -m pytest -p conftest \
  qualification/test_exo_group_registry.py -q
```

The registry conftest loads first and isolates its database, keys, token, file
store and authorized-keys paths. These controls use its real authenticated ASGI
API and SQLite state. No production credential, network, process or GPU is used.
They cover admitted versus serving, coordinator fencing, a fence between member
reads, foreign credential rotation, and an expired HOLD member remaining charged
and unsettled. Only the fixture creates test leases; the provider issues GETs.
