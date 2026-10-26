# Root preparation scheduler — staged, not installed

`RootPreparingArbiter` is an explicit alternate requester scheduler. Production
bootstrap and `Supervisor` still construct `Arbiter`, not this class. It stores
logical preparation/recovery identity in the existing inference journal; the
shared registry remains the only physical admission authority.

The first accepted call cannot wait for `owns_job` before requesting its root:
the original scheduler acquired job ownership only while marking a call running.
Preparation now selects the existing grant FIFO's next eligible linked job and
commits a logical slot with its exact job/call IDs while keeping the call queued.
It neither starts an execution nor makes a registry request. An earlier unlinked
call or an already-running legacy parent refuses preparation; do not skip it,
rewrite its IDs or silently adopt its physical ownership. Activate only after a
qualified old-runtime drain and explicit host/legacy-claim reconciliation.

The private journal freezes registry origin, actor and service. One nonterminal
preparation is allowed per resource. Concurrent preparation/reconnect reads the
same ID; the parent owns its scheduling place across model/tool gaps. The
preparation ID identifies one physical intent in `RootIntentStore`, alongside
its logical job ID. Explicit yield/cleanup can later put the same logical job
back into FIFO under a new physical preparation. An old completion is confined
to its old preparation and cannot close the resumed parent's new root.

## Trusted controller sequence

These are internal Python operations, not public API fields or a deployed loop.
No caller may manufacture a registry receipt or treat this as host permission.

1. `prepare_root(resource)` commits selection. Use its saved `id` with the root
   client, preserving the original job/grant/execution IDs. If the preparation
   is already ending, flush that same root end instead of starting a new root.
2. Explicitly `root_client.begin(preparation_id, resource)`. A queued registry
   root is not admission. Observe the same receipt until granted; never move
   the execution to running just to discover which root to request.
3. `confirm_root(preparation_id, receipt, resource_snapshot)` verifies the exact
   current full exclusive active envelope, finite unexpired health, resource
   hold/shared policy, provider identity, declared bytes and absence of another
   active/unsettled charge. Only fresh observations from the bound registry
   client are accepted. A failed check synchronously withdraws dispatch.
4. `dispatch_next` still requires the normal per-model/headroom permission. It
   rechecks the saved preparation inside the ordinary dispatch transaction and
   uses the existing accepted call state machine. Root bytes do not prove the
   model fits, owns its process or is safe to load. No model loader is wired yet.
5. Logical finish/yield drains all accepted children. Once drained, the
   preparation becomes `ending`, blocks dispatch and retains the logical slot.
   A running/uncertain child prevents this transition; no clock ends it.
6. The qualified physical controller stops/unloads/observes backend absence and
   ends/settles the exact registry root using the saved client identities. A
   cancelled never-admitted root may return a terminal no-claim receipt. Neither
   transport close nor consumer disappearance supplies cleanup evidence.
7. `settle_preparation` consumes the exact terminal root receipt. The terminal physical
   receipt and normal logical finish/yield commit together in this local journal. SQL failure keeps debt for the same terminal-receipt retry. Only
   after that boundary can another preparation be selected.

Withdrawal persists before a child transfer or failed observation. Restart has
no cached permission and requires a fresh exact root confirmation. An ordinary
`Arbiter` refuses a journal with unfinished preparations or an unfinished logical
job that already used this root protocol. Operator recovery must use the matching
qualified scheduler/identity; current bootstrap/recovery CLI is not wired for it.

## Limits before installation

Tests exercise actual temporary journal transactions, ordering, end/cancel races,
restart, stale completions, withdrawal, provider scope and logical rollback.
Separate pinned cross-repository cases connect preparation to the real staged
registry router/client and ledger with temporary tokens. Backend/model work in
those tests is synthetic, not Metal qualification.

Still required: source/supervisor/bootstrap and operator CLI integration; legacy
residency migration; actual loader/unload/renew accounting through one root;
physical absence/process identity/headroom; same-envelope delegated child/client
handoff; reconnect and explicit yield across the full native path; rollout and
rollback. No production capability, profile, queue, model or claim is changed.
The first standalone image and service endpoint remain Hermes-owned stages;
this protocol is not an additional first-image prerequisite.


The [staged whole-root model binding](root-models.md) now connects actual
ModelManager accounting and explicit drained-end to these control boundaries.
It remains explicitly attached/uninstalled; source/supervisor/bootstrap and
operator integration plus real host/process/cache qualification remain open.


## Explicit supervisor construction and phase wakeups

Supervisor and GatewayQueue accept an optional trusted internal journal_factory
with Arbiter's (path, stint_policy=...) constructor contract. Defaults still use
Arbiter. No HTTP field or current bootstrap selects this alternate scheduler.
Its existing provider-identity, legacy-drain and journal-reopen guards apply.

The synchronous on_resource_demand(resource) hook runs before the cached capacity
check in pump, including an unavailable capacity notification. It is a hint for
a controller to withdraw readiness and register coalesced service-owned work
through Supervisor.prepare; it is never an admission receipt or a polling loop.
An exception leaves the accepted call queued and does not dispatch backend work.
Callbacks must avoid recursively pumping their own demand hint when publishing
new readiness. This hook is deliberately separate from the old post-logical-end
on_job_boundary callback, whose default behavior is unchanged.

Owner end while the job is still owned also pumps the demand hook. A running
child finishes naturally; its completion pumps again, exposing physical ENDING
while the logical parent still owns its slot. An end between calls reaches that
same hook immediately. Cleanup registered via prepare stays service-owned after
an HTTP observer leaves and must drain before Supervisor.close. Neither a demand
hint nor a task completing stands in for positive backend absence and exact
physical settlement.

The qualified coalesced root controller/capacity source, registry-change and host
observation feeds, bootstrap/operator construction and migration are still to be
connected. These hooks alone do not activate whole-root scheduling or qualify
physical/native/image generation. Standalone Hermes first-image work remains
independent.
