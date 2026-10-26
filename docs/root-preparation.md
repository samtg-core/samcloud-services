# Root preparation scheduler — staged, not installed

`RootPreparingArbiter` is an explicit alternate requester scheduler. Production
bootstrap and Supervisor defaults still construct Arbiter. The explicit local
RootQueueStartup constructs this class via the trusted journal factory. It stores
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
   model fits, owns its process or is safe to load. RootModelBinding now wires
   the actual loader to these checks; physical host qualification remains separate.
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
qualified scheduler/identity. Explicit root startup is wired locally; the
operator CLI and deployed selection/recovery still need qualification.

## Limits before installation

Tests exercise actual temporary journal transactions, ordering, end/cancel races,
restart, stale completions, withdrawal, provider scope and logical rollback.
Separate pinned cross-repository cases connect preparation to the real staged
registry router/client and ledger with temporary tokens. Backend/model work in
those tests is synthetic, not Metal qualification.

Source/supervisor/manager accounting and explicit startup/lifespan integration
are now staged (root-bootstrap.md). Native main/tool/automatic-compaction flows
also have synthetic-provider HTTP evidence; separate staged-registry cases cover
physical control and runtime restart. Still required: deployed operator/native
qualification, legacy residency/accepted-work migration, actual physical absence/
process identity/headroom/health, same-envelope delegated child/client handoff,
real reconnect/yield and rollout/rollback. No production capability, profile, queue, model or claim is changed.
The first standalone image and service endpoint remain Hermes-owned stages;
this protocol is not an additional first-image prerequisite.


The [staged whole-root model binding](root-models.md) now connects actual
ModelManager accounting and explicit drained-end to these control boundaries.
It remains uninstalled. Explicit source/supervisor/startup integration is staged;
operator and deployed native/host/process/cache qualification remain open.


## Explicit supervisor construction and phase wakeups

Supervisor and GatewayQueue accept an optional trusted internal journal_factory
with Arbiter's (path, stint_policy=...) constructor contract. Defaults still use
Arbiter. No HTTP field or default bootstrap selects this alternate scheduler; only
trusted RootQueueStartup does.
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

The staged RootController now consumes these hooks (see root-controller.md).
The actual manager observation/source protocol and explicit startup/lifespan now
connect them. Direct registry push mapping, deployed operator selection and
legacy migration still need qualification. These hooks alone do not activate whole-root scheduling or qualify
physical/native/image generation. Standalone Hermes first-image work remains
independent.


## Single-call clients without an explicit parent

New root-mode intake can serve an ordinary execution request that omits job_key.
The journal derives one internal parent from that exact grant ID and call key.
Parent creation, its finish marker and execution intake commit in one transaction;
a failed execution INSERT leaves no orphan job or marker. No username, model,
HTTP connection or timing heuristic joins independent work together.

The internal parent is sealed for this one call. Its accepted child must actually
finish; unknown execution keeps ownership. Physical ENDING still requires strict
provider absence and exact root settlement before logical closure. Cancellation
before physical admission can end the saved intent without creating a claim.
A client doesn't need a new finish request or a holding connection.

Exact retries, including after restart or terminal cleanup, recover the same
execution/grant/job and cannot re-admit or regenerate. A changed payload is still
a conflict. A later call on the same work_session grant gets a distinct single-
call parent unless the client explicitly supplied a shared parent key. This does
not claim turn continuity for an older native client that never supplies parent
identity: all shipping Hermes profiles must still qualify the existing native
job-continuity binding before that stronger behavior is claimed.

Explicit parents retain their existing multi-call/compaction/reconnect lifecycle
and require their real finish/yield. Historical pre-upgrade unlinked rows are not
rewritten: their saved ACK remains readable under the original identity and
nonterminal ones must drain/reconcile before a root-mode rollout. The default
Arbiter intake behavior is unchanged.

Only root-mode capability responses add jobs.implicit_single_call_finish=true.
That is a reported intake contract, not a runtime activation switch or a grant
of physical capacity. Cross-service delegation remains false.
