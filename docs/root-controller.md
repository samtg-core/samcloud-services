# Root controller — staged, not activated

RootController explicitly connects one RootPreparingArbiter, Supervisor and
RootModelBinding for one physical resource. It is a capacity source and a
coalesced preparation driver, not an allocator. Only the shared registry admits
physical roots. No current service bootstrap, profile or capability selects it.
Explicit construction now attaches it to the manager's existing source protocol.
Standalone first native image/endpoint work stays independent.

## Construction and event contract

Construct the Supervisor (or GatewayQueue) with an explicit trusted root journal
factory. Attach RootModelBinding to its exact journal and a qualified cold
ModelManager, then construct RootController(supervisor, binding,
work_permitted=True/False). Unknown work permission is rejected. A mismatched or
already active supervisor/source is rejected rather than silently replaced.
Default bootstrap continues to use the existing journal/source.

The controller installs only the loop-local dispatch guards and synchronous
resource-demand hook. An enqueue, owner end, release or completed child already
pumps that hook. The existing manager stats_loop calls invalidate before collecting and
refresh after a physical sample; the controller implements that source protocol.
RootModelBinding notifies the same controller after an actual root-health pass,
including a failed read. Direct registry-change hints can also call
request(resource) on the service loop.
Unrelated resources are ignored. Event payloads never carry admission authority:
the controller reads the exact root, actual resource snapshot and host capacity.

Multiple hints while preparation is in flight coalesce into one service-owned
task. A newer hint withdraws readiness synchronously; an old observation cannot
publish dispatch after an epoch change. Already-started preload retains its exact
claim and may record known completion, but must pass fresh admission before a
child is dispatched. source_call drains an in-flight worker operation even if
its observer is cancelled. Supervisor.close refuses unfinished preparation.

The controller suppresses the demand hint from its own readiness publication,
so publishing cannot recursively start a prepare/pump loop. There is no idle
poll, elapsed-job deadline or self-scheduled retry. Queued or unknown roots wait
for the next authoritative event. drain() waits for the actual current task;
it does not submit another operation.

## Preparation and model selection

Bare single-call intake gets a durable sealed one-call parent; explicit parents
keep spanning calls (see root-preparation.md). No new finish verb is needed for
a standalone call, and uncertainty still retains it.

The scheduler first persists/selects the eligible logical parent without
marking its call running. Root admission uses the durable physical preparation
ID; lost replies recover the same root/claim. A queued root never permits model
loading or execution. Work permission false keeps accepted work but starts no
physical admission/load. Explicit permission restoration is a new event.

Only the owned parent's next accepted child is prepared, in the scheduler's
existing created/id order. Any running/uncertain execution on the resource
prevents a model switch. Ollama/VLM use the actual manager's strict ownership,
startup intent, same-envelope sizing/switch and known-completion paths. Unsupported
profiles or unknown startup debt pause; a later hint cannot silently launch a
second provider. Unrelated requesters cannot bypass the owning parent's children.

Before dispatch, fresh aggregate device use, positive usable host RAM, current
root/member and the exact managed model intent must agree. Cached dispatch guards
perform no network, SQL or loading inside the journal's transaction. Withdrawal,
expiry, changed model/lease/handle or paused work blocks another dispatch. These
are present-capacity observations, not promises about future inference peaks.

## Finish and uncertainty

Accepted children drain after explicit owner finish. Physical ENDING is handled
while the logical parent still owns its slot. The binding strictly unloads known
owned models, obtains fresh qualified process/cache absence and settles the exact
root before the scheduler closes/yields the parent. An actual completed physical
boundary lets the driver examine the next saved requester; no timer is involved.

False absence, failed control reads, startup uncertainty or release failure keep
the exact debt, accepted call and logical ownership. Errors expose only a bounded
reason class in last; no raw prompt, bearer or exception prose is logged. A new
external event can reconcile the saved intent. A lost committed settlement reply
reads the terminal root and finishes logical handback without a second release.
No cancel, success or absence is inferred from elapsed time or a lost observer.

## Manager observation and lifecycle

The existing15-second physical sampler and root-health loop are reused; no new
sampling/retry timer or second watcher is created. refresh schedules preparation
and returns immediately, so host reporting continues during a long preload.
Unknown/invalid samples withdraw readiness; a positive later sample can wake the
same saved work. Invalidation happens before collection, so a failed or cancelled
collector cannot leave stale permission. Health notifications do not mask the
maintenance result and carry no admission authority.

pause withdraws new work while retaining accepted jobs and exact claims. resume
requires a new positive host observation. Running/preparing work is not cancelled.
retains_model reports logical parent ownership to existing manager housekeeping.
Cached guards also refuse changed manager source/accounting identity. snapshot
exposes bounded source state, not prompts/credentials. Pause the source and drain tracked work before manager shutdown. While paused,
a completed physical boundary does not select another queued parent. The manager
requires positive physical drain before stopping observations; late hints then
cannot create preparation, and resume requires fresh startup. Queued work and
terminal-physical/pending-logical recovery remain durable (root-models.md).
The service lifespan awaits `manager.shutdown_async()` before runtime/journal
close; operator/bootstrap orchestration must use the same owned drain boundary. Source pause is not parent finish and never substitutes for physical
settlement.

## Remaining shipping work

Qualify any direct registry push-change mapping; the current host-sample path
already causes a fresh root/resource read, using the existing observation loop.
Keep finite registry health maintenance under its accepted root contract while
waiting/loading; those heartbeats are distinct from job deadlines. Integrate explicit bootstrap/operator
construction and graceful drain, and qualify migration of legacy accepted calls,
initial reservations and residency before enabling any live root policy.

Qualify actual host memory/process/cache sensors, cold restart/recovery, delegated
children, operator fencing, real native full-path execution and rollout/rollback.
Current tests use deterministic providers/sensors. Registry authentication, SQL,
lease promotion and control HTTP paths in cross-repo tests are real staged code,
but they are not GPU/Metal or deployed endpoint acceptance. Cross-service
capability advertisement remains false.

Explicit restart suspension (root-models.md) drains known owned backends with a
fresh cold proof while retaining the parent and exact root intent. It is not
finish/yield and sends no registry release. A paused source refuses new model
admission even after a late readiness hint. A reconstructed controller must
observe/recover the same saved root before dispatching the original accepted
child; no new claim is substituted for a lost acknowledgement.
