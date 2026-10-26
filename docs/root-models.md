# Whole-root model binding — staged, not installed

`RootModelBinding` explicitly attaches the actual `ModelManager` to one existing
registry root. The explicit staged RootQueueStartup can bind it to a qualified
supervisor; no configured shipping launcher selects it. The
registry root, service scheduler preparation and provider model intents are
separate records: only the registry allocates physical capacity.

Attachment requires a cold durable manager, matching registry/provider/resource
scope, no legacy reservation/residency controller or failed-release queue, a
positive fresh backend-absence callback and background maintenance not yet
started. It cannot be hot-installed over the legacy maintenance clock. The
manager's registry client is wrapped to refuse ordinary per-model reserve,
release, renew and rollover; reads/identity operations still use its own client.
The root client uses the same explicitly bound provider identity and origin.

## Model lifetime inside the parent

Only declared Ollama/VLM profiles are supported. Fresh exact root confirmation
comes from the preparation scheduler and the root client's persisted receipt.
Provider threads read scheduler metadata through their own read-only SQLite
connections; model intent transactions use operation-local private connections.
A saved receipt is not fresh permission after restart. Withdrawal, a newer
expired/ended receipt or changed current member blocks new backend work.

The manager checks work permission before preparing a model switch. Before load,
the binding checks fresh available host RAM, aggregate device use and its
declared full-root envelope. Cached per-model sizes alone are insufficient:
KV/context growth can outgrow a model's recorded load size. Unknown/invalid
physical readings defer before any load or eviction; missing usage is never
assumed to be zero. If a
new model needs room, strict owned-model unload must confirm backend absence;
the model's intent becomes gone, but the parent root is never released. The
46,000 MB text plus5,700 MB vision example therefore unloads the text model
inside the49,152 MB root instead of requesting a second claim behind itself.

An exact model-start intent commits before provider load. Unknown/crashed startup
remains loading/unknown debt and cannot silently launch again. Vision keeps its
owned process handle in manager inventory before health waiting and metadata
publication, including when the metadata link fails. Reported post-load sizing
is checked against the full-root budget; refusal preserves the model/lease link.
A post-load ownership loss is reconciliation, never a safe-to-repeat
pre-provider deferral. Routine heartbeat/status refresh withdraws new admission,
but an already-started load may finish under its same retained root/member.
Expired unsettled health under the frozen hold policy likewise retains ownership;
it never permits another start. Completion records residency without publishing
fresh permission. A terminal/released/settled claim or changed current holder
still preserves unknown model debt. Do not reuse new-admission freshness as the
completion condition of a provider operation that already owns its claim.
A positive post-load aggregate overrun retains unknown model/claim debt. A
post-load sensor gap preserves known provider completion while withdrawing new
admission; it does not retry that already-started load. Fresh physical data is
still required for subsequent dispatch. VLM's configured/profile figure is not
claimed as a measured peak.

Strict unload records the exact model intent gone before dropping its manager
pointer, without ordinary lease DELETE. Idle/foreign-demand cooldown clocks do
not return a parent root. Actual root finish/yield and model switches are explicit
controller boundaries. Unsupported llama/Whisper/exo paths refuse in root mode.
Cold restart with ambiguous models still needs qualified recovery; no PID/name
adoption or new load is inferred from empty manager inventory.

## Health and observation

`prepare_root` resumes the saved physical preparation and arms health from its
actual registry contract, including when physical/model readiness is unknown.
`refresh` withdraws permission before observation. `publish` validates the root
and supplies readiness; neither a read failure nor old cached success permits
new load. An explicitly attached controller receives a completed-health hint
(successful or failed) without replacing the maintenance result.
`renew_models` sends one exact root heartbeat, even with no resident
model, instead of resizing/replacing each model lease.

Root mode's maintenance waits for accepted health and schedules from one quarter
of that root's health interval, not the legacy per-model900-second sleep.
Repeated reads of the same receipt cannot postpone a heartbeat. Failed reads
leave a nonzero health-maintenance delay, not a busy retry loop. Unknown/expired
or delegated ownership cannot be renewed as a different holder or silently
readmitted. Before any root exists the health loop waits for a receipt event.
These clocks maintain physical health only; they never fail/cancel a queued job,
release capacity, truncate a turn or impose an elapsed-work deadline.

## Explicit end and recovery

`end_drained_root(preparation_id, observation=...)` requires the exact local
preparation to be ending with all accepted calls terminal. It persists/sends the
same physical end, strictly unloads known owned models, and invokes the supplied
fresh process/cache-absence observer. Empty manager inventory alone is not proof.
With accepted work drained plus positive provider absence, orphaned startup
intents can be recorded gone for cleanup; this never retries generation.

Only then does it settle the exact claim through the root client and acknowledge
physical completion to the logical scheduler. Unknown absence or failed release
keeps the root/parent end debt. A lost post-settlement reply recovers the same
terminal root without another release. A never-admitted terminal root with no
local model debt needs no global stop; a terminal root with unresolved local
models refuses to stop a later physical holder. A delegated current claim must
be cleaned up by that actual holder, not attested by its parent.

The observer callback is a qualified sensor contract, not an implementation of
host absence. Tests use deterministic providers/callbacks. Before activation,
qualify actual model process identity, GPU/cache release and pressure/headroom,
cold recovery, service-worker ownership and operator fencing on the shipping host.

## Shutdown boundary

ModelManager.shutdown invokes a root audit BEFORE pool release, model unload or
cancelling health tasks. An idle backend or empty manager inventory is not proof
that an admitted, queued, unknown or lost-ACK physical root ended. Unknown model
intent, unresolved execution, changed bound source, unmatched logical owner or non-terminal client
receipt refuses ordinary shutdown without changing claims or maintenance.

If a controller is attached, its source must first be paused and all supervisor
backend/input/control preparations drained. Queued accepted work may survive.
A preparation with no SDK intent row is known never submitted because the SDK
persists its intent before any HTTP attempt; shutdown suspends that durable
logical identity without closing it. Once an SDK intent exists, even a missing
reply requires an exact terminal receipt rather than an empty-inventory guess,
unless the explicit qualified suspension below has completed.

A positive physical terminal receipt may precede a failed local logical close.
With no model/execution/task debt, physical shutdown can proceed while preserving
that ending job for exact metadata recovery. It does not fabricate job success,
retry inference or discard a queued call.

Only after this audit does shutdown install a one-way observation fence, detach
the health callback and stop root-health waiting. Late host/health hints cannot
start another preparation or query a closed journal. An acknowledged stop can be
retried; a stopped source cannot resume/load models and needs fresh startup.
Source ownership stays bound even if the mutable manager source field is
cleared, and stopped/preparation entry points refuse before any new root submit.
The shutdown path creates/ends/releases no root itself. Native/service bootstrap,
operator orchestration and full native/host restart still need qualification.
The service lifespan now awaits `shutdown_async()` before runtime/journal close.
The root audit still precedes any task cancellation. That async boundary drains
all seven manager maintenance tasks (including the exo pool watcher), including `source_call` threads finishing
after cancellation. Concurrent/reconnected waiters join the same retained drain;
cancelling an observer cannot cancel it. A refused audit does not cancel
maintenance and may be retried after explicit reconciliation. No compute/job
deadline or inferred logical finish is added. Direct synchronous `shutdown()`
callers must not close shared clients/journals until maintenance is drained.

## Explicit restart suspension

`suspend_for_restart(observation=...)` is separate from parent finish and the
ordinary shutdown audit. A bound paused source, drained tracked work and no
running/uncertain execution are required. It strictly unloads only known owned
backends, refuses unresolved model-start debt, and requires a fresh positive
backend/cache absence observation. It then persists a current-process witness
bound to provider identity, model-intent history and unfinished preparations.
New backend work invalidates that witness. A source paused for drain cannot
admit a new load even if a late observation republishes readiness.

This permits control shutdown while keeping active, queued or lost-ack root
intents charged and durable. No registry end/release, parent completion, queued
call cancellation or replacement claim is sent. A fresh controller resumes the
same saved preparation/call using the existing root client recovery contract.
The old process witness cannot authorize another stop after restart. Unknown
execution or false/unknown absence keeps control alive. If health expires during
the outage, held capacity still needs the existing explicit qualified same-ID
recovery; elapsed time never proves absence or releases it.

Tests rebuild the manager, scheduler and controller over the same journals and
exercise actual staged registry admission/promotion/HTTP, with fake physical
providers. They do not qualify live bootstrap, credential recovery, host sensors,
a deployed maintenance-worker shutdown or a real process restart. The explicit
staged service startup/drain path is covered separately in root-bootstrap.md.

## Remaining integration

The staged RootController connects supervisor/model readiness and cleanup.
Explicit startup/lifespan wiring is tested (root-bootstrap.md). Actual deployed
launcher/native/operator qualification, legacy accepted-work/claim migration,
same-envelope delegated child client, real native recovery/yield and rollout/
rollback are still required. Existing capabilities continue to advertise no
cross-service delegation. First native standalone image/endpoint qualification
remains Hermes-owned and independent of these disabled platform components.
