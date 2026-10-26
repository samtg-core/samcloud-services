# Whole-root model binding — staged, not installed

`RootModelBinding` explicitly attaches the actual `ModelManager` to one existing
registry root. No supervisor, launcher, source or bootstrap selects it. The
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
new load. `renew_models` sends one exact root heartbeat, even with no resident
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

## Remaining integration

Source/supervisor/bootstrap/operator wiring, legacy accepted-work/claim migration,
same-envelope delegated child client, real native recovery/yield and rollout/
rollback are still required. Existing capabilities continue to advertise no
cross-service delegation. First native standalone image/endpoint qualification
remains Hermes-owned and independent of these disabled platform components.
