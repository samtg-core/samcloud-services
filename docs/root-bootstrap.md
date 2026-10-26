# Root startup — explicit local candidate, not deployed

`RootQueueStartup` connects the alternate root scheduler, actual GatewayQueue,
ModelManager binding and RootController to the existing server lifespan. No
launcher, environment default, registry router/policy or agent profile selects it.
The default QueueStartup and an absent startup binding retain their behavior.
A configured root startup failure propagates; it never adopts legacy models or
falls through to unjournaled direct dispatch.

## Trusted construction

The launcher supplies QueueSettings, its own RegistryRootClient/RootIntentStore,
an exact model-name to Backend map, an authoritative fresh cold callback and an
explicit boolean work permission. Only one local resource and OLLAMA/VLM are
qualified. Legacy per-model reservation overrides and finite shared queue waits
are rejected; exo is not supported by this root startup.

Before mounting, startup verifies the manager's bearer identity matches the root
client's durable principal, freshly checks actor-bound registry capabilities,
and reads the exact shared/hold resource. Fresh backend/cache absence is required.
A previous positive capability cannot survive a failed refresh. Cold absence is
not a capacity grant: startup/intake allocate nothing. The controller begins
with host observation UNKNOWN and only existing manager observations can make
it attempt root admission. The registry remains the only allocator.

A live/previously started manager, legacy accounting owner, active maintenance,
existing dispatcher or mismatched provider scope refuses attachment. A legacy
journal without a root identity and with unfinished accepted work requires an
explicit drain/migration; no identity is rewritten. The registry also refuses a
fresh root over existing ordinary per-model claims. A startup scope that removes a model with accepted unfinished work, changes its
resource, or leaves another unfinished physical root in the client journal also
refuses before mounting. Root-mode journal restart keeps its saved parent/
preparation/accepted call IDs. This is not a general legacy
migration or cold unknown-model-intent recovery mechanism.

Construction is synchronous after its observation awaits. A mount failure
restores the exact original routes and manager accounting/source fields and
releases the scheduler lock. It does not close the caller's root client on a
failed installation. On success the runtime owns that client until safe close.
Secrets stay with the service; no requester can choose the physical actor,
backend map, resource, startup mode or absence sensor through HTTP.

## Runtime shutdown and resume

The actual server lifespan calls runtime.quiesce, manager.shutdown_async, then
runtime.close in that order. Root quiescence is service-owned and shielded from
observer cancellation. It pauses dispatch and drains tracked workers/input/
preparation work without a clock. Unknown execution, held backend claim or false/
unknown cold proof returns not-drained and preserves control, journals and claims.
An explicit retry after reconciliation can observe again; there is no retry loop.

After known work drains, the runtime calls the qualified cold suspension API.
It strictly unloads owned backends and saves the suspension witness while keeping
the logical parent, pending calls and exact active/queued/lost-ack root intent.
It sends no end/release. An actual requester jobs/end call remains the explicit
multi-call completion boundary. Bare single calls retain their sealed parent
completion contract. Disconnect and service suspension are neither boundary.

Only the successful manager maintenance drain permits runtime.close. That closes
the supervisor journal and then the root HTTP client. A later controller reads
the same saved intent and original accepted child; it cannot reuse the previous
process's suspension witness as a new absence proof. Explicit resume before stop
requires fresh host evidence. If root health expired during the outage, held
capacity needs the existing qualified same-ID absence recovery before dispatch;
time never frees it or authorizes replacement admission.

`/status` reports a bounded `requester_root` snapshot: source state, saved
preparation/root/lease IDs, quiescing and closed flags. Saved IDs describe durable
intent/history and are not current physical admission proof. It exposes no input
prompts or bearer material. There is no second scheduling source or polling loop.

## Evidence and remaining work

Sixteen component tests cover the real server lifespan, API dispatch, cold/
identity/capability failures before mount, partial mount rollback, legacy-work
refusal, unknown execution retention and cancelled observer drain. Two new
cross-repository cases use actual staged registry authentication/SQL/HTTP to prove
explicit parent finish and queued-child runtime restart without a new root claim.
Physical providers, memory sensors and process identity are deterministic fixtures.

The native root round-trip suite additionally drives actual AIAgent, plugins,
queued SDK and local authenticated HTTP through this runtime for both execution
and stint providers. Ten cases cover a plain final response, vision tool child,
parallel tool children, interruption/resume and a child outlasting a deliberately
short generic test guard with its qualified per-tool override. All native calls
keep one exact physical root; the competing requester begins on a distinct root
only after native finish. These cases use synthetic registry/provider processes
and disable compaction; they are not compaction inference or deployed acceptance.

Four additional native cases enable automatic compaction using the actual
QueuedContextCompressor and stock summary dispatch/commit, not a patched compress
method. Both providers are tested with in-place compression and session rotation.
A competitor is queued during the real summary request; summary, later main/tool
calls and native finish keep one logical parent and exact physical root. Rotating
cases verify the committed native session end and owner alias. Only native finish
lets the competitor run, on its own different root. The fixture scales history
and threshold (1000 tokens) for testing; it says nothing about live trigger sizes.
It also verifies that compaction need not be the turn's first inference request.
Actual hardware/provider responses, long real health maintenance and deployed
profile behavior remain unqualified.

The later cross-service boundary and credential/job authorization requirements
are recorded in [root-child-handoff.md](root-child-handoff.md). Same-service vision
tool coverage must not be represented as image-service delegation acceptance.

No production root policy, token, launcher or profile is changed. Before rollout,
qualify actual host/process/cache sensors, native full turn/compaction/tool and
operator recovery, inherited legacy claims/accepted work, same-envelope delegated
children, and release/rollback. The first standalone native image and image
endpoint remain Hermes Samcloud's delivery and do not wait for this root project.
