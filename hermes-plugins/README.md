# Native queue components (offline)

The current rollout also preserves the existing same-gateway vision route. Its
registered `samcloud-queue-tool` provider binds to the exact native API/tool IDs
through tool-execution middleware; a positively returned response advances its
private intention, while an unknown observation reopens the saved intention.
It yields only that conversation's parent stints and never borrows their priority.
External auxiliary routes remain unchanged. Native auxiliary config must leave
the queued provider's credential in its registered environment binding: supplying
both `base_url` and `api_key` makes this Hermes version select anonymous custom
transport instead. The raw endpoint is retained, not changed to another provider.

The existing Ollama and mlx-vlm chat paths share one resource queue. Eligibility
is model-specific, using the existing model estimate/VLM catalog size and current
physical/registry fit, not a new whole-machine budget. Running/uncertain work
still owns the runner; only idle affinity can yield when its model cannot fit.
A service-owned vision read has no generation-age limit; uncertain completion
retains its admission and model claim. Native core, thread lifecycle, context
thresholds and provider versions are not modified by these components.

Queue-enabled compatibility routes require a stable `Idempotency-Key`; missing
intent is428 before dispatch (and before audio copying). A new random key per
transport retry would duplicate detached work. Native provider/job-API callers
already carry explicit intent keys. This requirement applies only when the new
dispatcher is enabled; legacy mode is unchanged. Migrate consumers before rollout.

Ollama streaming now also retains requested provider token counts through saved
queue events and the SDK, including the usage-only frame before DONE. Missing or
invalid counts stay unknown. A native handoff test verifies those counts reach
the stock compressor rather than only appearing in the gateway's log.

These are opt-in components, not installed in running Hermes homes. They require
the adjacent `ollama` queue modules on the runtime's Python import path and a
model service implementing the durable inference API. The current live gateway
has not been qualified or deployed with that API.

`inference_bootstrap.QueueRuntime.install` is the staged, explicit mounting
boundary: exact immutable model/resource settings, the durable API and the same
compatibility dispatcher installed together before serving. It refuses duplicate
installation or a manager with existing residency/lease ownership or a background
task history. It belongs before startup adoption, not as a hot conversion. In
durable mode the manager refuses legacy discovery/adoption lease allocation;
restart ownership must be reconciled through the durable path instead.
The real gateway lifespan now accepts a trusted `QueueStartup` binding before
adoption/maintenance. It verifies the gateway's own registry identity and requires
an explicit empty Ollama inventory; nonempty/unknown inventory refuses rather
than reloading/adopting it. Missing `models` in `/api/ps` is unknown, not empty.
An explicit launcher is available (not selected by the live launch agent):

```sh
python -m ollama.inference_launch --queue-config /absolute/queue.json --work-permission allow
```

It defaults to loopback, accepts explicit host/port, runs one process without
reload, and refuses an already-bound service. Config parsing opens no listener
or inference journal; actual lifespan still verifies own identity and cold state.
Live launcher configuration and real-provider qualification remain outstanding.
The cold-start binding now binds manager stats to a
capacity source and exact owned residency receipts to renewal. Cold admission
checks fresh physical and registry headroom; resident admission reuses its exact
reservation instead of charging it twice. Cached provider sizes are not physical
credit. Unknown measurements, permission, ownership or expiry withdraw admission
without canceling work. Collection failure also withdraws old permission.
Capacity begins unknown on every start; trusted ordered capacity
events can pump queued work without another consumer request. Old/duplicate
availability events cannot undo a newer withdrawal. Withdrawal does not cancel
active work, and restarting does not redispatch quarantined work. These tests
exercise receipt-backed admission synthetically, not real-provider memory safety.
The deployed launcher does not
select this startup binding yet; a failed install must fail startup, never fall back
to legacy direct dispatch. No timer is used to infer capacity.

`QueueRuntime.quiesce()` pauses dispatch and waits for actual service-owned
workers/input preparations, without an age deadline or canceling them when the
observer leaves. It reports unresolved journal ownership separately: an uncertain
execution is NOT drained merely because its task ended. New capacity events cannot
undo the pause. Resume requires fresh capacity evidence rather than reusing a
pre-handover permission. The gateway lifespan now validates the installed runtime
owns its manager/dispatcher and awaits quiescence before manager cleanup; unknown
ownership refuses lease cleanup and journal close. This is graceful shutdown,
not process survival or a guarantee under SIGKILL/power loss. Live startup remains
unwired, so this does not describe an enabled deployment.

`inference_residency.ResidencyStore` persists same-allocation rollover intent
before its outbound request. A lost reply/reopen recovers the same key and
predecessor, even after the predecessor's original expiry. Exact owner/service,
model/resource linkage and replacement metadata are checked; malformed replies
keep the pending intent. An ended replacement records uncovered accounting, NOT
terminated backend work. Restored disk receipts are not current confirmation.
Confirmation is tied to the exact observed receipt, not a mutable shared pointer;
sibling handovers cannot inherit it. This is a metadata store, not an enabled
renewal loop, physical working-set/KV sizing rule or capacity permission source.

`ResidencyRollover` owns the metadata request separately from observers. Joining
an in-flight handover reuses its task; an unknown reply keeps the saved operation
for an explicit retry, without release/reacquire or a fresh key. SQLite receipt
application remains on the owner loop. Observation/drain cancellation does not
cancel that request; normal close refuses while it is live. An actual service
cancellation can still leave the HTTP outcome unknown, with intent preserved.
The joined queue test withdraws the next dispatch during handover, preserves
existing compute, and pumps the queued peer after a valid receipt event. The
diagnostic callback is not sole permission: check exact current receipt AND
physical working-set policy. The cold-start path binds renewal now; production
launcher selection and real-provider qualification are still absent.

A real subprocess/socket recovery test now kills the SDK observer while its
service-owned job is running, then opens the same caller-persisted intent from
a new process. The blocked service worker stays alive; the second observer gets
its saved answer, with one execution and one dispatch. This qualifies that SDK/
durable-identity boundary only, NOT a native JSON-RPC restart, plugin-host cold
rebind, provider crash, physical lease handover or power-loss recovery.

`samcloud-queue-provider` uses native filesystem provider discovery. It supplies
`samcloud-queue` for execution-scoped main calls, `samcloud-queue-stint` for
conversation/model stints, and `samcloud-queue-aux` for explicitly bound auxiliary
work. The stint profile requires authenticated service capabilities before any
grant/execution write. Event-only StintPolicy() has no queue-age/compute deadline;
the optional timer experiments are not the first-rollout default.
All use the caller's normal profile-scoped credentials; no copied token files.
Main responses can carry genuine service-state frames. Auxiliary calls observe
the saved complete response instead of inventing summary tokens while queued.

Opt-in streaming observation starts after local durable-intent validation, before
the service necessarily accepts work. Its initial headers name local grant/call
keys, NOT an invented service execution ID. Successful submission emits a distinct
job_accepted control frame with the actual service grant/execution IDs. Successful
queue reads emit queue_status; lost/stalled contact emits adapter_status explicitly
labelled local_adapter and service_confirmation:unknown. Last-confirmed state is
historical metadata, not a claim that the service is still in that state. All
control frames have empty choices and no model usage/text; they consume no saved
provider cursor. The same pending read survives observation pulses. Reconnect
reuses intent; closing even an uniterated stream stops observation, not the saved
job. JSON/auxiliary complete-response policy remains separate. Real native tests
cross shortened stale windows before acceptance and during failed polls without
stale give-up or fabricated output; no day-long wall-clock acceptance is claimed.
Authentication/ownership/refusal evidence still produces explicit errors.

Compaction keys carry only the owning profile/session's allocated main stints.
Dependency enqueue and parent yield commit atomically; active/uncertain work is
not preempted. Saved dependency replay cannot yield a later main stint. Handoff
metadata is frozen with durable caller intent before submission, including an
empty parent set. No latest-conversation or prompt-equality lookup is used.

`POST /v1/inference/yield` accepts a durable grant key and expected logical-call
key. Owner isolation and the latest-call fence prevent a delayed native lifecycle
event from yielding newer work. A provider's finish_reason:stop is NOT treated
as the native turn ending: verification/continuation may still follow. The native
post_llm_call hook writes a private identifier-only outbox for its exact native
turn and that turn's submitted grant/call keys. A separate SDK-owned relay retries
delivery; the bounded callback performs no network I/O. Acknowledged or typed
stale events leave the pending set; missing/malformed/failed observations retain
them. Cursor rotation stops old unacknowledged events starving later pages.
Relay cancellation drains real SQLite work before its owner closes. Only the
opt-in stint profile starts this relay; other profiles are unchanged. Native
hook, temporary-home reopen and SDK/service handoff tests are not whole-RPC/
power-loss qualification. Vision/tool dependencies and detached background-review
source scopes still need qualification.

An opt-in stint profile can declare an external terminal dependency with
`sc-queue-wait -- <command>`. The general plugin strips that exact prefix before
normal native approval/tool dispatch, preserving the command tail and other tool
arguments. This is an annotation, not an installed shell executable or a duration
heuristic. Ordinary tools retain affinity. The callback records an identifier-only
yield against the originating session/turn/API/tool call; the SDK relay delivers
it while the tool waits, without ending the native turn or cancelling inference.
Missing origin metadata blocks the annotated tool rather than guessing a latest
call. Hermes can reuse a logical API counter: successful response intake freezes
each returned tool ID against its candidate call, not the API counter alone.
Refused attempts or invented tool IDs cannot authorize a tool-wait yield. A
returned tool's frozen binding cannot be rebound, even if its API counter recurs.
Native dispatch and service relay/handoff tests cover this contract; real
long-build/RPC/power-loss behavior and worker guidance are not yet qualified or
installed on the live cohort.

`samcloud-queue-hooks` is a separately enabled general plugin. Its observer
captures persisted source row IDs against the exact session/request identity;
execution middleware consumes that snapshot and binds it in the provider caller's
context. Observer hooks run on copied worker contexts, so setting a ContextVar
in the observer alone does not work. Missing/cross-session handoffs refuse before
service I/O. No "latest call" lookup or prompt matching is used.

`samcloud-queue-context` is explicitly selected with `context.engine`. It wraps
the stock compressor's `compress()` call in a held-source auxiliary scope; the
summary algorithm and host commit/stop fence remain native. Ordinary Qwen
configuration parity is tested for the65K ratio-only and262K/70K-cap profiles.
The latter is a configuration test, not a proposal to change a live threshold.
Constructor-only host overrides and other providers' special threshold policies
are not yet qualified. Preflight inputs not yet stored in native history get a
new explicit compaction operation, with the complete immutable snapshot saved
privately before inference. This is operation allocation, not prompt deduplication.
Normal full-turn preflight is tested both immediately and after a queued wait.
The native gateway's row writer/adoption helpers are also exercised: its
pre-acknowledged input keeps the accepted row address through preflight, so it
uses held-row identity rather than inventing an unpersisted compaction source.
These helper/full-agent tests are not a JSON-RPC process-crash qualification.

Private snapshots live under the owning profile's
`samcloud-queue/compaction-sources/` (directory700, files600). They contain full
held transcripts and must not be bundled with source or uploaded as telemetry.
`HostBindings.resume_compression(session_id, operation_id)` requires the exact
recorded operation and validates profile/session/snapshot; there is no newest-
source lookup. Automatic association with a restarted outer turn remains undone.
Stock gateway cold resume may schedule a newly synthesized continuation prompt;
that is new native work, not automatic re-entry into the previous inference.
Do not reuse an earlier call's identity for that different continuation body.

Saved, service-confirmed refusals now retain their SDK status/code in sync/async
JSON and streaming observation; authentication failures retain their status too.
Uncertain executions remain nonterminal in the job-API consumer: observation
waits for explicit reconciliation, retaining the same job and admission. They
are never replayed as fresh work. Stream control frames name uncertainty, not
successful inference; contradictory terminal evidence still fails explicitly.
Legacy compatibility responses are not this native waiting contract. The
Ollama adapter now qualifies only complete pre-stream400/404
JSON error responses as typed refusals, with fixed sanitized copy. Truncated or
malformed bodies,5xx, load errors and mid-stream errors remain unqualified.
Chat/completion streaming/non-streaming refusal claim release and non-reexecution
are tested, including an actual sync Ollama client→gateway→journal→SDK path.
No context-overflow category is guessed from arbitrary raw provider text. A saved,
explicit context_length_exceeded refusal now exercises the stock whole-turn
overflow handler: queued compaction, smaller second main request, final answer,
accepted input exactly once and original history rows retained. This includes
the gateway's accepted-row helpers and the direct caller, but uses a synthetic
service, not a production provider or actual RPC process. Preflight/overflow
whole-turn tests cover both execution-grant and conversation-stint profiles.

Admitted Ollama workers now select a service-owned read policy, without the
legacy generation-age/silence budgets. Connect/write availability bounds remain;
a real transport loss is unknown, not a completed request. This policy is chosen
by trusted queue code, not an HTTP body flag. The legacy path keeps its defaults.
Real-socket silence and scaled async-budget tests cover waiting; scheduler-cancel
tests keep both capacity and claims protected while the worker outlives its task.
Physical lease renewal over multi-hour jobs remains a live qualification item.

Queue-owned model resolution now offloads local health/list/load work only after
admission. A blocked real-resolver path leaves health/queue observations available;
a scheduler canceled during a load retains uncertain admission while the thread
outlives it. Legacy resolution keeps its existing serialization. These are mocked
backend tests, not live load/physical-lease continuity qualification.

Explicit model load/unload now join the same durable admission queue. Force
unload cannot bypass active or uncertain inference. The verified caller owns the
saved mutation result; an explicit idempotency key reopens it without repeating
the mutation, and changed input under that key conflicts. An HTTP observer leaving
does not cancel the service-owned mutation. Unmapped model resources fail closed.
Other backend adapters, resource mapping, leases and activation still need
qualification before shared-resource deployment.

`inference_inputs` stages private bounded uploads, verifies immutable per-grant
copies and retires them only from saved terminal journal state. Queued/running/
uncertain work, or another pending call using the same copy, keeps its input.
Audio now stages its input before enqueue and shares admission with chat/control.
Its worker holds the residency claim until confirmed completion or decode refusal;
unknown/canceled execution retains the claim and copy. Cleanup runs after the
saved result, verifies large bytes off-loop and rechecks metadata/references on
the loop. Five format replays, multipart routing, cancellation/unknown/refusal,
and a real local-socket owned-read/legacy-timeout control are tested. Queued
Whisper reads have no generation-age deadline; legacy defaults are unchanged.
Abandoned/conflicting uploads, retention quotas, bootstrap and actual backend/
multi-hour lease qualification remain unfinished. No live media was copied.

Still unfinished: automatic cold-turn source reconstruction and whole-turn
resume, real-provider error/overflow recovery, isolated plugin-host qualification,
installation/dependency packaging, every-backend arbitration and live rollout.
Normal queued summary/stop/reopen persistence is tested through the discovered
context engine. This does not establish summary quality or whole-process crash
recovery. Test fixtures use
`plugins.isolation: in_process`; do not infer host-process support from them.

Tests run via the stock Hermes checkout's `scripts/run_tests.sh` using the
PM-built test interpreter and `ollama/test_hermes_queue_client.py`. They use
temporary profile homes and a synthetic HTTP service, never live inference.
