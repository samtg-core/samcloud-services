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

`acquire.py` owns that metadata operation across observer disconnects. It waits
for the existing event consumer to confirm its cursor, then creates/advances
through the ordinary host queues. A committed `create_started` decision retains
ownership of the exact metadata operation even if its reply is lost and the
original caller subsequently cancels. An explicit event/demand/reconciliation
wakeup can resolve that same key; there is no timer or failure-driven retry.
The actor records returned group identity even if its event connection is lost
in flight. Contact loss withdraws authority until the cursor is reconciled.
Member identities are frozen in the inference journal before rank startup;
renewing expiry is permitted, replacing a lease or generation is not. Populated
unreleased draft journals without submission provenance refuse automatic
migration rather than guess whether an old create was submitted.

`demand.py` explicitly installs a preparation hook on the existing supervisor.
Its pump prepares new saved demand **before** checking capacity, so acquisition
cannot deadlock waiting for the capacity it must establish. It skips already
linked calls on capacity wakeups. The supervisor owns/drains acquisition tasks,
and preparation failures leave calls queued without direct-dispatch fallback.
Failures and recovery write deduplicated metadata events with no exception body.
The default supervisor path has no hook, so ordinary installed gateways retain
their existing behavior. The hook creates no public route or extra event loop.

`renew.py` maintains exact member leases in place, including partial ownership
while another member remains queued. Per-rank `renew_before_s` values must be
explicit, positive and below that member's declared TTL. Enabled or disabled
renewal policy commits to the same journal and cannot silently change on restart.
Fresh scoped projections supply the server observation time; conservative
continuous-clock deadlines subtract transport time and include host sleep.
Indefinite queue waits are not renewed. Expired members are not resurrected.

Renewal uses the existing ordinary lease endpoint, with fresh group checks before
and after. That endpoint is **not** a group-generation CAS: a POST racing a fence
can extend a lease row. The subsequent read rejects the old generation; the
renewal reply never grants execution permission. A successful but unextended
reply cannot drive an immediate retry loop. At the member's explicit lifetime
ceiling, no further renewal is attempted and an end-of-authority event is armed;
there is no automatic release/reacquisition.

`alarm.py` migrates the qualified Darwin continuous kqueue timer into this package.
One descriptor joins the existing service loop when renewal is explicitly enabled;
there is no watchdog thread/polling loop. Its flags are checked against the
installed SDK and its registration/disarm/replacement controls run against the
real native timer. An alarm failure withdraws permission. No timer kills compute
in this coordinator; host authority/cleanup remains a separate contract.

## Private instance preparation and observation

`prepare.py` migrates the qualified explicit placement builder. It uses the real
pinned EXO types in a preparation environment, validates the model's parent and
rank manifests, and assigns deterministic instance/runner IDs per group generation.
`instance.py` preserves both actual wire forms: snake_case for creating an instance,
and camelCase aliases for observing it. The gateway checker imports no EXO or MLX.

The immutable contract checks the exact plan/runtime pin and three launch bindings,
then refuses an extra instance (even of the same model), changed node/rank/card,
missing/extra/unready runner, prefill listener, instance link, unrelated/unfinished
task, or mismatched read-only discovery. EXO retains completed discovery records;
each must match that node's assigned shard. Historical terminal tasks are allowed,
including failed ones: idle is not evidence that any request succeeded.

These are configuration/observation checks. Runtime pin provenance, host identity/filesystem receipts,
fresh authenticated state and current lease authority must be verified separately.
No saved readiness result authorizes a later dispatch. The fixture is actual-type
serialization with synthetic node IDs, not observed running hosts. Its selected
source-pin manifest checks this API surface; it is not full runtime qualification.

`private.py` uses only an explicitly configured
loopback port. It reads EXO's own single-link, owned 0600 private key, not the
coordinator's plane credential; redirects, oversized/ambiguous JSON, failed reads
and invalid keys establish no observation. There is no cached readiness fallback,
retry, environment proxy or default transport deadline. Its inference POST must
first consume a once-only decision from the existing journal.

`calls.py` freezes the prepared instance per group intent in that same inference
journal, then records a once-only private submission decision only for the actual
dispatched execution. The canonical body must equal the saved chat request,
including its original stream mode. Reopening after a crash or losing the first frame cannot create
a second submission, even with zero saved output.

EXO creates a command UUID inside its chat handler, so it **cannot be known before
POST**. The pre-POST record includes execution, group generation, instance/contract
and canonical request digest, with command initially unknown. The actual UUID binds
once on first observation, before output is published. A changed/duplicate command
refuses; a late observation may improve uncertain-call provenance but cannot finish
it or release HOLD. The prepared baseline event index and exact end evidence are
also journaled; old draft rows without a baseline cannot invent it during migration.

`backend.py` is a supervisor-compatible executor for text chat, with streaming and
ordinary JSON responses. It binds the command from the private response header
before output. Streamed terminal frames stay provisional until a normal complete
stream, exact all-rank runtime end receipt and fresh exact idle state agree. A
confirmed backend error then fails cleanly and the next logical call can dispatch;
the residency group remains owned. Reset/empty/missing-terminal/changed-command
streams or unconfirmed end evidence stay uncertain, preserving HOLD and output.
The acquisition actor's current event cursor and lease permission are required;
runtime/host verification is an explicit injected effect, with no default success.

`startup.py` adds an explicit private-instance submission owner on the same
journal. It requires fresh admitted authority, an independently verified runtime
and the exact empty three-host namespace. The frozen contract/baseline commits
before the sole POST /instance; acceptance records the server command and matching
card, never runner readiness or serving permission. Crash, lost reply, malformed
acceptance and a second concurrent startup owner cannot re-post. GET reconciliation,
host launch and mark-serving remain separate integration work.

The leased runtime candidate also validates the prepared explicit pipeline/card
at POST /instance and avoids ModelCard.load's name-based fallback to Hugging Face
main. Ordinary EXO profiles retain their lookup. Actual handler tests cover the
pinned-card/ordinary paths and reject changed namespace, runner, endpoint, card,
layers, timeout, remote-code and image variants before any command. They use MLX
CPU and an injected command transport; no model/node/listening server.

## Staging before physical acquisition

`staging_files.py` is a standard-library host collector: exact files/aliases,
streamed hashes, stable inode/size/mtime/ctime, loading-view contents, local Darwin
filesystem and disk headroom. It follows no unpinned path component and writes
nothing. A trusted owner's verified cache can reuse a hash only after unchanged
kernel file identities; the native loader independently hashes all assigned bytes
before each actual model load. There is no elapsed-time shortcut or polling loop.

`staging.py` binds the real bundle/parent/config/sidecar bytes and three explicit
host/runtime policies. Each host supplies its own complete qualified runtime file
set: equal package versions do not imply equal native binary hashes. Runtime
directory entries, including the root, must match exactly; an extra importable
module refuses staging. Its reader must be an authenticated host channel supplied
by startup. Each host must match uid/boot, readonly files/directories, local APFS/HFS
and explicit scratch reserve. All observations are retained in the existing
journal before the first group submission; policy/runtime/host boot cannot change
under that intent. A missing verifier requests no leases. An already-owned lost
create reply can reconcile its exact saved key, while unconfirmed staging blocks
advance/start/serve. Local capacity remains usable before Slice is requested.

`host_channel.py` supplies that reader over explicit trusted host routes. Startup
pins the collector source, route/uid per host, interpreter and minimal local
environment. The fixed isolated stdlib probe receives policy JSON over stdin;
it never receives a plane bearer or inference body. Each bounded response must
match a fresh request nonce and collector pin. There is no shell interpolation of
policy data, default observation deadline, automatic retry or cached response.
Cancellation/oversized output reaps only this owned read-only observer. SSH key
and host-key trust configuration belong to the existing operator/service route;
neither queued callers nor this module select credentials or change host trust.

The attended route was exercised on all three real hosts: their exact12-entry
small-model copies and903-file candidate source copies are readonly, hash-matching
and host-local. A second fresh read reused only unchanged kernel identities.
The source copies live in new isolated directories; no node/model was started,
no lease was requested and no standing runtime was changed. This is source/model
file preparation, **not** the complete runtime qualification: interpreter,
dependencies/native libraries/config and host process provenance remain required.

The canonical bytes are the original staging format: sorted compact UTF-8 JSON,
**without a trailing newline**. A port regression had changed the parent/config
hashes and instance IDs. Frozen real bundle/parent fixtures now reproduce the
original request and literal instance ID, not just newly generated test data.
The native leased instance guard uses the same original identity format.

The current native candidate remains uncommitted and uninstalled under full
repository/native gates. The earlier source packets2254 and2256 are historical;
2256 contains the superseded newline identity guard. [Packet2257](https://cloud.samtg.xyz/api/v1/files/2257/raw) records
the corrected903-file candidate (SHA256 `70dce31a552727f008c355ab556498980f83344bbe3cbbf7d7b437b4bb343132`). CPU/type/HTTP controls are not distributed acceptance.

## Integration still required

This component is **not enabled by default startup and is not a complete EXO
provider**. Nothing in this PR has been installed on a host. It does not yet launch
ranks or load a model. Tests inject plane/private-runtime/host effects; the actual
registry qualification uses its isolated HTTP/SQLite implementation, not the fleet.
Remaining work:

1. Configure the implemented authenticated host reader for the service's exact
   complete deployed runtime and the existing plane event feed. Model/source
   files are sealed; dependency/native runtime qualification and process provenance
   remain. The attended operator route is not an unattended service credential.
2. Wire the implemented once-only private-instance submission to host rank startup,
   staging/runtime/file receipts and mark-serving after actual ready observations.
3. Wire the prepared instance contract to the pinned runtime's private namespace,
   authenticated API and host/file receipts. EXO's chat API selects by model, not
   a caller-supplied instance ID; check exact state before dispatch and persist
   execution-to-instance identity before posting inference. Bind the server-created
   command UUID on first observation, before publishing output.
4. Qualify the implemented executor and runtime barrier on native hosts. Transport
   loss, a missing terminal frame or unconfirmed backend end retains uncertainty
   and physical HOLD; reconnect/time never retries inference or releases the group.
5. Keep call completion distinct from the residency session. Release only after
   observed rank cleanup, exact evidence publication and acknowledged settlement
   of every member. Wire the implemented renewal policy to the pinned runtime
   configuration before longer runs. Standing long arcs still need their recorded
   protocol decision; there is no default idle-release/reacquire timer.

Production integration remains gated by the small-model native and multi-host
qualification, network confinement and the existing campaign boundary. CPU and
HTTP controls below are not evidence of distributed execution or 200 GB residency.

## Validation

From the model-service repository:

```sh
python -B -m unittest -v ollama.test_exo_group_plan \
  ollama.test_exo_group_member ollama.test_exo_group_clock \
  ollama.test_exo_group_authority ollama.test_exo_group_intents \
  ollama.test_exo_group_acquire ollama.test_exo_group_demand \
  ollama.test_exo_group_renew ollama.test_exo_group_alarm \
  ollama.test_exo_group_instance ollama.test_exo_group_calls \
  ollama.test_exo_group_private ollama.test_exo_group_backend \
  ollama.test_exo_group_install ollama.test_exo_group_startup \
  ollama.test_exo_group_staging ollama.test_exo_group_staging_files \
  ollama.test_exo_group_host_channel
```

Actual EXO-type controls require that candidate's Python environment, not the
gateway venv. From this repository, with `EXO_SOURCE` pointing at the qualified
candidate (21a54c5 plus the recorded patch):

```sh
"$EXO_SOURCE/.venv/bin/python" -B qualification/run_exo_instance_types.py \
  --exo-source "$EXO_SOURCE"
"$EXO_SOURCE/.venv/bin/python" -B qualification/run_exo_command_types.py \
  --exo-source "$EXO_SOURCE"
"$EXO_SOURCE/.venv/bin/python" -B qualification/run_exo_private_api_types.py \
  --exo-source "$EXO_SOURCE"
```

The runner checks selected source pins and uses a fresh private EXO root. Network,
process launch and native compute imports are forbidden. Thirteen controls exercise
the actual tagged request, State aliases, unequal ranks, ring guards and local
card round trip. Nine gateway controls exercise the same serialized fixture and
negative bindings without installing EXO into the gateway environment.
The second runner checks12actual native event-barrier controls under the same
CPU-only restrictions. The endpoint's whole-runtime/native qualification remains
separate. Both control sets also pass from the exported/reconstructed source.
Sixteen further controls use the actual chat/end HTTP handlers with command delivery
injected and MLX's default device forced to CPU. Network and process launch are
forbidden. They cover both response modes' command headers, duplicate submission
refusal, authentication, the event-driven end wait and the ordinary profile's404.
This handler check permits framework imports; it is distinct from the25controls
that forbid native compute imports. No model, node or listening server is started.

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
and unsettled. The acquisition controls additionally exercise actual ordinary
queue assembly behind an Obelisk blocker, same-key recovery of a lost committed
create reply after caller cancellation, and a logical dispatch through the real
supervisor only after lease-serving plus explicit injected-backend readiness.
The final backend is a CPU injection, not EXO/native qualification. No production
lease or inference is submitted by any of these controls. Renewal controls also
keep partial-group active IDs/queue positions, reject a fence racing POST, retain
expired HOLD without resurrection, and arm the explicit ceiling notification.
