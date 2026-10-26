# Registry root client — staged, not connected to admission

`ollama.inference_root_client` supplies durable control operations for the staged
registry explicit-end root protocol. Nothing constructs it in the manager,
bootstrap, native adapter or image service. It does not enable cross-service
inference capabilities. Do not activate requester jobs from these client tests.
First standalone image generation remains independent of compound-root support.

Create `RootIntentStore` at an owned private file, explicitly bound to the HTTPS
registry `/api/v1` origin, the provider's verified actor and registered service.
The `job_id` client argument identifies one durable physical root intent.
The staged `RootPreparingArbiter` supplies its saved preparation ID, bound to
the actual `requester_jobs.id`. Reconnect/interruption reuse that preparation;
explicit yield can later create a fresh physical preparation while preserving
the logical job. Never substitute a username, current conversation or latest
job, or overwrite the old intent. The journal stores no bearer, prompt or model
weights. Operation-local SQLite connections commit with `synchronous=FULL`.

`RegistryRootClient(store, token=own_provider_token)` creates a dedicated HTTP
client. Before registry mutations it reads authenticated
`GET /api/v1/capabilities/lease-roots`, requiring version1, the exact actor and
all recovery features. A missing/partial/incompatible server refuses. There is
no fallback to ordinary lease reservation. Redirects and environment proxies
are disabled; use the canonical HTTPS API origin without embedded credentials.

- `begin(job_id, resource)` commits a random submit key before sending intake.
  A lost reply retries that saved key; an acknowledged root is recovered by ID.
  A missing exact root raises rather than allocating a replacement.
- `observe(job_id, resource)` reads only the acknowledged root. A receipt is a
  historical observation, not dispatch permission. The qualified physical
  controller must separately check current exact registry admission, headroom,
  backend identity and absence/recovery conditions before loading.
- `end(job_id, resource)` commits sticky end intent before HTTP. It uses the
  submit key even without an intake acknowledgement, so it never creates a
  claim merely to end one. `begin` after reconnect flushes the saved end instead
  of submitting new work. A concurrent end cannot be cleared by an older open
  ACK; `begin` forwards the end before returning. Active/uncertain work remains
  charged until confirmed cleanup; a never-admitted receipt has no root/lease ID.
- `renew(..., lease_id=exact_current_member)` maintains finite health only.
  It does not change logical completion or acquire a new claim.
- `recover_absent` and `settle` require the exact lease and an explicit qualified
  absence-observation reference. Their operation keys/payloads persist before
  HTTP; lost replies repeat the same action. A locally ended root cannot be
  readmitted. These methods do not observe or stop a backend themselves.
- `close()` closes transport only. It never ends, cancels or releases work.

The finite HTTP observation timeout and short SQLite-lock wait do not impose a
queue/job deadline. Failure leaves durable intent and any accepted claim intact;
there is no polling thread, age takeover or automatic retry. Caller orchestration
must resume/flush its saved intents and reconcile terminal status. A saved
receipt, a client close or a missing consumer cannot authorize physical cleanup.

## Qualification limits

Unit tests cover client persistence, scope/contract refusal, concurrent prepare,
process reopening, lost replies, sticky end and exact operation keys. Separate
pinned two-checkout contract tests exercise this client against the actual staged
registry router, real bearer verifier and SQLite allocator, including lost
post-commit replies and one-time promotion. They are control-plane evidence,
not Metal/GPU or deployed endpoint acceptance.

Next integrate the controller with fresh physical admission before load, native
job completion/drain, exact backend cleanup, same-envelope child transfer and
operator reconciliation. Do not independently request an image claim while a
text root holds capacity and waits. That remains the unqualified self-deadlock
this component will help repair; this client alone does not fix or enable it.

The [staged preparation scheduler](root-preparation.md) now implements the local
pre-dispatch/end boundaries. It is not constructed by the service supervisor or
bootstrap. The actual model loader/capacity source and delegated child path still
need coherent integration and physical qualification.
