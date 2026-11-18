# Explicit EXO inference accommodations

Trusted `HostDeployment` profiles may set two optional controls. They are not
caller-selected request fields, queue deadlines or admission overrides.

- `prefill_step_size` requires the pinned native `inference_profile.py` control.
  It enters the compiled environment as `EXO_PREFILL_STEP_SIZE`. The qualified
  pipeline divides it across up to four ranks. Smaller chunks reduce temporary
  activation workspace without changing weight precision, cache precision or
  the served context. There is no new default; unset retains native behavior.
- `served_context_length` must be positive and no greater than the checkpoint's
  pinned `max_position_embeddings`. It requires the pinned native
  `served_context.py` enforcement module and generator. The owned metadata
  compiler puts the value in every explicit-instance model card; its response
  is rejected if it advertises a different window. Each worker receives the
  matching `EXO_SERVED_CONTEXT_LENGTH`, whose native guard counts the complete
  encoded prompt plus the requested output budget before cache lookup/prefill.
  Requests are refused rather than truncated. The checkpoint config is intact.

All fleet members must agree on both controls. Non-default values enter the
host configuration fingerprint and compiled launch environment, so changing a
control changes its binding. Existing profiles with both fields unset retain
their configuration digest, card window and environment behavior.

The service change must be paired with the reviewed native enforcement build
before enabling a served context. A filename alone is not a capability proof:
the full runtime source, native bindings, file inventory and transport remain
pinned and verified by the existing preparation path. No runtime dependency
installation, process launch, inference or source selection happens here.

For the proposed235B 6-bit profile,256-token prefill chunks and a64K window are
candidate settings only. Prove actual full-window KV/peak/device memory under
all admitted member leases before publishing that window or scoring cases.
A short prompt only demonstrates a short prompt. The upstream non-thinking
sampling settings must also be sent and verified in the eval's actual requests.

The registered EXO provider preserves explicitly supplied `top_p`, `top_k` and
`min_p` through both durable and compatibility queue intake. Probabilities must
be finite numbers in [0,1], and `top_k` a nonnegative integer; boolean/string
coercions are refused before creating a grant. Explicit null remains null.
Omitted options add no fields or defaults to the saved request, preserving its
existing identity. These native-only options do not change ordinary Ollama
requests. The provider forwards the persisted values to EXO; a qualification
must still inspect the actual native task options before claiming card settings.

A completed result and permission to dispatch again are distinct. After a valid
terminal response, exact all-rank command-end proof and the existing runtime /
idle-instance checks, a successful result stays successful even if admission
permission is invalidated. Fresh authority is still required before dispatch and
throughout an active stream. Missing/stale end evidence or interrupted output
remains uncertain. Physical leases still require positive cleanup and exact
settlement; recording a completed call does not release the group.

Validation includes no-effect deployment controls, mismatch refusal before any
host/plane request, changed configuration/environment bindings, a helper that
returns the wrong context, and actual pinned EXO wire types showing every card
uses the explicit window without modifying the parent. The live three-host
numerical gate still uses its previous source/runtime until a new candidate is
selected and qualified.
