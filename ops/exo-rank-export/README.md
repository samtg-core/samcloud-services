# Verified partial rank export

These data-only operators prepare byte-exact Qwen3 or Qwen3-MoE rank artifacts.
They import no native inference libraries, take no GPU lease and change no
gateway, model or service environment. Their manifest explicitly says the output
requires the qualified shard-aware adapter; it is not a standalone full model.

`rank_bundle.build_bundle` accepts the pinned parent manifest, tensor headers,
index and explicit contiguous layer placement. Each parent file is read and
SHA256-verified once while selected byte ranges stream to all assigned ranks.
Global layer names and packed quantisation bytes are preserved; every rank gets
the non-layer tensors. The output directory must be fresh and measured free
space must cover all output plus an explicit scratch reserve. Partial outputs
never receive a verification marker. `verify_bundle` rehashes the copied rank
against a pinned bundle manifest before that host uses it.

`build_completed_qwen_bundle.py` is the sealed-packet runner used for the 235B
rollout. Before any export, it requires the original downloader's full completion
receipt, exact file inventory, service ownership, read-only files/directory and
metadata hashes. It then delegates full weight hashing to the byte exporter.
The parent manifest pins the native config/tokenizer sidecars. The exporter
copies those exact bytes into each assigned loading view, writes the parent
manifest and the qualified local weight alias, then records export completion.
It does not publish a standard full-model index. The actual pinned native
metadata adapter accepts and byte-verifies both ranks of a small fixture built
this way without importing MLX or a model. A failed adapter-view copy receives
no export-complete receipt even if byte export had completed.
The operator substitutes the exact reviewed `PACKET` and `PLAN_SHA`; that plan
also pins both imported modules and all parent/catalogue/index inputs. A detached
launcher must independently pin the runner itself and record one process identity.
Unknown launch results are observed, never replayed over a partial destination.

The prepared rollout packet pins revision
`6dd2ef235d39fb976237fd9243184741d8407a55` of
`mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit`, with a candidate23/12/59 layer
allocation. At preparation time it has **not** run: the complete checkpoint is
still downloading. Candidate allocation does not establish resident fit, served
context, tool support or eval performance. Source, destination and packet pins
are retained in the owning rollout's audit state; copied files must be audited
again at each host. No automatic retry, model load or queue bypass is supplied.

```sh
python3 -m unittest discover -s ops/exo-rank-export -p 'test_*.py'
```

Twenty controls cover incomplete/seal/receipt/metadata/symlink/extra-file
refusals plus byte-exact export across tiny chunks, corrupt input/output,
interrupted publication, non-overwriting destinations, copied-rank verification,
and complete pinned native loading views.
