# Roster add: inkling-2-mini:26b-a5b

The candidate model for the hermes-eval trial (`inkling-2-mini:26b-a5b`) must be
on the gateway roster to be served alongside the baseline `qwen3.8:27b-mlx`.

## Current state (probed on wafer, 2026-11-22)

- `ordinary-queue.json` (the roster source, 399 bytes, 5 models →
  `wafer-services/gpu-metal`) is **sealed** (mode `-r--r--r--`, `read_sealed`).
  It cannot be hand-edited.
- `qwen3.8:27b-mlx` is **on the roster** (baseline, ready).
- `inkling-2-mini:26b-a5b` is present in Ollama on wafer but **not on the
  roster** — the one-line add.

## The add

Add the candidate to the roster via the **sealed-deploy path** under
`wafer-services` (ticket #135 / claude-wafer-services), not by hand-editing the
sealed file:

```jsonc
// ordinary-queue.json — add one model entry:
{
  "name": "inkling-2-mini:26b-a5b",
  "backend": "ollama",
  "gpu_pool": "wafer-services/gpu-metal"
}
```

Sealed-deploy it (the deploy tooling verifies + rolls out the named commit).
After rollout, assert the roster via the gateway's `/v1/models` /
`/api/tags` and confirm both arms are served before the eval.

## Verification

```sh
# from wafer, after the sealed-deploy rollout:
curl -s http://127.0.0.1:8800/v1/models | python3 -c 'import sys,json; print([m.get("id") for m in json.load(sys.stdin).get("data",[])])'
# expect both qwen3.8:27b-mlx and inkling-2-mini:26b-a5b present
```
