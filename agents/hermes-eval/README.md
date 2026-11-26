# hermes-eval — dedicated worker principal for model trials

A dedicated SAMcloud **worker principal on wafer** that runs `model_eval` model
trials against the gateway, **isolated from every other agent's inference
provider**. Created per admin directive (ticket #1139 / c1139): evals and model
trials run here so hermes-samcloud's own provider (`qwen3.8:27b-mlx` /
`samcloud-queue-stint`) is never swapped or disturbed.

## Design (settled c5325 point 6, task 1415)

hermes-eval is **native, not a container**. `model_eval` v1 accepts only the
`native-service` execution path; a `rail-service` container is refused until a
real rail adapter records delivery receipts. The eval runs "on the configured
host under its own worker principal," on the deployed runtime + queue plugins,
with tokens read from a 0600 file into named process-env variables. A
wake/rail seat (`samcloud-rail.prod`, host `cont-hermes-eval`) exists **only** so
hermes-eval can talk on the plane — it does not host the eval. No container image
is built (the previously-cited `samcloud/hermes-rail:3.18.0` does not exist on
slice-containers).

## Why a separate principal

Running a trial needs the gateway to serve the candidate model
(`inkling-2-mini:26b-a5b`) alongside the baseline (`qwen3.8:27b-mlx`). Doing that
on hermes-samcloud's own seat would mean swapping its inference provider —
explicitly forbidden (c1139: "don't switch your own inference provider"). A
dedicated principal keeps the eval's `OPENAI_BASE_URL` pointed at the gateway with
its own provider block, so hermes-samcloud's provider stays fixed.

## Principal definition

| Field | Value |
|-------|-------|
| Identity | `hermes-eval` (SAMcloud agent user 83, `role: agent`) |
| Host | **wafer** (`models-wafer.samtg.xyz`, `gpu-metal`) |
| Execution | `native-service` under hermes-eval's own worker principal |
| Coordination seat | wake `samcloud-rail.prod` on host `cont-hermes-eval` (plane talk only) |
| Provider | gateway `https://models-wafer.samtg.xyz/v1` (own block; NOT hermes-samcloud's) |
| Models served | `qwen3.8:27b-mlx` (baseline) + `inkling-2-mini:26b-a5b` (candidate) |
| Scope | `group:services` membership — what wafer's gateway demands of callers (`SC_REQUIRED_SCOPE`, CLAUDE.md). NOT `services/enroll`, NOT `device:slice-test`. |
| Eval harness | `model_eval` (`python -m model_eval run --config <arm>.json --run-dir runs/<arm>`) |

## Scope and security

Eval only. No chat, no delivery, no operator session. A clean workspace and a
fresh profile; no copy of any agent's session / memory / skills / auth / cron.
**No token appears on a command line or in a config** (c5325 #1): bearer headers
are read from a 0600 file (`-H @file`) and `SC_TOKEN` is delivered via a 0600 env
file. The wake seat is registered **under hermes-eval's own token**, not the
admin's (c5325 #2). The manifest binds model + digest + context window + hardware
+ revisions + identity so runs are reproducible and comparable.

## Enrollment (admin-provisioned, per c5325 on task 1415)

Two steps, in order:

1. **Invite** (admin-only), scope `group:services`:
   ```bash
   curl -X POST "$SC_BASE/users/invite" \
      -H @"$HOME/.samcloud/token.hdr" \
      -H 'Content-Type: application/json' \
      -d '{"username":"hermes-eval","role":"agent","scopes":["group:services"]}'
   ```
   The invite token is written to a 0600 state file, not the log.
2. **Accept** the invite to mint hermes-eval's token (lands in a 0600 file), then
   **register the wake seat under hermes-eval's own token**:
   ```bash
   curl -X POST "$SC_BASE/wake/register" \
      -H "Authorization: Bearer $(cat $HOME/.local/var/hermes-eval.token)" \
      -H 'Content-Type: application/json' \
      -d '{"wake_cmd":"/home/agent/.local/bin/samcloud-rail.prod","host":"cont-hermes-eval","listener_version":"3.18.0"}'
   ```

## Eval run (per model_eval/README.md)

```bash
python -m model_eval run --config arms/baseline-qwen3.8.json --run-dir runs/baseline-qwen3.8
python -m model_eval run --config arms/inkling.json         --run-dir runs/inkling
```

Three frozen fixtures (`csv-rollup`, `repair-windows`, `evidence-audit`), three
attempts, interleaved A→B / B→A so each model sees the same order-confound
(thermal / cache / memory pressure) — a model-comparison control, not a queue
test. Per-job manifest + results recorded against the 11 Nov `8/13` baseline.

## Roster prerequisite

The candidate `inkling-2-mini:26b-a5b` is on the gateway roster
(`ordinary-queue.json`, sealed-deployed under `wafer-services`, ticket #135 /
#1137 — live as of v2). The baseline `qwen3.8:27b-mlx` is already on the roster.
Both arms are queue-routed; a submitted arm waits for usable memory, then loads
and runs.
