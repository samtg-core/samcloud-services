# hermes-eval — dedicated satellite agent for model trials

A dedicated SAMcloud satellite agent that runs `model_eval` model trials
against the gateway, **isolated from every other agent's inference provider**.
This is the seat created per admin directive (ticket #1139 / c1139): evals and
model trials run here so hermes-samcloud's own provider
(`qwen3.8:27b-mlx` / `samcloud-queue-stint`) is never swapped or disturbed.

Pattern: `SPEC-satellite-agents.md` (container + SAMcloud identity + consume the
gateway via HTTP). Hermes-eval is a sibling of hermes-samcloud on the same
wake/rail stack.

## Why a separate seat

Running a trial needs the gateway to serve the candidate model
(`inkling-2-mini:26b-a5b`) alongside the baseline (`qwen3.8:27b-mlx`). Doing
that on hermes-samcloud's own seat would mean swapping its inference provider —
explicitly forbidden (c1139: "don't switch your own inference provider"). A
dedicated seat keeps the eval agent's `OPENAI_BASE_URL` pointed at the gateway
(`:8800`) with its own provider block, so hermes-samcloud's provider stays fixed.

## Seat definition

| Field | Value |
|-------|-------|
| Identity | `hermes-eval` (SAMcloud agent user, `role: agent`) |
| Container | `cont-hermes-eval` |
| Rail | `samcloud-rail.prod` (helper rail 3.18.0, image `ghcr.io/samcloud/hermes-rail:3.18.0`) |
| Provider | gateway `http://127.0.0.1:8800/v1` (own block; NOT hermes-samcloud's) |
| Models served | `qwen3.8:26b-mlx` (baseline) + `inkling-2-mini:26b-a5b` (candidate) |
| Envelope | `native-service` |
| Eval harness | `model_eval` (`python -m model_eval run --config <arm>.json --run-dir runs/<arm>`) |

## Scope

Eval only. No chat, no delivery, no operator session. A clean workspace and a
fresh profile; no copy of any agent's session / memory / skills / auth / cron.
The manifest binds model + digest + context window + hardware + revisions +
identity so runs are reproducible and comparable.

## Enrollment (admin-provisioned, per SPEC §4)

```bash
curl -X POST "$SC_BASE/users/invite" \
  -H "Authorization: Bearer $SC_ADMIN" \
  -H "Content-Type: application/json" \
  -d '{"username":"hermes-eval","role":"agent","scopes":["device:slice-test","services/enroll"]}'
```

## Container (per SPEC §3, §9)

```bash
docker create --name cont-hermes-eval \
  -e SC_TOKEN=$SC_HERMES_EVAL \
  -e OPENAI_BASE_URL=http://127.0.0.1:8800/v1 \
  -e OPENAI_API_KEY=*** \
  samcloud/hermes-rail:3.18.0
```

## Eval run (per model_eval/README.md)

```bash
python -m model_eval run --config arms/baseline-qwen3.8.json --run-dir runs/baseline-qwen3.8
python -m model_eval run --config arms/inkling.json        --run-dir runs/inkling
```

Three frozen fixtures (`csv-rollup`, `repair-windows`, `evidence-audit`), three
attempts, interleaved A→B / B→A so each model sees the same order-confound
(thermal / cache / memory pressure) — a model-comparison control, not a queue
test. Per-job manifest + results recorded against the 11 Nov `8/13` baseline.

## Roster prerequisite

The candidate `inkling-2-mini:26b-a5b` must be on the gateway roster
(`ordinary-queue.json`, sealed-deployed under `wafer-services`, ticket #135).
The baseline `qwen3.8:27b-mlx` is already on the roster.
