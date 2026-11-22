#!/usr/bin/env bash
# hermes-eval enrollment (admin-provisioned, per SPEC-satellite-agents.md §4, §9).
# Run as admin. Mints the hermes-eval identity + token, registers the wake seat,
# and creates the cont-hermes-eval container.
set -euo pipefail

SC_BASE="${SC_BASE:-https://cloud.samtg.xyz/api/v1}"
SC_ADMIN="${SC_ADMIN:-}"   # admin token — never logged
AGENT="hermes-eval"

# 1. Provision the agent user (admin one-time).
INVITE=$(curl -sS -X POST "$SC_BASE/users/invite" \
  -H "Authorization: Bearer $SC_ADMIN" \
  -H "Content-Type: application/json" \
  -d '{"username":"hermes-eval","role":"agent","scopes":["device:slice-test","services/enroll"]}')
echo "invite: $INVITE"

# 2. Register the wake seat (any auth).
curl -sS -X POST "$SC_BASE/wake/register" \
  -H "Authorization: Bearer $SC_ADMIN" \
  -H "Content-Type: application/json" \
  -d "{\"wake_cmd\":\"/home/agent/.local/bin/samcloud-rail.prod\",\"host\":\"cont-hermes-eval\",\"listener_version\":\"3.18.0\"}"
echo

# 3. Create the container (per §9).
docker create --name cont-hermes-eval \
  -e SC_TOKEN="$SC_HERMES_EVAL" \
  -e OPENAI_BASE_URL="http://127.0.0.1:8800/v1" \
  -e OPENAI_API_KEY=*** \
  -e LLM_MODEL="qwen3.8:27b-mlx" \
  -e AGENT_NAME="hermes-eval" \
  --cap-drop=ALL --cap-add=DAC_OVERRIDE --cap-add=CHOWN --cap-add=FOWNER \
  --security-opt=no-new-privileges --pids-limit=256 \
  samcloud/hermes-rail:3.18.0

echo "hermes-eval enrolled. Start: docker start cont-hermes-eval"
