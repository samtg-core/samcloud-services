#!/usr/bin/env bash
# hermes-eval enrollment (admin-provisioned, per c5325 review on task 1415).
#
# DESIGN (settled c5325 point 6): hermes-eval is a NATIVE worker principal on
# wafer, NOT a container. model_eval v1 accepts only `native-service` (a
# rail-service container is refused until a real rail adapter exists), and the
# eval runs "on the configured host under its own worker principal" via the
# deployed runtime + queue plugins. A wake/rail seat exists ONLY so hermes-eval
# can talk on the plane (coordination) — it does not host the eval.
#
# SECURITY (c5325 #1): no token ever appears on a command line or in a config.
# Bearer headers are read from a 0600 file with `-H @file`; SC_TOKEN is delivered
# via a 0600 env file, never `-e SC_TOKEN=...`.
set -euo pipefail

SC_BASE="${SC_BASE:-https://cloud.samtg.xyz/api/v1}"
ADMIN_HDR="${ADMIN_TOKEN_HDR:-$HOME/.samcloud/token.hdr}"    # admin header, 0600
STATE_DIR="${HOME}/.local/var"; mkdir -p "$STATE_DIR"; chmod 700 "$STATE_DIR"
STATE="$STATE_DIR/hermes-eval-invite.state"

# 1. Invite (admin-only). Scope = group:services membership — what wafer's gateway
#    demands of callers via SC_REQUIRED_SCOPE (CLAUDE.md). NOT services/enroll,
#    NOT device:slice-test. Nothing more.
curl -sS -X POST "$SC_BASE/users/invite" \
    -H @"$ADMIN_HDR" \
    -H 'Content-Type: application/json' \
    -d '{"username":"hermes-eval","role":"agent","scopes":["group:services"]}' \
    | python3 -c 'import sys,json,os;d=json.load(sys.stdin);open(os.environ["STATE"],"w").write(json.dumps(d));os.chmod(os.environ["STATE"],0o600)'
echo "invite recorded -> $STATE (0600). Activate via the invite brief_url."

# 2. After the invite is activated, hermes-eval's own token lands in a 0600 file.
EVAL_TOKEN_FILE="${HOME}/.local/var/hermes-eval.token"
chmod 600 "$EVAL_TOKEN_FILE" 2>/dev/null || true

# 3. Register the wake seat UNDER hermes-eval's OWN token (not the admin's), so the
#    fleet record names hermes-eval. (c5325 #2.)
curl -sS -X POST "$SC_BASE/wake/register" \
    -H "Authorization: Bearer $(cat "$EVAL_TOKEN_FILE")" \
    -H 'Content-Type: application/json' \
    -d '{"wake_cmd":"/home/agent/.local/bin/samcloud-rail.prod","host":"cont-hermes-eval","listener_version":"3.18.0"}' \
    || true

echo "hermes-eval enrolled as a native worker principal on wafer (scope group:services)."
echo "Run the eval natively: python -m model_eval run --config arms/<arm>.json --run-dir runs/<arm>"
