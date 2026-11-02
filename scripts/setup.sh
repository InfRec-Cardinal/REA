#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

NODE="${REA_NODE:-$(command -v node || true)}"
if [ -z "${REA_NODE:-}" ] && [ -x runtime/node/bin/node ]; then NODE="$HERE/runtime/node/bin/node"; fi
[ -x "$NODE" ] || { echo "no Node >= 22.19 found; install one or link runtime/node"; exit 1; }
"$NODE" -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 19) ? 0 : 1)' \
  || { echo "Node >= 22.19.0 is required; found $("$NODE" --version)"; exit 1; }
echo "node: $("$NODE" --version)"

REA_NODE="$NODE" bash "$HERE/scripts/setup-prime.sh"
PRIME="$HERE/runtime/prime-agent/node_modules/.bin/prime-agent"

[ -f .env ] || { cp env.example .env; chmod 600 .env; echo "created .env — set REA_API_KEY in it"; }

echo "running tests..."
PATH="$(dirname "$NODE"):$PATH" PRIME_AGENT_BIN="$PRIME" "$NODE" --test test/*.test.mjs 2>&1 | tail -8

cat <<EOF

Next (G1 service; see docs/robots/unitree-g1.md):
  1. set REA_API_KEY, REA_BASE_URL and REA_MODEL in $HERE/.env
  2. confirm the RealSense socket configured in .env is available
  3. $NODE server.mjs        # inference service; actuation stays off until G1_EXECUTE_ENABLED=1
EOF
