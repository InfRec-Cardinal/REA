#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${REA_NODE:-$(command -v node || true)}"
if [ -z "${REA_NODE:-}" ] && [ -x "$HERE/runtime/node/bin/node" ]; then NODE="$HERE/runtime/node/bin/node"; fi
[ -x "$NODE" ] || { echo "Node >= 22.19.0 is required"; exit 1; }
"$NODE" -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 19) ? 0 : 1)' \
  || { echo "Node >= 22.19.0 is required; found $("$NODE" --version)"; exit 1; }
SOURCE="${PRIME_AGENT_PACKAGE:-https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v0.8.0/prime-agent-0.8.0.tgz}"
PREFIX="$HERE/runtime/prime-agent"
PRIME="$PREFIX/node_modules/.bin/prime-agent"
if [ ! -x "$PRIME" ]; then
  NPM="${REA_NPM:-$(dirname "$NODE")/npm}"
  [ -x "$NPM" ] || NPM="$(command -v npm || true)"
  [ -x "$NPM" ] || { echo "npm is required to install Prime Agent"; exit 1; }
  mkdir -p "$PREFIX"
  PATH="$(dirname "$NODE"):$PATH" "$NPM" --prefix "$PREFIX" install --no-save --package-lock=false --omit=dev "$SOURCE"
fi
[ -x "$PRIME" ] || { echo "Prime Agent installation did not provide $PRIME"; exit 1; }
set +e
VERSION_OUTPUT=$(PATH="$(dirname "$NODE"):$PATH" "$PRIME" --version 2>&1)
VERSION_STATUS=$?
set -e
INSTALLED_VERSION=$(printf '%s\n' "$VERSION_OUTPUT" | sed -nE '/^[[:space:]]*[0-9]+\.[0-9]+\.[0-9]+[[:space:]]*$/p' | tail -n 1 | tr -d '[:space:]')
[ "$VERSION_STATUS" -eq 0 ] && [ "$INSTALLED_VERSION" = "0.8.0" ] || { echo "Prime Agent 0.8.0 required; found ${INSTALLED_VERSION:-unavailable}"; exit 1; }
echo "prime-agent: $INSTALLED_VERSION"
