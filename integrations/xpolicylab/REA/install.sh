#!/bin/bash
# Complete XPolicyLab/policy/REA as an XPolicyLab policy package and prepare the current (policy-side) environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# XPolicyLab's evaluation launchers are policy-agnostic (they read the policy name from their directory); reuse its own.
for name in eval.sh setup_eval_policy_server.sh setup_eval_env_client.sh deploy.py; do
    if [ ! -e "${SCRIPT_DIR}/${name}" ]; then
        cp "${XPL_ROOT}/policy/demo_policy/${name}" "${SCRIPT_DIR}/${name}"
        echo "[REA] copied ${name} from demo_policy"
    fi
done

python -m pip install -e "${XPL_ROOT}"
python -c "import numpy, PIL" 2>/dev/null || python -m pip install numpy pillow

NODE="${REA_NODE:-$(command -v node || true)}"
if [ -z "${NODE}" ] || ! "${NODE}" -e 'const [a, b] = process.versions.node.split(".").map(Number); process.exit(a > 22 || (a === 22 && b >= 19) ? 0 : 1)'; then
    echo "[REA] warning: REA needs Node.js 22.19 or newer on PATH (or REA_NODE) in this environment" >&2
fi
echo "[REA] set rea_root and rea_config in ${SCRIPT_DIR}/deploy.yml before evaluating"
