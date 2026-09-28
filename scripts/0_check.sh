#!/usr/bin/env bash
# Stage 0: check the environment (python, data, container engine, API key, EasyR1, Qwen3.5).
# Knobs: EASYR1_DIR, CONTROLLER_MODEL_PATH, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_python
require_data
require_container_engine
require_api_key
[ -n "${EASYR1_DIR:-}" ] && [ -d "${EASYR1_DIR}" ] || die "Set EASYR1_DIR to an EasyR1 checkout."
"${PYTHON}" "${SCRIPTS_DIR}/tools/preflight_qwen35.py" \
    --config "${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml" \
    ${CONTROLLER_MODEL_PATH:+--model "${CONTROLLER_MODEL_PATH}"}
info "[check] ok"
