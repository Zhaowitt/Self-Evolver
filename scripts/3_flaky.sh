#!/usr/bin/env bash
# Stage 3 (optional, costly): grade gold patches FLAKY_REPEATS times over the GRPO pool;
# instances whose official outcome varies or never resolves go to FLAKY_IDS (excluded by stage 5).
# Knobs: FLAKY_IDS, FLAKY_REPEATS, FLAKY_WORKERS, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -s "${FLAKY_IDS}" ]; then
    info "[flaky] reusing ${FLAKY_IDS}"
    exit 0
fi
require_python
require_container_engine
POOL="${SPLITS_DIR}/grpo_pool_prompts.jsonl"
"${PYTHON}" -m src.rl.easyr1_dataset --output "${POOL}" \
    --dataset full --split test --stage eval \
    --exclude-ids "${SPLITS_DIR}/train_exclude_ids.txt"
"${PYTHON}" "${SCRIPTS_DIR}/tools/detect_flaky_instances.py" \
    --jsonl "${POOL}" --dataset full --split test \
    --repeats "${FLAKY_REPEATS:-2}" --workers "${FLAKY_WORKERS:-4}" \
    --engine "${TEST_BACKEND}" --output "${FLAKY_IDS}"
