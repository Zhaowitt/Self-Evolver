#!/usr/bin/env bash
# Single-agent zero-shot baseline: one LLM call per issue; no skills, memory or
# evolution.
# Knobs: DATASET, SPLIT, NUM, RUN_DIR, TRAIN_IDS, SEED, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DATASET="${DATASET:-verified}"
SPLIT="${SPLIT:-test}"
NUM="${NUM:-}"
TRAIN_IDS="${TRAIN_IDS:-}"
RUN_DIR="${RUN_DIR:-${RUNS_DIR}/zero_shot-seed${SEED}}"

require_python
require_data
require_container_engine
require_api_key

run_benchmark "${RUN_DIR}" \
    --dataset "${DATASET}" --split "${SPLIT}" --stage eval --phase generate \
    --agent-mode single --skills off --memory off --task-evolution off --controller-mode off \
    --run-id "zero_shot-seed${SEED}" \
    ${NUM:+--num-instances "${NUM}"} \
    ${TRAIN_IDS:+--train-ids "${TRAIN_IDS}"}

info "Zero-shot run complete: ${RUN_DIR}"
