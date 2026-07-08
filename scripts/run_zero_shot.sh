#!/usr/bin/env bash
# Single-agent static baseline: one LLM call over the issue and the top files by
# lexical overlap produces a patch. No worker roles, no skills, no memory, no
# evolution. Grades in-loop with official container semantics.
#
# Knobs: DATASET, SPLIT, NUM (instances; default all), SEED, TEST_BACKEND,
#        TRAIN_IDS (ids to exclude as a contamination guard).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

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
