#!/usr/bin/env bash
# Static multi-agent baseline: the fixed Inspector -> PatchGenerator -> Verifier
# -> Judge repair loop over a fixed seed skill bank, with no memory writes and no
# task or skill evolution. Grades in-loop with official container semantics.
#
# Knobs: DATASET, SPLIT, NUM (instances; default all), SEED, TEST_BACKEND,
#        SKILLS (static|off), TRAIN_IDS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET="${DATASET:-verified}"
SPLIT="${SPLIT:-test}"
NUM="${NUM:-}"
SKILLS="${SKILLS:-static}"
TRAIN_IDS="${TRAIN_IDS:-}"
RUN_DIR="${RUN_DIR:-${RUNS_DIR}/mas_static-seed${SEED}}"

require_python
require_data
require_container_engine
require_api_key

run_benchmark "${RUN_DIR}" \
    --dataset "${DATASET}" --split "${SPLIT}" --stage eval --phase generate \
    --agent-mode mas --skills "${SKILLS}" --memory off --task-evolution off --controller-mode off \
    --run-id "mas_static-seed${SEED}" \
    ${NUM:+--num-instances "${NUM}"} \
    ${TRAIN_IDS:+--train-ids "${TRAIN_IDS}"}

info "Static multi-agent run complete: ${RUN_DIR}"
