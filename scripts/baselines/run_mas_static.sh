#!/usr/bin/env bash
# Static multi-agent baseline: fixed repair loop over the seed skill bank; no
# memory writes or evolution.
# Knobs: DATASET, SPLIT, NUM, SKILLS (static|off), SKILLS_DIR, RUN_DIR,
#        TRAIN_IDS, SEED, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

DATASET="${DATASET:-verified}"
SPLIT="${SPLIT:-test}"
NUM="${NUM:-}"
SKILLS="${SKILLS:-static}"
TRAIN_IDS="${TRAIN_IDS:-}"
RUN_DIR="${RUN_DIR:-${RUNS_DIR}/mas_static-seed${SEED}}"
# The static baseline uses the pristine seed skills, whatever ./skills holds.
export SELF_EVOLVER_SKILLS_DIR="${SKILLS_DIR:-${SEED_SKILLS_DIR}}"

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
