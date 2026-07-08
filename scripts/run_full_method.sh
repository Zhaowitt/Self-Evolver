#!/usr/bin/env bash
# Full method: task evolution and skill evolution together, driven by the shared
# hard-case buffer and Reflector. Train stage samples an evolving TaskPool and
# mines hard-case clusters into utility-gated skill updates; the evolved bank is
# then frozen for a held-out eval stage.
#
# Knobs: TRAIN_DATASET/TRAIN_SPLIT, NUM_ROLLOUTS, EVAL_DATASET/EVAL_SPLIT,
#        NUM_EVAL, VALIDATE_SKILLS, SEED, TEST_BACKEND, RESET_SKILLS (0/1),
#        RUN_EVAL (0/1), TRAIN_IDS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRAIN_DATASET="${TRAIN_DATASET:-full}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
EVAL_DATASET="${EVAL_DATASET:-verified}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_EVAL="${NUM_EVAL:-}"
VALIDATE_SKILLS="${VALIDATE_SKILLS:-0}"
RESET_SKILLS="${RESET_SKILLS:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
TRAIN_IDS="${TRAIN_IDS:-}"
TRAIN_RUN_DIR="${RUNS_DIR}/full_method-train-seed${SEED}"
EVAL_RUN_DIR="${RUNS_DIR}/full_method-seed${SEED}"

require_python
require_data
require_container_engine
require_api_key

[ "${RESET_SKILLS}" = "1" ] && reset_skill_bank

info "Train stage: ${NUM_ROLLOUTS} rollouts with task+skill evolution over ${TRAIN_DATASET}[${TRAIN_SPLIT}]"
run_benchmark "${TRAIN_RUN_DIR}" \
    --dataset "${TRAIN_DATASET}" --split "${TRAIN_SPLIT}" --stage train --phase generate \
    --agent-mode mas --skills evolve --memory on --task-evolution on --controller-mode off \
    --num-instances "${NUM_ROLLOUTS}" --run-id "full_method-train-seed${SEED}" \
    ${VALIDATE_SKILLS:+--validate-skills "${VALIDATE_SKILLS}"}

if [ "${RUN_EVAL}" = "1" ]; then
    info "Frozen eval stage over ${EVAL_DATASET}[${EVAL_SPLIT}] (evolved bank snapshot)"
    run_benchmark "${EVAL_RUN_DIR}" \
        --dataset "${EVAL_DATASET}" --split "${EVAL_SPLIT}" --stage eval --phase generate \
        --agent-mode mas --skills static --memory off --task-evolution off --controller-mode off \
        --run-id "full_method-seed${SEED}" \
        ${NUM_EVAL:+--num-instances "${NUM_EVAL}"} \
        ${TRAIN_IDS:+--train-ids "${TRAIN_IDS}"}
fi

info "Full-method experiment complete (train: ${TRAIN_RUN_DIR}, eval: ${EVAL_RUN_DIR})"
