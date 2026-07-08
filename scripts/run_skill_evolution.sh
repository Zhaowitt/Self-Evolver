#!/usr/bin/env bash
# Skill-evolution baseline. Train stage runs the repair loop while the Reflector
# mines hard-case clusters into skill create/refine/deprecate proposals, gated by
# execution utility (and optional held-out validation). The evolved skill bank is
# then snapshotted and frozen for a held-out eval stage.
#
# Knobs: TRAIN_DATASET/TRAIN_SPLIT, NUM_ROLLOUTS, EVAL_DATASET/EVAL_SPLIT,
#        NUM_EVAL, VALIDATE_SKILLS (held-out replay count), SEED, TEST_BACKEND,
#        RESET_SKILLS (0/1), RUN_EVAL (0/1), TRAIN_IDS.
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
TRAIN_RUN_DIR="${RUNS_DIR}/skill_evolution-train-seed${SEED}"
EVAL_RUN_DIR="${RUNS_DIR}/skill_evolution-seed${SEED}"

require_python
require_data
require_container_engine
require_api_key

[ "${RESET_SKILLS}" = "1" ] && reset_skill_bank

info "Train stage: ${NUM_ROLLOUTS} rollouts with skill evolution over ${TRAIN_DATASET}[${TRAIN_SPLIT}]"
run_benchmark "${TRAIN_RUN_DIR}" \
    --dataset "${TRAIN_DATASET}" --split "${TRAIN_SPLIT}" --stage train --phase generate \
    --agent-mode mas --skills evolve --memory on --task-evolution off --controller-mode off \
    --num-instances "${NUM_ROLLOUTS}" --run-id "skill_evolution-train-seed${SEED}" \
    ${VALIDATE_SKILLS:+--validate-skills "${VALIDATE_SKILLS}"}

if [ "${RUN_EVAL}" = "1" ]; then
    info "Frozen eval stage over ${EVAL_DATASET}[${EVAL_SPLIT}] (evolved bank snapshot)"
    run_benchmark "${EVAL_RUN_DIR}" \
        --dataset "${EVAL_DATASET}" --split "${EVAL_SPLIT}" --stage eval --phase generate \
        --agent-mode mas --skills static --memory off --task-evolution off --controller-mode off \
        --run-id "skill_evolution-seed${SEED}" \
        ${NUM_EVAL:+--num-instances "${NUM_EVAL}"} \
        ${TRAIN_IDS:+--train-ids "${TRAIN_IDS}"}
fi

info "Skill-evolution experiment complete (train: ${TRAIN_RUN_DIR}, eval: ${EVAL_RUN_DIR})"
