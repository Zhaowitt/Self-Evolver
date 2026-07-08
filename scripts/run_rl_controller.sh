#!/usr/bin/env bash
# Controller RL pipeline. The high-level Controller (task/skill/budget cues) is
# trained with EasyR1 GRPO against the online execution reward, then served and
# evaluated frozen. EasyR1 is an external trainer (not vendored), so this script
# does the in-repo parts for real -- build the prompt datasets and run the frozen
# eval -- and drives the GRPO launch only when EASYR1_DIR points at a checkout.
#
# Knobs: TRAIN_DATASET/TRAIN_SPLIT, NUM_TRAIN, VAL_DATASET/VAL_SPLIT, NUM_VAL,
#        EXCLUDE_IDS, EASYR1_DIR, EASYR1_CONFIG, CONTROLLER_BASE_URL,
#        RUN_EVAL (0/1), EVAL_DATASET/EVAL_SPLIT, NUM_EVAL, SEED, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRAIN_DATASET="${TRAIN_DATASET:-full}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
NUM_TRAIN="${NUM_TRAIN:-}"
VAL_DATASET="${VAL_DATASET:-lite}"
VAL_SPLIT="${VAL_SPLIT:-dev}"
NUM_VAL="${NUM_VAL:-}"
EXCLUDE_IDS="${EXCLUDE_IDS:-${DATA_DIR}/eval_ids.txt}"
EASYR1_DIR="${EASYR1_DIR:-}"
EASYR1_CONFIG="${EASYR1_CONFIG:-${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_DATASET="${EVAL_DATASET:-verified}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_EVAL="${NUM_EVAL:-}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/controller_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/controller_val.jsonl}"

require_python
require_data
mkdir -p "${DATA_DIR}"

# 1) Build the Controller prompt datasets (real, offline; no LLM call).
EXCLUDE_ARG=""
if [ -f "${EXCLUDE_IDS}" ]; then
    EXCLUDE_ARG="--exclude-ids ${EXCLUDE_IDS}"
    info "Excluding eval ids in ${EXCLUDE_IDS} from the training prompts"
fi
info "Building training prompts -> ${TRAIN_JSONL}"
# shellcheck disable=SC2086
"${PYTHON}" -m src.rl.easyr1_dataset \
    --output "${TRAIN_JSONL}" \
    --dataset "${TRAIN_DATASET}" --split "${TRAIN_SPLIT}" --stage train \
    ${NUM_TRAIN:+--num-instances "${NUM_TRAIN}"} \
    ${EXCLUDE_ARG}
info "Building validation prompts -> ${VAL_JSONL}"
"${PYTHON}" -m src.rl.easyr1_dataset \
    --output "${VAL_JSONL}" \
    --dataset "${VAL_DATASET}" --split "${VAL_SPLIT}" --stage eval \
    ${NUM_VAL:+--num-instances "${NUM_VAL}"}

# 2) GRPO training with EasyR1. The online reward runs a full repair episode plus
#    one container eval per policy sample, so the container engine and an API key
#    for the worker LLM must be available on the trainer host.
REWARD_FN="src/reward/online_reward.py:compute_score"
cat <<EOF
--------------------------------------------------------------------------------
EasyR1 GRPO launch (run from an EasyR1 checkout; not vendored here):

  export SELF_EVOLVER_TEST_BACKEND=${TEST_BACKEND}
  export SELF_EVOLVER_REWARD_CONFIG=${REPO_ROOT}/configs/reward_config.yaml
  # Serve the controller policy (vLLM) so rollouts can query it:
  #   python -m vllm.entrypoints.openai.api_server --model <controller-ckpt> --port 8000
  # Point EasyR1 at:
  #   data.train_files = ${TRAIN_JSONL}
  #   data.val_files   = ${VAL_JSONL}
  #   worker.reward.reward_function = ${REPO_ROOT}/${REWARD_FN}
  # Example config: ${EASYR1_CONFIG}
--------------------------------------------------------------------------------
EOF

if [ -n "${EASYR1_DIR}" ]; then
    [ -d "${EASYR1_DIR}" ] || die "EASYR1_DIR='${EASYR1_DIR}' does not exist."
    require_container_engine
    require_api_key
    export SELF_EVOLVER_TEST_BACKEND="${TEST_BACKEND}"
    export SELF_EVOLVER_REWARD_CONFIG="${REPO_ROOT}/configs/reward_config.yaml"
    info "Launching EasyR1 GRPO from ${EASYR1_DIR}"
    ( cd "${EASYR1_DIR}" && "${PYTHON}" -m verl.trainer.main \
        config="${EASYR1_CONFIG}" \
        data.train_files="${TRAIN_JSONL}" \
        data.val_files="${VAL_JSONL}" \
        worker.reward.reward_function="${REPO_ROOT}/${REWARD_FN}" )
else
    info "EASYR1_DIR not set: datasets built; skipping the GRPO launch."
fi

# 3) Frozen evaluation with the (served) controller policy.
if [ "${RUN_EVAL}" = "1" ]; then
    require_container_engine
    require_api_key
    [ -n "${CONTROLLER_BASE_URL:-}" ] || warn \
        "CONTROLLER_BASE_URL is not set; --controller-mode llm will use the OPENAI_* endpoint."
    EVAL_RUN_DIR="${RUNS_DIR}/rl_controller-seed${SEED}"
    info "Frozen eval with the LLM controller over ${EVAL_DATASET}[${EVAL_SPLIT}]"
    run_benchmark "${EVAL_RUN_DIR}" \
        --dataset "${EVAL_DATASET}" --split "${EVAL_SPLIT}" --stage eval --phase generate \
        --agent-mode mas --skills static --memory off --task-evolution off --controller-mode llm \
        --run-id "rl_controller-seed${SEED}" \
        ${NUM_EVAL:+--num-instances "${NUM_EVAL}"}
    info "Controller eval complete: ${EVAL_RUN_DIR}"
fi
