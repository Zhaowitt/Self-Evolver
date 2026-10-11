#!/usr/bin/env bash
# Controller RL pipeline. The Controller selects grounded skills and an optional
# short strategy; the worker iteration cap is system-owned (maximum three).
# trained with EasyR1 GRPO against the online execution reward, then served and
# evaluated frozen. EasyR1 is an external trainer (not vendored), so this script
# does the in-repo parts for real -- build the prompt datasets and run the frozen
# eval -- and drives the GRPO launch only when EASYR1_DIR points at a checkout.
#
# Knobs: TRAIN_DATASET/TRAIN_SPLIT, NUM_TRAIN, VAL_DATASET/VAL_SPLIT, NUM_VAL,
#        EXCLUDE_IDS, EASYR1_DIR, EASYR1_CONFIG, CONTROLLER_MODEL_PATH,
#        CONTROLLER_BASE_URL, QWEN35_PREFLIGHT,
#        REWARD_WORKERS, REBUILD_DATASETS (0/1),
#        HARD_CASE_START_OFFSET (byte offset for round reflection),
#        TASK_SIGNALS_REPORT (previous-round reflection report),
#        TASK_POOL_UNIFORM_FRACTION, MAX_INSTANCE_REPEATS,
#        TASK_POOL_STATE, RESET_TASK_POOL (0/1),
#        EASYR1_EXPERIMENT_NAME (optional fresh checkpoint namespace),
#        EASYR1_SAVE_CHECKPOINT_PATH (optional isolated checkpoint root),
#        EASYR1_LOAD_CHECKPOINT_PATH (optional prior-round global_step_*),
#        EASYR1_MAX_STEPS (optional cumulative trainer step target),
#        EASYR1_VAL_FREQ/EASYR1_VAL_BATCH_SIZE (optional validation overrides),
#        EASYR1_GPU_MEMORY_UTILIZATION/EASYR1_NUM_GPUS/
#        EASYR1_TENSOR_PARALLEL_SIZE (optional hardware overrides),
#        ENABLE_SKILL_REFLECTION (0/1), REFLECT_MIN_CLUSTER_SIZE, ROUND_ID,
#        RUN_EVAL (0/1), EVAL_DATASET/EVAL_SPLIT, NUM_EVAL, SEED, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRAIN_DATASET="${TRAIN_DATASET:-full}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
NUM_TRAIN="${NUM_TRAIN:-}"
VAL_DATASET="${VAL_DATASET:-full}"
VAL_SPLIT="${VAL_SPLIT:-dev}"
NUM_VAL="${NUM_VAL:-}"
EXCLUDE_IDS="${EXCLUDE_IDS:-${DATA_DIR}/eval_ids.txt}"
EASYR1_DIR="${EASYR1_DIR:-}"
EASYR1_CONFIG="${EASYR1_CONFIG:-${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml}"
CONTROLLER_MODEL_PATH="${CONTROLLER_MODEL_PATH:-}"
QWEN35_PREFLIGHT="${QWEN35_PREFLIGHT:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_DATASET="${EVAL_DATASET:-verified}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_EVAL="${NUM_EVAL:-}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/controller_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/controller_val.jsonl}"
ENABLE_SKILL_REFLECTION="${ENABLE_SKILL_REFLECTION:-1}"
ENABLE_SKILL_CREDIT="${ENABLE_SKILL_CREDIT:-1}"
REWARD_WORKERS="${REWARD_WORKERS:-4}"
REBUILD_DATASETS="${REBUILD_DATASETS:-1}"
REFLECT_MIN_CLUSTER_SIZE="${REFLECT_MIN_CLUSTER_SIZE:-3}"
ROUND_ID="${ROUND_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export SELF_EVOLVER_ROUND_ID="${ROUND_ID}"
SKILL_REFLECTION_DIR="${SKILL_REFLECTION_DIR:-${RUNS_DIR}/skill_reflections}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_ROOT}}"
HARD_CASES_PATH="${HARD_CASES_PATH:-${WORKSPACE_DIR}/hard_cases.jsonl}"
# The online worker, prompt builder, and post-round Reflector must share one
# canonical buffer. Keep checkout placement independent via
# SELF_EVOLVER_REWARD_WORKSPACE.
export SELF_EVOLVER_HARD_CASES_PATH="${HARD_CASES_PATH}"
TASK_SIGNALS_REPORT="${TASK_SIGNALS_REPORT:-}"
TASK_POOL_STATE="${TASK_POOL_STATE:-${DATA_DIR}/controller_task_pool.json}"
TASK_POOL_UNIFORM_FRACTION="${TASK_POOL_UNIFORM_FRACTION:-0.5}"
MAX_INSTANCE_REPEATS="${MAX_INSTANCE_REPEATS:-3}"
RESET_TASK_POOL="${RESET_TASK_POOL:-0}"
SKILL_SNAPSHOT_DIR="${SKILL_SNAPSHOT_DIR:-${RUNS_DIR}/skill_bank_at_start}"
EASYR1_EXPERIMENT_NAME="${EASYR1_EXPERIMENT_NAME:-}"
EASYR1_SAVE_CHECKPOINT_PATH="${EASYR1_SAVE_CHECKPOINT_PATH:-}"
EASYR1_LOAD_CHECKPOINT_PATH="${EASYR1_LOAD_CHECKPOINT_PATH:-}"
EASYR1_MAX_STEPS="${EASYR1_MAX_STEPS:-}"
EASYR1_VAL_FREQ="${EASYR1_VAL_FREQ:-}"
EASYR1_VAL_BATCH_SIZE="${EASYR1_VAL_BATCH_SIZE:-}"
EASYR1_GPU_MEMORY_UTILIZATION="${EASYR1_GPU_MEMORY_UTILIZATION:-}"
EASYR1_NUM_GPUS="${EASYR1_NUM_GPUS:-}"
EASYR1_TENSOR_PARALLEL_SIZE="${EASYR1_TENSOR_PARALLEL_SIZE:-}"

require_python
require_data
mkdir -p "${DATA_DIR}"

# Freeze an auditable bank snapshot once per round. Checkpoint resumes reuse it
# instead of overwriting the round's original skill boundary.
if [ ! -e "${SKILL_SNAPSHOT_DIR}/.complete" ]; then
    mkdir -p "${SKILL_SNAPSHOT_DIR}"
    cp -a "${REPO_ROOT}/skills/." "${SKILL_SNAPSHOT_DIR}/"
    touch "${SKILL_SNAPSHOT_DIR}/.complete"
    info "Captured round-start skill bank -> ${SKILL_SNAPSHOT_DIR}"
fi

# 1) Build the Controller prompt datasets (real, offline; no LLM call).
EXCLUDE_ARG=""
EXCLUDE_ARGS=()
if [ -n "${SELF_EVOLVER_FIXED_VAL_DIR:-}" ]; then
    exclude_id_args EXCLUDE_ARGS "${VAL_INCLUDE_IDS}" required
    exclude_id_args EXCLUDE_ARGS "${SFT_IDS}" required
fi
if [ -f "${EXCLUDE_IDS}" ]; then
    EXCLUDE_ARG="--exclude-ids ${EXCLUDE_IDS}"
    info "Excluding eval ids in ${EXCLUDE_IDS} from the training prompts"
fi
if [ "${REBUILD_DATASETS}" = "1" ]; then
    if [ "${RESET_TASK_POOL}" = "1" ]; then
        rm -f "${TASK_POOL_STATE}" "${TASK_POOL_STATE}.ingestion.json"
        info "Reset TaskPool state at ${TASK_POOL_STATE}"
    fi
    info "Building training prompts -> ${TRAIN_JSONL}"
    TASK_SIGNAL_ARGS=()
    if [ -n "${TASK_SIGNALS_REPORT}" ]; then
        [ -s "${TASK_SIGNALS_REPORT}" ] || die \
            "TASK_SIGNALS_REPORT='${TASK_SIGNALS_REPORT}' is missing or empty."
        TASK_SIGNAL_ARGS=(
            --task-signals-report "${TASK_SIGNALS_REPORT}"
            --sampling-seed "${SEED}"
            --task-pool-uniform-fraction "${TASK_POOL_UNIFORM_FRACTION}"
            --max-instance-repeats "${MAX_INSTANCE_REPEATS}"
        )
        info "Applying previous-round TaskPool signals from ${TASK_SIGNALS_REPORT}"
    fi
    # shellcheck disable=SC2086
    "${PYTHON}" -m src.rl.easyr1_dataset \
        --output "${TRAIN_JSONL}" \
        --dataset "${TRAIN_DATASET}" --split "${TRAIN_SPLIT}" --stage train \
        --task-pool-state "${TASK_POOL_STATE}" \
        ${NUM_TRAIN:+--num-instances "${NUM_TRAIN}"} \
        "${TASK_SIGNAL_ARGS[@]}" \
        "${EXCLUDE_ARGS[@]}" \
        ${EXCLUDE_ARG}
    if [ -z "${SELF_EVOLVER_FIXED_VAL_DIR:-}" ]; then
        info "Building validation prompts -> ${VAL_JSONL}"
        "${PYTHON}" -m src.rl.easyr1_dataset \
            --output "${VAL_JSONL}" \
            --dataset "${VAL_DATASET}" --split "${VAL_SPLIT}" --stage eval \
            ${NUM_VAL:+--num-instances "${NUM_VAL}"}
    fi
else
    [ -s "${TRAIN_JSONL}" ] || die "REBUILD_DATASETS=0 but TRAIN_JSONL='${TRAIN_JSONL}' is missing or empty."
    if [ -z "${SELF_EVOLVER_FIXED_VAL_DIR:-}" ]; then
        [ -s "${VAL_JSONL}" ] || die "REBUILD_DATASETS=0 but VAL_JSONL='${VAL_JSONL}' is missing or empty."
    fi
    info "Reusing frozen Controller prompt datasets for checkpoint resume"
fi

if [ -n "${SELF_EVOLVER_FIXED_VAL_DIR:-}" ]; then
    [ "${VAL_DATASET}/${VAL_SPLIT}/${NUM_VAL}" = full/dev/225 ] || \
        die "Production validation must be Full/dev with NUM_VAL=225."
    "${PYTHON}" -m src.rl.fixed_validation \
        --source-dir "${SELF_EVOLVER_FIXED_VAL_DIR}" --state-root "${STATE_ROOT:-${RUNS_DIR}}" \
        --sft-manifest "${SELF_EVOLVER_SFT_MANIFEST}" \
        --output "${VAL_JSONL}" --train-jsonl "${TRAIN_JSONL}" \
        --skills-dir "${SKILL_SNAPSHOT_DIR}" --model "${CONTROLLER_MODEL_PATH}" \
        --config "${EASYR1_CONFIG}"
fi

if [ "${SELF_EVOLVER_REQUIRE_READY_IMAGES:-0}" = 1 ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/tools/check_training_images_ready.py" \
        --task-pool "${TASK_POOL_STATE}" --train-jsonl "${TRAIN_JSONL}" \
        --source-dir "${SELF_EVOLVER_FIXED_VAL_DIR}" --cache-dir "${SIF_CACHE_DIR}" \
        --health-report "${SELF_EVOLVER_VALIDATION_HEALTH_REPORT:-${SELF_EVOLVER_FIXED_VAL_DIR}/validation_health.json}"
fi

# 2) GRPO training with EasyR1. Each candidate patch uses the frozen F2P subset
#    gate; promising/final patches receive a full official F2P+P2P evaluation.
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
    if [ "${QWEN35_PREFLIGHT}" = "1" ]; then
        PREFLIGHT_ARGS=(--config "${EASYR1_CONFIG}")
        if [ -n "${CONTROLLER_MODEL_PATH}" ]; then
            PREFLIGHT_ARGS+=(--model "${CONTROLLER_MODEL_PATH}")
        fi
        info "Checking Controller model compatibility before EasyR1 launch"
        "${PYTHON}" "${REPO_ROOT}/scripts/preflight_qwen35.py" "${PREFLIGHT_ARGS[@]}"
    fi
    export SELF_EVOLVER_TEST_BACKEND="${TEST_BACKEND}"
    export SELF_EVOLVER_REWARD_CONFIG="${REPO_ROOT}/configs/reward_config.yaml"
    export SELF_EVOLVER_ROLLOUT_JSONL="${RUNS_DIR}/online_rollouts.jsonl"
    export SELF_EVOLVER_ENABLE_SKILL_EVOLUTION="${ENABLE_SKILL_CREDIT}"
    export SELF_EVOLVER_REWARD_WORKERS="${REWARD_WORKERS}"
    export WORKSPACE_DIR
    HARD_CASE_START_OFFSET="${HARD_CASE_START_OFFSET:-}"
    if [ -z "${HARD_CASE_START_OFFSET}" ]; then
        HARD_CASE_START_OFFSET=0
        if [ -f "${HARD_CASES_PATH}" ]; then
            HARD_CASE_START_OFFSET="$(wc -c < "${HARD_CASES_PATH}")"
        fi
    fi
    # Docker runs outside the training container; keep bind-mounted eval files
    # under the shared workspace while leaving the original backend unchanged.
    export TMPDIR="${TMPDIR:-${RUNS_DIR}}"
    mkdir -p "${TMPDIR}"
    info "Launching EasyR1 GRPO from ${EASYR1_DIR}"
    EASYR1_OVERRIDES=()
    if [ -n "${CONTROLLER_MODEL_PATH}" ]; then
        EASYR1_OVERRIDES+=(worker.actor.model.model_path="${CONTROLLER_MODEL_PATH}")
    fi
    if [ -n "${EASYR1_EXPERIMENT_NAME}" ]; then
        EASYR1_OVERRIDES+=(trainer.experiment_name="${EASYR1_EXPERIMENT_NAME}")
    fi
    if [ -n "${EASYR1_SAVE_CHECKPOINT_PATH}" ]; then
        EASYR1_OVERRIDES+=(trainer.save_checkpoint_path="${EASYR1_SAVE_CHECKPOINT_PATH}")
    fi
    if [ -n "${EASYR1_LOAD_CHECKPOINT_PATH}" ]; then
        EASYR1_OVERRIDES+=(trainer.load_checkpoint_path="${EASYR1_LOAD_CHECKPOINT_PATH}")
    fi
    if [ -n "${EASYR1_MAX_STEPS}" ]; then
        EASYR1_OVERRIDES+=(trainer.max_steps="${EASYR1_MAX_STEPS}")
    fi
    if [ -n "${EASYR1_VAL_FREQ}" ]; then
        EASYR1_OVERRIDES+=(trainer.val_freq="${EASYR1_VAL_FREQ}")
    fi
    if [ -n "${EASYR1_VAL_BATCH_SIZE}" ]; then
        EASYR1_OVERRIDES+=(data.val_batch_size="${EASYR1_VAL_BATCH_SIZE}")
    fi
    if [ -n "${EASYR1_GPU_MEMORY_UTILIZATION}" ]; then
        EASYR1_OVERRIDES+=(worker.rollout.gpu_memory_utilization="${EASYR1_GPU_MEMORY_UTILIZATION}")
    fi
    if [ -n "${EASYR1_NUM_GPUS}" ]; then
        EASYR1_OVERRIDES+=(trainer.n_gpus_per_node="${EASYR1_NUM_GPUS}")
    fi
    if [ -n "${EASYR1_TENSOR_PARALLEL_SIZE}" ]; then
        EASYR1_OVERRIDES+=(worker.rollout.tensor_parallel_size="${EASYR1_TENSOR_PARALLEL_SIZE}")
    fi
    ( cd "${EASYR1_DIR}" && "${PYTHON}" -m verl.trainer.main \
        config="${EASYR1_CONFIG}" \
        data.train_files="${TRAIN_JSONL}" \
        data.val_files="${VAL_JSONL}" \
        worker.reward.reward_function="${REPO_ROOT}/${REWARD_FN}" \
        "${EASYR1_OVERRIDES[@]}" )
    "${PYTHON}" -m src.rl.update_task_pool_after_grpo \
        --task-pool-state "${TASK_POOL_STATE}" \
        --train-jsonl "${TRAIN_JSONL}" \
        --rollouts "${SELF_EVOLVER_ROLLOUT_JSONL}" \
        --start-offset 0 \
        --round-id "${ROUND_ID}"
    if [ "${ENABLE_SKILL_REFLECTION}" = "1" ]; then
        REFLECTION_REPORT="${SKILL_REFLECTION_DIR}/${ROUND_ID}.json"
        info "Reflecting over GRPO round ${ROUND_ID} -> ${REFLECTION_REPORT}"
        "${PYTHON}" -m src.rl.reflect_after_grpo \
            --hard-cases "${HARD_CASES_PATH}" \
            --start-offset "${HARD_CASE_START_OFFSET}" \
            --output "${REFLECTION_REPORT}" \
            --round-id "${ROUND_ID}" \
            --min-cluster-size "${REFLECT_MIN_CLUSTER_SIZE}"
    fi
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
