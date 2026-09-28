#!/usr/bin/env bash
# One GRPO round (called by 5_grpo.sh): build prompt datasets, train with EasyR1
# against the online reward, update the TaskPool and reflect.
# Knobs: every ${VAR:-default} below (ablations: SAMPLING_MODE, TRAIN_SUBSET_GATE, CLUSTER_LEVELS).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

TRAIN_DATASET="${TRAIN_DATASET:-full}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
NUM_TRAIN="${NUM_TRAIN:-}"
VAL_DATASET="${VAL_DATASET:-lite}"
VAL_SPLIT="${VAL_SPLIT:-dev}"
NUM_VAL="${NUM_VAL:-}"
EXCLUDE_IDS="${EXCLUDE_IDS:-${DATA_DIR}/eval_ids.txt}"
EASYR1_DIR="${EASYR1_DIR:-}"
EASYR1_CONFIG="${EASYR1_CONFIG:-${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml}"
CONTROLLER_MODEL_PATH="${CONTROLLER_MODEL_PATH:-}"
QWEN35_PREFLIGHT="${QWEN35_PREFLIGHT:-1}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/controller_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/controller_val.jsonl}"
ENABLE_SKILL_REFLECTION="${ENABLE_SKILL_REFLECTION:-1}"
ENABLE_SKILL_CREDIT="${ENABLE_SKILL_CREDIT:-1}"
REWARD_WORKERS="${REWARD_WORKERS:-8}"
REBUILD_DATASETS="${REBUILD_DATASETS:-1}"
REFLECT_MIN_CLUSTER_SIZE="${REFLECT_MIN_CLUSTER_SIZE:-3}"
# Ablation switches for the dataset builder and reward workers.
export SELF_EVOLVER_SAMPLING_MODE="${SAMPLING_MODE:-u_f}"
export SELF_EVOLVER_TRAIN_SUBSET_GATE="${TRAIN_SUBSET_GATE:-1}"
export SELF_EVOLVER_CLUSTER_LEVELS="${CLUSTER_LEVELS:-2}"

# A round keeps one id across checkpoint resumes (REBUILD_DATASETS=0): the
# frozen prompts, skill credit, reflection and baselines are all keyed by it.
CURRENT_ROUND_FILE="${RUNS_DIR}/current_round_id"
if [ -z "${ROUND_ID:-}" ]; then
    if [ "${REBUILD_DATASETS}" = "0" ] && [ -s "${CURRENT_ROUND_FILE}" ]; then
        ROUND_ID="$(cat "${CURRENT_ROUND_FILE}")"
    else
        ROUND_ID="$(date -u +%Y%m%dT%H%M%SZ)"
    fi
fi
mkdir -p "${RUNS_DIR}"
printf '%s\n' "${ROUND_ID}" > "${CURRENT_ROUND_FILE}"
export SELF_EVOLVER_ROUND_ID="${ROUND_ID}"
ROUND_STATE_DIR="${RUNS_DIR}/rounds/${ROUND_ID}"
mkdir -p "${ROUND_STATE_DIR}"
SKILL_REFLECTION_DIR="${SKILL_REFLECTION_DIR:-${RUNS_DIR}/skill_reflections}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_ROOT}}"
HARD_CASES_PATH="${HARD_CASES_PATH:-${WORKSPACE_DIR}/hard_cases.jsonl}"
# One canonical hard-case buffer for the online worker, prompt builder and Reflector.
export SELF_EVOLVER_HARD_CASES_PATH="${HARD_CASES_PATH}"
TASK_SIGNALS_REPORT="${TASK_SIGNALS_REPORT:-}"
TASK_POOL_STATE="${TASK_POOL_STATE:-${DATA_DIR}/controller_task_pool.json}"
TASK_POOL_UNIFORM_FRACTION="${TASK_POOL_UNIFORM_FRACTION:-0.5}"
MAX_INSTANCE_REPEATS="${MAX_INSTANCE_REPEATS:-3}"
RESET_TASK_POOL="${RESET_TASK_POOL:-0}"
SKILL_SNAPSHOT_DIR="${SKILL_SNAPSHOT_DIR:-${ROUND_STATE_DIR}/skill_bank_at_start}"
EASYR1_EXPERIMENT_NAME="${EASYR1_EXPERIMENT_NAME:-}"
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

# Snapshot the live bank once per round; checkpoint resumes keep the original.
use_skill_bank "${SELF_EVOLVER_SKILLS_DIR:-${RUNS_DIR}/skills}"
if [ ! -e "${SKILL_SNAPSHOT_DIR}/.complete" ]; then
    mkdir -p "${SKILL_SNAPSHOT_DIR}"
    cp -a "${SELF_EVOLVER_SKILLS_DIR}/." "${SKILL_SNAPSHOT_DIR}/"
    touch "${SKILL_SNAPSHOT_DIR}/.complete"
    info "Captured round-start skill bank -> ${SKILL_SNAPSHOT_DIR}"
fi

# 1) Build prompt datasets (offline). Eval/validation/SFT ids must never enter
# GRPO prompts; EXCLUDE_IDS is required since full/test contains Verified test.
EXCLUDE_ARGS=()
if [ "${ALLOW_MISSING_EXCLUDE_IDS:-0}" = "1" ]; then
    exclude_id_args EXCLUDE_ARGS "${EXCLUDE_IDS}" optional
else
    exclude_id_args EXCLUDE_ARGS "${EXCLUDE_IDS}" required
fi
if [ -n "${SFT_IDS:-}" ]; then
    exclude_id_args EXCLUDE_ARGS "${SFT_IDS}" required
else
    SFT_IDS="${DATA_DIR}/controller_sft_manifest_ids.txt"
    exclude_id_args EXCLUDE_ARGS "${SFT_IDS}" optional
fi
if [ "${REBUILD_DATASETS}" = "1" ]; then
    if [ "${RESET_TASK_POOL}" = "1" ]; then
        rm -f "${TASK_POOL_STATE}" "${TASK_POOL_STATE}.ingestion.json"
        info "Reset TaskPool state at ${TASK_POOL_STATE}"
    fi
    info "Building training prompts -> ${TRAIN_JSONL}"
    # Seed varies per round but is reproducible.
    ROUND_SEED=$(( SEED * 100003 + $(printf '%s' "${ROUND_ID}" | cksum | cut -d' ' -f1) % 100003 ))
    TASK_SIGNAL_ARGS=(
        --sampling-seed "${ROUND_SEED}"
        --task-pool-uniform-fraction "${TASK_POOL_UNIFORM_FRACTION}"
        --max-instance-repeats "${MAX_INSTANCE_REPEATS}"
    )
    if [ -n "${TASK_SIGNALS_REPORT}" ]; then
        [ -s "${TASK_SIGNALS_REPORT}" ] || die \
            "TASK_SIGNALS_REPORT='${TASK_SIGNALS_REPORT}' is missing or empty."
        TASK_SIGNAL_ARGS+=(--task-signals-report "${TASK_SIGNALS_REPORT}")
        info "Applying previous-round TaskPool signals from ${TASK_SIGNALS_REPORT}"
    fi
    "${PYTHON}" -m src.rl.easyr1_dataset \
        --output "${TRAIN_JSONL}" \
        --dataset "${TRAIN_DATASET}" --split "${TRAIN_SPLIT}" --stage train \
        --task-pool-state "${TASK_POOL_STATE}" \
        ${NUM_TRAIN:+--num-instances "${NUM_TRAIN}"} \
        ${TASK_SIGNAL_ARGS[@]+"${TASK_SIGNAL_ARGS[@]}"} \
        ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"}
    info "Building validation prompts -> ${VAL_JSONL}"
    # VAL_INCLUDE_IDS pins the fixed validation manifest; SFT ids stay out.
    VAL_ARGS=()
    [ -n "${VAL_INCLUDE_IDS:-}" ] && VAL_ARGS+=(--include-ids "${VAL_INCLUDE_IDS}")
    exclude_id_args VAL_ARGS "${SFT_IDS}" optional
    "${PYTHON}" -m src.rl.easyr1_dataset \
        --output "${VAL_JSONL}" \
        --dataset "${VAL_DATASET}" --split "${VAL_SPLIT}" --stage eval \
        ${NUM_VAL:+--num-instances "${NUM_VAL}"} \
        ${VAL_ARGS[@]+"${VAL_ARGS[@]}"}
    # Marks this round's prompts as frozen: a crash resume must reuse them.
    touch "${ROUND_STATE_DIR}/.datasets_built"
else
    [ -s "${TRAIN_JSONL}" ] || die "REBUILD_DATASETS=0 but TRAIN_JSONL='${TRAIN_JSONL}' is missing or empty."
    [ -s "${VAL_JSONL}" ] || die "REBUILD_DATASETS=0 but VAL_JSONL='${VAL_JSONL}' is missing or empty."
    info "Reusing frozen Controller prompt datasets for checkpoint resume"
fi

# 2) GRPO training with EasyR1.
REWARD_FN="src/reward/online_reward.py:compute_score"
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
        "${PYTHON}" "${SCRIPTS_DIR}/tools/preflight_qwen35.py" "${PREFLIGHT_ARGS[@]}"
    fi
    if [ "${PREWARM_IMAGES:-1}" = "1" ] && { [ "${TEST_BACKEND}" = "apptainer" ] || [ "${TEST_BACKEND}" = "docker" ]; }; then
        # Avoid stalling reward workers on cold multi-GB image conversions mid-step.
        info "Prewarming eval images for this round (PREWARM_IMAGES=0 skips)"
        "${PYTHON}" "${SCRIPTS_DIR}/tools/prewarm_images.py" \
            --jsonl "${TRAIN_JSONL}" --jsonl "${VAL_JSONL}" \
            --engine "${TEST_BACKEND}" --workers "${PREWARM_WORKERS:-4}" \
            || warn "Some images failed to prewarm; they will be pulled lazily."
    fi
    # Ray workers import the reward file from the EasyR1 checkout; `import src` needs this.
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export SELF_EVOLVER_TEST_BACKEND="${TEST_BACKEND}"
    # Per-arm reward config without editing the tracked file.
    export SELF_EVOLVER_REWARD_CONFIG="${REWARD_CONFIG:-${REPO_ROOT}/configs/reward_config.yaml}"
    export SELF_EVOLVER_ROLLOUT_JSONL="${RUNS_DIR}/online_rollouts.jsonl"
    export SELF_EVOLVER_ENABLE_SKILL_EVOLUTION="${ENABLE_SKILL_CREDIT}"
    export SELF_EVOLVER_REWARD_WORKERS="${REWARD_WORKERS}"
    export WORKSPACE_DIR
    # Record where this round's hard cases begin on first launch and reuse it
    # on resume; recomputing it after a crash would skip the round's records.
    OFFSET_FILE="${ROUND_STATE_DIR}/hard_case_start_offset"
    HARD_CASE_START_OFFSET="${HARD_CASE_START_OFFSET:-}"
    if [ -z "${HARD_CASE_START_OFFSET}" ] && [ -s "${OFFSET_FILE}" ]; then
        HARD_CASE_START_OFFSET="$(cat "${OFFSET_FILE}")"
    fi
    if [ -z "${HARD_CASE_START_OFFSET}" ]; then
        HARD_CASE_START_OFFSET=0
        if [ -f "${HARD_CASES_PATH}" ]; then
            HARD_CASE_START_OFFSET="$(wc -c < "${HARD_CASES_PATH}")"
        fi
    fi
    [ -s "${OFFSET_FILE}" ] || printf '%s\n' "${HARD_CASE_START_OFFSET}" > "${OFFSET_FILE}"
    # Docker runs outside the training container: keep bind-mounted eval files shared.
    export TMPDIR="${TMPDIR:-${RUNS_DIR}}"
    ensure_short_tmpdir
    info "Launching EasyR1 GRPO from ${EASYR1_DIR}"
    # The experiment seed reaches the trainer's data shuffling too.
    EASYR1_OVERRIDES=(data.seed="${SEED}")
    if [ -n "${CONTROLLER_MODEL_PATH}" ]; then
        EASYR1_OVERRIDES+=(worker.actor.model.model_path="${CONTROLLER_MODEL_PATH}")
    fi
    if [ -n "${EASYR1_SAVE_CHECKPOINT_PATH:-}" ]; then
        EASYR1_OVERRIDES+=(trainer.save_checkpoint_path="${EASYR1_SAVE_CHECKPOINT_PATH}")
    fi
    if [ -n "${EASYR1_EXPERIMENT_NAME}" ]; then
        EASYR1_OVERRIDES+=(trainer.experiment_name="${EASYR1_EXPERIMENT_NAME}")
    fi
    if [ -n "${EASYR1_LOAD_CHECKPOINT_PATH}" ]; then
        EASYR1_OVERRIDES+=(trainer.load_checkpoint_path="${EASYR1_LOAD_CHECKPOINT_PATH}")
    fi
    if [ -n "${EASYR1_MAX_STEPS}" ]; then
        EASYR1_OVERRIDES+=(trainer.max_steps="${EASYR1_MAX_STEPS}")
    fi
    if [ -n "${EASYR1_VAL_BEFORE_TRAIN:-}" ]; then
        EASYR1_OVERRIDES+=(trainer.val_before_train="${EASYR1_VAL_BEFORE_TRAIN}")
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
        ${EASYR1_OVERRIDES[@]+"${EASYR1_OVERRIDES[@]}"} )
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
