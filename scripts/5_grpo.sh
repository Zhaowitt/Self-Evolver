#!/usr/bin/env bash
# Stage 5: multi-round Controller GRPO for one arm (all state under ARM_DIR). Each round
# freezes prompts + skills, trains STEPS_PER_ROUND steps, then reflects; resumable.
# Starts from the stage-4 SFT model when present; merges ${ARM_DIR}/controller_hf at the end.
# Knobs: EXPERIMENT, SEED, ROUNDS, STEPS_PER_ROUND, NUM_TRAIN, NUM_VAL, VAL_BEFORE_TRAIN,
#        EASYR1_DIR, CONTROLLER_MODEL_PATH, SKIP_SFT, plus every tools/grpo_round.sh knob
#        (ablations: TRAIN_SUBSET_GATE, REWARD_CONFIG, CLUSTER_LEVELS, SAMPLING_MODE).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROUNDS="${ROUNDS:-3}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-64}"
EASYR1_CONFIG="${EASYR1_CONFIG:-${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml}"
export EASYR1_CONFIG
CKPT_DIR="${ARM_DIR}/checkpoints"
# CONTROLLER_MODEL_PATH is the base model; GRPO starts from the stage-4 SFT model when present.
if [ "${SKIP_SFT:-0}" != "1" ] && [ -e "${SFT_DIR}/merged/config.json" ]; then
    export CONTROLLER_MODEL_PATH="${SFT_DIR}/merged"
fi
# Verified + validation (+ flaky) never enter GRPO prompts; validation is the fixed holdout.
EXCLUDE_IDS="${EXCLUDE_IDS:-${SPLITS_DIR}/train_exclude_ids.txt}"
[ -s "${FLAKY_IDS}" ] && EXCLUDE_IDS="${EXCLUDE_IDS} ${FLAKY_IDS}"
export EXCLUDE_IDS
export VAL_DATASET="${VAL_DATASET:-full}" VAL_SPLIT="${VAL_SPLIT:-test}"
export VAL_INCLUDE_IDS="${VAL_INCLUDE_IDS:-${SPLITS_DIR}/in_domain_val_ids.txt}"
info "[grpo] ${ARM_NAME}: ${ROUNDS} rounds from ${CONTROLLER_MODEL_PATH:-the config base model}"

[ -n "${EASYR1_DIR:-}" ] && [ -d "${EASYR1_DIR}" ] || die "EASYR1_DIR must point at an EasyR1 checkout."
require_python
mkdir -p "${ARM_DIR}/data" "${ARM_DIR}/workspace"
# A round's frozen sample is exactly what its steps consume (steps x prompts/step).
if [ -z "${NUM_TRAIN:-}" ]; then
    prompts_per_step="$("${PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["data"]["rollout_batch_size"])' "${EASYR1_CONFIG}")"
    NUM_TRAIN=$((STEPS_PER_ROUND * prompts_per_step))
fi

# Per-arm state; arms share only read-only inputs.
export SELF_EVOLVER_SKILLS_DIR="${ARM_DIR}/skills"
export SELF_EVOLVER_CLUSTER_REGISTRY="${ARM_DIR}/cluster_registry.json"
export SELF_EVOLVER_CLUSTER_BASELINES="${ARM_DIR}/cluster_baselines.json"
export SELF_EVOLVER_REWARD_WORKSPACE="${ARM_DIR}/workspace/online_reward"
use_skill_bank "${SELF_EVOLVER_SKILLS_DIR}"

latest_checkpoint() {
    [ -d "${CKPT_DIR}" ] || return 0
    find "${CKPT_DIR}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1
}

previous_round=""
for ((round = 1; round <= ROUNDS; round++)); do
    round_id="${ARM_NAME}-r${round}"
    round_dir="${ARM_DIR}/rounds/${round_id}"
    if [ -e "${round_dir}/.done" ]; then
        info "Round ${round}/${ROUNDS} (${round_id}) already complete; skipping"
        previous_round="${round_id}"
        continue
    fi
    mkdir -p "${round_dir}"

    rebuild=1
    if [ -e "${round_dir}/.datasets_built" ]; then
        rebuild=0  # crash resume: keep this round's frozen prompts
        info "Resuming round ${round}/${ROUNDS} with its frozen prompts"
    else
        # A new round has a new frozen dataset: do not resume the previous
        # round's dataloader position inside it (EasyR1 restores dataloader.pt).
        ckpt="$(latest_checkpoint)"
        if [ -n "${ckpt}" ] && [ -f "${ckpt}/dataloader.pt" ]; then
            mv "${ckpt}/dataloader.pt" "${ckpt}/dataloader.previous_round.pt"
            info "Reset dataloader state of ${ckpt} for the new round's dataset"
        fi
    fi

    signals=""
    if [ -n "${previous_round}" ] && [ -s "${ARM_DIR}/skill_reflections/${previous_round}.json" ]; then
        signals="${ARM_DIR}/skill_reflections/${previous_round}.json"
    fi

    info "=== ${ARM_NAME}: round ${round}/${ROUNDS} (steps up to $((round * STEPS_PER_ROUND))) ==="
    RUNS_DIR="${ARM_DIR}" \
    ROUND_ID="${round_id}" \
    REBUILD_DATASETS="${rebuild}" \
    NUM_TRAIN="${NUM_TRAIN}" \
    TRAIN_JSONL="${round_dir}/controller_train.jsonl" \
    VAL_JSONL="${ARM_DIR}/data/controller_val.jsonl" \
    TASK_POOL_STATE="${ARM_DIR}/data/task_pool.json" \
    HARD_CASES_PATH="${ARM_DIR}/hard_cases.jsonl" \
    WORKSPACE_DIR="${ARM_DIR}/workspace" \
    TASK_SIGNALS_REPORT="${signals}" \
    SKILL_REFLECTION_DIR="${ARM_DIR}/skill_reflections" \
    EASYR1_EXPERIMENT_NAME="${ARM_NAME}" \
    EASYR1_SAVE_CHECKPOINT_PATH="${CKPT_DIR}" \
    EASYR1_MAX_STEPS="$((round * STEPS_PER_ROUND))" \
    EASYR1_VAL_FREQ="${STEPS_PER_ROUND}" \
    EASYR1_VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-$([ "${round}" -eq 1 ] && echo true || echo false)}" \
        bash "${SCRIPTS_DIR}/tools/grpo_round.sh"

    touch "${round_dir}/.done"
    previous_round="${round_id}"
done

# Merge the final checkpoint (FSDP shards + LoRA) into a servable HF model.
final_ckpt="$(latest_checkpoint)"
[ -n "${final_ckpt}" ] || die "No checkpoint under ${CKPT_DIR}; did training run?"
# Re-merge whenever the final checkpoint changed (e.g. ROUNDS was raised).
if [ "$(cat "${ARM_DIR}/controller_hf/.source_checkpoint" 2>/dev/null)" != "${final_ckpt}" ]; then
    info "Merging ${final_ckpt} into ${ARM_DIR}/controller_hf"
    "${PYTHON}" "${EASYR1_DIR}/scripts/model_merger.py" --local_dir "${final_ckpt}/actor"
    rm -rf "${ARM_DIR}/controller_hf"
    cp -a "${final_ckpt}/actor/huggingface" "${ARM_DIR}/controller_hf"
    printf '%s\n' "${final_ckpt}" > "${ARM_DIR}/controller_hf/.source_checkpoint"
fi
info "[grpo] ${ARM_NAME} done: controller=${ARM_DIR}/controller_hf skills=${SELF_EVOLVER_SKILLS_DIR}"
