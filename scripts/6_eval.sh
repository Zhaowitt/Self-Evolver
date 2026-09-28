#!/usr/bin/env bash
# Stage 6: serve ${ARM_DIR}/controller_hf with vLLM, frozen pass@1 generation on
# Verified test with the arm's evolved skills, then official grading.
# Knobs: NUM_EVAL, VLLM_PORT, VLLM_ARGS, CONTROLLER_TEMPERATURE, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EVAL_DIR="${ARM_DIR}/eval"
VLLM_PORT="${VLLM_PORT:-8011}"
VLLM_ARGS="${VLLM_ARGS:---max-model-len 16384 --gpu-memory-utilization 0.85}"
require_python
require_container_engine
require_api_key
export SELF_EVOLVER_SKILLS_DIR="${ARM_DIR}/skills"

if [ ! -e "${EVAL_DIR}/.generated" ]; then
    MODEL="${ARM_DIR}/controller_hf"
    [ -e "${MODEL}/config.json" ] || die "No trained Controller at ${MODEL}; run scripts/5_grpo.sh."
    info "[eval] serving ${MODEL} on port ${VLLM_PORT}"
    # shellcheck disable=SC2086
    "${PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --served-model-name controller --port "${VLLM_PORT}" \
        ${VLLM_ARGS} > "${ARM_DIR}/vllm_controller.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT
    waited=0
    until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
        kill -0 "${VLLM_PID}" 2>/dev/null || die "vLLM exited; see ${ARM_DIR}/vllm_controller.log"
        sleep 5; waited=$((waited + 5))
        [ "${waited}" -lt 1200 ] || die "vLLM did not become ready in 20 min"
    done
    export CONTROLLER_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" CONTROLLER_MODEL=controller \
        CONTROLLER_API_KEY=EMPTY CONTROLLER_TEMPERATURE="${CONTROLLER_TEMPERATURE:-0.0}"
    info "[eval] frozen generation on Verified test"
    run_benchmark "${EVAL_DIR}" \
        --dataset verified --split test --stage eval --phase generate \
        --agent-mode mas --skills static --memory off --task-evolution off \
        --controller-mode llm --run-id "${ARM_NAME}-eval" \
        ${NUM_EVAL:+--num-instances "${NUM_EVAL}"}
    touch "${EVAL_DIR}/.generated"
    kill "${VLLM_PID}" 2>/dev/null || true
fi
info "[eval] official grading"
PREDICTIONS="${EVAL_DIR}/predictions.json" RUN_ID="${ARM_NAME}-eval" \
RUN_DIR="${EVAL_DIR}/graded" DATASET=verified SPLIT=test \
    bash "${SCRIPTS_DIR}/tools/evaluate.sh"
