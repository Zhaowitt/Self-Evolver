#!/usr/bin/env bash
# Run EXPERIMENTS for one SEED and write a metrics table per experiment.
# Knobs: EXPERIMENTS, SEED, TEST_BACKEND, RUNS_DIR, BUDGET, PRICE_PER_TOKEN,
#        HARD_IDS, METRICS_DIR, RESET_SKILLS, CONTINUE_ON_ERROR, FRESH_RUN,
#        plus every per-experiment script knob.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

EXPERIMENTS="${EXPERIMENTS:-zero_shot mas_static task_evolution skill_evolution full_method}"
BUDGET="${BUDGET:-${MAX_ITERATIONS:-3}}"
PRICE_PER_TOKEN="${PRICE_PER_TOKEN:-0.0}"
HARD_IDS="${HARD_IDS:-${DATA_DIR}/hard_ids.txt}"
METRICS_DIR="${METRICS_DIR:-${RUNS_DIR}/metrics}"
RESET_SKILLS="${RESET_SKILLS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
# Never resume predictions from an earlier invocation.
export FRESH_RUN="${FRESH_RUN:-1}"
FAILED_RUNS=()

require_python

run_metrics() {
    local exp="$1"
    local d="${RUNS_DIR}/${exp}-seed${SEED}"
    if [ ! -f "${d}/rollouts.jsonl" ] || [ ! -f "${d}/final_summary.json" ]; then
        warn "no scored run for '${exp}'; skipping metrics"
        return 0
    fi
    mkdir -p "${METRICS_DIR}"
    local -a hard=()
    [ -f "${HARD_IDS}" ] && hard=(--hard-ids "${HARD_IDS}")
    info "metrics for '${exp}' -> ${METRICS_DIR}/${exp}.json"
    "${PYTHON}" -m src.benchmark.metrics \
        --rollouts "${d}/rollouts.jsonl" \
        --report "${d}/final_summary.json" \
        --budget "${BUDGET}" \
        --price-per-token "${PRICE_PER_TOKEN}" \
        "${hard[@]}" \
        --output "${METRICS_DIR}/${exp}.json"
}

for exp in ${EXPERIMENTS}; do
    script="${SCRIPTS_DIR}/baselines/run_${exp}.sh"
    if [ ! -f "${script}" ]; then
        warn "no script for experiment '${exp}' (${script}); skipping"
        continue
    fi
    info "=== experiment=${exp} seed=${SEED} ==="
    if RUNS_DIR="${RUNS_DIR}" RESET_SKILLS="${RESET_SKILLS}" bash "${script}"; then
        run_metrics "${exp}"
    else
        warn "experiment '${exp}' failed"
        FAILED_RUNS+=("${exp}-seed${SEED}")
        [ "${CONTINUE_ON_ERROR}" = "1" ] || die "stopping after failed run (CONTINUE_ON_ERROR=0)"
    fi
done

info "All experiments finished. Metrics under ${METRICS_DIR}"
if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
    die "${#FAILED_RUNS[@]} run(s) failed: ${FAILED_RUNS[*]}"
fi
