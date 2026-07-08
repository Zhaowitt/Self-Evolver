#!/usr/bin/env bash
# Run the experiment set across seeds and emit a metrics table per experiment
# (resolved rate, pass@k over seeds, success-under-budget, cost-to-success,
# hard-case success rate, average tokens). Evolution experiments reset the skill
# bank to seed state before each run so seeds are independent.
#
# Knobs: EXPERIMENTS, SEEDS, TEST_BACKEND, RUNS_DIR, BUDGET, PRICE_PER_TOKEN,
#        HARD_IDS, METRICS_DIR, RESET_SKILLS, CONTINUE_ON_ERROR, plus any knob
#        the per-experiment scripts read (DATASET, EVAL_DATASET, NUM_ROLLOUTS...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXPERIMENTS="${EXPERIMENTS:-zero_shot mas_static task_evolution skill_evolution full_method}"
SEEDS="${SEEDS:-0 1 2}"
BUDGET="${BUDGET:-${MAX_ITERATIONS:-3}}"
PRICE_PER_TOKEN="${PRICE_PER_TOKEN:-0.0}"
HARD_IDS="${HARD_IDS:-${DATA_DIR}/hard_ids.txt}"
METRICS_DIR="${METRICS_DIR:-${RUNS_DIR}/metrics}"
RESET_SKILLS="${RESET_SKILLS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

require_python

run_metrics() {
    local exp="$1"
    local seed d
    local -a rollouts=() reports=()
    for seed in ${SEEDS}; do
        d="${RUNS_DIR}/${exp}-seed${seed}"
        if [ -f "${d}/rollouts.jsonl" ] && [ -f "${d}/final_summary.json" ]; then
            rollouts+=("${d}/rollouts.jsonl")
            reports+=("${d}/final_summary.json")
        fi
    done
    if [ "${#rollouts[@]}" -eq 0 ]; then
        warn "no scored runs for '${exp}'; skipping metrics"
        return 0
    fi
    mkdir -p "${METRICS_DIR}"
    local -a hard=()
    [ -f "${HARD_IDS}" ] && hard=(--hard-ids "${HARD_IDS}")
    info "metrics for '${exp}' over ${#rollouts[@]} run(s) -> ${METRICS_DIR}/${exp}.json"
    "${PYTHON}" -m src.benchmark.metrics \
        --rollouts "${rollouts[@]}" \
        --report "${reports[@]}" \
        --budget "${BUDGET}" \
        --price-per-token "${PRICE_PER_TOKEN}" \
        "${hard[@]}" \
        --output "${METRICS_DIR}/${exp}.json"
}

for exp in ${EXPERIMENTS}; do
    script="${_COMMON_DIR}/run_${exp}.sh"
    if [ ! -f "${script}" ]; then
        warn "no script for experiment '${exp}' (${script}); skipping"
        continue
    fi
    for seed in ${SEEDS}; do
        info "=== experiment=${exp} seed=${seed} ==="
        if [ "${CONTINUE_ON_ERROR}" = "1" ]; then
            SEED="${seed}" RUNS_DIR="${RUNS_DIR}" RESET_SKILLS="${RESET_SKILLS}" \
                bash "${script}" || warn "experiment '${exp}' seed=${seed} failed"
        else
            SEED="${seed}" RUNS_DIR="${RUNS_DIR}" RESET_SKILLS="${RESET_SKILLS}" \
                bash "${script}"
        fi
    done
    run_metrics "${exp}"
done

info "All experiments finished. Metrics under ${METRICS_DIR}"
