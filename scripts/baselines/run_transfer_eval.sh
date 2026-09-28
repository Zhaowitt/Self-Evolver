#!/usr/bin/env bash
# Frozen transfer eval of an evolved skill bank on SWE-bench-Live (temporal) and
# Multi-SWE-bench (cross-language).
# Knobs: LIVE_DATASET, LIVE_SPLIT, MULTI_DATASET, NUM_EVAL, BENCHMARKS,
#        SKILLS_DIR, TRAIN_IDS, SEED, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

LIVE_DATASET="${LIVE_DATASET:-lite}"
LIVE_SPLIT="${LIVE_SPLIT:-test}"
MULTI_DATASET="${MULTI_DATASET:-full}"
NUM_EVAL="${NUM_EVAL:-}"
BENCHMARKS="${BENCHMARKS:-swebench_live multi_swe_bench}"
TRAIN_IDS="${TRAIN_IDS:-}"

require_python
require_data
require_container_engine

# Transfer evaluates the full method's evolved bank (override with SKILLS_DIR).
SKILLS_DIR="${SKILLS_DIR:-${RUNS_DIR}/full_method-seed${SEED}-skills}"
[ -d "${SKILLS_DIR}" ] || die "Skill bank ${SKILLS_DIR} not found; run baselines/run_full_method.sh first or set SKILLS_DIR."
export SELF_EVOLVER_SKILLS_DIR="${SKILLS_DIR}"

for benchmark in ${BENCHMARKS}; do
    run_dir="${RUNS_DIR}/transfer_${benchmark}-seed${SEED}"
    if [ "${benchmark}" = "multi_swe_bench" ]; then
        dataset="${MULTI_DATASET}"; split="test"
    else
        dataset="${LIVE_DATASET}"; split="${LIVE_SPLIT}"
    fi
    info "Frozen transfer eval on ${benchmark}[${dataset}/${split}] -> ${run_dir}"
    BENCHMARK="${benchmark}" run_benchmark "${run_dir}" \
        --dataset "${dataset}" --split "${split}" --stage eval --phase generate \
        --agent-mode mas --skills static --memory off --task-evolution off --controller-mode off \
        --run-id "transfer_${benchmark}-seed${SEED}" \
        ${NUM_EVAL:+--num-instances "${NUM_EVAL}"} \
        ${TRAIN_IDS:+--train-ids "${TRAIN_IDS}"}
done

info "Transfer predictions written under ${RUNS_DIR}/transfer_*. Grade SWE-bench-Live"
info "with scripts/tools/evaluate.sh; grade Multi-SWE-bench with its official Docker harness"
info "(the runner writes a harness-ready patch file and prints the exact command)."
