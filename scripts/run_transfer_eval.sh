#!/usr/bin/env bash
# Held-out transfer evaluation: after skills have evolved on SWE-bench (run
# run_full_method.sh first, or SKILLS are already in ./skills), evaluate the
# FROZEN skill bank on held-out distributions the agent never trained on:
#   - SWE-bench-Live : temporal / contamination-free held-out (still Python)
#   - Multi-SWE-bench: cross-language transfer (Go, Rust, Java, JS, TS, C, C++)
# Each benchmark runs with --stage eval so the skill bank is snapshotted and
# read-only. This isolates "did evolution learn transferable strategy" from
# "did it adapt to the eval set".
#
# Knobs: LIVE_DATASET/LIVE_SPLIT, MULTI_DATASET, NUM_EVAL, SEED, TEST_BACKEND,
#        BENCHMARKS (subset of "swebench_live multi_swe_bench"), TRAIN_IDS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LIVE_DATASET="${LIVE_DATASET:-lite}"
LIVE_SPLIT="${LIVE_SPLIT:-test}"
MULTI_DATASET="${MULTI_DATASET:-full}"
NUM_EVAL="${NUM_EVAL:-}"
BENCHMARKS="${BENCHMARKS:-swebench_live multi_swe_bench}"
TRAIN_IDS="${TRAIN_IDS:-}"

require_python
require_data
require_container_engine

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
info "with scripts/evaluate.sh; grade Multi-SWE-bench with its official Docker harness"
info "(the runner writes a harness-ready patch file and prints the exact command)."
