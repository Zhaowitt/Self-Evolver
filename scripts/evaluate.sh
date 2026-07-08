#!/usr/bin/env bash
# Grade an existing predictions file with official SWE-bench semantics. On a
# host with a Docker engine (TEST_BACKEND auto|docker) this drives the official
# batch harness for canonical leaderboard numbers; otherwise it grades each
# prediction per-instance through the same swebench code via apptainer or the
# host backend.
#
# Knobs: PREDICTIONS (required), BENCHMARK, DATASET, SPLIT, RUN_ID, EVAL_WORKERS,
#        RUN_DIR, TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PREDICTIONS="${PREDICTIONS:-}"
DATASET="${DATASET:-verified}"
SPLIT="${SPLIT:-test}"
RUN_ID="${RUN_ID:-evaluate-seed${SEED}}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
RUN_DIR="${RUN_DIR:-${RUNS_DIR}/${RUN_ID}}"

[ -n "${PREDICTIONS}" ] || die "set PREDICTIONS=/path/to/predictions.json (from a generate run)."
[ -f "${PREDICTIONS}" ] || die "predictions file not found: ${PREDICTIONS}"

require_python
require_data
require_container_engine

run_benchmark "${RUN_DIR}" \
    --dataset "${DATASET}" --split "${SPLIT}" --phase evaluate \
    --predictions-path "${PREDICTIONS}" --run-id "${RUN_ID}" \
    --eval-workers "${EVAL_WORKERS}"

info "Evaluation complete: ${RUN_DIR}/final_summary.json"
