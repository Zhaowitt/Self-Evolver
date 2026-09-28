#!/usr/bin/env bash
# Stage 7: resolved rate, pass@k, cost and evolution curve -> ${ARM_DIR}/metrics.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EVAL_DIR="${ARM_DIR}/eval"
SUMMARY="${EVAL_DIR}/graded/final_summary.json"
[ -f "${SUMMARY}" ] || SUMMARY="${EVAL_DIR}/final_summary.json"
[ -f "${EVAL_DIR}/rollouts.jsonl" ] || die "No eval rollouts under ${EVAL_DIR}; run scripts/6_eval.sh."
"${PYTHON}" -m src.benchmark.metrics \
    --rollouts "${EVAL_DIR}/rollouts.jsonl" --report "${SUMMARY}" \
    --output "${ARM_DIR}/metrics.json"
info "[metrics] ${ARM_DIR}/metrics.json"
