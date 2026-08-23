#!/usr/bin/env bash
# Build the reviewed Controller SFT artifact. Training is launched by the
# external trainer selected by the experiment environment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PROMPTS="${PROMPTS:-${DATA_DIR}/controller_sft_prompts.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-${DATA_DIR}/controller_sft_annotations.jsonl}"
OUTPUT="${OUTPUT:-${DATA_DIR}/controller_sft.jsonl}"
MANIFEST="${MANIFEST:-${DATA_DIR}/controller_sft_manifest.json}"

"${PYTHON}" "${REPO_ROOT}/scripts/build_controller_sft_dataset.py" \
  --prompts "${PROMPTS}" \
  --annotations "${ANNOTATIONS}" \
  --output "${OUTPUT}" \
  --manifest "${MANIFEST}" \
  --skills-dir "${REPO_ROOT}/skills" \
  "$@"

info "Controller SFT data ready: ${OUTPUT} (config: ${REPO_ROOT}/configs/controller_sft.yaml)"
