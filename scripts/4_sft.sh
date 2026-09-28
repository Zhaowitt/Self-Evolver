#!/usr/bin/env bash
# Stage 4: Controller SFT warm start -> ${SFT_DIR}/merged (stage 5 starts GRPO from it).
# Labels: ANNOTATOR_MODELS=<a>,<b> cross-model annotates SFT_NUM full/train instances, or
# SFT_ANNOTATIONS + SFT_IDS supply reviewed labels. SKIP_SFT=1 skips the stage.
# Knobs: SKIP_SFT, ANNOTATOR_MODELS, SFT_NUM, SFT_SPLIT, SFT_ANNOTATIONS, SFT_IDS,
#        SFT_CONFIG, CONTROLLER_MODEL_PATH.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ "${SKIP_SFT:-0}" = "1" ]; then
    info "[sft] skipped (SKIP_SFT=1)"
    exit 0
fi
if [ -e "${SFT_DIR}/merged/config.json" ]; then
    info "[sft] reusing ${SFT_DIR}/merged"
    exit 0
fi
require_python
SFT_SPLIT="${SFT_SPLIT:-train}"   # full/train never overlaps the GRPO pool or Verified
DATA="${DATA_DIR}/controller_sft"
PROMPTS="${DATA}/prompts.jsonl"
ANNOTATIONS="${SFT_ANNOTATIONS:-${DATA}/annotations.jsonl}"
mkdir -p "${DATA}"

if [ -n "${SFT_ANNOTATIONS:-}" ]; then
    [ -n "${SFT_IDS:-}" ] || die "[sft] SFT_ANNOTATIONS needs SFT_IDS"
    "${PYTHON}" -m src.rl.easyr1_dataset --output "${PROMPTS}" \
        --dataset full --split "${SFT_SPLIT}" --stage train --include-ids "${SFT_IDS}"
elif [ ! -s "${ANNOTATIONS}" ]; then
    [ -n "${ANNOTATOR_MODELS:-}" ] || die "[sft] set ANNOTATOR_MODELS=<model_a>,<model_b>, SFT_ANNOTATIONS + SFT_IDS, or SKIP_SFT=1"
    "${PYTHON}" "${SCRIPTS_DIR}/tools/annotate_controller_sft.py" \
        --models "${ANNOTATOR_MODELS}" --num "${SFT_NUM:-300}" \
        --split "${SFT_SPLIT}" --seed "${SEED}" --output-dir "${DATA}"
fi

# Validation and final-eval ids must never become SFT labels.
EXCLUDE_ARGS=()
exclude_id_args EXCLUDE_ARGS "${DATA_DIR}/eval_ids.txt ${SPLITS_DIR}/in_domain_val_ids.txt" required
"${PYTHON}" "${SCRIPTS_DIR}/tools/build_controller_sft_dataset.py" \
    --prompts "${PROMPTS}" --annotations "${ANNOTATIONS}" \
    --output "${DATA}/train.jsonl" --val-output "${DATA}/val.jsonl" \
    --manifest "${DATA}/manifest.json" --skills-dir "${REPO_ROOT}/skills" \
    "${EXCLUDE_ARGS[@]}"
MODEL_ARGS=()
[ -n "${CONTROLLER_MODEL_PATH:-}" ] && MODEL_ARGS=(--model "${CONTROLLER_MODEL_PATH}")
"${PYTHON}" "${SCRIPTS_DIR}/tools/train_controller_sft.py" \
    --config "${SFT_CONFIG:-${REPO_ROOT}/configs/controller_sft.yaml}" \
    --train "${DATA}/train.jsonl" --val "${DATA}/val.jsonl" \
    --manifest "${DATA}/manifest.json" --output-dir "${SFT_DIR}" \
    ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
info "[sft] Controller warm start: ${SFT_DIR}/merged"
