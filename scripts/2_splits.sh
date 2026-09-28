#!/usr/bin/env bash
# Stage 2: id splits. data/eval_ids.txt = Verified test (final eval), data/train_ids.txt =
# full/train (baselines' --train-ids guard), and under SPLITS_DIR the in-domain validation
# holdout and the GRPO exclusion list (Verified + validation [+ SFT_IDS]).
# Knobs: SPLITS_DIR, VAL_PER_REPO, SFT_IDS, HARD_CASES.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -s "${SPLITS_DIR}/train_exclude_ids.txt" ] && [ -s "${SPLITS_DIR}/in_domain_val_ids.txt" ]; then
    info "[splits] reusing ${SPLITS_DIR}"
    exit 0
fi
require_python
require_data
mkdir -p "${DATA_DIR}"

"${PYTHON}" - \
    full train verified test "${DATA_DIR}" "${HARD_CASES:-}" <<'PY'
import json
import sys
from pathlib import Path

from src.benchmark import datasets

train_ds, train_split, eval_ds, eval_split, out_dir, hard_cases = sys.argv[1:7]
out = Path(out_dir)


def ids(dataset, split):
    return [row["instance_id"] for row in datasets.iter_rows(dataset, split)]


train_ids = ids(train_ds, train_split)
eval_ids = ids(eval_ds, eval_split)
(out / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
(out / "eval_ids.txt").write_text("\n".join(eval_ids) + "\n", encoding="utf-8")
print(f"train_ids: {len(train_ids)}  eval_ids: {len(eval_ids)}")

hard = []
if hard_cases and Path(hard_cases).exists():
    for line in Path(hard_cases).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = rec.get("instance_id")
        if iid and iid not in hard:
            hard.append(iid)
hard_text = "\n".join(hard) + "\n" if hard else "# populate from a run's hard_cases.jsonl (HARD_CASES=...)\n"
(out / "hard_ids.txt").write_text(hard_text, encoding="utf-8")
print(f"hard_ids: {len(hard)}")

overlap = sorted(set(train_ids) & set(eval_ids))
if overlap:
    sys.stderr.write(
        f"CONTAMINATION: {len(overlap)} instance id(s) appear in both train and eval, e.g. "
        f"{overlap[:5]}\n"
    )
    raise SystemExit(1)
print("contamination check: OK (no shared instance ids)")
PY

SFT_ARGS=()
[ -n "${SFT_IDS:-}" ] && SFT_ARGS=(--sft-ids "${SFT_IDS}")
"${PYTHON}" "${SCRIPTS_DIR}/tools/prepare_splits.py" \
    --verified-ids "${DATA_DIR}/eval_ids.txt" \
    --output-dir "${SPLITS_DIR}" \
    --per-repo "${VAL_PER_REPO:-4}" \
    ${SFT_ARGS[@]+"${SFT_ARGS[@]}"}
info "[splits] written to ${DATA_DIR} and ${SPLITS_DIR}"
