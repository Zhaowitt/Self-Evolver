#!/usr/bin/env bash
# Build the instance-id lists the experiments consume:
#   data/train_ids.txt  ids used for evolution (default: SWE-bench full train)
#   data/eval_ids.txt   held-out evaluation ids (default: SWE-bench Verified test)
#   data/hard_ids.txt   hard-case ids for hard_case_success_rate (seeded from a
#                       prior run's hard_cases.jsonl when HARD_CASES is given)
# and verify the eval set shares no instance id with the training set
# (contamination guard consumed by --train-ids).
#
# Knobs: TRAIN_DATASET/TRAIN_SPLIT, EVAL_DATASET/EVAL_SPLIT, DATA_DIR, HARD_CASES.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_python
require_data

TRAIN_DATASET="${TRAIN_DATASET:-full}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_DATASET="${EVAL_DATASET:-verified}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
HARD_CASES="${HARD_CASES:-}"

mkdir -p "${DATA_DIR}"
info "train = ${TRAIN_DATASET}[${TRAIN_SPLIT}], eval = ${EVAL_DATASET}[${EVAL_SPLIT}] -> ${DATA_DIR}"

"${PYTHON}" - \
    "${TRAIN_DATASET}" "${TRAIN_SPLIT}" \
    "${EVAL_DATASET}" "${EVAL_SPLIT}" \
    "${DATA_DIR}" "${HARD_CASES}" <<'PY'
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

info "Splits written."
info "Pass --train-ids ${DATA_DIR}/train_ids.txt (TRAIN_IDS=... in the eval scripts)"
info "to drop any training id from an eval set as a contamination guard."
