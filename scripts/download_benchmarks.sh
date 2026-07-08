#!/usr/bin/env bash
# Download the SWE-bench dataset family to $SWEBENCH_DATA_DIR as save_to_disk
# copies, in the exact local layout the loaders expect. Idempotent: a dataset
# whose local directory already holds data is skipped.
#
# Knobs: SWEBENCH_DATA_DIR, DATASETS (space-separated subset of
#        "lite verified full swebench_live swebench_pro multi_swe_bench
#         multi_swe_bench_flash").
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_python
DATASETS="${DATASETS:-lite verified full swebench_live swebench_pro multi_swe_bench multi_swe_bench_flash}"

info "Downloading benchmarks to ${SWEBENCH_DATA_DIR}: ${DATASETS}"
mkdir -p "${SWEBENCH_DATA_DIR}"

"${PYTHON}" - ${DATASETS} <<'PY'
import sys

from src.benchmark import datasets as reg

root = reg.swebench_data_dir()
root.mkdir(parents=True, exist_ok=True)

for key in sys.argv[1:]:
    spec = reg.DATASETS.get(key)
    if spec is None:
        raise SystemExit(f"unknown dataset {key!r}; known: {', '.join(sorted(reg.DATASETS))}")
    target = root / spec.local_dir
    if target.exists() and any(target.iterdir()):
        print(f"skip {key}: already present at {target}")
        continue
    if not spec.hf_name:
        raise SystemExit(f"{key}: no hub name to download from")

    if spec.jsonl_glob:
        # Multi-SWE-bench: load_dataset cannot unify the per-language JSONL, so
        # fetch the raw files (full) or dump a single flash.jsonl (subset).
        if key == "multi_swe_bench_flash":
            from datasets import load_dataset
            import json
            print(f"downloading {spec.hf_name} -> {target}/flash.jsonl")
            data = load_dataset(spec.hf_name, split="train")
            target.mkdir(parents=True, exist_ok=True)
            with (target / "flash.jsonl").open("w", encoding="utf-8") as fh:
                for row in data:
                    fh.write(json.dumps(dict(row)) + "\n")
        else:
            from huggingface_hub import snapshot_download
            print(f"downloading {spec.hf_name} raw JSONL -> {target}")
            snapshot_download(spec.hf_name, repo_type="dataset",
                              local_dir=str(target), allow_patterns=["*/*.jsonl"])
        print(f"saved {key} -> {target}")
        continue

    from datasets import load_dataset

    if spec.single_split:
        print(f"downloading {spec.hf_name}[{spec.single_split}] -> {target}")
        data = load_dataset(spec.hf_name, split=spec.single_split)
    else:
        print(f"downloading {spec.hf_name} (all splits) -> {target}")
        data = load_dataset(spec.hf_name)
    data.save_to_disk(str(target))
    print(f"saved {key} -> {target}")
PY

info "Benchmark download complete."
