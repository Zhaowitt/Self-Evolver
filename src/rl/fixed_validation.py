"""Pin all SWE-bench Full/dev tasks and safely prepare resumable validation prompts.

This module never changes training data, checkpoints, skills, or the old
validation file. Source preparation runs once on CPU; later launches use only
the pinned local source, even when Hugging Face is unavailable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

EXPECTED_COUNT = 225


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def record_id(record: dict) -> str:
    value = record.get("instance_id") or record.get("extra_info", {}).get("instance_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing instance_id in dataset record")
    return value.strip()


def assert_disjoint(validation_ids: set[str], other_ids: set[str], label: str) -> None:
    overlap = sorted(validation_ids & other_ids)
    if overlap:
        raise ValueError(f"Validation overlaps {label}: {len(overlap)} tasks: {overlap[:10]}")


def sft_ids(manifest: Path) -> set[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ids = set(data.get("instance_ids", [])) | set(data.get("val_instance_ids", []))
    if not ids or not all(isinstance(item, str) and item for item in ids):
        raise ValueError(f"Missing/invalid SFT IDs in {manifest}")
    return ids


def check_training(validation_ids: set[str], state_root: Path,
                   train_jsonl: Path | None = None) -> dict[str, int]:
    paths = set(state_root.glob("rounds/*/controller_train.jsonl"))
    legacy = state_root / "data/controller_train.jsonl"
    if legacy.exists():
        paths.add(legacy)
    if train_jsonl is not None:
        if not train_jsonl.is_file():
            raise FileNotFoundError(f"Training dataset missing: {train_jsonl}")
        paths.add(train_jsonl)
    counts = {}
    for path in sorted(paths):
        ids = {record_id(row) for row in read_jsonl(path)}
        assert_disjoint(validation_ids, ids, str(path))
        counts[str(path)] = len(ids)
    # Check actual executions too, not just potentially rebuilt prompt files.
    rollout_paths = set(state_root.glob("runs/*/online_rollouts.jsonl"))
    if (state_root / "online_rollouts.jsonl").exists():
        rollout_paths.add(state_root / "online_rollouts.jsonl")
    for path in sorted(rollout_paths):
        ids = {record_id(row) for row in read_jsonl(path) if row.get("stage") == "train"}
        assert_disjoint(validation_ids, ids, str(path))
        counts[str(path)] = len(ids)
    return counts


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(source_dir: Path) -> list[dict]:
    metadata = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if (metadata.get("dataset"), metadata.get("split"), metadata.get("count")) != (
            "full", "dev", EXPECTED_COUNT):
        raise ValueError("Validation source must be SWE-bench Full/dev, all 225 tasks")
    for filename in ("source.jsonl", "instance_ids.txt", "sft_ids.txt"):
        if digest(source_dir / filename) != metadata["sha256"][filename]:
            raise ValueError(f"Pinned validation source changed: {filename}")
    rows = read_jsonl(source_dir / "source.jsonl")
    ids = [record_id(row) for row in rows]
    frozen_ids = (source_dir / "instance_ids.txt").read_text().splitlines()
    if len(rows) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT or ids != frozen_ids:
        raise ValueError("Validation source has missing, duplicate, or unexpected tasks")
    return rows


def prepare_source(source_dir: Path, state_root: Path, sft_manifest: Path) -> list[dict]:
    if (source_dir / "manifest.json").exists():
        return load_source(source_dir)
    from src.benchmark.datasets import load_split

    rows = [dict(row) for row in load_split("full", "dev")]
    ids = {record_id(row) for row in rows}
    if len(rows) != EXPECTED_COUNT or len(ids) != EXPECTED_COUNT:
        raise ValueError("Official Full/dev does not contain exactly 225 unique tasks")
    sft = sft_ids(sft_manifest)
    assert_disjoint(ids, sft, "SFT train/validation")
    assert_disjoint(ids, {record_id(row) for row in load_split("full", "test")},
                    "Full/test training source")
    check_training(ids, state_root)
    source_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(source_dir / "source.jsonl", "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    atomic_write(source_dir / "instance_ids.txt", "".join(record_id(row) + "\n" for row in rows))
    atomic_write(source_dir / "sft_ids.txt", "".join(item + "\n" for item in sorted(sft)))
    metadata = {
        "dataset": "full", "split": "dev", "count": EXPECTED_COUNT,
        "hub_dataset": "princeton-nlp/SWE-bench",
        "repos": dict(Counter(row["repo"] for row in rows)),
        "sha256": {name: digest(source_dir / name)
                   for name in ("source.jsonl", "instance_ids.txt", "sft_ids.txt")},
    }
    # Written last: an interrupted preparation must never appear complete.
    atomic_write(source_dir / "manifest.json", json.dumps(metadata, indent=2) + "\n")
    return load_source(source_dir)


def check_records(records: list[dict], source_rows: list[dict]) -> None:
    expected = {record_id(row): row for row in source_rows}
    ids = [record_id(row) for row in records]
    if len(ids) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT or set(ids) != set(expected):
        raise ValueError("Validation prompts must cover exactly all 225 pinned Full/dev IDs")
    for record in records:
        extra = record["extra_info"]
        truth = json.loads(record["ground_truth"])
        row = expected[record_id(record)]
        for payload in (extra, truth, truth["extra_info"]):
            if payload.get("stage") != "eval" or payload.get("split") != "dev":
                raise ValueError("Validation payload must use stage=eval and split=dev")
            if payload.get("instance_id") != row["instance_id"]:
                raise ValueError("Validation payload instance_id mismatch")
        for field, source_field in (("repo_name", "repo"), ("base_commit", "base_commit"),
                                    ("problem_statement", "problem_statement"),
                                    ("test_patch", "test_patch"), ("FAIL_TO_PASS", "FAIL_TO_PASS"),
                                    ("PASS_TO_PASS", "PASS_TO_PASS")):
            if truth.get(field) != row.get(source_field):
                raise ValueError(f"Validation ground_truth {field} mismatch for {row['instance_id']}")
        if not isinstance(record.get("prompt"), str) or not record["prompt"]:
            raise ValueError("Missing validation prompt")


def check_prompt_lengths(records: list[dict], tokenizer, maximum: int) -> int:
    lengths = [(record_id(row), len(tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True,
        return_dict=False)))
        for row in records]
    overlong = [(item, length) for item, length in lengths if length > maximum]
    if overlong:
        raise ValueError(f"Full/dev tasks would be filtered at max_prompt_length={maximum}: {overlong}")
    return max(length for _, length in lengths)


def ensure_prompts(output: Path, source_rows: list[dict], skills_dir: Path,
                   tokenizer, maximum: int) -> dict:
    reused = output.exists()
    if reused:
        records = read_jsonl(output)
    else:
        from src.benchmark.datasets import row_to_issue
        from src.rl.easyr1_dataset import build_easyr1_prompt_record
        from src.skills.skill_bank import SkillBank
        from src.skills.skill_selector import SkillSelector

        if not skills_dir.is_dir():
            raise FileNotFoundError(f"Round-start skill bank missing: {skills_dir}")
        selector = SkillSelector(SkillBank(skills_dir))
        if not selector.skill_bank.active():
            raise ValueError("Round-start skill bank has no active skills")
        records = [build_easyr1_prompt_record(
            row_to_issue(row), stage="eval", split="dev", skill_selector=selector,
            hard_cases=[]) for row in source_rows]
    check_records(records, source_rows)
    longest = check_prompt_lengths(records, tokenizer, maximum)
    metadata_path = output.with_suffix(".manifest.json")
    if reused:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("sha256") != digest(output):
            raise ValueError("Frozen validation prompts changed; refusing silent rebuild")
        if metadata.get("max_prompt_tokens") != longest:
            metadata["max_prompt_tokens"] = longest
            atomic_write(metadata_path, json.dumps(metadata, indent=2) + "\n")
    else:
        atomic_write(output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        atomic_write(metadata_path, json.dumps({
            "dataset": "full", "split": "dev", "count": EXPECTED_COUNT,
            "sha256": digest(output), "max_prompt_tokens": longest,
            "skills_dir": str(skills_dir), "hard_cases": "disabled_for_fixed_validation",
        }, indent=2) + "\n")
    return {"output": str(output), "count": len(records), "reused": reused,
            "max_prompt_tokens": longest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--sft-manifest", type=Path, required=True)
    parser.add_argument("--prepare-source", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--skills-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    rows = (prepare_source(args.source_dir, args.state_root, args.sft_manifest)
            if args.prepare_source else load_source(args.source_dir))
    ids = {record_id(row) for row in rows}
    sft = sft_ids(args.sft_manifest)
    pinned_sft = set((args.source_dir / "sft_ids.txt").read_text().splitlines())
    if sft != pinned_sft:
        raise ValueError("SFT exclusion IDs differ from pinned source manifest")
    assert_disjoint(ids, sft, "SFT train/validation")
    checked = check_training(ids, args.state_root, args.train_jsonl)
    result = {"dataset": "full", "split": "dev", "count": len(ids),
              "training_sources_checked": len(checked), "training_overlap": 0, "sft_overlap": 0}
    if args.output is not None:
        if not all((args.skills_dir, args.model, args.config, args.train_jsonl)):
            parser.error("--output requires --skills-dir, --model, --config, --train-jsonl")
        import yaml
        from transformers import AutoTokenizer

        config = yaml.safe_load(args.config.read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        result.update(ensure_prompts(args.output, rows, args.skills_dir, tokenizer,
                                     config["data"]["max_prompt_length"]))
    print("FIXED_VALIDATION_READY " + json.dumps(result))


if __name__ == "__main__":
    main()
