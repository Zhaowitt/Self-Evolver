#!/usr/bin/env python3
"""Join reviewed Controller labels with frozen prompts (``src.rl.easyr1_dataset``) into SFT JSONL.

Annotations, one per line: {"instance_id": ..., "selected_skill_ids": [0..2 candidate ids],
"strategy": optional, "review_status": "cross_model_agreed"|"human_confirmed", "skill_bank_hash": optional}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controller.injection import ground_controller_skills  # noqa: E402
from src.controller.parser import parse_controller_response  # noqa: E402
from src.controller.prompt_builder import CONTROLLER_PROMPT_FORMAT  # noqa: E402
from src.controller.schema import MAX_SELECTED_SKILLS  # noqa: E402
from src.skills.skill_bank import SkillBank  # noqa: E402

MIN_CANDIDATES = 4
MAX_CANDIDATES = 6
MAX_STRATEGY_CHARS = 300
ACCEPTED_REVIEW_STATUSES = {"cross_model_agreed", "human_confirmed"}
FORBIDDEN_FIELDS = {"budget", "target_failure_type", "memory_query", "task_wrapper"}
# Labels collapsed onto one answer teach a constant policy, not skill matching.
MAX_TOP_COMBO_SHARE = 0.5
MAX_EMPTY_SHARE = 0.4


def build_sft_dataset(
    prompts_path: Path,
    annotations_path: Path,
    output_path: Path,
    manifest_path: Path,
    skills_dir: Path,
    exclude_ids: set[str] | None = None,
    val_output_path: Path | None = None,
    val_fraction: float = 0.1,
    allow_unreviewed: bool = False,
) -> int:
    prompt_rows = _jsonl(prompts_path)
    annotation_rows = _jsonl(annotations_path)
    prompts, prompt_dupes = _index(prompt_rows, _instance_id)
    annotations, annotation_dupes = _index(annotation_rows, _annotation_id)
    excluded = set(exclude_ids or set())
    bank_hash = _skill_bank_hash(skills_dir)
    bank = SkillBank(skills_dir=skills_dir)
    active_ids = {skill.id for skill in bank.active()}

    rows = []
    rejected = []

    def reject(instance_id: str, reason: str) -> None:
        rejected.append({"instance_id": instance_id, "reason": reason})

    for instance_id in sorted(annotations):
        annotation = annotations[instance_id]
        prompt = prompts.get(instance_id)
        if not instance_id:
            reject(instance_id, "annotation without instance_id")
            continue
        if instance_id in annotation_dupes:
            reject(instance_id, "duplicate annotation rows")
            continue
        if instance_id in prompt_dupes:
            reject(instance_id, "duplicate prompt rows")
            continue
        if instance_id in excluded:
            reject(instance_id, "contamination exclusion")
            continue
        if not prompt:
            reject(instance_id, "missing prompt")
            continue
        extra_info = prompt.get("extra_info") or {}
        if extra_info.get("prompt_format") != CONTROLLER_PROMPT_FORMAT:
            reject(
                instance_id,
                f"prompt_format must be {CONTROLLER_PROMPT_FORMAT}; rebuild prompts "
                "with the current src.rl.easyr1_dataset",
            )
            continue
        status = str(annotation.get("review_status") or "")
        if not allow_unreviewed and status not in ACCEPTED_REVIEW_STATUSES:
            reject(instance_id, f"unreviewed annotation (review_status={status!r})")
            continue
        forbidden = sorted(FORBIDDEN_FIELDS & set(annotation))
        if forbidden:
            reject(instance_id, f"annotation uses removed Controller fields: {forbidden}")
            continue
        if annotation.get("skill_bank_hash") and annotation["skill_bank_hash"] != bank_hash:
            reject(instance_id, "annotation was made against a different skill bank")
            continue

        truth = _dict_value(prompt.get("ground_truth"))
        candidates = [
            str(item)
            for item in (truth.get("allowed_skill_ids") or extra_info.get("allowed_skill_ids") or [])
        ]
        if not MIN_CANDIDATES <= len(set(candidates)) <= MAX_CANDIDATES:
            reject(
                instance_id,
                f"candidate skill count must be {MIN_CANDIDATES}..{MAX_CANDIDATES}, "
                f"got {len(set(candidates))}",
            )
            continue
        missing = sorted(set(candidates) - active_ids)
        if missing:
            reject(instance_id, f"candidates not active in the frozen skill bank: {missing}")
            continue
        prompt_text = prompt.get("prompt")
        if not isinstance(prompt_text, str) or not all(
            f'"id": "{skill_id}"' in prompt_text for skill_id in candidates
        ):
            reject(instance_id, "prompt text does not list the recorded candidates")
            continue

        raw_selected = annotation.get("selected_skill_ids")
        if raw_selected is None or not isinstance(raw_selected, list):
            reject(instance_id, "selected_skill_ids must be a list (may be empty)")
            continue
        selected = [str(item).strip() for item in raw_selected]
        if (
            len(selected) > MAX_SELECTED_SKILLS
            or len(set(selected)) != len(selected)
            or any(item not in candidates for item in selected)
        ):
            reject(instance_id, "invalid selected skill ids")
            continue
        strategy = str(annotation.get("strategy") or "").strip()
        if len(strategy) > MAX_STRATEGY_CHARS or "\n" in strategy:
            reject(instance_id, f"strategy must be one line of <= {MAX_STRATEGY_CHARS} chars")
            continue

        target: dict[str, Any] = {"selected_skill_ids": selected}
        if strategy:
            target["strategy"] = strategy
        response = json.dumps(target, ensure_ascii=False)
        # Same parse+ground path as online, so SFT never teaches a rejected output.
        signal = ground_controller_skills(
            parse_controller_response(response),
            skill_bank=bank,
            allowed_skill_ids=candidates,
        )
        if signal is None or signal.parse_error or signal.selected_skill_ids != selected:
            reject(instance_id, "target does not round-trip through parser/grounding")
            continue

        rows.append(
            {
                "prompt": prompt_text,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response},
                ],
                "instance_id": instance_id,
                "repo_name": truth.get("repo_name") or extra_info.get("repo_name") or "",
                "candidate_skill_ids": candidates,
                "review_status": status,
                "skill_bank_hash": bank_hash,
            }
        )

    train_rows, val_rows = _split(rows, val_fraction if val_output_path else 0.0)
    _write_jsonl(output_path, train_rows)
    if val_output_path:
        _write_jsonl(val_output_path, val_rows)
    ids_path = manifest_path.with_name(manifest_path.stem + "_ids.txt")
    ids_path.write_text("".join(f"{row['instance_id']}\n" for row in rows), encoding="utf-8")

    diagnostics = _diagnostics(rows)
    manifest = {
        "schema_version": "controller_sft_manifest_v2",
        "prompt_format": CONTROLLER_PROMPT_FORMAT,
        "count": len(rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "instance_ids": [row["instance_id"] for row in rows],
        "val_instance_ids": [row["instance_id"] for row in val_rows],
        "ids_file": str(ids_path),
        "skill_bank_hash": bank_hash,
        "active_skill_ids": sorted(active_ids),
        "candidate_skill_count_range": [
            min((len(row["candidate_skill_ids"]) for row in rows), default=0),
            max((len(row["candidate_skill_ids"]) for row in rows), default=0),
        ],
        "diagnostics": diagnostics,
        "rejected_count": len(rejected),
        "rejected_reasons": dict(Counter(item["reason"] for item in rejected)),
        "rejected": rejected,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    combos: Counter = Counter()
    skills: Counter = Counter()
    repos: Counter = Counter()
    empty = strategy = first_listed = 0
    for row in rows:
        target = json.loads(row["response"])
        selected = target["selected_skill_ids"]
        combos["+".join(sorted(selected)) or "<none>"] += 1
        skills.update(selected)
        repos[row["repo_name"] or "unknown"] += 1
        empty += not selected
        strategy += bool(target.get("strategy"))
        listed = _listed_order(row["prompt"], row["candidate_skill_ids"])
        if selected and set(selected) == set(listed[: len(selected)]):
            first_listed += 1
    warnings = []
    if total:
        top_combo, top_count = combos.most_common(1)[0]
        if top_count / total > MAX_TOP_COMBO_SHARE:
            warnings.append(
                f"label collapse: {top_combo!r} is {top_count}/{total} of targets"
            )
        if empty / total > MAX_EMPTY_SHARE:
            warnings.append(f"empty selections are {empty}/{total} of targets")
        candidates = {skill for row in rows for skill in row["candidate_skill_ids"]}
        never = sorted(candidates - set(skills))
        if never:
            warnings.append(f"candidate skills never selected: {never}")
    else:
        warnings.append("no accepted SFT rows")
    return {
        "selection_combo_counts": dict(combos.most_common()),
        "skill_selection_counts": dict(skills.most_common()),
        "repo_counts": dict(repos.most_common()),
        "empty_selection_rate": round(empty / total, 4) if total else 0.0,
        "strategy_rate": round(strategy / total, 4) if total else 0.0,
        # With per-instance shuffled candidates this should stay near chance.
        "selected_equals_first_listed_rate": round(first_listed / total, 4) if total else 0.0,
        "warnings": warnings,
    }


def _listed_order(prompt_text: str, candidates: list[str]) -> list[str]:
    positions = {skill: prompt_text.find(f'"id": "{skill}"') for skill in candidates}
    return sorted(candidates, key=lambda skill: positions[skill])


def _split(rows: list[dict[str, Any]], val_fraction: float):
    """Deterministic, repo-stratified validation split by instance-id hash."""
    if val_fraction <= 0 or len(rows) < 2:
        return rows, []
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_repo.setdefault(row["repo_name"] or "unknown", []).append(row)
    train, val = [], []
    for repo in sorted(by_repo):
        ranked = sorted(by_repo[repo], key=lambda row: _hash(row["instance_id"]))
        n_val = int(round(len(ranked) * val_fraction))
        val.extend(ranked[:n_val])
        train.extend(ranked[n_val:])
    if not val:
        ranked = sorted(train, key=lambda row: _hash(row["instance_id"]))
        val, train = ranked[:1], ranked[1:]
    return train, val


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"))
    parser.add_argument("--exclude-ids", type=Path, action="append", default=[])
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Accept annotations without review_status (pilot runs only).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when label-distribution warnings are raised.",
    )
    args = parser.parse_args()
    exclusions: set[str] = set()
    for path in args.exclude_ids:
        exclusions.update(_read_ids(path))
    count = build_sft_dataset(
        args.prompts,
        args.annotations,
        args.output,
        args.manifest,
        args.skills_dir,
        exclusions,
        val_output_path=args.val_output,
        val_fraction=args.val_fraction,
        allow_unreviewed=args.allow_unreviewed,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(f"Wrote {count} reviewed Controller SFT records ({manifest['rejected_count']} rejected)")
    for warning in manifest["diagnostics"]["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if count == 0 or (args.strict and manifest["diagnostics"]["warnings"]):
        sys.exit(1)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(rows: list[dict[str, Any]], key_fn) -> tuple[dict[str, dict[str, Any]], set[str]]:
    index: dict[str, dict[str, Any]] = {}
    dupes: set[str] = set()
    for row in rows:
        key = key_fn(row)
        if key in index:
            dupes.add(key)
        index[key] = row
    return index, dupes


def _instance_id(row: dict[str, Any]) -> str:
    truth = _dict_value(row.get("ground_truth"))
    return str(truth.get("instance_id") or row.get("instance_id") or "")


def _annotation_id(row: dict[str, Any]) -> str:
    return str(row.get("instance_id") or row.get("id") or "")


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _read_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def _skill_bank_hash(skills_dir: Path) -> str:
    """Hash skill content plus the active set; usage stats are excluded as they change every rollout."""
    digest = hashlib.sha256()
    for path in sorted(skills_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    active = sorted(skill.id for skill in SkillBank(skills_dir=skills_dir).active())
    digest.update(("active:" + ",".join(active)).encode("utf-8"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
