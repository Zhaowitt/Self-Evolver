"""Local-first loading of the SWE-bench dataset family.

Datasets are downloaded once with ``datasets.save_to_disk`` under
``$SWEBENCH_DATA_DIR`` (default ``<repo_root>/benchmarks``) and loaded from
there with ``load_from_disk``; when a split is absent locally we fall back to
the HuggingFace hub. Splits are validated up front with an actionable error
that lists what is actually available, and loaded splits are cached by
``(dataset, split)`` so a runner can hold several splits at once.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from src.environment.models import Issue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """A registered dataset: where it lives locally, on the hub, and its splits."""

    name: str
    local_dir: str
    hf_name: Optional[str]
    splits: Tuple[str, ...]          # every split that can be loaded (local or hub)
    local_splits: Tuple[str, ...]    # splits present in the local save_to_disk copy
    single_split: Optional[str] = None  # local dir is a bare Dataset for this split
    jsonl_glob: Optional[str] = None    # local dir holds raw JSONL files (glob) instead
                                        # of a save_to_disk copy; used by Multi-SWE-bench


# Local layout (see $SWEBENCH_DATA_DIR): swe_bench_full_train holds only the
# full dataset's train split as a bare save_to_disk Dataset; the others are
# DatasetDicts. Live/Pro repos are not in swebench's registry, so their eval
# lives in dedicated runners.
DATASETS: Dict[str, DatasetSpec] = {
    "lite": DatasetSpec(
        "lite", "swe_bench_lite", "princeton-nlp/SWE-bench_Lite",
        splits=("dev", "test"), local_splits=("dev", "test"),
    ),
    "verified": DatasetSpec(
        "verified", "swe_bench_verified", "princeton-nlp/SWE-bench_Verified",
        splits=("test",), local_splits=("test",),
    ),
    "full": DatasetSpec(
        "full", "swe_bench_full_train", "princeton-nlp/SWE-bench",
        splits=("train", "dev", "test"), local_splits=("train",), single_split="train",
    ),
    "swebench_live": DatasetSpec(
        "swebench_live", "swe_bench_live", "SWE-bench-Live/SWE-bench-Live",
        splits=("test", "lite", "verified", "full"),
        local_splits=("test", "lite", "verified", "full"),
    ),
    "swebench_pro": DatasetSpec(
        "swebench_pro", "swe_bench_pro", "ScaleAI/SWE-bench_Pro",
        splits=("test",), local_splits=("test",),
    ),
    # Multi-SWE-bench ships as per-repo JSONL files (one dir per language), which
    # datasets.load_dataset cannot unify into one Arrow table, so we read the raw
    # JSONL. ``full`` is all 1632 non-Python instances across 7 languages;
    # ``flash`` is the balanced 300-instance subset.
    "multi_swe_bench": DatasetSpec(
        "multi_swe_bench", "multi_swe_bench_full", "ByteDance-Seed/Multi-SWE-bench",
        splits=("test",), local_splits=("test",), jsonl_glob="*/*.jsonl",
    ),
    "multi_swe_bench_flash": DatasetSpec(
        "multi_swe_bench_flash", "multi_swe_bench_flash", "ByteDance-Seed/Multi-SWE-bench-flash",
        splits=("test",), local_splits=("test",), jsonl_glob="flash.jsonl",
    ),
}

_CACHE: Dict[Tuple[str, str], Any] = {}


def swebench_data_dir() -> Path:
    """Local benchmark data root: ``$SWEBENCH_DATA_DIR`` or ``<repo>/benchmarks``."""
    env_dir = os.getenv("SWEBENCH_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(__file__).resolve().parents[2] / "benchmarks"


def get_spec(dataset: str) -> DatasetSpec:
    spec = DATASETS.get(dataset)
    if spec is None:
        raise ValueError(
            f"unknown dataset {dataset!r}; registered datasets: "
            f"{', '.join(sorted(DATASETS))}."
        )
    return spec


def available_splits(dataset: str) -> Tuple[str, ...]:
    return get_spec(dataset).splits


def validate_split(dataset: str, split: str) -> None:
    """Reject an unavailable ``(dataset, split)`` with a listing of what exists."""
    splits = available_splits(dataset)
    if split not in splits:
        hint = (
            " For SWE-bench training data use --dataset full --split train."
            if split == "train"
            else ""
        )
        raise ValueError(
            f"dataset {dataset!r} has no {split!r} split; available splits: "
            f"{', '.join(splits)}.{hint}"
        )


def load_split(dataset: str, split: str):
    """Load one split, preferring the local copy and caching by (dataset, split)."""
    validate_split(dataset, split)
    key = (dataset, split)
    if key in _CACHE:
        return _CACHE[key]
    spec = get_spec(dataset)
    data = _load_local(spec, split)
    if data is None:
        data = _load_hub(spec, split)
    _CACHE[key] = data
    logger.info("Loaded %d instances from %s[%s]", len(data), dataset, split)
    return data


def _load_local(spec: DatasetSpec, split: str):
    if split not in spec.local_splits:
        return None
    path = swebench_data_dir() / spec.local_dir
    if not path.exists():
        return None
    if spec.jsonl_glob:
        return _load_jsonl_dir(path, spec.jsonl_glob)
    from datasets import load_from_disk

    loaded = load_from_disk(str(path))
    if hasattr(loaded, "keys"):  # DatasetDict keyed by split name
        return loaded[split] if split in loaded else None
    return loaded if split == spec.single_split else None


def _load_jsonl_dir(path: Path, glob: str) -> list:
    """Read every row from the JSONL files under ``path`` matching ``glob``.

    Multi-SWE-bench encodes each instance's language in the directory name
    (``<lang>/<repo>_dataset.jsonl``) rather than a row field, so we inject it.
    """
    import json

    rows: list = []
    for jsonl in sorted(path.glob(glob)):
        language = jsonl.parent.name if jsonl.parent != path else None
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if language and "language" not in row:
                    row["language"] = language
                rows.append(row)
    return rows


def _load_hub(spec: DatasetSpec, split: str):
    if spec.jsonl_glob:
        raise RuntimeError(
            f"dataset {spec.name!r} is missing locally under "
            f"{swebench_data_dir() / spec.local_dir}. Its raw JSONL files must be "
            f"fetched with snapshot_download (datasets.load_dataset cannot unify "
            f"its per-language schema); run scripts/download_benchmarks.sh."
        )
    if not spec.hf_name:
        raise RuntimeError(
            f"dataset {spec.name!r} split {split!r} is not available locally under "
            f"{swebench_data_dir() / spec.local_dir} and has no hub fallback. "
            f"Download it to $SWEBENCH_DATA_DIR first."
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets package required. Install with: pip install datasets") from exc
    logger.info("Local copy missing; loading %s[%s] from the hub", spec.hf_name, split)
    return load_dataset(spec.hf_name, split=split)


def iter_rows(
    dataset: str,
    split: str,
    limit: Optional[int] = None,
    exclude_ids: Optional[set] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield raw instance rows, dropping excluded ids before the limit is applied."""
    exclude = set(exclude_ids or ())
    count = 0
    for row in load_split(dataset, split):
        if row.get("instance_id") in exclude:
            continue
        yield dict(row)
        count += 1
        if limit is not None and count >= limit:
            break


# SWE-bench-Live/Pro carry these extra fields; keep them so downstream runners
# can build the right eval (test command, per-instance image tag, ...).
_EXTRA_METADATA_FIELDS = (
    "test_cmds",
    "log_parser",
    "dockerhub_tag",
    "before_repo_set_cmd",
    "selected_test_files_to_run",
    "repo_language",
)


# Source-file globs per language, used by the Inspector to list candidate files
# in a repo. SWE-bench is Python-only; Multi-SWE-bench spans 7 more languages.
LANGUAGE_GLOBS: Dict[str, Tuple[str, ...]] = {
    "python": ("**/*.py",),
    "c": ("**/*.c", "**/*.h"),
    "cpp": ("**/*.cpp", "**/*.cc", "**/*.hpp", "**/*.h"),
    "c++": ("**/*.cpp", "**/*.cc", "**/*.hpp", "**/*.h"),
    "go": ("**/*.go",),
    "java": ("**/*.java",),
    "javascript": ("**/*.js", "**/*.jsx", "**/*.mjs"),
    "js": ("**/*.js", "**/*.jsx", "**/*.mjs"),
    "typescript": ("**/*.ts", "**/*.tsx"),
    "ts": ("**/*.ts", "**/*.tsx"),
    "rust": ("**/*.rs",),
}


def source_globs(language: Optional[str]) -> Tuple[str, ...]:
    """Source-file globs for a repo's language; defaults to Python."""
    return LANGUAGE_GLOBS.get((language or "python").lower(), ("**/*.py",))


def _msb_row_to_issue(row: Dict[str, Any]) -> Issue:
    """Map a Multi-SWE-bench row (org/repo/number/base/f2p_tests/...) to an Issue."""
    title = row.get("title", "")
    body = row.get("body", "")
    description = f"{title}\n\n{body}".strip()
    base = row.get("base") or {}
    fail_to_pass = sorted((row.get("f2p_tests") or {}).keys())
    pass_to_pass = sorted((row.get("p2p_tests") or {}).keys())
    metadata: Dict[str, Any] = {
        "language": row.get("language"),
        "repo_language": row.get("language"),
        "org": row.get("org"),
        "repo": row.get("repo"),
        "number": row.get("number"),
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "gold_patch": row.get("fix_patch"),
    }
    return Issue(
        id=row["instance_id"],
        description=description,
        repo_name=f"{row.get('org')}/{row.get('repo')}",
        base_commit=base.get("sha"),
        test_patch=row.get("test_patch"),
        metadata=metadata,
    )


def row_to_issue(row: Dict[str, Any], use_hints: bool = False) -> Issue:
    """Convert a raw dataset row to an Issue (tolerant of the F2P/P2P casing).

    SWE-bench Lite/Verified/full use ``FAIL_TO_PASS``/``PASS_TO_PASS``; Pro uses
    the lowercase names; Multi-SWE-bench uses ``f2p_tests``/``p2p_tests`` and a
    ``base`` dict. ``hints_text`` is only surfaced when ``use_hints`` is set
    (default off: the human hints are excluded to keep the task honest).
    """
    if "f2p_tests" in row:
        return _msb_row_to_issue(row)
    fail_to_pass = row.get("FAIL_TO_PASS", row.get("fail_to_pass"))
    pass_to_pass = row.get("PASS_TO_PASS", row.get("pass_to_pass"))
    metadata: Dict[str, Any] = {
        "version": row.get("version"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
    }
    for field in _EXTRA_METADATA_FIELDS:
        if row.get(field) is not None:
            metadata[field] = row[field]
    return Issue(
        id=row["instance_id"],
        description=row.get("problem_statement", ""),
        repo_name=row.get("repo"),
        base_commit=row.get("base_commit"),
        hints=row.get("hints_text") if use_hints else None,
        test_patch=row.get("test_patch"),
        metadata=metadata,
    )
