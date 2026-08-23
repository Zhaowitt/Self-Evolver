"""Two-layer failure clustering with fielded complete-link similarity."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from src.config import get_config
from src.memory.hard_case_buffer import HardCaseRecord
from src.skills.embedding_client import EmbeddingClient, cosine_similarity, local_text_similarity
from src.skills.file_lock import file_lock


SECONDARY_SIMILARITY_THRESHOLD = 0.62
MIN_CLUSTER_SIZE = 3
MIN_UNIQUE_ISSUES = 2
MIN_COHESION = 0.62
FIELD_WEIGHTS = {
    "summary": 0.55,
    "error_signature": 0.25,
    "verification_pattern": 0.10,
    "retry_pattern": 0.10,
}


@dataclass
class FailureCluster:
    """A secondary pattern cluster within one exact mechanism bucket."""

    members: List[HardCaseRecord]
    mechanism: str = "unknown"
    cluster_id: str = ""
    cohesion: float = 1.0
    shared_pattern: Dict[str, object] = field(default_factory=dict)
    representative_indices: List[int] = field(default_factory=list)
    diagnosis_available: bool = True

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def instance_ids(self) -> List[str]:
        return [record.issue_id for record in self.members if record.issue_id]

    @property
    def unique_issue_count(self) -> int:
        return len(set(self.instance_ids))

    @property
    def dominant_failure_type(self) -> str:
        return self.mechanism

    @property
    def representatives(self) -> List[HardCaseRecord]:
        return [self.members[index] for index in self.representative_indices]

    def summary_text(self) -> str:
        return "\n".join(cluster_summary(record) for record in self.members)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "mechanism": self.mechanism,
            "size": self.size,
            "unique_issue_count": self.unique_issue_count,
            "cohesion": round(self.cohesion, 6),
            "shared_pattern": self.shared_pattern,
            "representative_issue_ids": [item.issue_id for item in self.representatives],
        }


def cluster_summary(record: HardCaseRecord) -> str:
    diagnosis = record.failure_diagnosis or {}
    return str(diagnosis.get("summary") or "").strip()


def cluster_records(
    records: List[HardCaseRecord],
    embedding_client: Optional[EmbeddingClient] = None,
    similarity_threshold: float = SECONDARY_SIMILARITY_THRESHOLD,
    cosine_threshold: Optional[float] = None,
    lexical_threshold: Optional[float] = None,
    registry_path: Optional[Path] = None,
) -> List[FailureCluster]:
    """Bucket exactly by mechanism, then run deterministic complete-link.

    The old threshold parameters are accepted for call compatibility; the new
    fielded threshold is selected explicitly via ``similarity_threshold``.
    """
    if not records:
        return []
    buckets: Dict[str, List[HardCaseRecord]] = {}
    legacy: List[HardCaseRecord] = []
    for record in records:
        diagnosis = record.failure_diagnosis or {}
        mechanism = str(diagnosis.get("mechanism") or "").strip()
        if not mechanism:
            legacy.append(record)
            continue
        buckets.setdefault(mechanism, []).append(record)

    clusters: List[FailureCluster] = []
    for mechanism in sorted(buckets):
        members = sorted(
            buckets[mechanism],
            key=lambda item: (item.issue_id, item.created_at, cluster_summary(item)),
        )
        similarity = _fielded_similarity_matrix(members, embedding_client)
        for indices in _complete_link(similarity, similarity_threshold):
            grouped = [members[index] for index in indices]
            pair_scores = [
                similarity[left][right]
                for offset, left in enumerate(indices)
                for right in indices[offset + 1 :]
            ]
            cohesion = min(pair_scores) if pair_scores else 1.0
            representatives = _representative_indices(grouped, similarity, indices)
            shared = _shared_pattern(grouped)
            clusters.append(
                FailureCluster(
                    members=grouped,
                    mechanism=mechanism,
                    cohesion=cohesion,
                    shared_pattern=shared,
                    representative_indices=representatives,
                )
            )

    # Legacy records remain readable but cannot qualify for v2 reflection.
    for record in legacy:
        clusters.append(
            FailureCluster(
                members=[record],
                mechanism="unknown",
                diagnosis_available=False,
                shared_pattern={"legacy_schema": True},
                representative_indices=[0],
            )
        )

    registry = StableClusterRegistry(registry_path) if registry_path else None
    for cluster in clusters:
        cluster.cluster_id = (
            registry.resolve(cluster) if registry and cluster.diagnosis_available else _stable_id(cluster)
        )
    if registry:
        registry.save()
    clusters.sort(key=lambda item: (item.mechanism, -item.size, item.cluster_id))
    return clusters


def qualifying_clusters(
    clusters: List[FailureCluster],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_unique_issues: int = MIN_UNIQUE_ISSUES,
    min_cohesion: float = MIN_COHESION,
) -> List[FailureCluster]:
    """Apply evidence gates before a cluster can reach the Reflector."""
    return [
        cluster
        for cluster in clusters
        if cluster.diagnosis_available
        and cluster.mechanism != "environment_failure"
        and cluster.size >= min_cluster_size
        and cluster.unique_issue_count >= min_unique_issues
        and cluster.cohesion >= min_cohesion
    ]


def default_cluster_registry_path() -> Path:
    configured = os.getenv("SELF_EVOLVER_CLUSTER_REGISTRY", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(get_config().environment.workspace_dir) / "cluster_registry.json"


class StableClusterRegistry:
    """Keep stable IDs and reflection decisions across round boundaries."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or default_cluster_registry_path())
        self.entries = self._load()

    def resolve(self, cluster: FailureCluster) -> str:
        signature = _pattern_text(cluster.shared_pattern)
        candidates = [
            (cluster_id, entry)
            for cluster_id, entry in self.entries.items()
            if entry.get("mechanism") == cluster.mechanism
        ]
        ranked = sorted(
            (
                local_text_similarity(signature, str(entry.get("signature") or "")),
                cluster_id,
            )
            for cluster_id, entry in candidates
        )
        if ranked and ranked[-1][0] >= 0.72:
            cluster_id = ranked[-1][1]
        else:
            cluster_id = _stable_id(cluster)
        previous = dict(self.entries.get(cluster_id) or {})
        self.entries[cluster_id] = {
            **previous,
            "mechanism": cluster.mechanism,
            "signature": signature,
            "shared_pattern": cluster.shared_pattern,
            "last_size": cluster.size,
            "last_unique_issue_count": cluster.unique_issue_count,
            "last_cohesion": round(cluster.cohesion, 6),
        }
        return cluster_id

    def record_decision(self, cluster_id: str, decision: Dict[str, object]) -> None:
        if cluster_id in self.entries:
            self.entries[cluster_id]["last_reflector_decision"] = decision

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "stable_cluster_registry_v1", "clusters": self.entries}
        with file_lock(self.path):
            temp = self.path.with_name(self.path.name + ".tmp")
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, self.path)

    def _load(self) -> Dict[str, Dict[str, object]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        clusters = payload.get("clusters") if isinstance(payload, dict) else None
        return dict(clusters) if isinstance(clusters, dict) else {}


def _fielded_similarity_matrix(
    records: Sequence[HardCaseRecord],
    embedding_client: Optional[EmbeddingClient],
) -> List[List[float]]:
    fields = [_diagnosis_fields(record) for record in records]
    summary_vectors = None
    if embedding_client:
        summaries = [item["summary"] for item in fields]
        if all(summaries):
            summary_vectors = embedding_client.embed_many(summaries)
    size = len(records)
    matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            scores: Dict[str, float] = {}
            for name in FIELD_WEIGHTS:
                a, b = fields[left][name], fields[right][name]
                if not a or not b:
                    continue
                if name == "summary" and summary_vectors:
                    scores[name] = cosine_similarity(summary_vectors[left], summary_vectors[right])
                else:
                    scores[name] = local_text_similarity(a, b)
            total_weight = sum(FIELD_WEIGHTS[name] for name in scores)
            value = (
                sum(scores[name] * FIELD_WEIGHTS[name] for name in scores) / total_weight
                if total_weight
                else 0.0
            )
            matrix[left][right] = matrix[right][left] = max(0.0, min(1.0, value))
    return matrix


def _diagnosis_fields(record: HardCaseRecord) -> Dict[str, str]:
    diagnosis = record.failure_diagnosis or {}
    evidence = diagnosis.get("evidence") if isinstance(diagnosis.get("evidence"), dict) else {}
    errors = diagnosis.get("error_signatures") or []
    verification = {
        "f2p_progress": evidence.get("f2p_progress") or [],
        "p2p_regression": evidence.get("p2p_regression") or [],
        "patch_applied": evidence.get("patch_applied") or [],
    }
    retry = evidence.get("retry_pattern") or {}
    return {
        "summary": str(diagnosis.get("summary") or ""),
        "error_signature": " ".join(str(item) for item in errors),
        "verification_pattern": (
            json.dumps(verification, sort_keys=True)
            if any(verification.values())
            else ""
        ),
        "retry_pattern": json.dumps(retry, sort_keys=True) if retry else "",
    }


def _complete_link(matrix: List[List[float]], threshold: float) -> List[List[int]]:
    groups: List[List[int]] = [[index] for index in range(len(matrix))]
    while True:
        choices: List[tuple[float, tuple[int, ...], int, int]] = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                score = min(matrix[a][b] for a in groups[left] for b in groups[right])
                if score >= threshold:
                    choices.append((score, tuple(groups[left] + groups[right]), left, right))
        if not choices:
            break
        _, _, left, right = max(choices, key=lambda item: (item[0], tuple(-x for x in item[1])))
        groups[left] = sorted(groups[left] + groups[right])
        groups.pop(right)
    return sorted(groups, key=lambda group: (group[0], len(group)))


def _representative_indices(
    grouped: List[HardCaseRecord],
    matrix: List[List[float]],
    original_indices: List[int],
) -> List[int]:
    medoid_position = max(
        range(len(grouped)),
        key=lambda pos: (
            sum(matrix[original_indices[pos]][other] for other in original_indices),
            grouped[pos].issue_id,
        ),
    )
    result = [medoid_position]
    ranked = sorted(
        range(len(grouped)),
        key=lambda pos: matrix[original_indices[medoid_position]][original_indices[pos]],
        reverse=True,
    )
    for position in ranked:
        if position == medoid_position:
            continue
        if grouped[position].issue_id in {grouped[item].issue_id for item in result}:
            continue
        result.append(position)
        if len(result) >= 3:
            break
    return result


def _shared_pattern(records: Sequence[HardCaseRecord]) -> Dict[str, object]:
    summaries = [cluster_summary(record) for record in records]
    token_sets = [set(_tokens(summary)) for summary in summaries if summary]
    common = sorted(set.intersection(*token_sets)) if token_sets else []
    errors: Dict[str, int] = {}
    for record in records:
        for signature in (record.failure_diagnosis or {}).get("error_signatures") or []:
            errors[str(signature)] = errors.get(str(signature), 0) + 1
    shared_errors = [
        value for value, count in sorted(errors.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:5]
    return {
        "summary_keywords": common[:20],
        "shared_error_signatures": shared_errors,
        "verification_patterns": sorted(
            {
                _diagnosis_fields(record)["verification_pattern"]
                for record in records
            }
        )[:5],
        "retry_patterns": sorted(
            {_diagnosis_fields(record)["retry_pattern"] for record in records}
        )[:5],
    }


def _stable_id(cluster: FailureCluster) -> str:
    digest = hashlib.sha256(
        f"{cluster.mechanism}\0{_pattern_text(cluster.shared_pattern)}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{cluster.mechanism}-{digest}"


def _pattern_text(pattern: Dict[str, object]) -> str:
    return json.dumps(pattern, sort_keys=True, ensure_ascii=False)


def _tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower()) if len(token) > 2]
