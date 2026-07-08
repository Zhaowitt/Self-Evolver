"""Failure clustering over hard-case records via union-find similarity.

Two hard cases join the same cluster when their failure summaries are similar:
embedding cosine >= COSINE_THRESHOLD when an EmbeddingClient yields vectors,
otherwise lexical overlap >= LEXICAL_THRESHOLD. Clusters of at least
MIN_CLUSTER_SIZE members are the ones that trigger the Reflector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.memory.hard_case_buffer import HardCaseRecord
from src.skills.embedding_client import (
    EmbeddingClient,
    cosine_similarity,
    local_text_similarity,
)

COSINE_THRESHOLD = 0.85
LEXICAL_THRESHOLD = 0.55
MIN_CLUSTER_SIZE = 3


@dataclass
class FailureCluster:
    """A group of hard cases sharing a failure pattern."""

    members: List[HardCaseRecord]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def instance_ids(self) -> List[str]:
        return [record.issue_id for record in self.members if record.issue_id]

    @property
    def dominant_failure_type(self) -> str:
        counts: Dict[str, int] = {}
        for record in self.members:
            counts[record.failure_type] = counts.get(record.failure_type, 0) + 1
        return max(sorted(counts), key=lambda key: counts[key]) if counts else "unknown"

    def summary_text(self) -> str:
        return "\n".join(cluster_summary(record) for record in self.members)


class _UnionFind:
    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def cluster_summary(record: HardCaseRecord) -> str:
    """Text used to compare two hard cases: failure type, reason, and evidence."""
    parts = [record.failure_type, record.reason]
    parts.extend(record.verification_statuses)
    parts.extend(record.errors)
    return " ".join(part for part in parts if part).strip()


def cluster_records(
    records: List[HardCaseRecord],
    embedding_client: Optional[EmbeddingClient] = None,
    cosine_threshold: float = COSINE_THRESHOLD,
    lexical_threshold: float = LEXICAL_THRESHOLD,
) -> List[FailureCluster]:
    """Union-find clustering of hard cases; returns clusters largest first."""
    if not records:
        return []
    summaries = [cluster_summary(record) for record in records]
    similar = _pair_similarity_fn(summaries, embedding_client, cosine_threshold, lexical_threshold)

    union_find = _UnionFind(len(records))
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if similar(i, j):
                union_find.union(i, j)

    groups: Dict[int, List[HardCaseRecord]] = {}
    for index, record in enumerate(records):
        groups.setdefault(union_find.find(index), []).append(record)

    clusters = [FailureCluster(members=members) for members in groups.values()]
    clusters.sort(key=lambda cluster: (-cluster.size, cluster.instance_ids[:1]))
    return clusters


def qualifying_clusters(
    clusters: List[FailureCluster],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> List[FailureCluster]:
    """Clusters large enough to trigger the Reflector."""
    return [cluster for cluster in clusters if cluster.size >= min_cluster_size]


def _pair_similarity_fn(
    summaries: List[str],
    embedding_client: Optional[EmbeddingClient],
    cosine_threshold: float,
    lexical_threshold: float,
) -> Callable[[int, int], bool]:
    vectors = embedding_client.embed_many(summaries) if embedding_client else None
    if vectors:
        return lambda i, j: cosine_similarity(vectors[i], vectors[j]) >= cosine_threshold
    return lambda i, j: local_text_similarity(summaries[i], summaries[j]) >= lexical_threshold
