"""Two-layer mechanism bucketing and deterministic complete-link clustering."""

from __future__ import annotations

from src.memory.hard_case_buffer import HardCaseRecord
from src.reflection.clustering import cluster_records, qualifying_clusters
from src.skills.embedding_client import EmbeddingClient


def _rec(issue_id: str, mechanism: str, summary: str) -> HardCaseRecord:
    return HardCaseRecord(
        issue_id=issue_id,
        failure_type=mechanism,
        failure_diagnosis={
            "mechanism": mechanism,
            "summary": summary,
            "error_signatures": [],
            "evidence": {},
            "confidence": 0.9,
        },
    )


def test_embedding_cosine_clusters_by_category():
    # Pure, deterministic embed_fn: one-hot over category tokens in the summary.
    def embed_fn(text: str):
        return [1.0 if token in text else 0.0 for token in ("alpha", "beta")]

    client = EmbeddingClient(embed_fn=embed_fn)
    records = [
        _rec("a1", "incomplete_fix", "alpha problem"),
        _rec("a2", "incomplete_fix", "alpha problem"),
        _rec("a3", "incomplete_fix", "alpha problem"),
        _rec("b1", "patch_application_failure", "beta problem"),
        _rec("b2", "patch_application_failure", "beta problem"),
    ]
    clusters = cluster_records(records, embedding_client=client)
    assert sorted(c.size for c in clusters) == [2, 3]

    triggering = qualifying_clusters(clusters, min_cluster_size=3)
    assert len(triggering) == 1
    assert set(triggering[0].instance_ids) == {"a1", "a2", "a3"}


def test_complete_link_prevents_transitive_chaining():
    # A~B and B~C, but A!~C: complete-link must not merge all three.
    vectors = {
        "veca": [1.0, 0.0],
        "vecb": [0.8660254, 0.5],
        "vecc": [0.5, 0.8660254],
    }

    def embed_fn(text: str):
        for key, vec in vectors.items():
            if key in text:
                return vec
        return [0.0, 0.0]

    client = EmbeddingClient(embed_fn=embed_fn)
    records = [
        _rec("x", "incomplete_fix", "veca"),
        _rec("y", "incomplete_fix", "vecb"),
        _rec("z", "incomplete_fix", "vecc"),
    ]
    clusters = cluster_records(records, embedding_client=client, similarity_threshold=0.85)
    assert sorted(cluster.size for cluster in clusters) == [1, 2]


def test_lexical_fallback_clusters_similar_text():
    reason = "git apply failed with malformed hunk context near the fix"
    records = [
        _rec("p1", "patch_application_failure", reason),
        _rec("p2", "patch_application_failure", reason),
        _rec("p3", "patch_application_failure", reason),
        _rec("t1", "incomplete_fix", "assertion mismatch on returned value structure"),
    ]
    clusters = cluster_records(records, embedding_client=None)
    triggering = qualifying_clusters(clusters, min_cluster_size=3)
    assert len(triggering) == 1
    assert set(triggering[0].instance_ids) == {"p1", "p2", "p3"}
    assert triggering[0].dominant_failure_type == "patch_application_failure"


def test_dissimilar_records_stay_singletons():
    records = [
        _rec("a", "incomplete_fix", "completely unrelated alpha topic"),
        _rec("b", "patch_generation_failure", "utterly different beta subject matter"),
    ]
    clusters = cluster_records(records, embedding_client=None)
    assert all(c.size == 1 for c in clusters)
    assert qualifying_clusters(clusters) == []


def test_empty_records_yield_no_clusters():
    assert cluster_records([]) == []
