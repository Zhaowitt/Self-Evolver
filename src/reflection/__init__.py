"""Failure clustering and post-execution reflection (lazy public exports)."""

__all__ = [
    "FailureCluster",
    "cluster_records",
    "qualifying_clusters",
    "Reflector",
    "ReflectionResult",
]


def __getattr__(name):
    # Lazy imports let the hard-case writer use the leaf diagnosis module
    # without causing hard_case_buffer <-> clustering package cycles.
    if name in {"FailureCluster", "cluster_records", "qualifying_clusters"}:
        from src.reflection import clustering

        return getattr(clustering, name)
    if name in {"Reflector", "ReflectionResult"}:
        from src.reflection import reflector

        return getattr(reflector, name)
    raise AttributeError(name)
