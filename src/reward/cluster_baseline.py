"""Round-frozen baselines for cluster-relative skill contribution."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.config import get_config
from src.skills.file_lock import file_lock


CLUSTER_BASELINE_SCHEMA_VERSION = "cluster_baseline_v1"


def default_cluster_baseline_path() -> Path:
    configured = os.getenv("SELF_EVOLVER_CLUSTER_BASELINES", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(get_config().environment.workspace_dir) / "cluster_baselines.json"


class ClusterBaselineStore:
    """Persist EMA task rewards, updated once after a completed round."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or default_cluster_baseline_path())

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        clusters = payload.get("clusters") if isinstance(payload, dict) else None
        return dict(clusters) if isinstance(clusters, dict) else {}

    def frozen_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Detached copy, safe to reuse throughout one round."""
        return json.loads(json.dumps(self.load()))

    def update_round(
        self,
        samples: Iterable[Mapping[str, Any]],
        round_id: str,
        alpha: float = 0.3,
    ) -> Dict[str, Dict[str, Any]]:
        """Update baselines from contribution-free ``task_reward`` values."""
        grouped: Dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            cluster_id = str(sample.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            try:
                grouped[cluster_id].append(float(sample.get("task_reward")))
            except (TypeError, ValueError):
                continue
        if not grouped:
            return self.load()

        with file_lock(self.path):
            clusters = self.load()
            for cluster_id, values in sorted(grouped.items()):
                mean = sum(values) / len(values)
                old = dict(clusters.get(cluster_id) or {})
                if round_id and old.get("last_round_id") == round_id:
                    continue  # this round was already folded in (rerun)
                old_count = int(old.get("count", 0) or 0)
                old_ema = float(old.get("ema", 0.0) or 0.0)
                ema = mean if old_count <= 0 else (1.0 - alpha) * old_ema + alpha * mean
                clusters[cluster_id] = {
                    "ema": round(ema, 8),
                    "count": old_count + len(values),
                    "last_round_id": round_id,
                    "last_round_count": len(values),
                    "last_round_mean_task_reward": round(mean, 8),
                }
            self._save(clusters)
        return clusters

    def _save(self, clusters: Dict[str, Dict[str, Any]]) -> None:
        payload = {
            "schema_version": CLUSTER_BASELINE_SCHEMA_VERSION,
            "clusters": {key: clusters[key] for key in sorted(clusters)},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, self.path)
