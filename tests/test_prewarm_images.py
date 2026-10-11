"""CPU cache fill is incremental, disk-bounded, and independent of training."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prewarm_images", ROOT / "scripts/tools/prewarm_images.py")
prewarm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prewarm)


def setup_backend(monkeypatch, tmp_path, count=225, cached=18):
    rows = [{"instance_id": f"task-{i}"} for i in range(count)]
    cache = tmp_path / "sif"
    cache.mkdir()
    calls = []

    class Backend:
        sif_cache_dir = cache

        def __init__(self, engine):
            assert engine == "apptainer"

        def image_key(self, instance):
            return instance["instance_id"]

        def sif_path(self, key):
            return cache / f"{key}.sif"

        def ensure_image(self, instance):
            key = instance["instance_id"]
            calls.append(key)
            self.sif_path(key).write_text("complete image")
            return key

    for row in rows[:cached]:
        (cache / f"{row['instance_id']}.sif").write_text("existing image")
    monkeypatch.setattr(prewarm, "ContainerTestBackend", Backend)
    monkeypatch.setattr(prewarm, "load_source", lambda _path: rows)
    report = tmp_path / "progress.json"
    monkeypatch.setattr(sys, "argv", ["prewarm", "--fixed-validation-source", str(tmp_path / "source"),
                                    "--workers", "1", "--report", str(report)])
    return rows, calls, report, Backend


def test_full_dev_downloads_only_207_missing_and_preserves_existing(monkeypatch, tmp_path):
    _, calls, report, _ = setup_backend(monkeypatch, tmp_path)
    protected = tmp_path / "controller_train.jsonl"
    protected.write_text("frozen training")
    existing = tmp_path / "sif/task-0.sif"
    before = existing.stat().st_mtime_ns
    prewarm.main()
    result = json.loads(report.read_text())
    assert result["target_count"] == 225
    assert result["initial_cached"] == 18
    assert result["initial_missing"] == 207
    assert result["completed_count"] == 207
    assert result["status"] == "complete"
    assert len(calls) == 207
    assert existing.stat().st_mtime_ns == before
    assert protected.read_text() == "frozen training"


def test_low_disk_stops_without_attempting_more_images(monkeypatch, tmp_path):
    _, calls, report, _ = setup_backend(monkeypatch, tmp_path)
    sys.argv += ["--min-free-gib", "256"]
    monkeypatch.setattr(prewarm.shutil, "disk_usage", lambda _path: SimpleNamespace(free=255 * 1024**3))
    with pytest.raises(SystemExit) as exc:
        prewarm.main()
    assert exc.value.code == 2
    assert not calls
    assert json.loads(report.read_text())["status"] == "disk_safety_stop"


def test_transient_error_retries_and_keeps_completed_images(monkeypatch, tmp_path):
    _, _, report, backend = setup_backend(monkeypatch, tmp_path, count=1, cached=0)
    original = backend.ensure_image
    attempts = []

    def fail_once(self, instance):
        attempts.append(instance)
        if len(attempts) == 1:
            raise RuntimeError("transient registry failure")
        return original(self, instance)

    monkeypatch.setattr(backend, "ensure_image", fail_once)
    sys.argv += ["--attempts", "2", "--retry-delay", "0"]
    prewarm.main()
    assert len(attempts) == 2
    assert json.loads(report.read_text())["status"] == "complete"


def test_dry_run_does_not_download_or_write_progress(monkeypatch, tmp_path):
    _, calls, report, _ = setup_backend(monkeypatch, tmp_path)
    sys.argv += ["--dry-run"]
    prewarm.main()
    assert not calls
    assert not report.exists()
