"""Production launches must reject incomplete or stale CPU image preparation."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('images_ready', ROOT / 'scripts/tools/check_training_images_ready.py')
ready = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ready)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    cache = tmp_path / 'cache'
    source.mkdir()
    cache.mkdir()
    (source / 'manifest.json').write_text('pinned validation')
    ids = [f'repo__task-{i}' for i in range(225)]
    monkeypatch.setattr(ready, 'load_source', lambda _: [{'instance_id': i} for i in ids])
    pool = tmp_path / 'pool.json'
    pool.write_text(json.dumps({'base_instances': {'train__task-1': {}}}))
    for i in [*ids, 'train__task-1']:
        (cache / (('sweb.eval.x86_64.' + i).replace('__', '_1776_') + '.sif')).write_bytes(b'image')
    results = []
    for i in ids:
        path = cache / (('sweb.eval.x86_64.' + i).replace('__', '_1776_') + '.sif')
        st = path.stat()
        results.append({'instance_id': i, 'status': 'passed', 'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns})
    health = tmp_path / 'health.json'
    health.write_text(json.dumps({'status': 'ready', 'target_count': 225, 'results': results,
                                'source_manifest_sha256': hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
                                'evaluator_sha256': hashlib.sha256((ROOT / 'src/environment/test_backend.py').read_bytes()).hexdigest()}))
    return pool, source, cache, health


def test_complete_preparation_is_ready(preparation):
    assert ready.check_ready(*preparation) == {'training': 1, 'validation': 225, 'ready': True}


def test_missing_training_image_blocks_launch(preparation):
    pool, source, cache, health = preparation
    (cache / 'sweb.eval.x86_64.train_1776_task-1.sif').unlink()
    with pytest.raises(RuntimeError, match='1 task images missing'):
        ready.check_ready(*preparation)


def test_changed_validation_image_blocks_launch(preparation):
    _, _, cache, _ = preparation
    (cache / 'sweb.eval.x86_64.repo_1776_task-1.sif').write_bytes(b'different environment')
    with pytest.raises(RuntimeError, match='changed or failed'):
        ready.check_ready(*preparation)


def test_incomplete_health_report_blocks_launch(preparation):
    _, _, _, health = preparation
    report = json.loads(health.read_text())
    report['results'].pop()
    health.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='missing or duplicate'):
        ready.check_ready(*preparation)


def test_round_checks_do_not_wait_for_unselected_pool_tasks(preparation, tmp_path):
    pool, _, _, _ = preparation
    data = json.loads(pool.read_text())
    data['base_instances']['future__task-2'] = {}
    pool.write_text(json.dumps(data))
    prompts = tmp_path / 'round.jsonl'
    prompts.write_text(json.dumps({'extra_info': {'instance_id': 'train__task-1'}}) + '\n')
    assert ready.check_ready(*preparation, train_jsonl=prompts)['training'] == 1
    with pytest.raises(RuntimeError, match='1 task images missing'):
        ready.check_ready(*preparation)


def test_round_missing_selected_image_still_blocks(preparation, tmp_path):
    _, _, cache, _ = preparation
    prompts = tmp_path / 'round.jsonl'
    prompts.write_text(json.dumps({'extra_info': {'instance_id': 'train__task-1'}}) + '\n')
    (cache / 'sweb.eval.x86_64.train_1776_task-1.sif').unlink()
    with pytest.raises(RuntimeError, match='1 task images missing'):
        ready.check_ready(*preparation, train_jsonl=prompts)
    assert ready.check_ready(*preparation, validation_only=True)['training'] == 0


def test_missing_training_metadata_is_not_treated_as_ready(preparation, tmp_path):
    prompts = tmp_path / 'round.jsonl'
    prompts.write_text(json.dumps({'extra_info': {}}) + '\n')
    with pytest.raises(ValueError, match='instance ID'):
        ready.check_ready(*preparation, train_jsonl=prompts)
