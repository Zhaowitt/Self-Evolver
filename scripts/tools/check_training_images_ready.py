#!/usr/bin/env python3
"""Block a production launch unless every eligible image and validation control is ready."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.fixed_validation import load_source


def check_ready(task_pool, source_dir, cache_dir, health_report, train_jsonl=None, validation_only=False):
    pool = json.loads(task_pool.read_text())['base_instances']
    training = set(pool)
    if validation_only:
        training = set()
    elif train_jsonl is not None:
        training = set()
        for line in train_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get('instance_id') or row.get('extra_info', {}).get('instance_id')
            if not isinstance(instance_id, str) or instance_id not in pool:
                raise ValueError('Training prompt has a missing or unknown instance ID')
            training.add(instance_id)
        if not training:
            raise ValueError('Training prompt file is empty')
    rows = load_source(source_dir)
    validation = {r['instance_id'] for r in rows}
    missing = []
    for i in sorted(training | validation):
        path = cache_dir / (('sweb.eval.x86_64.' + i.lower()).replace('__', '_1776_') + '.sif')
        if not path.is_file() or not path.stat().st_size:
            missing.append(i)
    if missing:
        raise RuntimeError(f'{len(missing)} task images missing: {missing[:20]}')
    health = json.loads(health_report.read_text())
    manifest_hash = hashlib.sha256((source_dir / 'manifest.json').read_bytes()).hexdigest()
    evaluator_hash = hashlib.sha256((ROOT / 'src/environment/test_backend.py').read_bytes()).hexdigest()
    if (health.get('status') != 'ready' or health.get('target_count') != 225
            or health.get('source_manifest_sha256') != manifest_hash
            or health.get('evaluator_sha256') != evaluator_hash):
        raise RuntimeError('All 225 validation images need current passing controls')
    checks = {r['instance_id']: r for r in health['results']}
    if len(health['results']) != 225 or set(checks) != validation:
        raise RuntimeError('Validation health report has missing or duplicate IDs')
    for i, receipt in checks.items():
        path = cache_dir / (('sweb.eval.x86_64.' + i.lower()).replace('__', '_1776_') + '.sif')
        st = path.stat()
        if (receipt.get('status') != 'passed' or receipt.get('size_bytes') != st.st_size
                or receipt.get('mtime_ns') != st.st_mtime_ns):
            raise RuntimeError(f'Validation image changed or failed since controls: {i}')
    return {'training': len(training), 'validation': len(validation), 'ready': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-pool', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--health-report', type=Path, required=True)
    scope = p.add_mutually_exclusive_group()
    scope.add_argument('--train-jsonl', type=Path, help='Check only this round\'s actual training tasks')
    scope.add_argument('--validation-only', action='store_true')
    a = p.parse_args()
    try:
        print('IMAGES_READY ' + json.dumps(check_ready(
            a.task_pool, a.source_dir, a.cache_dir, a.health_report,
            train_jsonl=a.train_jsonl, validation_only=a.validation_only)))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'IMAGES_NOT_READY: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
