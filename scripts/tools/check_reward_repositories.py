"""Check exact task commits before spending GPU time on online rewards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reward.online_reward import _repo_mirror, ensure_repository_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=Path, action='append', required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    targets = set()
    for path in args.jsonl:
        for line in path.open():
            row = json.loads(line)
            truth = row['ground_truth']
            if isinstance(truth, str):
                truth = json.loads(truth)
            targets.add((truth['repo_name'], truth['base_commit']))
    mirrors = {repo: _repo_mirror(args.workspace, repo) for repo, _ in sorted(targets)}
    def check(target):
        repo, commit = target
        ensure_repository_commit(mirrors[repo], commit, 'origin')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, sorted(targets)))
    print('REPOSITORIES_READY ' + json.dumps({'commits': len(targets), 'repositories': len(mirrors)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
