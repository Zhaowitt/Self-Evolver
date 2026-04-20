"""
SWE-bench Lite - Phase 1 补跑 + Phase 2 按 repo 分批评估

Phase 1: 只跑 predictions.json 中 patch 为空的 instance（补跑）
Phase 2: 按 repo 分批构建 Docker 镜像，每批跑完后清理 env/instance 镜像，保留 base 镜像
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("/root/Self-Evolver/run_swebench_v2.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("run_swebench_v2")

sys.path.insert(0, "/root/Self-Evolver")

from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness.run_evaluation import main as swebench_eval
from swebench.harness.utils import load_swebench_dataset

from src.config import get_config
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.orchestrator.orchestrator import ExecutionOrchestrator
from src.benchmark.swebench_runner import SWEBenchRunner

# ── 配置 ───────────────────────────────────────────────────────────────────
DATASET_NAME      = "princeton-nlp/SWE-bench_Lite"
MODEL_NAME        = "self-evolver-gpt4o"
RUN_ID            = "se-lite-v1"
OUTPUT_DIR        = Path("/root/Self-Evolver/swebench_results")
MAX_AGENT_WORKERS = 4   # Phase 1 并发数（磁盘安全起见不超过 5）
MAX_ITERATIONS    = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
predictions_path = OUTPUT_DIR / "predictions.json"

# ── Phase 1: 补跑空 patch 的 instance ──────────────────────────────────────

def cleanup_repo_dir(repo_dir: Path):
    """删除 repo 目录以释放磁盘空间。"""
    try:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup {repo_dir}: {e}")


def generate_patch_for_instance(raw_instance: dict, workspace: Path) -> dict:
    """Clone repo，运行 Self-Evolver，返回 prediction dict，完成后清理 repo 目录。"""
    instance_id = raw_instance["instance_id"]
    log = logging.getLogger(f"worker.{instance_id}")
    log.info("Starting")

    issue = Issue(
        id=instance_id,
        description=raw_instance["problem_statement"],
        repo_name=raw_instance["repo"],
        base_commit=raw_instance["base_commit"],
        hints=raw_instance.get("hints_text"),
        test_patch=raw_instance.get("test_patch"),
        metadata={
            "fail_to_pass": raw_instance.get("FAIL_TO_PASS", "[]"),
            "pass_to_pass": raw_instance.get("PASS_TO_PASS", "[]"),
        },
    )

    repo_dir = workspace / instance_id.replace("/", "_")
    patch = ""
    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        env = ProjectEnvironment(repo_dir)

        if not any(repo_dir.iterdir()):
            repo_url = f"https://github.com/{issue.repo_name}.git"
            log.info(f"Cloning {repo_url}")
            if not env.clone_repo(repo_url):
                raise RuntimeError(f"Clone failed: {repo_url}")

        if issue.base_commit:
            env.checkout_commit(issue.base_commit)

        test_cmd = SWEBenchRunner._build_test_cmd(issue)
        if test_cmd:
            env.test_cmd = test_cmd

        llm = LLMClient()
        orch = ExecutionOrchestrator(env=env, llm_client=llm, max_iterations=MAX_ITERATIONS)
        result = orch.run(issue)

        if result.final_patch:
            patch = result.final_patch.content
            log.info(f"Patch generated ({len(patch)} chars). success={result.success}")
        else:
            for rec in reversed(result.iteration_records):
                if rec.patch_result and rec.patch_result.patch_content:
                    patch = rec.patch_result.patch_content
                    log.info(f"Using last attempted patch ({len(patch)} chars)")
                    break
            if not patch:
                log.warning("No patch generated")

    except Exception as e:
        log.error(f"Exception: {e}", exc_info=True)
    finally:
        # 每个 instance 处理完后立即清理 repo 目录，释放磁盘
        cleanup_repo_dir(repo_dir)
        log.info(f"Cleaned up repo dir: {repo_dir}")

    return {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: MODEL_NAME,
        KEY_PREDICTION: patch,
    }


def run_phase1():
    logger.info("=" * 60)
    logger.info("Phase 1: 补跑空 patch 的 instance")
    logger.info("=" * 60)

    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_swebench_dataset(DATASET_NAME, "test")
    logger.info(f"Loaded {len(dataset)} instances")

    # 加载已有 predictions
    predictions: dict = {}
    if predictions_path.exists():
        with open(predictions_path) as f:
            for pred in json.load(f):
                predictions[pred[KEY_INSTANCE_ID]] = pred
        logger.info(f"Loaded {len(predictions)} existing predictions")

    # 找出需要补跑的：patch 为空或不存在
    todo = [
        inst for inst in dataset
        if not predictions.get(inst["instance_id"], {}).get(KEY_PREDICTION, "").strip()
    ]
    logger.info(f"Instances to (re)run: {len(todo)}")

    if not todo:
        logger.info("No empty patches found, skipping Phase 1.")
        return predictions

    workspace = Path(tempfile.mkdtemp(prefix="swebench_workspace_v2_"))
    logger.info(f"Workspace: {workspace}")

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_AGENT_WORKERS) as pool:
        futures = {
            pool.submit(generate_patch_for_instance, inst, workspace): inst["instance_id"]
            for inst in todo
        }
        for future in as_completed(futures):
            iid = futures[future]
            try:
                pred = future.result()
                predictions[iid] = pred  # 覆盖旧的空 patch
            except Exception as e:
                logger.error(f"{iid} worker failed: {e}")
                predictions[iid] = {
                    KEY_INSTANCE_ID: iid,
                    KEY_MODEL: MODEL_NAME,
                    KEY_PREDICTION: "",
                }
            completed += 1
            logger.info(f"Progress: {completed}/{len(todo)}")
            # 增量保存
            with open(predictions_path, "w") as f:
                json.dump(list(predictions.values()), f, indent=2)

    # 清理 workspace
    cleanup_repo_dir(workspace)

    non_empty = sum(1 for p in predictions.values() if p.get(KEY_PREDICTION, "").strip())
    logger.info(f"Phase 1 done. Non-empty patches: {non_empty}/{len(predictions)}")
    return predictions


# ── Phase 2: 按 repo 分批 Docker 评估 ──────────────────────────────────────

def cleanup_docker_images(except_base: bool = True):
    """
    删除所有 sweb.eval.* 和 sweb.env.* 镜像，保留 sweb.base.* 镜像。
    """
    logger.info("Cleaning up Docker images...")
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True
        )
        images_to_remove = []
        for line in result.stdout.strip().splitlines():
            if "sweb.eval." in line or "sweb.env." in line:
                images_to_remove.append(line)
            elif not except_base and "sweb.base." in line:
                images_to_remove.append(line)

        if images_to_remove:
            logger.info(f"Removing {len(images_to_remove)} Docker images...")
            subprocess.run(
                ["docker", "rmi", "-f"] + images_to_remove,
                capture_output=True
            )
        # 清理悬空层
        subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
        logger.info("Docker cleanup done.")
    except Exception as e:
        logger.warning(f"Docker cleanup error: {e}")


def get_disk_free_gb() -> float:
    stat = os.statvfs("/")
    return stat.f_bavail * stat.f_frsize / (1024 ** 3)


def run_phase2(predictions: dict):
    logger.info("=" * 60)
    logger.info("Phase 2: 按 repo 分批 Docker 评估")
    logger.info("=" * 60)

    # 按 repo 分组 instance_ids
    repo_groups: dict[str, list[str]] = defaultdict(list)
    for iid in predictions:
        repo = iid.rsplit("__", 1)[0]   # e.g. "django__django"
        repo_groups[repo].append(iid)

    # 按 instance 数量从多到少排序（先跑大的）
    sorted_repos = sorted(repo_groups.items(), key=lambda x: -len(x[1]))

    logger.info(f"Total repos to evaluate: {len(sorted_repos)}")
    for repo, iids in sorted_repos:
        logger.info(f"  {repo}: {len(iids)} instances")

    # 汇总所有 resolved 结果
    all_resolved = []
    all_failed = []

    for i, (repo, instance_ids) in enumerate(sorted_repos, 1):
        free_gb = get_disk_free_gb()
        logger.info(f"[{i}/{len(sorted_repos)}] Evaluating repo: {repo} "
                    f"({len(instance_ids)} instances) | Disk free: {free_gb:.1f}GB")

        # 仅评估有非空 patch 的 instance
        non_empty_ids = [
            iid for iid in instance_ids
            if predictions.get(iid, {}).get(KEY_PREDICTION, "").strip()
        ]
        empty_ids = [iid for iid in instance_ids if iid not in non_empty_ids]

        if empty_ids:
            logger.info(f"  Skipping {len(empty_ids)} instances with empty patches")
        if not non_empty_ids:
            logger.info(f"  No non-empty patches for {repo}, skipping Docker eval")
            all_failed.extend(instance_ids)
            continue

        logger.info(f"  Evaluating {len(non_empty_ids)} instances")

        try:
            swebench_eval(
                dataset_name=DATASET_NAME,
                split="test",
                instance_ids=non_empty_ids,
                predictions_path=str(predictions_path),
                max_workers=2,            # 保守起见，每批 2 个并发
                force_rebuild=False,
                cache_level="env",        # env 镜像在同一批内复用
                clean=False,
                open_file_limit=4096,
                run_id=RUN_ID,
                timeout=get_config().docker.timeout,
                namespace=None,
                rewrite_reports=False,
                modal=False,
                report_dir=str(OUTPUT_DIR),
            )
            logger.info(f"  Evaluation done for {repo}")
        except Exception as e:
            logger.error(f"  Evaluation failed for {repo}: {e}")

        # 每批结束后清理 Docker 镜像
        cleanup_docker_images(except_base=True)
        free_after = get_disk_free_gb()
        logger.info(f"  Disk free after cleanup: {free_after:.1f}GB")

    # 最终汇总：解析所有 run_instance.log 里的结果
    logger.info("=" * 60)
    logger.info("Generating final summary...")
    summarize_results()


def summarize_results():
    """解析 run_evaluation 日志，统计 resolved 数量并输出最终结果。"""
    from pathlib import Path
    import json

    eval_dir = Path("/root/Self-Evolver/logs/run_evaluation") / RUN_ID / MODEL_NAME
    if not eval_dir.exists():
        logger.warning(f"Evaluation directory not found: {eval_dir}")
        return

    resolved = []
    failed = []
    errored = []

    for instance_dir in sorted(eval_dir.iterdir()):
        iid = instance_dir.name
        report_file = instance_dir / "report.json"
        log_file = instance_dir / "run_instance.log"

        if report_file.exists():
            try:
                data = json.loads(report_file.read_text())
                if data.get(iid, {}).get("resolved", False):
                    resolved.append(iid)
                else:
                    failed.append(iid)
                continue
            except Exception:
                pass

        # Fallback: 从 log 里判断
        if log_file.exists():
            content = log_file.read_text()
            if "resolved" in content.lower():
                resolved.append(iid)
            elif "BuildImageError" in content or "No space" in content:
                errored.append(iid)
            else:
                failed.append(iid)

    total = len(resolved) + len(failed) + len(errored)
    logger.info(f"Final Results:")
    logger.info(f"  Total evaluated:  {total}")
    logger.info(f"  Resolved:         {len(resolved)}")
    logger.info(f"  Failed:           {len(failed)}")
    logger.info(f"  Errored:          {len(errored)}")
    if total > 0:
        logger.info(f"  Resolve rate:     {len(resolved)/total*100:.1f}%")
    logger.info(f"  (out of 300 total in SWE-bench Lite)")
    if total > 0:
        logger.info(f"  Overall score:    {len(resolved)/300*100:.1f}%")

    # 保存汇总
    summary = {
        "resolved": resolved,
        "failed": failed,
        "errored": errored,
        "total_evaluated": total,
        "total_instances": 300,
        "resolve_rate": len(resolved) / total if total > 0 else 0,
        "overall_score": len(resolved) / 300,
    }
    summary_path = OUTPUT_DIR / "final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


# ── 入口 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both",
                        help="Which phase to run (default: both)")
    args = parser.parse_args()

    predictions = {}
    if predictions_path.exists():
        with open(predictions_path) as f:
            for pred in json.load(f):
                predictions[pred[KEY_INSTANCE_ID]] = pred

    if args.phase in ("1", "both"):
        predictions = run_phase1()

    if args.phase in ("2", "both"):
        run_phase2(predictions)

    logger.info("All done.")
