# Self-Evolver

当前正式 GRPO 的核心修复、训练参数和不依赖集群调度的启动方法见 [复现说明](docs/GRPO_REPRODUCTION_ZH.md)。

## 安装

```bash
pip install -e .
cp .env.example .env
```

## 分阶段运行

```bash
export EASYR1_DIR=/path/to/EasyR1 EXPERIMENT=full_method SEED=0
bash scripts/0_check.sh
bash scripts/1_download.sh
bash scripts/2_splits.sh
bash scripts/3_flaky.sh
ANNOTATOR_MODELS=gpt-4o,gpt-4.1 bash scripts/4_sft.sh
ROUNDS=3 STEPS_PER_ROUND=64 bash scripts/5_grpo.sh
bash scripts/6_eval.sh
bash scripts/7_metrics.sh
```

## 一键运行

```bash
ANNOTATOR_MODELS=gpt-4o,gpt-4.1 bash scripts/pipeline.sh
STAGES="grpo eval metrics" bash scripts/pipeline.sh
```

## 消融

```bash
EXPERIMENT=no_subset_gate  TRAIN_SUBSET_GATE=0 bash scripts/pipeline.sh
EXPERIMENT=old_reward      REWARD_CONFIG=configs/reward_config_legacy.yaml bash scripts/pipeline.sh
EXPERIMENT=no_sft          SKIP_SFT=1 bash scripts/pipeline.sh
EXPERIMENT=one_level       CLUSTER_LEVELS=1 bash scripts/pipeline.sh
EXPERIMENT=no_contribution REWARD_CONFIG=configs/reward_config_no_contribution.yaml bash scripts/pipeline.sh
EXPERIMENT=old_sampling    SAMPLING_MODE=legacy bash scripts/pipeline.sh
```

## 基线

```bash
bash scripts/baselines/run_zero_shot.sh
bash scripts/baselines/run_mas_static.sh
bash scripts/baselines/run_task_evolution.sh
bash scripts/baselines/run_skill_evolution.sh
bash scripts/baselines/run_full_method.sh
bash scripts/baselines/run_all_experiments.sh
DATASETS="swebench_live multi_swe_bench" bash scripts/1_download.sh
bash scripts/baselines/run_transfer_eval.sh
PREDICTIONS=runs/<run>/predictions.json bash scripts/tools/evaluate.sh
```

## NeSI

```bash
ANNOTATOR_MODELS=gpt-4o,gpt-4.1 sbatch --account=<project> scripts/slurm_nesi.sl
```
