# 当前 GRPO 代码与复现说明

本分支基于 `integration/teacher-sft-grpo-20260928` 的 `da4e28f`，整理了正式训练使用的核心修复。保留通用训练入口和检查工具；NeSI 排队、GPU 保留、挂接 allocation、一次性镜像修复和历史任务迁移脚本仍由本地管理。

## 核心变化

- 固定 SWE-bench Full/dev 全部 225 个任务，校验数据摘要，检查与 SFT 和实际 RL 训练记录的交集。每轮生成验证提示，恢复同一轮时复用冻结提示；过长提示直接报错，避免静默丢弃任务。
- 验证期间的基础设施错误直接中止评测，不用零分或组均值填充；训练期间保留原来的组均值处理。
- 测试后端统一 Unicode locale，处理 pydicom 参数化测试路径、pvlib 数值内核差异和 PyVista 离屏渲染；修复后的 SIF 摘要参与缓存键，缓存格式升级。
- 每个候选 checkout 从 Git mirror 创建，获取确切 base commit，同一 checkout 的完整 episode 加锁；失败 checkout 隔离保存后重建，兼容跨文件系统克隆。
- 支持独立检查点目录和逐轮累计步数；只有达到该轮目标检查点才写完成标记。
- 通用镜像预热支持跳过已缓存镜像、重试、进度记录和磁盘余量保护。

## 正式实验设置

| 设置 | 当前值 |
|---|---|
| Controller | Qwen3.5-4B 的 SFT 模型，LoRA rank 16 / alpha 32 |
| 学习率、优势估计 | `1e-5`、RLOO |
| 每步提示数、每提示采样数 | 2、6 |
| 每轮训练步数、抽样提示数 | 64、320（允许同一任务重复抽样） |
| 训练来源、验证来源 | Full/test、Full/dev 全部 225 个任务 |
| 验证每任务候选数、批大小 | 2、2 |
| reward 并行、worker 最大迭代 | 4、3 |
| 保存频率、保留数量 | 每 2 步、最近 2 个可恢复检查点 |

A100 正式配置见 `configs/easyr1_online_grpo_a100_40gb_formal.yaml`。H100 配置保留相同训练参数，但使用兼容的 rollout 内核。配置中的检查点默认目录为相对路径，启动时应指定绝对路径覆盖。

当前 NeSI 环境版本记录（用于对应日志，不代表在其他机器上均已验证）：Python 3.12.14、PyTorch `2.11.0+cu126`、vLLM `0.20.3.dev0+gbc150f502.d20260823.cu126`、transformers `5.8.0`、swebench `4.1.0`、numpy `2.3.5`、ray `2.55.1`、datasets `5.0.1`、peft `0.19.1`、verl `0.3.3.dev0`。EasyR1 是外部 checkout；模型、数据、镜像和 API 凭据单独配置。

## 不依赖集群调度脚本的启动方法

以下命令在已经分配 GPU、激活兼容 Python 环境后执行；先配置 `.env` 中的 worker API 信息。把示例绝对路径换成自己的目录。

```bash
export PYTHON=python
export EASYR1_DIR=/absolute/path/to/EasyR1
export CONTROLLER_MODEL_PATH=/absolute/path/to/sft-model
export ARM_DIR=/absolute/path/to/grpo-state
export CKPT_DIR=/absolute/path/to/grpo-checkpoints
export SWEBENCH_DATA_DIR=/absolute/path/to/swebench-datasets
export SIF_CACHE_DIR=/absolute/path/to/sif-cache
export EASYR1_CONFIG="${PWD}/configs/easyr1_online_grpo_a100_40gb_formal.yaml"
export SELF_EVOLVER_SFT_MANIFEST="${CONTROLLER_MODEL_PATH}/artifacts/data/manifest.json"
export SELF_EVOLVER_FIXED_VAL_DIR="${ARM_DIR}/data/validation_full_dev225"
export EXCLUDE_IDS=/absolute/path/to/train_exclude_ids.txt

python -m src.rl.fixed_validation \
  --prepare-source --source-dir "${SELF_EVOLVER_FIXED_VAL_DIR}" \
  --state-root "${ARM_DIR}" --sft-manifest "${SELF_EVOLVER_SFT_MANIFEST}"

export VAL_INCLUDE_IDS="${SELF_EVOLVER_FIXED_VAL_DIR}/instance_ids.txt"
export SFT_IDS="${SELF_EVOLVER_FIXED_VAL_DIR}/sft_ids.txt"
export VAL_DATASET=full VAL_SPLIT=dev NUM_VAL=225
export EASYR1_VAL_BATCH_SIZE=2 REWARD_WORKERS=4
export NUM_TRAIN=320 SKIP_SFT=1 VAL_BEFORE_TRAIN=false
ROUNDS=3 STEPS_PER_ROUND=64 bash scripts/5_grpo.sh
```

`train_exclude_ids.txt` 应包含保留给最终评测的任务；固定验证和 SFT ID 还会单独强制排除。模型目录需要对应的 SFT 数据 manifest，不能用其他模型的 manifest 替代。首次固定源准备需要本地官方数据或 Hugging Face 访问；之后验证源在本地冻结，manifest 摘要改变会报错。

同一轮中断后重复启动命令，会保留该轮训练提示并恢复检查点。增加 `ROUNDS` 可继续后续轮次，例如第四轮累计到 256 步。恢复原有实验还需要其冻结提示、技能状态、TaskPool、完成标记、检查点及一致的环境；本分支不包含 NeSI 历史目录迁移逻辑。

## 镜像与仓库检查

`prewarm_images.py` 下载注册表镜像并检查缓存覆盖率，不能替代 gold patch / empty patch 控制测试。部分现有验证镜像经过本地依赖修复；在其他机器上直接重新下载原始镜像，不等于拿到了日志对应的修复环境。修复后的 SIF、`.verified.json` 回执和控制测试报告属于独立实验产物，应随环境说明另外交付。

如果已有完整的镜像控制报告，可以开启正式训练前的检查：

```bash
export SELF_EVOLVER_REQUIRE_READY_IMAGES=1
export SELF_EVOLVER_VALIDATION_HEALTH_REPORT=/absolute/path/to/validation_health.json
```

报告需要覆盖全部 225 个验证任务，并与固定源 manifest、当前测试后端代码摘要、SIF 大小和修改时间一致；迁移镜像后应重新运行控制测试，不能直接沿用旧报告。默认报告位置为 `${SELF_EVOLVER_FIXED_VAL_DIR}/validation_health.json`。训练入口使用仓库内的 `scripts/tools/check_training_images_ready.py` 和 `scripts/tools/check_reward_repositories.py`，不依赖未提交的本地诊断脚本。

日志另外交付，不提交到 Git。第一轮旧验证只有 18 个任务；第二、三轮为 Full/dev 225 个任务，每任务 2 个候选，两种设置不能直接比较百分比。进行中的第四轮日志属于截取时刻的快照。

## 本分支验证范围

以下 CPU 回归检查覆盖本次核心修复及相邻 reward 逻辑，不需要申请 GPU：

```bash
python -m pytest -q \
  tests/test_environment_backend.py tests/test_config_cache.py \
  tests/test_fixed_validation.py tests/test_reward_repository_recovery.py \
  tests/test_training_images_ready.py tests/test_validation_infra_guard.py \
  tests/test_prewarm_images.py tests/test_rl_online_reward.py \
  tests/test_preflight_qwen35.py
bash -n scripts/5_grpo.sh scripts/run_rl_controller.sh scripts/tools/grpo_round.sh
```

整理时上述 93 个测试全部通过，shell 语法和差异空白检查通过。旧分支的 `tests/test_rl_dataset.py` 导入已移除的数据集符号，单独在 `da4e28f` 上重现相同收集错误；本次未扩大范围修复整个历史测试集，因此这里的通过结果不代表全仓库测试通过。
