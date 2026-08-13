# Self-Evolver 阶段实验总结

> 数据截点：完成 `global_step_960`（Round 11）。正在重跑的 Round 12 不计入本文结果。

## 1. 实验目标与方法

本实验在 SWE-bench `full/test` 上训练一个高层 **Controller**。Controller 不直接编写代码，而是根据任务选择已有 skill、生成修复策略并分配执行预算；`gpt-4o-mini` 作为 Inspector / Patch Generator，生成补丁后由 SWE-bench Docker harness 执行 F2P（FAIL_TO_PASS）和 P2P（PASS_TO_PASS）测试。在线 reward 再用于 GRPO 更新 Controller。

每轮结束后，系统还会把失败任务聚类：一方面更新 TaskPool 中下一轮的任务采样权重，另一方面由 Reflector 创建、更新或淘汰 skill。

```text
SWE-bench task
  -> Controller（选择 skill / strategy / budget）
  -> gpt-4o-mini（检查代码并生成 patch）
  -> SWE-bench harness（F2P / P2P）
  -> Online reward
  -> GRPO 更新 Controller
  -> TaskPool + Skill evolution
```

### 主要配置

| 配置项 | 设置 |
|---|---|
| Controller | Qwen2.5-1.5B-Instruct + LoRA |
| Worker / Patch Generator | `gpt-4o-mini` |
| 训练算法 | GRPO |
| 每轮训练长度 | 64 optimizer steps |
| Rollout batch size | 4 |
| 每个任务的 rollout 数 | 3 |
| 每轮候选任务 | 320 |
| 每轮实际 rollout | 约 740–760 |
| 最大修复迭代 | 3 |
| Validation | 固定 18 个任务，每题 1 个 rollout |
| TaskPool 采样 | 50% 均匀采样 + 50% 失败加权采样 |

64 step 是一次自进化周期，而不是一次独立训练。模型每轮都从上一轮 checkpoint 继续，因此本文 Round 6–11 实际对应连续的 step 576–960。

Online reward 为：

```text
0.5 * resolved
+ 0.2 * F2P fraction
+ 0.1 * P2P no-regression
+ 0.1 * cost efficiency
+ 0.1 * process score
```

因此 reward 非零不一定代表任务完全解决，`Resolved` 才表示完整通过目标测试且没有回归。

## 2. 分轮实验结果

| Round | Steps | Rollouts | 不同任务 | Train reward | Train resolved | Val reward | Val resolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | 576–640 | 739 | 193 | 0.1740 | 82/739（11.10%） | 0.1272 | 1/18（5.56%） |
| 7 | 640–704 | 747 | 194 | 0.1588 | 68/747（9.10%） | 0.1453 | 1/18（5.56%） |
| 8 | 704–768 | 745 | 192 | 0.1744 | 82/745（11.01%） | 0.1540 | 1/18（5.56%） |
| 9 | 768–832 | 753 | 203 | 0.1826 | 88/753（11.69%） | 0.1416 | 1/18（5.56%） |
| 10 | 832–896 | 761 | 219 | **0.1992** | **104/761（13.67%）** | 0.1292 | 1/18（5.56%） |
| 11 | 896–960 | 755 | **240** | 0.1790 | 84/755（11.13%） | 0.1215 | 1/18（5.56%） |

Round 10 取得了目前最高的训练 reward 和 resolved rate，但 Round 11 又有所回落，说明训练指标仍有明显波动。固定 validation 在所有轮次均为 1/18，当前没有证据表明训练提升已经稳定泛化到验证任务。由于 validation 只有 18 条，每多解决一个任务就会改变 5.56 个百分点，因此它更适合作为低成本健康检查，而不是最终结论。

## 3. 每轮任务分布变化

下表统计实际进入训练 rollout 的 repository 分布。为保持可读性，仅展开占比较高的四个仓库，其余 8 个仓库合并为“其他”。

| Round | Django | SymPy | scikit-learn | Sphinx | 其他 8 个仓库 | 不同任务 |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 23.5% | 11.5% | 12.9% | 5.7% | 46.4% | 193 |
| 7 | 27.0% | 15.0% | 9.8% | 6.2% | 42.0% | 194 |
| 8 | 28.1% | 17.3% | 9.7% | 7.5% | 37.4% | 192 |
| 9 | 26.3% | 15.8% | 10.4% | 12.0% | 35.5% | 203 |
| 10 | 29.0% | 14.3% | 10.5% | 6.7% | 39.5% | 219 |
| 11 | 29.4% | 12.3% | 9.3% | 8.3% | 40.7% | 240 |

每轮均覆盖 12 个 repository。Django 始终占比最高（约 24%–29%），这是原始任务分布与失败加权共同作用的结果；但仍有约 71%–76% 的 rollout 来自其他仓库。Round 9–11 的不同任务数从 203 增长到 240，表明 TaskPool 加权并没有把训练完全锁定在少量 hard cases 上。

Task evolution 的具体机制为：Reflector 从失败聚类中产生 `instance_boosts`，下一轮用 50% TaskPool 加权采样重访 hard cases，同时保留 50% 均匀采样维持覆盖度，每个实例每轮最多重复 3 次。

## 4. Skill Evolution 结果

| Round | Hard cases | 失败簇 | 有效簇 | Skill 操作 |
|---|---:|---:|---:|---|
| 6 | 657 | 45 | 5 | 更新 `max_iterations_management`；创建 `unknown_failure_analysis` |
| 7 | 679 | 55 | 5 | 更新 `unknown_failure_analysis`；创建 `test_failure_analysis` |
| 8 | 663 | 50 | 5 | 更新上述两个 failure-analysis skills |
| 9 | 665 | 44 | 5 | 更新 `test_failure_analysis`；重新创建 `max_iterations_management` |
| 10 | 657 | 37 | 3 | 更新两个 failure-analysis skills；淘汰 `max_iterations_management` |
| 11 | 671 | 45 | 6 | 再次更新两个 failure-analysis skills |

下面是 Reflector 实际生成并写入 Skill Bank 的内容，而不是人工整理后的模板。

### 示例 1：Unknown Failure Analysis

```markdown
# Unknown Failure Analysis

## Description
This skill provides techniques for identifying and addressing unknown failures
in test cases, particularly those that lack clear error messages. It focuses on
systematic troubleshooting and hypothesis testing.

## How to Apply
1. Gather all relevant information about the failure, including test case details
   and environment settings.
2. Formulate hypotheses about potential causes based on the context of the failure.
3. Systematically test each hypothesis by modifying one variable at a time and
   observing the outcomes.
4. Document findings and refine hypotheses based on results to narrow down the
   root cause.
5. Share insights with the team to enhance collective knowledge on handling
   similar unknown failures in the future.

Target failure type: unknown
```

### 示例 2：Test Failure Analysis

```markdown
# Test Failure Analysis

## Description
This skill provides a structured approach to analyze and resolve test failures,
particularly those related to configuration limits like max_iterations. It
emphasizes understanding the context of failures and adjusting parameters accordingly.

## How to Apply
1. Review the test failure logs to identify the specific conditions under which
   the failure occurs.
2. Check the configuration settings, especially parameters like max_iterations,
   to ensure they are set appropriately for the test case.
3. Adjust the parameters as needed and rerun the tests to observe if the failure persists.
4. If the failure continues, consider other dependencies or environmental factors
   that may be influencing the test outcome.

Target failure type: test_failure
```

这些结果证明 skill 的创建、更新和淘汰流程已经接通。不过生成内容目前仍偏通用，对具体 repository 和代码模式的指导有限。这可能是 skill evolution 已持续发生、但 validation 尚未明显改善的原因之一。

## 5. 阶段结论

1. 系统已经实现连续 GRPO、在线 SWE-bench reward、TaskPool evolution 和 Skill evolution；每轮约产生 750 条真实修复轨迹，而不是只训练 64 个任务。
2. Train resolved 在 9.10%–13.67% 之间波动，Round 10 最好，但尚未形成持续上升趋势。
3. Validation 始终为 1/18，暂时不能证明 Controller 的提升已经泛化。
4. TaskPool 在提高 hard-case 重采样概率的同时仍维持了 12 个仓库的覆盖，后期每轮不同任务数有所增加。
5. Reflector 每轮都能从约 650 个 hard cases 中产生有效失败簇并更新 skill，但现有 skill 内容较泛化，其实际贡献仍需要通过更大的 validation 或消融实验验证。

当前最稳妥的结论是：**自进化训练闭环已经真实运行，训练分布上出现了局部改善，但尚无充分证据证明 held-out 泛化能力稳定提升。**
