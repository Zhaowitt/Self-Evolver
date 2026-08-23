# Self-Evolver 下一阶段代码修改方案（老师思路对齐版）

> 日期：2026-08-23  
> 适用分支：`feature/in-loop-backend-server-changes`  
> 目标：根据 Round 6–11 实验结果和会议讨论，收敛下一阶段的代码修改范围，提高在线训练效率、reward 的目标一致性、skill credit 的可信度以及 TaskPool 的能力缺口采样效果。

## 1. 本阶段的核心结论

下一阶段不训练 Reflector、不替换 harness，也不加入复杂的过程判分。主线收敛为以下七项：

1. 训练时，每个 candidate patch 先执行该任务自己的小型 F2P 子集；子集通过后，再对同一个 patch 执行完整 official F2P + P2P。
2. 完整 official outcome 用于最终 reward 和 `resolved`；F2P 子集只负责过滤明显失败的中间 patch、指导是否 retry。
3. 只对 Controller 做小规模、浅层 SFT；Reflector 继续使用较强 API 模型，不做 Reflector SFT。
4. 总 reward 改为“任务结果正奖励 − token 成本 + cluster-relative skill contribution”；Controller 不再输出 budget，Worker 使用系统固定的最多 3 轮硬上限。
5. 删除没有结果闭环的 Controller 顶层 `target_failure_type`；Controller 的活跃 action 收敛为 `selected_skill_ids` 和可选短 `strategy`。
6. 将失败表示改为结构化 `failure_diagnosis`，先按一级 capability mechanism 分桶，再在桶内形成稳定的二级 failure-pattern cluster。
7. 下一轮任务严格由“50% 未执行任务随机探索 + 50% 已观察失败任务按一级标签重采样”组成；二级 cluster 用于 Reflector 和 contribution baseline，不要求给完整 TaskPool 预标 `expected_capability_label`。

暂时保持不变：

- `max_selected_skills = 2`；
- `max_iterations = 3`，由系统配置固定控制；
- 每轮 64 optimizer steps；
- Worker 仍使用当前 coding workflow；
- 不加入 skill quality reward；
- 不加入 worker-skill alignment reward；
- 不立即替换为 mini-SWE-agent；
- 暂不新增细粒度 timing profiling。

本阶段明确不做：

- 不让 Controller 输出 `budget` 或顶层 `target_failure_type`；
- 不为全部未执行任务预测 `expected_capability_label`；
- 不让每个新 cluster 自动创建一个 skill；
- 不让 Reflector 根据单个 cluster 直接淘汰旧 skill。

---

## 2. 训练验证：每个任务使用自己的 F2P 子集

### 2.1 子集的含义

F2P 子集不是从一轮的几百个 rollout 中只挑 1–2 个 patch 验证，而是：

> 每个 rollout 的每个 candidate patch 都验证，但只执行当前任务 `FAIL_TO_PASS` 列表中的少量测试。

不同任务拥有不同的 F2P 列表，因此不会让所有任务运行同一组测试。系统只使用统一的选择算法，为每个任务生成它自己的训练子集。

示例：

```text
Task A 完整 F2P：A1 A2 A3 A4
Task A 训练子集：A2 A4

Task B 完整 F2P：B1
Task B 训练子集：B1

Task C 完整 F2P：C1 C2 C3
Task C 训练子集：C1 C3
```

### 2.2 子集选择规则

新增确定性的 `F2PSubsetSelector`，初版规则如下：

1. `len(FAIL_TO_PASS) <= subset_size` 时运行全部 F2P；
2. 否则最多选择 `subset_size` 个测试，初始默认值建议为 2；
3. 如果测试来自不同文件或测试类，优先覆盖不同测试文件/测试类；
4. 剩余位置使用 `instance_id + round_id` 的稳定 hash 选择；
5. 同一 instance 在同一 round 内的所有 GRPO samples 使用完全相同的子集；
6. 不允许根据生成出来的 patch 或第一次测试结果重新挑选更容易通过的测试；
7. 可以跨 round 确定性轮换子集，以逐步覆盖更多 F2P。

建议在数据中同时保留完整列表和训练子集：

```json
{
  "FAIL_TO_PASS": ["T1", "T2", "T3", "T4"],
  "TRAIN_FAIL_TO_PASS_SUBSET": ["T2", "T4"],
  "F2P_SUBSET_POLICY": "stable_hash_v1"
}
```

### 2.3 训练阶段执行流程

```text
Candidate Patch
    ↓
patch apply + canonical diff 检查
    ↓
执行 TRAIN_FAIL_TO_PASS_SUBSET
    ├─ 子集未全部通过且仍有轮次 → Judge → retry
    ├─ 子集全部通过 → 对同一 patch 执行完整 official F2P + P2P
    │      ├─ official resolved → 本次 rollout 结束
    │      └─ 未解决且仍有轮次 → 将完整结果反馈给 Judge → retry
    └─ 最后一轮 → 无论子集结果如何，都对最终 patch 执行一次完整 official evaluation
```

F2P 子集是完整验证之前的过滤层，不是完整验证的替代品。明显失败的中间 patch 不再消耗完整 F2P + P2P evaluation；只有子集通过的 promising patch，以及达到最后一轮时的最终 patch，才执行完整 official evaluation。

典型调用量从：

```text
patch 1 → full
patch 2 → full
patch 3 → full
```

变为：

```text
patch 1 → subset fail
patch 2 → subset fail
patch 3 → subset pass → full
```

这样通常可以把每个 rollout 的完整 official evaluation 从最多 3 次降低到约 1 次，同时保留真实的 F2P、P2P 和 resolved reward。

### 2.4 训练指标与正式指标必须分开命名

训练阶段需要同时保存子集筛选结果和最终完整评估结果。子集结果只用于 retry routing：

```text
subset_f2p_passed
subset_f2p_total
subset_f2p_fraction
evaluation_scope = "subset"
```

当同一 patch 进入完整评估后，记录：

```text
f2p_passed / f2p_total
p2p_passed / p2p_total
official_resolved
evaluation_scope = "full"
```

子集通过不能写成 official resolved；`R_task` 只使用完整 official outcome。若前几轮子集失败，则直接 retry；最后一轮对可用的非空最终 patch 始终补一次完整评估，保证正常完成的 rollout 都有一致的最终 reward 口径。若最后仍没有可应用的非空 patch，则直接记为执行失败和零 outcome，不调用完整测试。

### 2.5 Backend 实现要求

不能只缩短用于 grading 的 `FAIL_TO_PASS` 元数据列表。必须保证容器中的实际测试命令也只执行所选测试，否则可能只减少计分项而没有减少真实运行时间。

需要为 backend 增加明确接口：

```python
run_subset_eval(instance, model_patch, selected_f2p_ids)
run_full_eval(instance, model_patch)
```

不同仓库的测试 ID 格式和最小可执行单元可能不同：

```text
能执行单个 test function → 执行 test function
只能执行 test file/module → 执行最小 test file/module
```

因此选择逻辑和测试命令生成逻辑应分离：

```text
F2PSubsetSelector
    ↓ selected test IDs
Repository-aware Test Command Builder
    ↓ executable command
Container Backend
```

### 2.6 相关代码修改

- `src/environment/test_backend.py`
  - 增加 subset/full 两种 evaluation scope；
  - 支持实际只执行指定测试；
  - 返回结果时标记 `evaluation_scope` 和 `selected_f2p_ids`。
- `src/workers/verifier.py`
  - 训练阶段读取 F2P 子集；
  - 子集通过后切换到 full official evaluation；
  - full evaluation 失败时把 F2P/P2P 结果交给 Judge；
  - 最后一轮强制执行一次 full official evaluation。
- `src/rl/online_rollout_runner.py`
  - subset outcome 只用于 retry routing；
  - train/validation/final reward 都使用各自最终的 full official outcome。
- `src/reward/online_reward.py`
  - 移除每个 iteration 固定执行 full evaluation 的语义；
  - 接收最终 full outcome 计算 reward。
- `src/rl/easyr1_dataset.py`
  - 生成并冻结每个训练 instance 的 `TRAIN_FAIL_TO_PASS_SUBSET`。

---

## 3. Reward：结果、成本和 Skill Contribution

### 3.1 基础任务奖励

基础 reward 使用最终完整 official evaluation 的真实结果：

```text
R_outcome
= 0.3 × official_resolved
+ 0.7 × official_f2p_fraction
```

`official_resolved` 仍要求全部 F2P 通过且没有 P2P regression。P2P 不再作为一个额外正奖励项，但继续通过 `official_resolved` 影响最终 outcome。

### 3.2 Token 成本是负奖励

当前 `cost_efficiency` 会因为 token 少而提供正奖励，可能出现“任务没有修好，但因为便宜而拿到正分”的情况。改为真正的负项：

```text
token_cost
= λ_cost × min(total_worker_tokens / token_budget, 1)

R_task
= R_outcome - token_cost
```

初版可以从较小的 `λ_cost` 开始，例如 0.05 或 0.1，并通过单变量实验调整。

`total_worker_tokens` 统计 Inspector、Patch Generator 和 Judge 的 token；Controller policy 自身的 generation cost 单独记录，不必混入 Worker cost。

### 3.3 固定三轮硬上限

本阶段暂不让 Controller 输出 `budget`。原因是预算预测会扩大 Controller 输出 schema、增加 SFT 标注字段和 RL action 负担，而当前更优先的问题是 skill 选择质量与 contribution credit。

统一使用系统配置：

```text
max_iterations = 3
```

`ExecutionOrchestrator` 最多生成和验证 3 个 candidate patches。达到 3 轮后必须结束，不再设计软预算、超预算 gate 或 budget reward。

Controller 的活跃输出只保留：

```text
selected_skill_ids
optional short strategy
```

从 Controller action 中移除：

```text
budget
target_failure_type（Controller 顶层预测）
memory_query
task_wrapper
```

这里移除的是 Controller 自己预测的顶层 `target_failure_type`。Skill Bank 内部用于描述适用范围的 legacy `target_failure_type` 暂时保留，并逐步映射到 `applicable_capability_label`；它不是 Controller 的预测目标。Controller 只输出 skill ID，skill 的标题、procedure 和适用标签均从冻结的 Skill Bank 中回填，不能由 Controller 自由生成。

为兼容旧 checkpoint，parser 可以暂时容忍历史响应中的 `budget`、`target_failure_type`、`memory_query` 和 `task_wrapper` 字段，但 prompt、SFT target、GRPO action、worker guidance 和 reward 都忽略这些字段。兼容路径只负责读取旧结果，不能继续影响执行。

### 3.4 Skill contribution 从正式训练开始进入总 reward

对于当前 rollout，先找到它所属的稳定 cluster，然后计算：

```text
C_skill = R_task - B_cluster
```

其中 `B_cluster` 是该类任务在之前轮次中的历史任务奖励 baseline。

总 reward：

```text
R_total
= R_task
+ λ_contribution × C_skill
```

建议将 `λ_contribution` 配置化。初版可以从 0.1 或 0.2 开始，不直接假设最终最优权重。

最终可以裁剪到有限区间，例如：

```text
R_total ∈ [-1, 1]
```

不能继续强制裁剪到 `[0, 1]`，否则 token cost 和负 contribution 无法形成真正的负信号。

### 3.5 Cluster baseline 更新规则

一轮训练开始时冻结所有 baseline。本轮的所有 rollout 都使用相同的旧 baseline：

```text
C_skill = R_task - B_cluster_old
```

一轮结束后再统一更新：

```text
B_cluster_new
= 0.7 × B_cluster_old
+ 0.3 × mean(R_task of current-round rollouts assigned to cluster)
```

必须使用不含 contribution 的 `R_task` 更新 baseline，不能使用 `R_total`，否则会形成 contribution 与 baseline 互相放大的循环。

如果 cluster 没有历史 baseline，或样本量低于最小阈值：

```text
C_skill = 0
```

Round 6–11 只有在完整 rollout artifact 足以重建 v2 diagnosis 时才能用于离线初始化；旧 hard-case summary 本身不能直接初始化 baseline。没有可靠历史证据的 cluster 从 `baseline_count = 0` 开始，并在样本不足时令 `C_skill = 0`。

### 3.6 多 Skill 的 credit

Controller 一次最多选择两个 skill。当前 contribution 能可靠评价的是整个 Controller action，即 skill 组合，而不是其中某一个 skill 的独立因果贡献。

因此建议记录：

```text
skill_set_key = sorted(selected_skill_ids)
skill_set_contribution = C_skill
```

不要简单给两个 skill 各记一次完整独立贡献。单 skill 统计可以保留，但需要标记为 joint credit，避免将组合收益误解释为单个 skill 的因果效果。

### 3.7 Reward 代码修改

- `configs/reward_config.yaml`
  - 改为 outcome、cost 和 contribution 配置；
  - 删除 `p2p_no_regression` 独立正权重；
  - 删除 `cost_efficiency` 正权重；
  - 删除 `process` 正权重。
- `src/reward/reward_components.py`
  - 增加 `token_cost`；
  - subset 指标只进入过程日志，不进入最终 `R_task`；
  - 明确区分 subset 和 full outcome。
- `src/reward/reward_model.py`
  - 计算 `R_task`、`C_skill` 和 `R_total`；
  - 移除 `[0,1]` 的非负裁剪；
  - 不再维护与 skill store 重复的全局 EMA baseline。
- `src/orchestrator/orchestrator.py`
  - 统一使用系统固定的 `max_iterations = 3`；
  - 不再从 ControllerSignal 读取执行预算。
- `src/controller/schema.py`、`src/controller/prompt_builder.py`
  - 从当前 Controller 输出 schema 和 prompt 中移除 `budget`、顶层 `target_failure_type`、`memory_query` 和 `task_wrapper`；
  - 活跃输出只保留 `selected_skill_ids` 和可选短 `strategy`；
  - parser 暂时兼容旧响应中的历史字段，但不再使用。
- `src/skills/skill_store.py`
  - 存储 cluster/skill-set contribution；
  - 不再使用全任务共享的 global reward EMA 作为 skill contribution。

---

## 4. Controller SFT：小规模、浅层、只训练选择行为

### 4.1 只做 Controller SFT

本阶段不做 Reflector SFT。Reflector 继续使用较强模型 API，通过 prompt、cluster evidence 和 contribution 信号改进 skill，并根据长期 contribution 决定保留或淘汰。

Controller SFT 的目标是让小模型先学会：

- 输出合法 JSON；
- 从候选 skill 中选择相关 skill；
- 不选择明显无关 skill。

建议主要监督：

```json
{
  "selected_skill_ids": ["skill_a"]
}
```

可以保留一句简短 strategy，但不要把大量 reasoning、Reflector 行为或完整 skill generation 混入同一个 SFT 目标。

### 4.2 SFT 数据来源

1. 使用较强 coding model 对少量任务标注相关 skill；
2. 可使用两个模型交叉标注，只保留一致或经人工快速确认的样本；
3. 从历史成功 rollout 反推当时有效的 skill 组合；
4. 优先质量，不追求传统 post-training 规模；
5. 初版使用数百条高质量数据和较浅训练深度；
6. SFT-only task、GRPO task、validation task、final-eval task 必须互斥。

### 4.3 Candidate Skills 数量需要增加

当前数据构建只向 Controller 提供最多两个 candidate skills，而 Controller 最多也选择两个，选择任务过于简单。

建议：

```text
candidate_skill_count = 4–6
max_selected_skills = 2
```

候选中应包含相关项和干扰项，SFT 才真正训练 task-to-skill matching。

### 4.4 Skill Bank 冻结与版本

SFT 标注必须绑定 Skill Bank 版本：

```text
skill_bank_revision
skill_bank_hash
candidate_skill_ids
```

如果 SFT 标签引用的 skill 在 GRPO 开始前已被重命名、淘汰或覆盖，SFT 会训练出无效选择。因此 SFT warm-up 和随后的首轮 GRPO 应共享冻结的初始 Skill Bank snapshot。

### 4.5 新增代码

- `scripts/build_controller_sft_dataset.py`
- `scripts/run_controller_sft.sh`
- `configs/controller_sft.yaml`
- SFT 数据 manifest 和 contamination audit 文件
- GRPO launcher 中的 SFT checkpoint 加载入口
- 修改 `src/controller/schema.py` 和 `src/controller/prompt_builder.py`，使 SFT 与在线 GRPO 使用同一个无 budget 输出 schema

---

## 5. Structured Failure Diagnosis 与两层 Cluster

### 5.1 Terminal status、termination reason 与 failure mechanism 分离

以下信息只是验证结果或停止原因，不得继续作为 root cause 或聚类主特征：

```text
max_iterations
judge_give_up
tests_failed
new_issues
unknown
```

每条 failed trajectory 生成一个结构化 `failure_diagnosis`，其中只选择一个最主要的 mechanism。一级 taxonomy 与 TaskPool 的 observed failure label 共用同一套枚举：

```text
localization_failure
wrong_root_cause
patch_generation_failure
patch_application_failure
incomplete_fix
regression_introduced
test_interpretation_failure
environment_failure
unknown
```

标签必须尽量互斥：Inspector 没有形成可用定位属于 `localization_failure`；已形成定位并生成可应用 patch、但完整 F2P 始终没有进展时才考虑 `wrong_root_cause`；F2P 有部分进展但停滞属于 `incomplete_fix`；P2P 退化优先归入 `regression_introduced`。纯容器、依赖或 backend 故障归入 `environment_failure`，不伪装成 worker repair skill 缺口。`max_iterations` 只保存在 `termination_reason`。

### 5.2 `failure_diagnosis` schema

Hard case schema 升级为向后兼容的 v2。至少保存：

```json
{
  "failure_diagnosis": {
    "schema_version": "failure_diagnosis_v1",
    "mechanism": "incomplete_fix",
    "summary": "The patch fixed one target behavior but repeatedly missed a sibling branch.",
    "error_signatures": [
      "assert expected <type> but got <type>"
    ],
    "evidence": {
      "f2p_progress": [
        {"iteration": 1, "scope": "subset", "passed": 0, "total": 2},
        {"iteration": 2, "scope": "full", "passed": 1, "total": 2}
      ],
      "p2p_regression": [false, false],
      "patch_applied": [true, true],
      "verification_statuses": ["tests_failed", "tests_failed"],
      "retry_pattern": ["regenerate_patch_same_location", "reinspect"],
      "judge_categories": ["tests_failed", "likely_wrong_location"],
      "termination_reason": "max_iterations"
    },
    "confidence": 0.82,
    "source": "rule_and_judge"
  }
}
```

错误签名需要去除临时路径、行号、随机 ID、时间戳等不稳定信息。完整原始日志继续保存在 rollout artifact 中，cluster 只使用截断、规范化后的诊断证据。

### 5.3 诊断生成时机与证据来源

诊断必须基于完整 failed trajectory，而不是直接采用 Controller 的预测或最后一次 verifier status。流程为：

```text
ExecutionResult
    ↓ 最终 full official outcome（若存在可用 patch）
trajectory evidence extractor
    ↓
高置信度规则分类
    ├─ 可确定 → 直接形成 diagnosis
    └─ 歧义案例 → round 后由强模型在固定 enum 内批量判定
```

运行时已有但当前 hard-case 写入会丢失的证据需要补齐：

- 每轮 suspected files / locations 和 root-cause analysis 摘要；
- raw/canonical patch 是否存在、modified files；
- patch applied、apply strategy 和规范化 diagnostic；
- subset/full evaluation scope；
- 每轮 F2P 进展与 full P2P regression；
- Judge failure category、route 和简短 feedback；
- retry pattern 和 termination reason。

Hard case 不再在 Orchestrator 刚退出循环时用粗粒度 reason 直接定型；应在 online rollout runner 得到最终 official outcome 后构建 diagnosis 并写入。Capability-gap 统计基于所有 failed train rollouts，而不是只基于 HardCaseBuffer admission 后的子集；HardCaseBuffer 仍用于保存和检索高价值失败案例。

在 online reward 阶段不额外增加一次强模型调用。当前 rollout 只有在能用规则或已有 stable cluster 高置信匹配时才计算 cluster contribution；歧义或新模式本轮 `C_skill = 0`，round 后批量诊断并用于下一轮。

### 5.4 第一层：按 mechanism 精确分桶

第一层不做语义聚类：

```text
failed trajectory
    ↓ structured failure diagnosis
bucket[failure_diagnosis.mechanism]
```

一级 mechanism 的用途是：

- 统计 observed capability gap；
- 给下一轮已观察失败任务分配采样 quota；
- 保证不同根因不会因为共享 `tests_failed/max_iterations` 被合并；
- 限定第二层 cluster 和 skill 的适用范围。

### 5.5 第二层：同一 mechanism 内形成具体 failure pattern

第二层只比较同一 mechanism 的记录，初版固定使用“字段化相似度矩阵 + complete-link agglomerative clustering”，不再把 average-link、centroid 和 union-find 同时保留为可选主路径。

每条记录先构造四组独立特征：

```text
summary_text         = failure_diagnosis.summary
error_signatures     = normalized error signatures
verification_pattern = F2P progress + P2P regression + patch-applied sequence
retry_pattern        = Judge category/route sequence
```

`max_iterations`、`tests_failed`、`test_failure` 等公共词不进入 `summary_text`。`termination_reason` 可以留在 evidence 中供 Reflector 查看，但不参与第二层相似度。

对同一 mechanism 内任意两条记录计算：

```text
pair_similarity
= 0.55 × summary_similarity
+ 0.25 × error_signature_similarity
+ 0.10 × verification_pattern_similarity
+ 0.10 × retry_pattern_similarity
```

其中：

- `summary_similarity`：有 embedding 时使用 cosine；没有 embedding 时使用现有 lexical similarity；
- `error_signature_similarity`：对规范化 signature 集合计算 Jaccard/lexical similarity；
- `verification_pattern_similarity`：比较 F2P 是无进展、部分进展后停滞还是通过后 P2P regression，以及 patch-applied 序列；
- `retry_pattern_similarity`：比较 Judge category/route 序列；
- 某个字段双方都缺失时不按 0 分惩罚，而是对剩余可用字段的权重重新归一化；可用证据不足时记录保持 singleton，不交给 Reflector。

随后运行 deterministic complete-link：

```text
cluster_similarity(A, B)
= min(pair_similarity(x, y) for x in A for y in B)

每一步合并相似度最高且 >= secondary_similarity_threshold 的两个 cluster
直到没有 cluster pair 达到阈值
```

这意味着只有两个 cluster 中最不相似的那一对记录也达到阈值时才允许合并。因此 A≈B、B≈C、但 A!≈C 时，A/B/C 不会通过 B 被串成一个大 cluster。相似度相同的 merge 使用稳定的 record/cluster ID 排序打破平局，确保同样输入得到同样输出。

`secondary_similarity_threshold` 必须配置化，并用一小批人工核对的同模式/异模式 pair 校准。初版优先 cluster precision 而不是 recall：宁可留下 singleton 或较小 cluster，也不要重新产生一个几百条的泛化大 cluster。

只有满足以下条件的二级 cluster 才交给 Reflector：

```text
record_count >= 3
unique_issue_count >= 2       # 更严格实验可设为 3
cluster_cohesion >= threshold
mechanism != environment_failure（纯基础设施故障）
```

同一 instance 的三次重复失败不能单独证明一个跨任务可复用 pattern。

每个 qualifying cluster 的 prototype 使用 medoid（与本 cluster 其他成员平均相似度最高的记录），representative cases 取 medoid 及距离它最近、但 issue ID 不同的 1–2 条记录。`shared_pattern` 从成员共同的 summary/error/verification/retry evidence 中提取，并随代表案例一起交给 Reflector。

### 5.6 Stable Cluster Registry

每个 qualifying 二级 cluster 持久化：

```json
{
  "cluster_id": "wrong_root_cause_003",
  "mechanism": "wrong_root_cause",
  "shared_pattern": "The worker patches the visible exception site without tracing the state producer.",
  "common_error_signatures": ["..."],
  "common_retry_pattern": ["reinspect", "regenerate_patch_same_location"],
  "representative_issue_ids": ["..."],
  "member_count": 12,
  "unique_issue_count": 7,
  "reward_baseline": 0.17,
  "baseline_count": 12,
  "last_skill_decision": "refine:failure_localization"
}
```

代表案例沿用 5.5 的规则：一个 medoid，加上与它最相近且 issue ID 不同的 1–2 条记录，而不是简单取最早五条。Registry prototype 由 `shared_pattern` 的表示和 medoid evidence 组成；新一轮二级 cluster 先与 registry 中同 mechanism prototype 匹配，避免同一 pattern 每轮获得新 ID。

成功 rollout 为计算 contribution，可以在结束后匹配已有 stable cluster；低置信度时安全回退为 `C_skill = 0`。成功 rollout 不用于创建 failure cluster，但可以更新已匹配 cluster 的 reward baseline。

Round 6–11 的旧 `hard_cases.jsonl` 缺少足够 diagnosis evidence，不能直接初始化 v2 cluster。只有能从对应完整 rollout artifact 重建证据时才允许离线 backfill；否则新 registry 从空 baseline 开始。

### 5.7 相关代码

- 新增 `src/reflection/failure_diagnosis.py`：schema、证据提取、规则优先诊断和 error-signature normalization；
- 新增 `src/reflection/trajectory_labeler.py`：只批量处理规则无法确定的歧义记录；
- 新增 `src/reflection/cluster_registry.py`：稳定 ID、prototype、cohesion、baseline 和最近 skill decision；
- 修改 `src/reflection/clustering.py`：mechanism bucket + 字段化 pair similarity + deterministic complete-link clustering；
- 修改 `src/memory/hard_case_buffer.py`：hard-case v2 与 legacy read compatibility；
- 修改 `src/orchestrator/orchestrator.py`：不再过早写入缺少最终 outcome 的粗粒度 hard case；
- 修改 `src/rl/online_rollout_runner.py`、`src/rl/rollout_writer.py`：保存紧凑 trajectory 和最终诊断所需证据；
- 修改 `src/reflection/reflector.py`：接收 mechanism、shared pattern、代表性案例及其 diagnosis/evidence。

---

## 6. Task Sampling：50% Unseen Exploration + 50% Observed-Failure Replay

### 6.1 两个候选池

下一轮冻结数据由两个明确、互斥的候选池构成：

```text
U = {task | attempts == 0}
F = {task | 已执行、最新有效结果为 failed、已有 observed_failure_label}
```

其中 `observed_failure_label` 直接等于失败 trajectory 的一级 `failure_diagnosis.mechanism`。未执行任务没有 trajectory，不要求也不预测 capability label。

### 6.2 固定 50/50 组合

```text
50%：从 U 中 repository-balanced 随机抽取
50%：从 F 中先按 observed failure label 分配 quota，再在 label 内抽取
```

第一部分负责探索新任务；新任务执行失败后得到真实 `observed_failure_label`，从下一轮起才有资格进入 F。第二部分负责针对已观察到的能力缺口复练，初版不主动预测“尚未执行但可能属于同一 failure type”的 Task C/D。

### 6.3 Failure gap 与 label quota

一轮结束后，对全部有效 failed train rollouts 的一级 mechanism 计数：

```text
gap[c]
= failed_rollouts_with_label_c
/ all_valid_failed_rollouts
```

weighted half 先根据 gap 分配 label quota，再在各 label 的 F 子池内按 repository、instance failure history 和去重约束抽取。不得继续先给每个 instance 一个权重后直接从完整 TaskPool 混抽，因为那样不能保证 50/50 边界和 label 比例。

保护条件：

```text
max_label_fraction = 0.40
min_samples_per_active_label >= 1
max_instance_repeats_per_round = 3
```

纯外部基础设施导致的 `environment_failure` 先标记为 `sampling_eligible = false`，不通过重复抽样训练 worker；如果诊断显示是 agent 的工具使用错误，则可以保留在 active label 中。

二级 cluster 初版不决定总体 quota。可以在同一 label 内用 cluster ID 做多样性去重，但不得让某个大 cluster 再次垄断 weighted half。

### 6.4 不做完整 TaskPool 的 expected-label 预测

本阶段明确不新增：

- 强模型对完整 TaskPool 的 `expected_capability_label` 预标；
- 未执行 task 与 cluster prototype 的全量匹配；
- capability label 预测器。

这样会牺牲“主动发现未执行但属于相同能力缺口的新任务”的定向能力，但避免额外标注成本、标签噪声和新的实验变量。50% unseen random 会持续发现新任务；只有观察到系统明显过拟合旧失败实例时，再把 expected-label prediction 作为独立增强实验，而不是当前主线依赖。

### 6.5 边界与回退

- U 数量不足时，用最低 attempts 的非成功任务补足探索半区；
- F 或某个 active label 数量不足时，将空余 quota 先分配给其他 active label，仍不足再回填 U；
- 同一 round 的冻结 JSONL 必须记录每条样本来自 `unseen_random` 还是 `observed_failure_replay`；
- sampling gap 统计使用全部失败 rollout，但抽样时对单个 instance 设置 repeat cap，避免 rollout 数量放大成 task-ID 垄断；
- 已经稳定解决的 task 不继续停留在 F，历史 label 可以保留用于审计。

### 6.6 相关代码

- `src/tasks/task_pool.py`
  - 保存 attempts、latest outcome、observed failure label 和 sampling eligibility；
  - 新增 unseen/observed-failure 两个显式候选池；
  - 移除以 `cluster_hits` 直接提高完整池内具体 instance 权重的主路径。
- `src/rl/easyr1_dataset.py`
  - 先各自物化 50% unseen 与 50% failure-label quota，再合并、shuffle 和冻结下一轮 JSONL；
  - 记录 `sampling_source` 和 `observed_failure_label`。
- `src/reflection/reflector.py`
  - `task_signals` 不再只输出 `instance_boosts`，而是输出 observed `capability_gaps`；
- `configs/task_evolution.yaml`
  - 增加 `unseen_fraction`、`failure_replay_fraction`、`max_label_fraction`、`min_samples_per_active_label` 和 repeat cap。

---

## 7. Skill Evolution：Cluster → Match → Create / Refine / No-op

### 7.1 当前行为与问题

当前实现不是“每产生一个 cluster 就必然创建一个 skill”。它会把所有 qualifying clusters 和当前 SkillBank 一起放进一次 Reflector prompt，由模型自由输出最多三个 `create/update/deprecate` proposal。

但当前没有强制：

- 每个 cluster 都有显式决策；
- proposal 指向哪个 stable cluster；
- create 前先匹配现有 skill；
- 已被覆盖或不可复用时输出 no-op；
- update 前读取已有 skill 的完整 procedure。

当前 Reflector 只看到已有 skill 的 id/title/summary，不看到 `When to Use`、`How to Apply` 和 `Constraints`，因此很难做可靠 refinement。当前 parser 又只检查 title/summary/content 非空，落盘层只自动补 H1 和 legacy target marker，无法保证生成内容达到 seed skill 的结构质量。

### 7.2 新的逐 cluster 决策流程

```text
Qualifying Secondary Cluster
    ↓ stable cluster ID + shared pattern + representative evidence
与同 mechanism 的 active skills 做 deterministic top-k matching
    ↓ 最相似的 1–3 个 candidate skills（包含完整结构化内容）
Batch Reflector
    ↓ 每个 cluster 恰好一个决策
┌────────────────┬────────────────┬─────────────────┐
│ create         │ refine         │ no-op           │
│ 新可复用模式   │ 已有 skill 缺项 │ 已覆盖/不可复用 │
└────────────────┴────────────────┴─────────────────┘
```

匹配先按 `mechanism/applicable_capability_label` 过滤，再比较 cluster shared pattern 与 skill 的 Description、When to Use、How to Apply 和 Constraints。SkillBank 活跃数量仍有硬上限，因此 top-k 比较的计算开销很小，不需要增加一次 LLM 调用。

### 7.3 三种 action 的准入规则

`create` 仅在以下条件同时成立时允许：

- cluster 满足 record count、unique issue count 和 cohesion 门槛；
- pattern 在多个任务中具有 procedural reuse；
- 现有 skill 没有覆盖关键步骤；
- 能从 representative evidence 提炼出明确的触发条件、行动步骤和验证方式。

`refine` 用于已有 skill 方向正确、但 cluster 暴露出缺失步骤或约束的情况。必须指定 `matched_skill_id`，读取旧 skill 全文，并记录修改了哪些 section；不能用一个泛化新 skill 绕过 refinement。

`no-op` 用于：

- 已有 skill 已充分覆盖；
- cluster cohesion 或 diagnosis confidence 不足；
- 只有单个任务的特殊实现细节；
- 纯 environment/tooling 故障；
- 无法形成可执行、可验证的通用 procedure。

`no-op` 需要持久化到 cluster registry，避免同一个 pattern 每轮被反复提交。Cluster evidence 显著变化后可以重新评估。

### 7.4 Reflector 输出 schema

Reflector 不再直接返回一个没有 cluster 对应关系的 `skill_updates` 列表，而是返回：

```json
{
  "cluster_decisions": [
    {
      "cluster_id": "wrong_root_cause_003",
      "matched_skill_id": "failure_localization",
      "match_score": 0.84,
      "action": "refine",
      "reason": "The existing skill does not guard against traceback-frame anchoring.",
      "skill_update": {
        "skill_id": "failure_localization",
        "title": "Failure Localization",
        "description": "...",
        "purpose": "...",
        "when_to_use": ["..."],
        "how_to_apply": ["..."],
        "constraints": ["..."],
        "action_type": "LOCALIZE",
        "applicable_capability_label": "wrong_root_cause"
      },
      "evidence_refs": ["issue_a", "issue_b"]
    }
  ]
}
```

每个输入 cluster 必须恰好出现一次；只有 `create/refine` 转成可写入 proposal，`no-op` 只记录决策。Proposal 必须带 `cluster_id`、evidence refs 和 matched skill provenance。

### 7.5 与 seed skill 一致的 canonical 模板

不要继续让 LLM 自由生成整段 Markdown。Reflector 输出结构化 section，再由代码统一渲染：

```markdown
# Skill Title

## Description

...

## Purpose

...

## When to Use

- ...

## How to Apply

- ...

## Constraints

- ...

Action type: LOCALIZE only.
```

强制 quality lint：

- 所有 section 和 Action type 必须存在；
- `When to Use`、`How to Apply`、`Constraints` 均不能为空；
- How to Apply 至少包含 2–3 条可执行步骤；
- 不接受只有“分析错误、检查代码、运行测试”的空泛模板；
- 关键步骤必须能追溯到 cluster shared pattern 或 representative evidence；
- create/refine 后再次执行 exact hash 和 semantic dedup。

Worker prompt 当前主要注入 summary 和 `How to Apply`。新模板落盘后，建议同时注入截断后的关键 Constraints；Purpose 和完整 evidence 只用于 SkillBank/Reflector，不把 worker prompt 变得过长。

### 7.6 激活、refinement 与 retirement

- create/refine proposal 先经过 schema lint、evidence grounding 和 dedup；
- refine 写入前归档旧 revision，并保留 contribution 历史和来源 cluster；
- 如果开启 candidate replay，只在 1–2 个代表案例上验证，不为每个 cluster 增加一整套大规模评估；
- Reflector 不再输出 `deprecate`。单个 cluster 不能证明 skill 应被淘汰；
- retirement 继续由足够 trial 数、连续多轮负 contribution 和 active-skill cap 共同决定；
- 新 skill 的最终价值仍由后续 task reward/contribution 衡量，不新增独立的 LLM skill-quality reward。

SkillBank 在一轮 GRPO 内冻结。Failure diagnosis、二级聚类和 Reflector batch decision 在 round boundary 完成，产生的 create/refine 只进入下一轮 snapshot，不在同一轮中途改变 Controller 候选集合。

### 7.7 负担控制

该设计增加的是少量离线匹配、schema 校验和 registry 元数据，不增加 Controller action，也不要求每个 cluster 单独调用 LLM。保持每个 reflection batch 一次强模型调用，只把每个 cluster 的 top-k candidate skill 全文加入 prompt。

显著成本更高的 representative replay 保持可配置；默认先运行 deterministic lint + match + dedup。由于 no-op 会阻止无意义创建和重复 refinement，长期会减少 SkillBank 膨胀、Controller 候选干扰和无价值验证。

### 7.8 相关代码

- `src/reflection/reflector.py`
  - 改为逐 cluster 的 batch decision；
  - 输入 shared pattern、representative evidence 和 top-k matched skills；
  - 输出 create/refine/no-op，移除 cluster-driven deprecate。
- `src/skills/proposals.py`
  - 新增 `cluster_id`、matched skill、structured sections 和 evidence refs；
  - 校验每个输入 cluster 恰好一个 decision。
- `src/skills/skill_matcher.py`（新增）
  - mechanism filter + semantic/lexical top-k matching。
- `src/skills/skill_store.py`
  - 用 canonical renderer 生成与 seed skill 一致的 Markdown；
  - 保存 revision、source cluster、match decision 和 provenance。
- `src/skills/skill_bank.py`
  - 解析 Purpose、When to Use、Constraints、Action type 和 applicable label；
- `src/skills/skill_evolver.py`
  - 只应用 create/refine proposal；
  - retirement 继续走长期 contribution 规则。
- `src/controller/injection.py`
  - 注入 canonical `How to Apply` 和必要 Constraints，不信任模型生成的 skill metadata。

---

## 8. Validation 设计

当前 18 条 validation 过少，且“每仓库选择 prompt 最短的一条”不能保证任务机制多样性。

建议建立固定的 36–48 条 validation：

- 覆盖不同 repository；
- 覆盖不同 capability label；
- 覆盖不同 F2P 数量；
- 覆盖不同问题长度和估计难度；
- 人工只检查覆盖和代表性，不根据模型是否容易通过来挑选；
- 生成固定 ID manifest，并从 SFT/GRPO 中排除；
- Validation 使用完整 official F2P + P2P；
- 降低执行频率，例如每轮一次或每若干轮一次，而不是扩大集合后仍每 8 steps 全量运行。

训练阶段的 subset success 不能替代这套 validation。主要趋势以 official validation resolved 为准。

修改：

- `scripts/select_low_cost_validation.py`
  - 从“每仓库最短 prompt”改为固定、分层、可审计的多样性选择；
- `scripts/prepare_phase_b_splits.py`
  - 将 SFT、validation 和 final-eval IDs 统一加入 contamination exclusion；
- `configs/easyr1_online_grpo_example.yaml`
  - 调整 validation frequency 和固定 validation 文件。

---

## 9. 暂不实施的内容

### 9.1 暂不新增细粒度 Timing Profiling

会议判断主要耗时已经明确来自 backend verification。当前阶段不新增以下细粒度字段：

```text
controller_ms
inspector_ms
patch_generator_ms
judge_ms
container_start_ms
```

保留代码已有的整体 duration 即可。为保证实验口径可审计，只需记录非 timing 元数据：

```text
evaluation_scope
selected_f2p_ids
subset_f2p_passed / total
official_eval 是否执行
```

### 9.2 暂不做 Reflector SFT

Reflector 继续由强模型负责。改进重点是给它更好的 trajectory representation 和更细的 cluster，而不是再训练一个生成 skill 的小模型。

### 9.3 暂不加入 Quality / Alignment Reward

两者依赖额外 LLM judgment，噪声大，并且可能偏离最终任务目标。本阶段唯一增加的 skill-specific signal 是 cluster-relative contribution。

### 9.4 暂不增加单次 Skill 选择数量和 Active Bank 上限

Controller 仍最多选择两个 skill，active SkillBank 仍维持当前硬上限。Reflector 可以在满足证据门槛时 create/refine，但不得通过提高 active cap 掩盖重复、泛化或低质量 skill。先提高选择质量、credit 可信度和 skill 内容质量，再考虑 Adaptive-K 或扩大 active bank。

### 9.5 暂不替换 Harness

mini-SWE-agent 只作为后期 transfer/harness ablation，不进入本阶段主训练流程。

---

## 10. 推荐实施顺序

### PR 1：数据边界与 Validation 冻结

- 建立 SFT/GRPO/validation/final-eval 互斥 manifest；
- 建立 36–48 条多样性 validation；
- 固定首轮实验数据和 Skill Bank snapshot。

### PR 2：F2P Subset Verification

- 实现 `F2PSubsetSelector`；
- 实现 backend subset execution；
- 子集失败时直接 retry，不运行 full evaluation；
- 子集通过时，对同一 patch 运行完整 official F2P + P2P；
- 最后一轮强制运行一次完整 official evaluation；
- 最终 reward 只读取 full official outcome。

### PR 3：Reward 与固定执行上限

- 固定 `max_iterations = 3`；
- 从 Controller prompt/SFT/action/worker guidance 中移除 `budget`、顶层 `target_failure_type`、`memory_query` 和 `task_wrapper`；
- Controller 活跃输出只保留 `selected_skill_ids` 和可选短 `strategy`；
- outcome reward 改为 0.3 official resolved + 0.7 official F2P fraction；
- token cost 改为负项；
- 允许负 reward。

### PR 4：Controller SFT

- 构建少量高质量标注；
- 候选 skill 增至 4–6；
- 训练浅层 Controller SFT；
- 从 SFT checkpoint 开始 GRPO。

### PR 5：Structured Diagnosis 与两层 Stable Cluster

- 扩展 rollout/hard-case schema，保存 F2P/P2P、patch apply、Judge 和 retry evidence；
- 基于完整失败 trajectory 生成单一 `failure_diagnosis.mechanism`；
- 规则优先，歧义记录在 round 后由强模型按固定 enum 批量判定；
- 第一层按 mechanism 精确分桶，第二层使用字段化 pair similarity 和 deterministic complete-link clustering；
- 增加 unique-issue、cohesion 和 environment filtering 门槛；
- 建立 stable cluster registry、shared pattern 和 representative cases；
- 历史数据只在能从完整 rollout 重建 diagnosis 时 backfill。

### PR 6：Structured Skill Evolution

- 为每个 qualifying cluster 匹配 top-k active skills；
- Reflector 一次 batch 输出每个 cluster 的 create/refine/no-op；
- 使用 canonical seed-skill renderer 和强制 quality lint；
- 保存 source cluster、matched skill 和 evidence provenance；
- 从 Reflector action 中移除 deprecate，retirement 只走长期 contribution；
- 新 skill/refinement 只进入下一轮冻结的 SkillBank snapshot。

### PR 7：Contribution 进入总 Reward

- 一轮内冻结 cluster baseline；
- `C_skill = R_task - B_cluster`；
- contribution 从本 PR 的正式训练开始进入 `overall`；
- 一轮结束后用 `R_task` 更新 baseline；
- 同时记录 joint skill-set contribution。

> PR 3–7 可以分开开发和测试，但在新的正式训练配置中，skill contribution 必须进入总 reward，不能只做事后统计。

### PR 8：50/50 Task Sampling

- 显式划分 unseen pool 与 observed-failure pool；
- 50% 从 unseen pool repository-balanced 随机抽取；
- 50% 按 observed failure mechanism 分配 label quota 并在 label 内抽样；
- 加 label 上限、最低覆盖和 instance repeat cap。
- 不给完整 TaskPool 预测 expected capability label。

---

## 11. Ablation 顺序

为了区分每项改动的效果，建议按以下顺序运行：

| 实验 | F2P 子集 | 新 Reward/固定预算 | Controller SFT | 两层 Cluster | Structured Skill Evolution | Contribution | 50/50 Sampling |
|---|---|---|---|---|---|---|---|
| A | × | × | × | × | × | × | × |
| B | ✓ | × | × | × | × | × | × |
| C | ✓ | ✓ | × | × | × | × | × |
| D | ✓ | ✓ | ✓ | × | × | × | × |
| E | ✓ | ✓ | ✓ | ✓ | × | × | × |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | × | × |
| G | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| H | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

主要比较：

- B vs A：F2P 子集是否降低训练耗时，official validation 是否明显受损；
- C vs B：结果/成本 reward 是否减少 reward hacking；
- D vs C：浅层 Controller SFT 是否改善 schema 和 skill 选择；
- E vs D：两层 cluster 是否降低最大 cluster 占比并提高 pattern cohesion；
- F vs E：create/refine/no-op 和 canonical template 是否提高 skill 具体性、降低重复创建；
- G vs F：cluster-relative contribution 是否改善 skill-conditioned policy；
- H vs G：50% unseen + 50% observed-failure sampling 是否提高样本效率和 held-out performance。

---

## 12. 测试与验收标准

### 12.1 F2P 子集

- 同一 instance + round 的子集完全可复现；
- 同一 round 的三个 GRPO samples 使用同一子集；
- 不同 round 可以按规则轮换；
- 子集中的每个测试确实被 backend 执行；
- subset outcome 不会被写成 official resolved；
- 子集通过后，同一 patch 会执行完整 F2P + P2P；
- 最后一轮即使子集失败也会执行完整 official evaluation；
- train/validation/final 的最终 reward 都来自 full outcome。

### 12.2 Reward

- 子集无进展且有 token 消耗时，reward 可以为负；
- Worker 不会执行超过系统固定的 3 轮；
- Controller response 不再要求或使用 `budget`、顶层 `target_failure_type`、`memory_query` 或 `task_wrapper`；
- 旧 checkpoint 中的历史字段可被 parser 容忍，但不会改变迭代数或 worker guidance；
- contribution 正负方向正确；
- baseline 更新只使用 `R_task`；
- 同一 round 内 baseline 不变化；
- `compute_score()["overall"]` 包含 contribution，而不是只写入 skill metadata。

### 12.3 SFT

- SFT/GRPO/validation/final-eval ID 无交集；
- 输出 JSON schema 正确率达到预设阈值；
- 所选 skill 必须来自 candidate list；
- skill Bank hash 与标注 manifest 一致。

### 12.4 Failure Diagnosis、Cluster 与 Sampling

- 每个 failed trajectory 只有一个 `failure_diagnosis.mechanism`，并保留 confidence/source；
- `max_iterations` 只出现在 termination reason，不会成为 mechanism 或二级 cluster 主特征；
- F2P progress、P2P regression、patch applied 和 retry pattern 能从 iteration records 正确提取；
- legacy hard case 可读取，但不会伪造缺失的 v2 diagnosis；
- 不同 mechanism 的记录绝不会进入同一个二级 cluster；
- 二级 cluster 不使用 single-link 传递合并，并满足 cohesion 门槛；
- qualifying cluster 同时满足 record count 和 unique issue count；
- cluster ID 跨 round 稳定；
- 成功 rollout 可以匹配已有 cluster 或安全回退为 contribution 0；
- 下一轮数据明确由 50% `unseen_random` 和 50% `observed_failure_replay` 组成；
- unseen task 不要求 `expected_capability_label`；
- observed failure label 的采样比例不超过上限；
- 每个 active label 有最低覆盖；
- 单个 instance 不超过 repeat cap；
- U/F 任一候选池不足时按明确回退规则补齐并记录实际比例。

### 12.5 Skill Evolution

- 每个输入 Reflector 的 qualifying cluster 恰好产生一个 create/refine/no-op decision；
- create/refine 都带 stable cluster ID、matched skill 和 representative evidence provenance；
- create 不能绕过 top-k existing-skill matching；
- refine 会读取旧 skill 全文、归档旧 revision，并保留历史 credit；
- no-op 不写 SkillBank，并保存在 cluster registry；
- 新写入 skill 必须包含 Description、Purpose、When to Use、How to Apply、Constraints 和 Action type；
- 缺 section、空泛 procedure、重复 skill 或 evidence 不足的 proposal 被拒绝；
- Reflector 不再根据单个 cluster 输出 deprecate；
- 一轮内 SkillBank snapshot 不变化，create/refine 只在下一轮可见。

---

## 13. 最终系统数据流

```text
SWE TaskPool
    ├─ U: attempts == 0
    └─ F: observed failed + failure mechanism
           ↓
    50% unseen random + 50% observed-failure label quota
Frozen Round Dataset
    ↓
SFT-warmed Controller
    ↓ selected_skill_ids + optional short strategy
Coding Worker
    ↓ fixed max_iterations = 3
    ↓
Candidate Patch
    ↓
Per-task F2P Subset Verification
    ├─ fail and iterations remain → retry
    ├─ pass → Full Official F2P + P2P
    │            ├─ resolved → finish rollout
    │            └─ unresolved and iterations remain → retry
    └─ final iteration → Full Official F2P + P2P
                 ↓
      R_task = outcome - token cost
             ↓
      match previous stable cluster
             ├─ confident → C_skill = R_task - cluster baseline
             └─ uncertain/new pattern → C_skill = 0
             ↓
          GRPO reward

All Train Rollouts
    ↓ compact trajectory + final official evidence
Structured Failure Diagnosis
    ↓ mechanism bucket
Within-mechanism Semantic Sub-clustering
    ↓
Stable Secondary Clusters
    ├─ update cluster baselines
    ├─ update next-round observed capability gaps
    └─ cluster → top-k existing skill matching
                    ↓
             Batch Reflector
             ├─ create
             ├─ refine
             └─ no-op
                    ↓
             Next-round frozen SkillBank

Fixed Validation / Final Evaluation
    ↓ full official F2P + P2P
Official Resolved and Generalization Metrics
```

本阶段的核心不是让 Controller 预测更多字段，而是让每个信号含义清楚：F2P 子集只过滤明显失败的中间 patch，最终 reward 使用完整 official outcome；Controller 只选择 skill，执行轮数由系统固定为最多 3 轮；一级 failure mechanism 服务 observed-failure sampling，二级稳定 cluster 服务 contribution baseline 和具体 skill evolution；SkillBank 只接受有证据的 create/refine，其他 cluster 明确 no-op。
