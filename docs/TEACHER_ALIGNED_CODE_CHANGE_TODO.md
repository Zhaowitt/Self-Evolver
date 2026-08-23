# Self-Evolver 简短 Todo List

## P0：冻结实验口径

- [ ] 用真实实验 ID 运行 split 脚本，产出 SFT、GRPO、validation、final-eval 的互斥 manifest（代码已检查交集）。
- [x] 固定 `max_iterations = 3`，从 Controller prompt、SFT target 和 GRPO action 中移除 `budget`。
- [x] 从 Controller 活跃输出移除顶层 `target_failure_type`、`memory_query`、`task_wrapper`；只保留 `selected_skill_ids` 和可选短 `strategy`。
- [x] Parser 兼容旧响应但忽略旧字段，Orchestrator 和 worker guidance 不再使用它们。
- [ ] 用真实任务运行分层选择脚本，物化 36–48 条 validation；代码已强制完整 official F2P + P2P。

## P1：两阶段验证

- [x] 实现确定性的 `F2PSubsetSelector`，同一 instance/round 的所有 rollout 使用同一子集。
- [x] Backend 真正只执行选中的 F2P tests，而不只是缩短 grading 列表。
- [x] 每个 patch 先跑 F2P 子集：失败则 retry，通过则对同一 patch 跑完整 F2P + P2P。
- [x] 最后一轮无论子集结果如何，都运行一次完整 official evaluation。
- [x] 最终 GRPO reward 只使用 full official outcome；subset outcome 只用于 retry routing。

## P2：Reward 与 Controller SFT

- [x] 实现 `R_task = 0.3 × official_resolved + 0.7 × official_f2p_fraction - token_cost`。
- [x] 实现 `C_skill = R_task - B_cluster`，并将 contribution 加入 `overall`。
- [x] 一轮内冻结 cluster baseline，轮末只用不含 contribution 的 `R_task` 更新。
- [x] 提供仅监督 skill 选择/短 strategy 的 Controller SFT 构建器与训练配置；强制 candidate skills 为 4–6、最多选择 2 个。
- [ ] 完成人工审核标注并实际物化数百条高质量 Controller SFT 数据。

## P3：Structured Diagnosis 与两层 Cluster

- [x] 扩展 rollout/hard-case schema，保存 F2P/P2P、patch apply、Judge、retry 和 termination evidence。
- [x] 为所有 failed rollout 生成单一 `failure_diagnosis.mechanism`、summary、error signatures 和 confidence；`max_iterations` 只作 termination reason。
- [x] 第一层按 mechanism 精确分桶；第二层计算 summary/error/verification/retry 的字段化相似度，并使用 deterministic complete-link 聚类。
- [x] qualifying 二级 cluster 同时检查 record count、unique issue count 和 cohesion，并生成 shared pattern 与代表案例。
- [x] 建立跨 round 稳定的 cluster registry；旧数据不伪造 diagnosis，也不进入 v2 Reflector。

## P4：Task Sampling

- [x] 显式建立 `U={attempts==0}` 与 `F={observed failed + label}` 两个候选池。
- [x] 下一轮严格物化 50% `unseen_random` + 50% `observed_failure_replay`。
- [x] Weighted half 先分配 observed failure label quota，再在 label 内按 repository/instance 抽样。
- [x] 实现 label 上限、最低覆盖、instance repeat cap 和 U/F 不足时的回退规则。
- [x] 不为完整 TaskPool 预测 `expected_capability_label`。

## P5：Skill Evolution

- [x] 每个 qualifying 二级 cluster 先匹配 top-k active skills，再由一次 batch Reflector 输出 create/refine/no-op。
- [x] Reflector 读取 matched skill 的完整内容；每个输入 cluster 恰好一个 decision，并保存 evidence provenance。
- [x] 用结构化 section + canonical renderer 强制生成与 seed skill 相同的 Description/Purpose/When to Use/How to Apply/Constraints/Action type。
- [x] 增加 quality lint、semantic dedup、revision archive；no-op 不写 SkillBank。
- [x] 从 Reflector action 中移除 deprecate；retirement 只由长期 contribution/trials 控制。
- [x] 一轮内冻结 SkillBank，create/refine 只进入下一轮 snapshot。

## 最小验收

- [x] 明显失败的中间 patch 不再触发 full evaluation。
- [x] 每个产生可用非空最终 patch 的 rollout 都有一次 final full official outcome；无有效 patch 的 rollout 明确记零 outcome。
- [x] Controller 输出不包含或不使用 `budget`、顶层 `target_failure_type`、`memory_query`、`task_wrapper`。
- [x] `compute_score()["overall"]` 同时包含 task reward、token cost 和 skill contribution。
- [x] 一级 mechanism 只控制 observed-failure sampling，二级 cluster 只控制 pattern/Reflector/baseline。
- [x] 新 skill 不再由 cluster 自动创建，所有 qualifying cluster 都显式走 create/refine/no-op。

## 仍需在实验环境执行

- [ ] 在可用的 Docker/Apptainer official image 中做 1–2 个代表性任务的 smoke test，核对仓库特定测试命令与实际日志解析。
- [ ] 用最终 ID 清单生成互斥 manifests、固定 validation JSONL 和人工审核后的 Controller SFT JSONL。
