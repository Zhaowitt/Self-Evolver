# Self-Evolver 目前实验结果总结

> 数据截点：完成 `global_step_960`（Round 11）。正在重跑的 Round 12 不计入本文结果。

## 1. 我们现在是怎么训练的

目前我们在 SWE-bench `full/test` 上训练一个 **Controller**。Controller 本身不直接改代码，它主要负责从已有 skill 中做选择，并给出修复思路和执行预算。真正检查代码、生成 patch 的模型是 `gpt-4o-mini`。生成 patch 后，再用 SWE-bench 的 Docker 测试环境检查 F2P（原来失败的测试是否修好）和 P2P（原来通过的测试是否仍然通过），最后把得到的 reward 用于 GRPO 训练 Controller。

每轮训练结束后，系统还会整理这一轮的失败任务。一部分结果用来调整下一轮 TaskPool 的采样权重，让比较难的任务更容易再次被抽到；另一部分交给 Reflector，用来创建、更新或淘汰 skill。

```text
SWE-bench task
  -> Controller（选择 skill / strategy / budget）
  -> gpt-4o-mini（检查代码并生成 patch）
  -> SWE-bench harness（F2P / P2P）
  -> Online reward
  -> GRPO 更新 Controller
  -> TaskPool + Skill evolution
```

### 目前使用的主要参数

| 配置项 | 设置 |
|---|---|
| Controller | Qwen2.5-1.5B-Instruct + LoRA |
| Worker / Patch Generator | `gpt-4o-mini` |
| 训练算法 | GRPO |
| 每轮训练长度 | 64 optimizer steps |
| Rollout batch size | 4 |
| 每个任务的 rollout 数 | 3 |
| 每轮冻结的训练任务 | 320 |
| 每轮实际 rollout | 约 740–760 |
| 最大修复迭代 | 3 |
| Validation | 固定 18 个任务，每题 1 个 rollout |
| TaskPool 采样 | 50% 均匀采样 + 50% 失败加权采样 |

这里的 64 step 只是一次更新周期，并不是每一轮都从头训练。下一轮会继续读取上一轮的 checkpoint，所以 Round 6–11 实际上是一段连续的训练，对应 step 576–960。

Online reward 为：

```text
0.5 * resolved
+ 0.2 * F2P fraction
+ 0.1 * P2P no-regression
+ 0.1 * cost efficiency
+ 0.1 * process score
```

所以 reward 大于 0 不一定说明任务已经完全修好；只有 `Resolved` 才表示目标测试通过，同时没有引入新的回归。

## 2. 每轮训练结果

| Round | Steps | Rollouts | 不同任务 | Train reward | Train resolved | Val reward | Val resolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | 576–640 | 739 | 193 | 0.1740 | 82/739（11.10%） | 0.1272 | 1/18（5.56%） |
| 7 | 640–704 | 747 | 194 | 0.1588 | 68/747（9.10%） | 0.1453 | 1/18（5.56%） |
| 8 | 704–768 | 745 | 192 | 0.1744 | 82/745（11.01%） | 0.1540 | 1/18（5.56%） |
| 9 | 768–832 | 753 | 203 | 0.1826 | 88/753（11.69%） | 0.1416 | 1/18（5.56%） |
| 10 | 832–896 | 761 | 219 | **0.1992** | **104/761（13.67%）** | 0.1292 | 1/18（5.56%） |
| 11 | 896–960 | 755 | **240** | 0.1790 | 84/755（11.13%） | 0.1215 | 1/18（5.56%） |

从表里可以看到，Round 10 的训练 reward 和 resolved rate 是目前最高的，但 Round 11 又下降了一些，说明结果还不稳定。固定 validation 每一轮都是 1/18，所以现在还不能说明训练效果已经稳定地提升到了没参与训练的任务上。另外，validation 只有 18 条，多通过 1 条就会变化 5.56%，因此目前它主要用于快速观察训练有没有明显问题，不能单独作为最终结果。

## 3. 每轮抽到了哪些任务

下表统计了实际进入训练 rollout 的仓库分布。为了让表格短一些，只单独列出占比较高的四个仓库，其余 8 个放在“其他”中。

| Round | Django | SymPy | scikit-learn | Sphinx | 其他 8 个仓库 | 不同任务 |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 23.5% | 11.5% | 12.9% | 5.7% | 46.4% | 193 |
| 7 | 27.0% | 15.0% | 9.8% | 6.2% | 42.0% | 194 |
| 8 | 28.1% | 17.3% | 9.7% | 7.5% | 37.4% | 192 |
| 9 | 26.3% | 15.8% | 10.4% | 12.0% | 35.5% | 203 |
| 10 | 29.0% | 14.3% | 10.5% | 6.7% | 39.5% | 219 |
| 11 | 29.4% | 12.3% | 9.3% | 8.3% | 40.7% | 240 |

每轮都覆盖到了 12 个仓库。Django 的比例一直最高，大约是 24%–29%，这既和原始数据分布有关，也受到失败任务加权的影响。不过仍然有 71%–76% 的 rollout 来自其他仓库。Round 9–11 的不同任务数从 203 增加到 240，说明训练没有一直重复少数几个困难任务。

Task evolution 的做法是：Reflector 根据失败任务给出 `instance_boosts`，下一轮一半使用较均匀的采样，一半使用 TaskPool 权重采样。这样既能再次训练困难任务，也能继续看到不同类型的任务。为了避免某一个任务出现太多，每个任务一轮最多重复 3 次。

### 3.1 上一轮的困难任务有多少被再次抽到

这里需要注意，“50% 失败加权采样”不表示 320 条任务中一定有 160 条困难任务。代码会先相对均匀地抽 160 条，再从**完整 TaskPool**中按权重抽剩下的 160 条。困难任务只是更容易被抽到，不是一定会被抽到。由于最终数据被打乱后没有保存每条任务来自哪一半，所以下表直接统计“上一轮被加权、并且这一轮确实再次出现”的任务。

| Round | 上轮被加权的任务 | 本轮再次抽到的任务 | 对应训练记录 | 占 320 条任务 |
|---|---:|---:|---:|---:|
| 6 | 180 | 46 | 53 | 16.6% |
| 7 | 177 | 42 | 47 | 14.7% |
| 8 | 180 | 41 | 43 | 13.4% |
| 9 | 175 | 35 | 35 | 10.9% |
| 10 | 185 | 42 | 44 | 13.8% |
| 11 | 199 | 47 | 52 | 16.3% |

下面是一些实际被再次抽到的任务。`本轮次数` 表示这个任务在本轮 320 条数据中出现了几次；`失败记录数` 表示它上一轮在有效失败簇中出现了几条失败记录。

| Round | 实例 | 本轮次数 | 失败记录数 | 问题摘要 |
|---|---|---:|---:|---|
| 6 | `scikit-learn__scikit-learn-12784` | 3 | 2 | KNeighborsRegressor 在不同 `n_jobs` 下结果不同 |
| 6 | `astropy__astropy-8263` | 2 | 3 | 单位转换错误：`s / m` 被判定为不可转换 |
| 6 | `psf__requests-774` | 2 | 2 | 未捕获 IDNA 编码产生的 `UnicodeError` |
| 7 | `pylint-dev__pylint-5730` | 2 | 3 | 文档参数扩展发生命令行选项冲突 |
| 7 | `scikit-learn__scikit-learn-11235` | 2 | 3 | StandardScaler 一致性问题 |
| 7 | `django__django-13341` | 2 | 2 | TemplateView 不应自动把 kwargs 放入 context |
| 8 | `sympy__sympy-19093` | 2 | 3 | mutable BlockDiagMatrix 表达式回归 |
| 8 | `scikit-learn__scikit-learn-14450` | 2 | 2 | PLS 遇到常量特征时报 NaN 错误 |
| 8 | `matplotlib__matplotlib-13859` | 1 | 6 | 零宽度 figure 导致 libpng 崩溃 |
| 9 | `django__django-11279` | 1 | 6 | index/constraint 名称支持 app label 与 class |
| 9 | `pallets__flask-4575` | 1 | 6 | 将 `redirect` 移到 Flask app 对象 |
| 9 | `psf__requests-2466` | 1 | 6 | PyInstaller 兼容问题 |
| 10 | `sphinx-doc__sphinx-7923` | 2 | 2 | `rst_epilog` 导致 pot 文件引用错误 |
| 10 | `sympy__sympy-13988` | 2 | 2 | `Integral.as_sum()` 应返回 `Sum` 对象 |
| 10 | `django__django-13355` | 1 | 6 | 优化 `django.forms.widgets.Media.__add__` |
| 11 | `astropy__astropy-7218` | 2 | 4 | `HDUList.copy()` 错误返回普通 list |
| 11 | `sympy__sympy-15241` | 2 | 3 | Derivative 变量规范化问题 |
| 11 | `pytest-dev__pytest-7499` | 2 | 2 | 改善 `raises` 匹配失败时的提示 |
| 11 | `django__django-12212` | 1 | 9 | 反序列化异常被未赋值的局部变量 `pk` 掩盖 |

### 3.2 失败任务被分成了哪些组

系统会把这一轮值得保留的失败记录拿出来分组。分组时会比较 `failure_type`、失败原因、测试状态和错误信息。当前没有配置 embedding，所以使用文字相似度进行判断，相似度阈值是 0.55。一个组至少要有 3 条记录，才会交给 Reflector。下表列出了 Round 6–11 所有达到这个条件的失败组。

| Round | 失败组 | 记录数 | 主要失败类型 | 代表实例/特点 |
|---|---:|---:|---|---|
| 6 | C1 | 596 | `test_failure` | `max_iterations` 主簇；如 `sympy__sympy-24102`、`django__django-13615`、`astropy__astropy-12962` |
| 6 | C2 | 6 | `test_failure` | `django__django-12830` |
| 6 | C3 | 4 | `unknown` | `matplotlib__matplotlib-25631` |
| 6 | C4 | 3 | `unknown` | `django__django-11446` |
| 6 | C5 | 3 | `unknown` | `scikit-learn__scikit-learn-13472` |
| 7 | C1 | 613 | `test_failure` | `max_iterations` 主簇；如 `sympy__sympy-15567`、`scikit-learn__scikit-learn-25601`、`matplotlib__matplotlib-13859` |
| 7 | C2 | 3 | `unknown` | `django__django-12184` |
| 7 | C3 | 3 | `unknown` | `pydata__xarray-4182` |
| 7 | C4 | 3 | `unknown` | `pytest-dev__pytest-8906` |
| 7 | C5 | 3 | `unknown` | `sympy__sympy-20691`、`django__django-13886` |
| 8 | C1 | 596 | `test_failure` | `max_iterations` 主簇；如 `sympy__sympy-17696`、`django__django-11517`、`scikit-learn__scikit-learn-7760` |
| 8 | C2 | 6 | `unknown` | `pydata__xarray-5131` |
| 8 | C3 | 3 | `unknown` | `matplotlib__matplotlib-18869` |
| 8 | C4 | 3 | `regression_introduced` | `matplotlib__matplotlib-25772`、`pallets__flask-4575` |
| 8 | C5 | 3 | `unknown` | `pytest-dev__pytest-7648` |
| 9 | C1 | 602 | `test_failure` | `max_iterations` 主簇；如 `pytest-dev__pytest-5254`、`pallets__flask-4935`、`mwaskom__seaborn-2576` |
| 9 | C2 | 5 | `test_failure` | `django__django-11527` |
| 9 | C3 | 5 | `test_failure` | `sphinx-doc__sphinx-9261` |
| 9 | C4 | 3 | `unknown` | `sphinx-doc__sphinx-9128` |
| 9 | C5 | 3 | `test_failure` | `sympy__sympy-13624` |
| 10 | C1 | 607 | `test_failure` | `max_iterations` 主簇；如 `mwaskom__seaborn-2848`、`pytest-dev__pytest-7499`、`django__django-16735` |
| 10 | C2 | 8 | `unknown` | `django__django-12212` |
| 10 | C3 | 3 | `unknown` | `scikit-learn__scikit-learn-14898`、`pylint-dev__pylint-5175` |
| 11 | C1 | 608 | `test_failure` | `max_iterations` 主簇；如 `sympy__sympy-24638`、`sphinx-doc__sphinx-10757`、`django__django-13097` |
| 11 | C2 | 5 | `unknown` | `astropy__astropy-14907`、`matplotlib__matplotlib-20788`、`pydata__xarray-6548` |
| 11 | C3 | 5 | `unknown` | `scikit-learn__scikit-learn-14898`、`pylint-dev__pylint-5175`、`sympy__sympy-14575` |
| 11 | C4 | 3 | `unknown` | `django__django-12212` |
| 11 | C5 | 3 | `test_failure` | `pallets__flask-5063` |
| 11 | C6 | 3 | `unknown` | `sympy__sympy-17103` |

从这个结果也能看出一个比较明显的问题：每轮大约 90% 的 hard cases 都进入了同一个很大的组，因为它们都包含 `max_iterations`、`tests_failed` 或 `new_issues` 这类共同信息。这样 Reflector 看到的主要是“测试失败”或“迭代次数用完”，而不是更具体的仓库、文件位置和代码问题，所以生成的 skill 也比较通用。

## 4. 每轮生成或更新了哪些 skill

| Round | Hard cases | 全部失败组 | 达到 3 条的组 | Skill 操作 |
|---|---:|---:|---:|---|
| 6 | 657 | 45 | 5 | 更新 `max_iterations_management`；创建 `unknown_failure_analysis` |
| 7 | 679 | 55 | 5 | 更新 `unknown_failure_analysis`；创建 `test_failure_analysis` |
| 8 | 663 | 50 | 5 | 更新上述两个 failure-analysis skills |
| 9 | 665 | 44 | 5 | 更新 `test_failure_analysis`；重新创建 `max_iterations_management` |
| 10 | 657 | 37 | 3 | 更新两个 failure-analysis skills；淘汰 `max_iterations_management` |
| 11 | 671 | 45 | 6 | 再次更新两个 failure-analysis skills |

下面两段是 Reflector 实际生成并写入 Skill Bank 的内容，没有经过人工改写。

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

从这些结果可以确认，skill 的创建、更新和淘汰流程已经能够运行。不过目前生成的内容还是比较通用，对具体仓库和代码问题的帮助有限。这可能也是 skill 一直在更新，但 validation 还没有明显提高的原因之一。

## 5. 目前的总结

1. 目前连续 GRPO、SWE-bench 在线 reward、TaskPool 更新和 skill 更新都已经能够跑通。每轮虽然只有 64 step，但实际会产生大约 750 条修复轨迹。
2. Train resolved 在 9.10%–13.67% 之间变化，Round 10 最好，但还没有一直上升。
3. Validation 每轮都是 1/18，因此目前还不能说明 Controller 对新任务的效果已经提高。
4. TaskPool 会提高困难任务再次被抽到的概率，同时每轮仍然覆盖 12 个仓库，后几轮看到的不同任务也有所增加。
5. 每轮实际再次抽到 35–47 个上一轮被加权的困难任务，对应 35–53 条训练记录，并不是一半训练数据都是困难任务。
6. Reflector 每轮都会根据约 650 条 hard cases 更新 skill，但大部分失败被合并到了一个 `max_iterations` 大组中，所以生成的 skill 还比较宽泛。后面需要继续观察这些 skill 是否真的能提高验证结果。

目前可以得出的结论是：**整个自进化训练流程已经跑通，训练结果有过一定提高，但波动比较明显，validation 也还没有提升，所以现在还不能说明模型已经获得了稳定的泛化能力。**
