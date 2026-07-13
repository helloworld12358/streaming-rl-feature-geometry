# 今晚统一总目标：Streaming RL Predictive Representation Properties

## 使用说明

这是当前 Codex 任务唯一的、完整的、最高优先级目标规范。

本文件同时覆盖：

1. 对当前已经开始但尚未完全收尾的核心 T-maze 任务进行检查、修复、测试、pilot、文档和 Git 收尾；
2. 在核心任务形成可靠 Git checkpoint 后，扩展到多个不同 latent-state geometry 的 continuing POMDP / streaming control 环境；
3. 比较 predictive representations 服从不同统计、几何、分布、时间和代数性质时，对严格 Streaming RL agent 的预测、状态表示和控制性能的影响；
4. 准备能够在远程 Linux 云服务器上运行完整大规模实验的代码、配置、脚本和中文傻瓜式教程；
5. 将全部有效工作提交并推送到现有 GitHub 仓库；
6. 本地只运行测试、smoke 和有限 pilot，不运行远程 full experiment。

当前本地仓库预期位于：

```text
D:\download\GitHub\streaming-rl-feature-geometry
```

现有 GitHub 仓库：

```text
https://github.com/helloworld12358/streaming-rl-feature-geometry.git
```

本文件下载后的预期本地路径：

```text
C:\Users\27768\Downloads\codex_complete_overnight_master_goal.md
```

如果实际下载文件名被浏览器追加了 `(1)` 或其他后缀，请先搜索：

```text
C:\Users\27768\Downloads\codex_complete_overnight_master_goal*.md
```

找到唯一正确文件后再读取。

不要重新 clone 仓库，不要创建新 GitHub 仓库，不要修改 origin，不要删除 `.git`，不要 force push。

---

# 0. 执行原则

你必须持续实际工作，不能只输出计划后停止。

每个阶段必须执行：

1. 检查当前状态；
2. 阅读相关代码和文档；
3. 做最小必要修改；
4. 运行对应测试；
5. 诊断失败；
6. 修复实现错误；
7. 保存真实结果；
8. 记录负结果；
9. 创建 Git checkpoint；
10. 再进入下一阶段。

只有遇到真正的外部阻塞时才暂停，例如：

- GitHub 必须由用户重新登录；
- 当前 Windows 权限禁止访问仓库；
- 网络完全不可用；
- 缺少不可替代的软件；
- 出现未知用户文件冲突，无法安全判断。

遇到可选方法失败时：

- 记录失败；
- 保留日志；
- 尝试有限、合理的修复；
- 不得让可选方法阻塞全部必需任务；
- 不得静默换成另一个方法；
- 不得伪造成功。

禁止以下命令：

```text
git reset --hard
git clean -fd
git push --force
```

禁止：

- 删除未知用户文件；
- 丢弃当前有效修改；
- 重建仓库；
- 重写 main 历史；
- 读取、显示或提交密码、token、SSH 私钥、Cookie 或其他凭据。

---

# 1. 任务分为两个严格顺序阶段

## 阶段 A：先完整收尾当前核心 T-maze 项目

阶段 A 完成之前，不得开始 Ringworld、Two-loop、Hidden-velocity 或其他跨环境扩展。

阶段 A 必须完成：

1. 审计当前工作区和当前已有代码；
2. 保存当前有效修改；
3. 修复 continuing T-maze；
4. 修复 raw GVF / fixed trace / controller；
5. 修复 standardization、decorrelation、whitening、Gaussian-inspired transform；
6. 修复结果隔离、probe、真实图表和验证逻辑；
7. 完成完整单元测试；
8. 完成 smoke；
9. 完成三 seed 有限 pilot；
10. 生成当前核心 Proposal；
11. 生成远程运行包和教程；
12. 创建清晰 commit；
13. push 当前核心分支；
14. 若认证允许，创建或更新 Draft PR；
15. 记录核心 commit SHA。

## 阶段 B：核心 checkpoint 后执行跨环境扩展

只有阶段 A 已经：

- 测试通过；
- smoke 完成；
- pilot 完成；
- 文档完成；
- 创建 commit；

之后，才进入阶段 B。

阶段 B 必须基于阶段 A 的核心 commit 创建或使用分支：

```text
codex/cross-environment-representation-priors
```

阶段 B 增加：

- Aliased Ringworld；
- Two-loop aliased POMDP；
- Hidden-velocity continuing control；
- 可选 switching hidden-context；
- categorical/simplex prior；
- circular prior；
- block categorical × circular prior；
- anisotropic covariance prior；
- 跨环境 smoke 和三 seed pilot；
- 跨环境 Proposal、报告、图表和远程 full 配置。

阶段 A 和阶段 B 不得并行修改同一批文件。

---

# 2. 开始时必须执行的审计

首先执行：

```text
git status
git branch --show-current
git remote -v
git log -1 --oneline
git diff --stat
git diff
```

验证：

```text
origin = https://github.com/helloworld12358/streaming-rl-feature-geometry.git
```

读取：

- `README.md`
- `AGENTS.md`
- 所有 proposal 文件
- 所有 research specification
- implementation report
- work log
- source code
- tests
- configs
- scripts
- workflows
- 当前结果摘要
- 当前分支和 PR 相关信息

创建或更新：

```text
docs/COMPLETE_TASK_RECONCILIATION.md
```

写清楚：

- 当前仓库实际状态；
- 当前分支；
- 当前 commit；
- 当前未提交修改；
- 当前已实现内容；
- 当前已知 bug；
- 当前失败 tests；
- 当前已有 smoke/pilot；
- 阶段 A 尚缺内容；
- 阶段 B 尚缺内容；
- 需要保留、修改、删除或替换的实现；
- 执行顺序；
- 科学风险；
- 工程风险。

完成该文档后继续实际实现，不能停下来等待用户回复。

---

# 3. 唯一研究主体

项目题目：

```text
Prescribed Statistical and Geometric Properties of Predictive Representations for Streaming Reinforcement Learning under Partial Observability
```

核心问题：

> 在严格 streaming、无 replay、部分可观测、continuing 的强化学习中，如果预先要求由 observation history 构造出的 predictive features / GVFs 满足特定的统计、几何或分布性质，是否能够改善隐藏状态信息保留、在线更新稳定性、样本效率和控制性能？

跨环境扩展问题：

> representation prior 的效果是否取决于 latent-state geometry？与任务 latent structure 匹配的 prior 是否优于通用 isotropic Gaussian prior？

核心数据流：

```text
partial observation/action/reward history
→ online trace / predictive features / GVFs
→ prescribed representation transform or constraint
→ online linear controller
```

主要自变量必须是：

```text
representation mathematical property
```

GVF question 和 horizon 只用于：

- 构造固定 predictive bank；
- 少量 feature-bank ablation；
- 少量 horizon mismatch ablation。

不得把项目改成：

- 哪个 GVF question 最好；
- 哪个 horizon 最好；
- 哪个 world model 最好；
- 哪个深度网络最好。

---

# 4. 硬性算法约束

## 4.1 严格 Streaming RL

每个 transition 到达时只能被 agent 用于一次在线更新。

禁止：

- replay buffer；
- experience replay；
- minibatch；
- batch training；
- offline representation fitting；
- 收集完整数据后再训练 agent；
- 使用完整轨迹做反向传播；
- 使用完整数据集 covariance；
- 使用未来 observation；
- 使用未来统计量；
- 根据 test seeds 调整方法；
- 删除失败 seeds。

允许维护有限维递推量：

- 当前 observation；
- 当前 action；
- 当前 reward；
- fixed trace state；
- GVF predictions；
- eligibility traces；
- running mean；
- running variance；
- running covariance；
- exponentially weighted third/fourth moments；
- predictor weights；
- controller weights；
- transform parameters；
- 明确的递归统计量。

当前时刻 representation 必须使用截至当前可用的因果历史，不能使用未来数据。

## 4.2 禁止深度学习

禁止：

- PyTorch；
- TensorFlow；
- JAX；
- MLP；
- CNN；
- Transformer；
- LSTM；
- GRU；
- deep RNN；
- autoencoder；
- deep world model；
- JEPA neural architecture；
- target network；
- GPU training。

使用：

- Python；
- NumPy；
- SciPy；
- pandas；
- matplotlib；
- pytest；
- Python 标准库。

predictor、representation 和 controller 必须是：

- 线性函数逼近；
- 小规模线性递归；
- 或具有明确在线更新公式的简单非深度方法。

## 4.3 禁止无关机制

不要实现：

- episodic retrieval memory；
- nearest-neighbor memory；
- replay；
- image observation；
- MuJoCo；
- Atari；
- model-based planning；
- Dyna；
- learned transition model；
- generative reconstruction；
- normalizing flow；
- 大型 benchmark suite。

---

# 5. 代码修改原则

遵循最小修改原则：

1. 不删除已有正确功能；
2. 不主动缩减已有功能；
3. 不重命名已有文件，除非确实无法避免；
4. 不重构无关代码；
5. 不增加无关工程功能；
6. 不增加静默 fallback；
7. 错误必须明确报错；
8. 不吞掉异常；
9. 代码与文档必须一致；
10. 负结果必须保留；
11. 不得为了通过测试降低科学正确性要求。

---

# 6. 阶段 A：核心 T-maze 环境

实现或修复严格 continuing partially observable T-maze。

每个 trial：

1. cue phase；
2. corridor phase；
3. junction；
4. outcome；
5. 立即进入下一 trial。

隐藏 cue：

```text
b ∈ {-1, +1}
```

cue 只在 trial 开头可见。

进入 corridor 后：

- cue 消失；
- junction observation 在两个 cue 条件下完全一致；
- 当前 observation 无法决定正确动作；
- agent 必须依赖内部 predictive state。

奖励：

- 正确：+1；
- 错误：-1；
- 其他：0。

outcome 后必须进入下一 trial。

trial 边界不得重置：

- predictor；
- controller；
- running statistics；
- learned representation；
- 默认 eligibility traces。

Delayed cue echo：

- 只能在 junction action 完成后出现；
- 主 agent 决策时不得访问 latent cue；
- oracle baseline 可访问 cue，但路径必须隔离。

环境 tests：

- phase transition；
- continuing transition；
- outcome → next trial；
- no cue leakage；
- delayed cue echo；
- reproducibility；
- reward correctness；
- corridor length；
- observation-only 不能直接读取 cue。

---

# 7. 阶段 A：Predictive features

主体优先使用：

```text
m_t = rho * m_(t-1) + B * o_t
x_t = [o_t, m_t, 1]
```

fixed trace 是 hand-designed memory。

在该输入上在线学习 linear GVF bank。

mixed GVF bank 可包含：

- future left cue echo；
- future right cue echo；
- future junction occupancy；
- future positive outcome；
- future negative outcome；
- 少量固定 horizon。

每个 GVF 记录：

- name；
- cumulant；
- continuation；
- effective horizon；
- interpretation；
- prediction；
- TD error。

实现：

- observation-only baseline；
- oracle baseline；
- trace-only baseline；
- raw GVF baseline。

不得使用 batch regression。

---

# 8. 阶段 A 和 B 共用的 representation properties

令：

```text
g_t = raw predictive vector
z_t = transformed representation
```

必须实现：

## R0 Raw

```text
z_t = g_t
```

## R1 RMS-matched raw

仅因果匹配整体 RMS scale，不改变 correlation structure。

## R2 Marginal standardization

目标：

```text
mean(z_i) ≈ 0
variance(z_i) ≈ 1
```

必须解决首样本和 warm-up，不得被 epsilon 异常放大。

## R3 Decorrelation

降低 off-diagonal covariance，明确区别于 whitening。

## R4 Whitening / covariance isotropy

目标：

```text
mean(z) ≈ 0
Cov(z) ≈ I
```

必须因果在线。

只能称为：

- whitening；
- second-order isotropy；
- covariance isotropy。

不得称为完整 Gaussian。

## R5 Gaussian-inspired online moment matching

目标：

- zero mean；
- identity covariance；
- skewness 接近 0；
- kurtosis 接近 3。

选择一种简单、低参数、单调、非神经、在线方法：

- signed power；
- Yeo–Johnson；
- asinh；
- sinh–arcsinh。

不得：

- offline quantile transform；
- normalizing flow；
- 使用 reward 调节 transform 参数。

必须在 synthetic streams 上测试：

- Gaussian；
- skewed；
- heavy-tailed；
- bimodal。

若失败：

- 保留失败报告；
- 不声称成功；
- 不把 R4 改名为 Gaussian；
- 继续其他必需任务。

## R6 Unit sphere

```text
z_t = g_t / (||g_t|| + epsilon)
```

## R7 Sparse

选择一个简单透明的：

- top-k；
- soft threshold；
- fixed threshold。

## R8 Bounded

例如：

```text
z_t = tanh(alpha * standardized_g_t)
```

alpha 预先固定。

---

# 9. 阶段 A 控制算法

使用严格在线线性控制：

优先：

```text
true-online SARSA(lambda)
```

若已有正确 semi-gradient SARSA(lambda)，可保留并明确说明。

controller input：

```text
phi_t = [observation_t, z_t, bias]
```

所有 conditions 共享：

- environment；
- raw predictive bank；
- action space；
- reward；
- exploration；
- interaction budget；
- seeds；
- 尽可能相同 learning rate。

必须包含：

- shared-learning-rate comparison；
- RMS-scale-matched comparison。

不得为不同 property 大规模单独调参。

---

# 10. 阶段 A 核心假设

在新 pilot 前写入：

```text
docs/research_specification.md
```

至少包含：

- scale hypothesis；
- redundancy hypothesis；
- isotropy hypothesis；
- Gaussian-moment hypothesis；
- binary/bimodal hypothesis；
- rank-not-enough hypothesis；
- mixed-bank hypothesis；
- magnitude-information hypothesis；
- adaptation hypothesis。

记录文档创建时间和 Git 状态。

---

# 11. 阶段 A 指标

Control：

- trial accuracy；
- moving-average accuracy；
- cumulative reward；
- final-window accuracy；
- time to threshold；
- across-seed standard error。

Prediction：

- per-GVF TD error；
- MSE；
- cue-echo error；
- outcome error；
- calibration（如适用）。

Hidden-state：

- cue decodability；
- cue margin；
- between/within variance ratio；
- junction classification accuracy。

Probe：

- 只用于诊断；
- train/test split；
- 不反馈给 agent；
- 不得只报告训练集拟合。

Representation：

- mean error；
- variance imbalance；
- off-diagonal correlation；
- eigenvalues；
- condition number；
- effective rank；
- isotropy error；
- skewness error；
- kurtosis error；
- norm statistics；
- sparsity；
- transform drift。

Update stability：

- TD-error variance；
- update norm；
- parameter norm；
- NaN/Inf；
- divergence flags。

---

# 12. 阶段 A 本地 profiles

## Smoke

- seed 0；
- 2000–5000 interactions；
- observation-only；
- oracle；
- raw；
- whitened；
- 单进程；
- 几分钟内完成。

## Pilot

- seeds `[0, 1, 2]`；
- 20000–50000 interactions；
- stationary T-maze；
- observation-only；
- oracle；
- trace；
- raw；
- RMS raw；
- standardized；
- decorrelated；
- whitened；
- Gaussian-inspired（若诊断通过）；
- 最多一个额外 property；
- 本地目标不超过约 30 分钟。

不得把主要 pilot 缩减为一个 seed。

## Remote full

只准备，不在本地运行。

必须同时要求：

```text
RL_RUN_CONTEXT=remote
--allow-full-run
```

缺一即明确报错。

---

# 13. 阶段 A 结果和图表

每次 run 唯一目录：

```text
results/<profile>/<run_id>/
```

包含：

- config；
- manifest；
- commit SHA；
- branch；
- dirty status；
- exact command；
- seed；
- metrics；
- logs；
- figures；
- exit status。

不得混入旧结果，不得覆盖。

图表必须来自真实数据，不得占位：

- learning curve；
- final accuracy；
- cumulative reward；
- TD error；
- cue decodability；
- eigenvalue spectrum；
- effective rank；
- isotropy error；
- condition number；
- skewness/kurtosis；
- feature norm；
- update norm；
- accuracy vs isotropy；
- accuracy vs rank；
- accuracy vs decodability；
- cue-conditioned projection。

使用 matplotlib，不使用 seaborn。

---

# 14. 阶段 A 文档

生成或更新：

```text
AGENTS.md
README.md
docs/research_specification.md
docs/proposal_en.md
docs/proposal_zh_explanation.md
docs/implementation_report.md
docs/REMOTE_RUN_GUIDE_ZH.md
docs/paper_outline.md
docs/poster_outline.md
docs/work_log.md
```

Proposal 主体必须是：

> prescribed statistical and geometric properties of predictive features in streaming RL

不得改成 GVF question comparison。

Pilot 结果必须明确：

- seeds；
- interaction budget；
- result path；
- commit；
- uncertainty；
- 只是 pilot，不是 final conclusion。

---

# 15. 阶段 A Git checkpoint

阶段 A 完成时：

1. 运行完整 tests；
2. 运行 smoke；
3. 运行 pilot；
4. `git diff --check`；
5. 检查 secrets；
6. 创建清晰 commit；
7. push 当前核心分支；
8. 若允许，创建或更新 Draft PR；
9. 记录：
   - branch；
   - commit SHA；
   - push status；
   - PR URL；
   - tests；
   - pilot path。

阶段 A 未形成 commit 之前不得进入阶段 B。

---

# 16. 阶段 B 研究问题

扩展研究：

```text
latent-state geometry
× representation prior
→ property fulfillment
→ latent accessibility
→ optimization stability
→ control performance
```

需要区分：

- task-agnostic prior；
- task-matched prior；
- property metric；
- predictive accuracy；
- latent decodability；
- control utility。

主动寻找：

- 更 isotropic 但控制更差；
- 更 Gaussian 但 categorical separation 更差；
- 更高 rank 但 reward 不提高；
- sphere 删除 magnitude；
- sparse 伤害连续插值；
- global prior 不如 block prior。

---

# 17. 阶段 B 环境 E1–E4

## E1 T-maze

保留阶段 A。

latent：

```text
binary categorical cue
```

task-matched prior：

```text
categorical/simplex or mixture-like representation
```

## E2 Aliased continuing Ringworld

位置：

```text
s_t ∈ {0, ..., N-1}
```

动作：

- clockwise；
- counterclockwise；
- 可选 stay。

大部分位置 observation alias。

少数 landmark 可区分。

控制任务必须：

- 当前 observation 不足以解决；
- 需要 history/predictive state；
- 有明确 oracle；
- CPU 轻量；
- continuing。

latent：

```text
circular phase S^1
```

task-matched prior：

```text
[cos(theta_hat), sin(theta_hat)]
```

theta_hat 必须从在线 predictive features 推断，不能使用真实 latent phase 作为 agent input。

## E3 Two-loop aliased POMDP

两个 loops：

- identity 不同；
- length 或 rhythm 不同；
- 大量 observation alias；
- correct action 依赖 identity、phase 或两者。

latent：

```text
discrete identity × circular phase
```

指标：

- identity decodability；
- phase error；
- joint-state decodability；
- control accuracy。

task-matched prior：

```text
categorical/simplex identity block
×
circular phase block
```

agent training 不得访问真实 identity 或 phase。

## E4 Hidden-velocity continuing control

建议动力学：

```text
x_(t+1) = x_t + v_t
v_(t+1) = rho * v_t + a_t + noise_t
```

agent 只观察 position，不观察 velocity。

离散 actions：

```text
{-a, 0, +a}
```

reward 可使用：

```text
-r = x^2 + eta*v^2 + xi*a^2
```

实现时保持数值稳定和 continuing。

latent：

```text
continuous approximately ellipsoidal/Gaussian hidden state
```

task-matched prior：

```text
anisotropic Gaussian / prescribed covariance
```

比较：

- raw；
- standardized；
- isotropic whitening；
- prescribed anisotropic covariance。

anisotropic target 必须由预先声明的动力学/feature 语义给出，不能根据最终 reward 调参。

## E5 可选 Switching hidden-context

只有 E1–E4 全部完成后才允许实现。

隐藏 context：

```text
c_t ∈ {1, ..., K}
```

task-matched prior：

```text
online mixture/simplex
```

不得延误必需任务。

---

# 18. 阶段 B predictive banks

所有环境使用统一接口：

- observation；
- action；
- reward；
- diagnostic latent state；
- event cumulants；
- continuation；
- oracle features。

每个环境提供：

1. compact task-relevant bank；
2. mixed redundant bank。

预测可以包括：

- future landmark occupancy；
- event occupancy；
- cue echo；
- rewarding event；
- negative event；
- fixed horizons；
- phase-related events；
- outcome predictions。

不得为了某个 transform 有利而专门改变维度。

---

# 19. 阶段 B task-matched priors

## M1 Categorical simplex / mixture-like

用于 E1 和可选 E5。

要求：

- nonnegative；
- normalized 或明确 categorical scores；
- agent learning 不使用 latent labels；
- streaming online。

## M2 Circular

用于 E2。

二维圆环 block：

```text
[cos(theta_hat), sin(theta_hat)]
```

theta_hat 来源于 predictive features。

## M3 Block categorical × circular

用于 E3。

不同子空间使用不同 prior。

## M4 Prescribed anisotropic covariance

用于 E4。

明确区别：

- conditioning；
- isotropy；
- task-relevant anisotropy。

---

# 20. 阶段 B 实验矩阵

不要完整笛卡尔积。

## Stage A：跨环境核心比较

环境：

- E1；
- E2；
- E3；
- E4。

representation：

- raw；
- RMS；
- standardized；
- whitened；
- Gaussian-inspired（若有效）；
- sphere；
- sparse。

decorrelation 和 bounded 保持实现和测试，可不全部进入本地跨环境 pilot。

## Stage B：task-matched prior

- E1：M1；
- E2：M2；
- E3：M3；
- E4：M4。

## Stage C：bank ablation

每个环境只比较：

- compact；
- mixed。

## Stage D：horizon mismatch

最多选择两个环境：

- matched；
- short。

## Stage E：remote non-stationary

- E1 corridor length change；
- 可选 E5 context switch。

---

# 21. 阶段 B 假设

在跨环境 pilot 前创建：

```text
docs/CROSS_ENV_HYPOTHESES.md
```

包含：

- universal conditioning hypothesis；
- geometry-matching hypothesis；
- binary/multimodal hypothesis；
- circular hypothesis；
- product-structure hypothesis；
- continuous-state hypothesis；
- rank-not-enough；
- isotropy-not-enough；
- magnitude-information；
- sparsity tradeoff；
- blockwise-prior；
- adaptation hypothesis。

记录时间和 Git 状态。

---

# 22. 阶段 B 指标

统一指标：

- cumulative reward；
- final reward；
- time to threshold；
- seed uncertainty；
- TD error；
- update norm；
- parameter norm；
- condition number；
- effective rank；
- isotropy；
- skewness；
- kurtosis；
- norm；
- sparsity；
- transform drift。

E1：

- cue decodability；
- cue margin；
- trial accuracy。

E2：

- phase decoding error；
- circular correlation；
- neighborhood preservation；
- control reward/accuracy。

E3：

- identity decodability；
- phase error；
- joint-state decodability；
- control accuracy。

E4：

- velocity decoding error；
- state-estimation error；
- control cost；
- stabilization rate；
- update variance。

E5：

- context decodability；
- switch detection delay；
- recovery time。

所有 offline probes：

- 只诊断；
- train/test split；
- deterministic；
- 不反馈 agent。

---

# 23. Property fulfillment diagnostics

在解释方法前验证 property 确实成立。

保存：

```text
results/diagnostics/cross_environment/
```

检查：

- standardization；
- decorrelation；
- whitening；
- Gaussian moments；
- circular unit norm 和 phase order；
- simplex nonnegative/normalized；
- block semantics；
- anisotropic covariance；
- sparsity；
- boundedness。

生成机器可读 pass/fail summary。

失败 condition 不得包装为成功。

---

# 24. 阶段 B 本地 profiles

## Cross-environment smoke

每个 E1–E4：

- one seed；
- short budget；
- observation-only；
- oracle；
- raw；
- whitened；
- task-matched prior。

## Cross-environment pilot

- seeds `[0, 1, 2]`；
- E1–E4；
- 核心 representations；
- task-matched prior；
- no remote non-stationarity；
- no full Cartesian product；
- 本地目标约 60 分钟以内。

若太慢：

1. 减少 interactions；
2. 保留 3 seeds；
3. 保留 E1–E4；
4. 保留 raw、RMS、standardized、whitened、Gaussian-inspired（若有效）、sphere、sparse 和 matched prior；
5. 推迟低优先级 conditions 到 remote。

## Remote full

只准备，不在本地运行。

建议：

- seeds 0–19；
- E1–E4；
- task-agnostic；
- task-matched；
- selected bank/horizon；
- E1 non-stationary；
- optional E5；
- CPU parallelism。

必须 remote guard。

---

# 25. 阶段 B 图表

必须来自真实数据：

1. control learning curves by environment；
2. final performance matrix；
3. matched prior vs generic；
4. rank vs performance；
5. isotropy vs performance；
6. decodability vs performance；
7. skew/kurtosis vs performance；
8. update stability；
9. eigenvalue spectra；
10. property fulfillment；
11. E1 cue clusters；
12. E2 circular phase；
13. E3 block representation；
14. E4 velocity prediction；
15. remote adaptation template。

使用 matplotlib。

---

# 26. 阶段 B 文档

生成：

```text
docs/CROSS_ENV_EXTENSION_SPEC.md
docs/CROSS_ENV_EXTENSION_RECONCILIATION.md
docs/CROSS_ENV_HYPOTHESES.md
docs/CROSS_ENV_EXPERIMENT_MATRIX.md
docs/CROSS_ENV_IMPLEMENTATION_REPORT.md
docs/CROSS_ENV_RESULTS.md
docs/proposal_cross_environment_en.md
docs/proposal_cross_environment_zh.md
docs/CROSS_ENV_REMOTE_RUN_GUIDE_ZH.md
docs/cross_environment_paper_outline.md
docs/cross_environment_poster_outline.md
```

不要覆盖原 Proposal，除非明确需要。

跨环境 Proposal 主体：

> prescribed representation priors across different latent-state geometries in streaming RL

必须区分：

- hypothesis；
- local pilot；
- remote planned full；
- failed method；
- unverified conclusion。

---

# 27. 远程云服务器交付

准备：

```text
scripts/bootstrap_remote.sh
scripts/run_full_remote.sh
scripts/aggregate_remote.sh
```

要求：

- Linux；
- Python 3.11；
- CPU 多核；
- no GPU assumption；
- `set -euo pipefail`；
- tests；
- smoke；
- remote guard；
- workers 参数；
- unique outputs；
- missing-seed detection；
- nonzero failure codes；
- logs；
- aggregation；
- packaging。

中文教程必须傻瓜式，包括：

1. SSH；
2. clone private repository；
3. checkout exact commit；
4. venv；
5. install；
6. tests；
7. smoke；
8. tmux；
9. stationary full；
10. non-stationary full；
11. logs；
12. process；
13. disk；
14. aggregation；
15. packaging；
16. scp 到 Windows；
17. commit/result path；
18. common errors。

使用占位符，不硬编码凭据。

---

# 28. Git checkpoints

每个验证阶段创建本地 commit。

阶段 A 完成后形成核心 commit。

阶段 B 推荐 commits：

1. specification；
2. common environment API；
3. Ringworld；
4. Two-loop；
5. Hidden velocity；
6. matched priors；
7. experiments；
8. diagnostics；
9. smoke；
10. pilot；
11. docs；
12. remote package。

最终：

- tests；
- smoke；
- pilot；
- diff check；
- secret scan；
- push；
- Draft PR。

不自动 merge。

---

# 29. 睡眠期间自治策略

不得因为可选内容失败而停止整个目标。

失败时：

1. 保存错误；
2. 增加测试；
3. 尝试有限修复；
4. 失败则标记 unavailable；
5. 继续必需任务。

如果 oracle 不能学习某环境：

- 暂停该环境科学比较；
- 修复环境。

如果 observation-only 能解决本应部分可观测任务：

- 检查 leakage；
- 修复任务。

如果 task-matched prior 使用 latent label 训练：

- 判定无效；
- 改为因果实现。

如果 push 失败：

- 保留 commit；
- 记录准确错误；
- 继续本地必需文档；
- 不伪造 push/PR。

---

# 30. 完成标准

阶段 A：

- [ ] 当前工作审计完成；
- [ ] T-maze 正确；
- [ ] no cue leakage；
- [ ] fixed trace；
- [ ] GVF bank；
- [ ] raw；
- [ ] RMS；
- [ ] standardization；
- [ ] decorrelation；
- [ ] whitening；
- [ ] Gaussian diagnostic；
- [ ] controller；
- [ ] tests；
- [ ] smoke；
- [ ] 3-seed pilot；
- [ ] real figures；
- [ ] Proposal；
- [ ] remote package；
- [ ] core commit；
- [ ] push/PR 或准确错误。

阶段 B：

- [ ] E1 verified；
- [ ] E2 implemented/tested；
- [ ] E3 implemented/tested；
- [ ] E4 implemented/tested；
- [ ] E5 implemented or deferred；
- [ ] common API；
- [ ] M1；
- [ ] M2；
- [ ] M3；
- [ ] M4；
- [ ] no latent leakage；
- [ ] diagnostics；
- [ ] cross-env smoke；
- [ ] 3-seed pilot；
- [ ] real figures；
- [ ] revised proposal；
- [ ] remote full configs；
- [ ] remote Chinese guide；
- [ ] tests；
- [ ] commits；
- [ ] push/PR 或准确错误；
- [ ] remote full 明确未运行，除非真实在 remote 执行。

---

# 31. 最终报告格式

最终报告：

1. local repository path；
2. starting branch/commit；
3. core branch/commit；
4. extension branch/commit；
5. origin；
6. push status；
7. PR URL；
8. implemented environments；
9. predictive banks；
10. representation methods；
11. matched priors；
12. tests exact results；
13. smoke commands/results；
14. pilot commands/runtime；
15. result paths；
16. property fulfillment；
17. control findings；
18. negative findings；
19. Gaussian method status；
20. Proposal paths；
21. figure paths；
22. remote scripts；
23. remote guide path；
24. exact remote full command；
25. remote full status；
26. remaining blockers。

不得声称：

- 未执行的 remote full 已完成；
- 未成功 push 已成功；
- 未创建 PR 已创建；
- whitening 等于完整 Gaussian；
- pilot 等于最终结论。

---

# 32. 立即执行方式

1. 读取本文件全文；
2. 将本文件原样复制到仓库：

```text
docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md
```

3. 报告：
   - source path；
   - destination path；
   - line count；
   - byte count；
   - SHA-256；
   - first heading；
   - last heading。
4. 创建 reconciliation；
5. 保留当前有效工作；
6. 先执行阶段 A；
7. 阶段 A commit 后再执行阶段 B；
8. 持续执行直到完成标准或真实外部阻塞。

不要只返回计划。
