# 英文 Proposal 中文逐段解释

## 1. 标题与核心问题

项目不是比较“哪一个 GVF 问题最好”，而是在同一套预测问题、同一控制器和同一环境下，只改变预测特征被赋予的统计或几何性质。核心问题是：在严格流式、部分可观测的强化学习里，规定特征尺度、相关性、协方差形状或范数，是否真的能让在线控制更好？

## 2. 为什么属于 Core RL

智能体必须在连续交互中行动、获得延迟奖励、在线学习预测并在线改进策略。每个 transition 只使用一次；没有 replay、minibatch、离线拟合、未来统计、深度网络或分析反馈。因此表示不是事后监督学习对象，而是 RL agent state 的一部分，最终由回报与决策质量检验。

## 3. 环境逻辑

T-maze 每轮开头短暂给出左/右 cue，随后只有一次 delayed echo。经过五步走廊后到达 junction，此时两类 trial 的观测完全相同，正确动作却取决于已经消失的 cue。outcome 之后直接进入下一 trial，构成 continuing task。oracle 直接看到隐藏 cue，只用于证明任务可学；其他方法绝不能获得该信息。

## 4. 预测特征逻辑

固定 12 维 trace 保存因果历史，十个线性 GVF 预测观测/echo、junction、正 outcome 和负 outcome，并使用两个固定 horizon。这样可以得到含冗余、相关、不同尺度和不同时间范围的预测向量，同时避免把研究变成调 GVF question 的项目。

## 5. 各方法的科学含义

- raw：保持 GVF 原始输出，是对照起点。
- RMS raw：只匹配整体 RMS，测试“纯尺度”影响。
- standardized：逐维在线零均值、单位方差。
- decorrelated：降低非对角相关，但保留边际尺度，从而区别于 whitening。
- whitened：在线将协方差推向单位阵，只声称二阶 isotropy，不声称高斯。
- Gaussian-inspired：白化后再用低参数、单调、有界的在线 moment shaping 尝试降低 skew/kurtosis 误差；这是高风险探索性 extension。
- unit sphere：固定样本范数，可能同时丢掉事件接近程度或置信度的幅值信息。

## 6. 假设和反例意识

部分假设认为尺度控制、去相关和白化会让线性 TD/SARSA 的更新更稳定；另一部分假设专门寻找反例：几何更漂亮不代表 cue 更容易被策略利用，尤其二值/双峰 latent structure 可能与 Gaussian-like prior 不匹配。项目预先允许负结果，pilot 后不为追求正结论改超参数。

## 7. 指标逻辑

主指标是末窗 junction accuracy、累计 reward、到达阈值时间和更新稳定性。表示指标只解释机制，包括相关性、eigenvalues、effective rank、isotropy、skew/kurtosis 和 transform drift。held-out cue probe 按 trial 分组划分训练/测试，只用于诊断，不能回馈智能体。

## 8. Pilot 的准确含义

正式本地 pilot 是 seeds 0、1、2，每条件 20,000 interactions，共 30 个独立 run；路径为 `results/pilot/preregistered-pilot-20260713-v3`。manifest 记录干净的发布前 SHA `e5f31cc`；因 GitHub 邮箱隐私保护仅重写提交元数据后，对应的相同代码 tree 为 `7687451`。oracle 达到 `0.985 ± 0.006`，证明任务和控制器上界有效。其余条件末窗均约为 0.52–0.57，误差范围重叠，没有证据支持任何预测变换优于 observation-only。

最重要的局部观察是：raw 的 isotropy error 为 `6.01 ± 0.05`，whitened 降到 `2.98 ± 0.07`；effective rank 从 `2.59 ± 0.02` 升到 `5.88 ± 0.08`，但控制并未清晰改善。这不是“白化失败”，而是“完成二阶性质不等于完成 RL 目标”的反例候选。与此同时，多数预测表示在 junction 的 held-out cue decoding 高于 0.5，说明表示包含可线性访问的信息，但 SARSA 在此短预算内没有稳定利用它。

## 9. 目前不能下的结论

三 seed pilot 不能证明 whitening 有害或无效，不能证明 raw 永远不优于观测基线，也不能把 cue probe 当成因果中介，更不能说 Gaussian-inspired 已实现真实高斯分布。pilot 只用于发现错误、估计方差和展示可能的几何—控制脱钩。最终统计判断要等待远程 20-seed full。

## 10. Full plan、算力与失败兜底

stationary full 为 20 seeds、200,000 interactions；non-stationary full 为 300,000 interactions，并在中点将 corridor 由 5 改为 9。均只在 CPU 远程服务器运行，必须同时满足环境变量和显式命令开关。若 Gaussian-inspired 不稳定，将保留失败测试和日志，并继续完成 raw、RMS、standardized、decorrelated、whitened；负结果不会删除。

## 11. 预期贡献

贡献不依赖出现“最好方法”。项目提供可复现的严格流式框架，并回答一个更基础的问题：task-agnostic 的统计几何先验在什么情况下与隐藏状态结构相容，在什么情况下只改善通用数字指标、却没有改善在线强化学习。
