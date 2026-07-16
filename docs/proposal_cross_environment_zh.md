# 研究计划：不同潜在状态几何下的流式强化学习表征先验

## 研究动机

预测特征常用秩、各向同性、边际矩或可视化结构来评价，但流式控制器真正需要的是可在线利用的任务信息、尺度与几何。本项目研究：在预测问题和线性控制器保持固定时，显式规定表征性质能否跨越二元、圆周、乘积和连续潜在结构改善控制。

## 方法边界

智能体严格流式、非深度：固定泄漏迹、线性 TD(0) 预测、单一因果或固定语义变换、线性 SARSA(lambda)。每个 transition 只处理一次；没有 replay、minibatch、target network、未来协方差、world model，也不把离线 probe 反馈给智能体。

环境包括 E1 continuing T-maze、E2 Ringworld、E3 two-loop identity × phase、E4 hidden velocity。observation-only 与 oracle 完全隔离。通用性质包括 RMS、边际标准化、decorrelation、作为二阶变换的 whitening、探索性的在线矩塑形、unit sphere、sparse 与 bounded；任务匹配先验包括 simplex、circle、simplex × circle block 和预先规定的 anisotropy。

## 可证伪假设

通用 conditioning 可能跨任务有效，但 task-matched prior 只有在预测 bank 真正恢复对应语义时才可能有益。高秩或低 isotropy error 本身不保证控制改善；unit norm 与 sparsity 可能删除幅值或冗余信息。任何方法若不能在 synthetic/real-stream diagnostics 中满足命名性质，就不得按成功方法解释。

## 当前本地 pilot

已完成 3-seed 本地 pilot，只作为诊断证据。E3 对 standardization、whitening、探索性 moment shaping 和 block prior 出现正向信号；E1 基本为 null；E2 不确定性较大；E4 多数变换不如 raw。圆周 prior 满足单位半径，但没有通过真实在线 phase-order 阈值，因此保留为失败/部分满足的方法，而不是降低阈值。

## Remote planned full

已准备 20-seed E1-E4 主矩阵、有限 compact/mixed 与 matched/short ablation，以及 E1 单次 corridor-length change。full 必须同时提供 `RL_RUN_CONTEXT=remote` 和 `--allow-full-run`。当前没有执行 remote full，因此 adaptation hypothesis 和确认性跨环境结论仍属未验证。

## 预期贡献

目标不是宣称某一种几何普遍最优，而是建立“性质是否真正满足—任务信息是否保留—在线控制是否改善”的可审计关系。负结果、null result 与反例均是正式贡献。

## 生产分析补充

确认性计划新增长尾 seed 分布与预注册 catastrophic failure，明确区分 fixed-alpha、仅由 tuning seeds 选择的 tuned-alpha 和因果 norm-scaled alpha，并加入最小修改的 informative hidden-velocity 环境。T-maze 与 Two-loop 的解释同时报告真实决策时刻、按 group held-out 的 probe。这些分析用于区分 representation 与 optimization 混杂，不允许在看到 evaluation seeds 后调参，也不改变线性严格 streaming 的研究问题。
