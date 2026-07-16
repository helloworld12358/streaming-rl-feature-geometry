# Hidden velocity informative：设计与本地 pilot 记录

## 原环境为什么容易接近

原 `hidden_velocity` 的位置成本权重为 1、速度成本权重仅为 0.25，惯性 `rho=0.9`，动作加速度较强。位置符号已经给 memoryless controller 一个有效的反应式信号；在有限训练预算中，oracle 的速度信息因此不总能形成稳定收益差距。

## 最小机制修改

没有叠加目标位置切换或巨大噪声。唯一候选方案保留同一任务结构，只做三项可解释调整：

- `rho: 0.90 -> 0.97`，延长速度影响；
- `velocity_cost_weight: 0.25 -> 1.0`，使速度估计与控制目标直接相关；
- 每 96–144 步施加一次 seed 控制、幅度 0.24–0.38 的有界速度 impulse，并定义 32 步 post-disturbance 窗口。

动作成本从 0.02 调整为 0.015，加速度从 0.12 调整为 0.09，避免控制因过高动作代价或过强单步反向动作而退化。观测仍只有 bias、归一化 position 与 position sign；velocity 只存在于 latent/oracle diagnostics，绝不进入普通 representation 的训练。

候选配置和最终采用配置均为 `configs/hidden_velocity_informative_pilot.json` 与四个正式 extension profile 中的 `hidden_velocity_informative.kwargs`。本轮只评估一个基于机制的候选，没有无限搜索。

## Pilot 协议

- conditions：`observation_only`、`oracle`、`raw`、`whitened`
- design seeds：200–204
- 每个 run：5000 interactions
- final window：500
- evaluation seeds 0–19 完全未使用
- 本地没有运行任何 formal full profile

完整的小型 seed 汇总保存在 `docs/pilot_results/hidden_velocity_informative_pilot_summary.csv`。

## Pilot 结果

| condition | mean final reward | mean final stabilization | mean position RMSE | mean velocity decoding R2 | mean control update norm |
|---|---:|---:|---:|---:|---:|
| observation_only | -0.9340 | 0.1188 | 0.8566 | -0.0023 | 0.0179 |
| oracle | -0.5396 | 0.1628 | 0.6469 | 1.0000 | 0.0114 |
| raw | -1.0167 | 0.2760 | 0.8475 | 0.1876 | 0.0977 |
| whitened | -1.5561 | 0.0800 | 1.1232 | 0.0733 | 0.1091 |

oracle 相对 observation-only 的 final reward gap 在 seeds 200–204 分别为 `+0.0820, -0.8288, +0.1227, +2.2248, +0.3711`，即 5 个 seed 中 4 个为正，均值约 `+0.3943`。所有 reward 与 update norm 有限，没有数值爆炸。predictive 条件并未被调到必然优于 baseline：这项负面/混合证据被保留，后续由独立 tuning seeds 的 alpha pipeline 检查优化混杂，而不是用 evaluation seeds 修改环境。

## 选择结论

该候选满足本地 pilot 的预设用途：环境可学习、oracle 在多数独立 design seeds 上优于 observation-only、真实 velocity 可由 oracle 完美取得、普通条件仍无 latent label、扰动和恢复字段可复现且数值有界。因此采用该配置进入正式 profile。此结论不宣称 predictive feature 已取得正结果；正式结论必须等待远端 evaluation seeds 0–19。
