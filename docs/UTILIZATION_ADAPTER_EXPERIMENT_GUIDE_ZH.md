# 固定非线性 utilization adapter 实验指南

## 修改前审计

本轮从 `codex/fix-streaming-rl-production-root-causes` 的精确提交
`a638ac62a6a9b1727f0dbe60bb1fdb9e0478da86` 创建本地分支
`codex/add-utilization-adapters`。切分支前 tracked 与 index 均为空；唯一未跟踪目录
`paper_from_existing_results/` 保持不变。修改前完整测试为 `122 passed`。

1. 跨环境 controller input 原为 `[o_t, s_t, 1]`；core T-maze 的 oracle 特例也等价于该定义。
2. 拼接顺序固定为 observation、controller state、bias。
3. `observation_only` 的 state 是空向量；oracle state 只在 oracle condition 读取；其他 condition 只使用在线预测/变换 state。core 的 `trace_only` 直接使用固定 trace state。
4. 每个 transition 的因果顺序是：用当前 state 选动作，environment step，predictor update，构造下一预测与 transform，构造下一 controller state，选下一动作，执行 accumulating-trace SARSA(λ) update。probe 不反馈训练。
5. environment 使用 run seed，predictor 使用 `seed + 17`，controller 使用 `seed + 31`，diagnostic reservoir 使用独立 seed。
6. 现有 `norm_scaled` SARSA(λ) 使用仅含当前/过去 feature squared norm 的 EMA，并在已登记上下界内得到 effective alpha。
7. core/cross 配置加载器均 fail closed；新 adapter 配置也拒绝未知名称、scope、字段或非固定超参数。
8. 旧 `compact_v2` 继续有效且默认行为不变；新实验使用独立 `adapter_summary_v1`。
9. 现有 runner 使用 process-level multiprocessing，未增加第二套 scheduler。
10. cross runner 的 resume/retry-invalid 会核对 config、commit、schema、identity、required files 与 runtime validity；失败证据先归档再重跑。
11. core T-maze v3 使用 continuing T-maze、固定 `trace_only`、20,000 interactions、decision-time junction trial，以及末 200 trials accuracy；本轮 formal adapter controller 改为统一 norm-scaled，旧 fixed-alpha 结果不作为 identity baseline。
12. 仓库没有可直接复用的新 commit norm-scaled formal result root。Stage A 必须由用户通过 `--baseline-root` 提供；缺失、重复或任一逐字段不兼容都会在新 RFF run 启动前停止。

## Controller input 与 adapter

修改后的唯一控制器输入定义为：

```text
[observation, adapter(controller_state), bias]
```

adapter 只接收当前 `controller_state`，不接收 observation、action、reward、future sample、latent label 或 probe output。三个固定选项为：

- `identity`：严格返回原 state，不复制、不 cast、不使用 RNG、不更新状态。
- `residual_rff`：`[s, sqrt(2/64) cos(Ws+b)]`，width 64，frequency scale 1.0；`W` 与 `b` 在 run 内只读。
- `tile_coding`：`[s, tile(s)]`，8 tilings、512 table entries、4 tiles per unit，每个 active increment 为 `1/sqrt(8)`，collision 累加。

空 state 对三个 adapter 都返回空向量，因此 observation-only 没有 RFF 或 tile 常量块。RFF 与 tile seed 使用
SHA-256 从 adapter version、environment、run seed 派生，不包含 condition，也不改变 environment、predictor 或 controller RNG。

## 正式矩阵与复用

| Stage | Logical cells | Baseline identity | Structural no-op | Stage A reuse | New runs |
|---|---:|---:|---:|---:|---:|
| A | 1120 | 560 | 80 | 0 | 480 |
| B1 | 120 | 0 | 20 | 0 | 100 |
| B2 | 150 | 0 | 20 | 90 | 40 |
| Total | 1390 | 560 | 120 | 90 | 620 |

Stage A 只包含 `tmaze`、`ringworld`、`two_loop`、`hidden_velocity`；明确不包含
`hidden_velocity_informative`。其科学字段逐项取自
`configs/cross_extension_norm_scaled_full.json`。Stage B1 是 core T-maze 的
`observation_only/raw/trace_only/oracle`。Stage B2 是 hidden velocity 的
`observation_only/raw/whitened/matched/oracle`。

baseline validator 核对 environment、condition、seed、interactions、final window、environment kwargs、horizons、GVF bank、predictive alpha、gamma、lambda、epsilon、base control alpha、norm scaling、transform 参数、controller input identity、config hash、source commit、runtime validity 与 run status。它不会硬编码历史 run name，也不会把旧 `cross-full-2940182f41c9` 当兼容 baseline。

## 本地 smoke、存储 pilot 与 dry-run

本地短 smoke 覆盖四个环境、三个 adapter 与
`observation_only/raw/matched/oracle`，每个 run 600 interactions，不生成正式图：

```bash
python scripts/run_utilization_experiment.py \
  --stage smoke --run-name utilization-smoke --workers 4
```

用真实 smoke run 生成轻量存储报告：

```bash
python scripts/report_utilization_storage.py \
  --run-root results/utilization_adapters/utilization-smoke \
  --output artifacts/utilization-smoke/storage_report.json \
  --inventory-csv artifacts/utilization-smoke/storage_inventory.csv
```

正式 dry-run 允许在本地运行，但 baseline 缺失时预期返回非零并只报告缺口，不启动实验：

```bash
python scripts/run_utilization_experiment.py \
  --stage all --run-name utilization-formal --dry-run \
  --baseline-root /path/to/completed-new-commit-norm-scaled-suite \
  --storage-report artifacts/utilization-smoke/storage_report.json
```

## 远端正式运行

formal 只能在 CPU 环境执行，并且必须同时提供双 gate、真实 storage report 与 baseline root：

```bash
RL_RUN_CONTEXT=remote bash scripts/run_utilization_remote.sh \
  --stage all \
  --allow-full-run \
  --baseline-root /path/to/completed-new-commit-norm-scaled-suite \
  --storage-report artifacts/utilization-smoke/storage_report.json \
  --run-name utilization-formal-<COMMIT_SHA> \
  --workers 16
```

`all` 严格按 Stage A、B1、B2 顺序执行；只在单一 stage 内并行。支持重复
`--stage`、`--resume` 和 `--retry-invalid`。不得使用 GPU，不得在本地伪装 remote gate，
不得在 baseline 缺失时自行补跑 identity full。

每个新 run 只保存 `config.json`、`manifest.json`、`runtime_validity.json`、
`summary.json`、`summary.csv` 和 `stdout.log`。阶段聚合只保存登记的 CSV/JSON/Markdown；
完整 formal 聚合只生成四张登记图。
