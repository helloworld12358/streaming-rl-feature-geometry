# 固定非线性 Utilization Adapter 实验指南

## 科学边界与接入点

本扩展不改变环境、GVF questions/horizons、predictor、representation transform、continuing on-policy SARSA(λ) 或原在线更新顺序。每个 transition 只处理一次；没有 replay、minibatch、future fitting、神经网络、MLP 或 GPU 依赖。Oracle state 只允许进入 oracle condition，latent/probe 只用于诊断。

旧 controller 的完整输入先按原顺序构造：

```text
x_t = [observation_t, condition-specific state_t, bias=1]
h_t = adapter(x_t)
action/update = linear SARSA(lambda)(h_t)
```

adapter 只能读取当前 `x_t`。它不读取 reward、future observation/transition、latent label 或 probe 输出。bias 位于完整输入最后一维，并同时保留在 residual 原始分支中。

三个固定 adapter：

- `identity`: `h_t = x_t`，返回同一 float64 ndarray，不创建 RNG。
- `residual_rff`: `h_t = [x_t, sqrt(2/64) cos(Wx_t+b)]`；`W ~ N(0,1)`、`b ~ U(0,2π)`，width=64、scale=1、residual=true。
- `tile_coding`: `h_t = [x_t, Tile(x_t)]`；8 tilings、512-entry stable-hash table、4 tiles/unit，每个 tiling 的增量为 `1/sqrt(8)`，collision 累加。

RFF/Tile seed 只由 environment 与 run seed 的稳定 SHA-256 派生，不含 condition；它使用独立 `numpy.random.Generator`，不消费 environment、predictor、transform 或 controller 的随机流。adapter 参数只读且没有 update。

## 参数来源

- Stage A：从 `configs/cross_full.json` 的 fixed-alpha 正式源复制四环境的 interactions、final window、seeds、controller/predictor、GVF bank/horizons、transform 与 environment kwargs，只选择预注册的七个 conditions。
- Stage B1：从 core T-maze v3/pilot 源复制 20,000 interactions、末 200 trials accuracy、fixed alpha、trace-only 定义与 predictor/controller 参数；固定使用正式 seed 列表前 10 个。
- Stage B2：从 Stage A 的 hidden-velocity fixed-alpha 源复制参数，使用与 B1 相同的 10 seeds。

不得按 adapter 调 alpha，不得 clipping、自动降参、fallback identity 或根据 smoke 结果改正式参数。

## 正式矩阵

| Stage | 逻辑矩阵 | Logical cells | 无 baseline 时新 runs |
|---|---:|---:|---:|
| A | 4 env × 7 condition × 2 adapter × 20 seed | 1120 | 1120 |
| B1 | 4 condition × 3 adapter × 10 seed | 120 | 120 |
| B2 | 5 condition × 3 adapter × 10 seed | 150 | 50（在同一计划中严格复用 A 的 100 个 identity/RFF cell） |
| Total |  | 1390 | 1290 |

若 Stage A 的 560 个历史 identity cell 全部兼容，则总新 runs 为 730：B1 的 40 个 identity、Stage A/B1 的 600 个 RFF、B1/B2 的 90 个 Tile。任何复用都由逐 cell 签名验证决定，不能硬编码认定。

## Baseline 与跨阶段复用

`scripts/validate_utilization_baseline.py` 和 launcher 的 `--baseline-root` 对每个候选 identity cell 检查 environment、condition、seed、interactions、final window、gamma/lambda/epsilon、control/predictive alpha、alpha mode、GVF bank/horizons、transform、完整 controller input、environment kwargs、evaluation、identity semantics、schema、config hash、source commit、manifest/runtime validity 和 run status。

候选 `cross-full-2940182f41c9` 不会被自动接受。缺失、重复或不兼容时记录具体字段和两侧值，将 identity 加入 pending；旧结果不会被修改。B2 通过 `--stage-a-root` 对 Stage A identity/RFF 做同样的规范化逐字段验证，只有完全一致才复用。

## 配置和工具

- Stage 0：`configs/utilization_adapter_smoke.json`
- Storage pilot：`configs/utilization_storage_pilot.json`
- Stage A：`configs/utilization_stage_a_full.json`
- Stage B1：`configs/utilization_stage_b1_full.json`
- Stage B2：`configs/utilization_stage_b2_full.json`
- Launcher：`scripts/run_utilization_experiment.py`
- Baseline validator：`scripts/validate_utilization_baseline.py`
- Storage report：`scripts/report_utilization_storage.py`
- Integrity audit：`scripts/audit_utilization_results.py`
- Aggregation：`scripts/aggregate_utilization_results.py`
- Plotting：`scripts/plot_utilization_results.py`

正式 run 使用 `adapter_summary_v1`，每个 run 仅保存 `config.json`、`manifest.json`、`runtime_validity.json`、`summary.json`、`summary.csv` 和 `stdout.log`。默认不保存逐步 observations、predictive vectors、weights trajectory、NPZ 或 per-seed figures。

## 本地验证

```powershell
python -m pytest -q tests/test_utilization_adapters.py tests/test_utilization_experiment.py
python -m pytest -q
python scripts/run_utilization_experiment.py --stage smoke --workers 4 --run-name local-utilization-smoke --output-root results/utilization_adapters --dry-run
python scripts/run_utilization_experiment.py --stage stage-a --run-name dry-stage-a --dry-run
python scripts/run_utilization_experiment.py --stage stage-b1 --run-name dry-stage-b1 --dry-run
python scripts/run_utilization_experiment.py --stage stage-b2 --run-name dry-stage-b2 --dry-run
python scripts/run_utilization_experiment.py --stage all --run-name dry-all --dry-run
```

本地只允许运行 tests、smoke、storage pilot 和 dry-run。Stage A/B1/B2 正式运行必须同时具有 `RL_RUN_CONTEXT=remote` 和 `--allow-full-run`，且必须传入真实 `--storage-report`。baseline 缺失不会绕过 full-run guard，也不会在本地自动启动正式 identity。

## 审计、聚合与四图

每个 stage 完成后 launcher 生成 `run_manifest.json`、`run_summaries.csv`、`aggregate_performance.csv`、`paired_adapter_differences.csv`、`runtime_summary.csv`、`reuse_validation.csv`、`failed_runs.csv` 和 `experiment_summary.md`。

全局聚合前必须运行 `scripts/audit_utilization_results.py`；它检查 expected/completed/valid/invalid/failed/incomplete、duplicate/extra、错误 axes/seeds、commit/config 混合、runtime validity、paired identity coverage 与 NaN/Inf/divergence。审计失败时禁止聚合。

正式只生成四张 PNG：

1. 四环境 Stage A identity/RFF mean ± SEM；
2. 各 environment × condition 的 paired RFF−identity，正值代表 RFF 更好；
3. Core T-maze 的 trace identity/RFF/Tile 与 oracle identity，指标为末 200 trials accuracy；
4. Hidden velocity raw/whitened/matched 的 identity/RFF/Tile，指标为 final-window reward。

不生成 per-seed 图，不跨环境平均原始 performance，不使用统计显著性语言。负面、无改善、invalid 和 failed 证据必须保留。
