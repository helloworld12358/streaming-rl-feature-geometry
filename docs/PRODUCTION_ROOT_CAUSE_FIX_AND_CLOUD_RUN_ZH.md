# Streaming RL 生产根因修复与云端运行手册

## 1. 科研目标与不可变约束

本项目研究在 partial observability、continuing、strictly streaming 的在线强化学习中，固定 GVF 问题所形成的预测表征能否改善线性控制。必须分别报告 prediction accuracy、representation geometry、task information、decision-time information、control performance 和 optimization stability；高 decodability 不是良好控制的充分条件。

实现保持一次处理每个 transition、无 replay、无 minibatch、无 deep network、无 future-data fitting。普通 agent 不得访问 cue、loop identity、hidden velocity 或其他 oracle state；离线 group-held-out probe 只做诊断，不反馈给 agent。负结果、invalid run 和数值失败均保留。

四项扩展保持不变：

1. hidden-velocity 的 position/velocity/action/total cost、RMSE、稳定区域、边界、settling 和 disturbance/recovery 指标；运行时强制 `reward = -total_cost`。
2. fixed、condition-specific lr-tune/lr-eval、norm-scaled 三条独立路径；tuning seeds 与 evaluation seeds 隔离，selected alpha 通过文件哈希和逐 run 匹配检查。
3. informative hidden velocity 仍是 position-only observation；真实 velocity 只用于 oracle、probe 和评估。
4. T-maze junction 与 two-loop action-relevant point 的 decision-time probe 仍为 offline、group-held-out，并报告 sample/group/train/test group 数。

正式规模不变：fixed-full 1000、lr-tune 1500、lr-eval 1000、norm-scaled 700，共 4200 runs。

## 2. 数值根因与修复

旧实现的 `gaussian_moment` 使用随时间变化的矩塑形参数，却用全历史累计统计归一化当前输出；whitening matrix 又在刷新点整块替换。非平稳预测流中，这会产生晚期 leverage spike。固定 SARSA(λ) 步长与累积 eligibility trace 随后把有限但过大的 feature norm 放大为 TD/update/parameter 爆炸。旧 manifest 仍把 `nan_count`、`inf_count` 和 `divergence_flag` 写成 0，因此直到最终聚合才失败。

修复保持 transform 的科研定义：

- whitening 仍是二阶变换；使用因果 covariance shrinkage 与平滑 matrix refresh，降低估计噪声和矩阵突变，不做 feature clipping；
- `gaussian_moment` 仍是无分布保证的探索性在线矩塑形；输出归一化改用与在线矩参数相同时间尺度的因果指数矩统计；
- 未提高 `extreme_finite_limit=1e12`，未 clip reward/feature/gradient/parameter，未替换 condition，未删 seed；
- 每个 transition 真实检查 controller/predictive parameters、TD error、update norm、feature norm、reward/cost、transform/covariance/moment state 和 effective alpha；首次 NaN、Inf 或有限极端值立即 fail closed。

每个 run 的 `runtime_validity.json` 保存 first failure identity、interaction、metric/value/threshold、feature/parameter/update norm、transform state、前序 context、failure classification 和 resume eligibility。invalid run 的 manifest 使用 `exit_status=invalid`，不会再伪装成 completed。

### 本地 before/after 与最终验证证据

基线 commit `bb8cb48f3e8818262d582be3ff6661eda0826026` 上，T-maze `gaussian_moment` seed 13 在原 200,000 interactions、固定 α 下的最大 control update 为 `5.320254234653238e66`，最终 parameter norm 为 `1.3430143965076576e66`；旧系统仍错误写成无 divergence。仅应用上述因果 transform 修复后，同一 case 的对应值为 `0.487714942894621` 与 `3.0610716222506213`。

完整历史失败集合按原正式 horizon 重跑 13 cases：T-maze gaussian seeds 4/8/13/14，Ringworld gaussian 9/10/15，Two-loop gaussian 0/1/2/7/10，hidden-velocity whitened 1。结果为 13/13 passed、0 failed；interactions 范围 200,000–260,000，跨 case 最大 control update `2.076655225938364`、最大最终 parameter norm `7.321885228058665`、最大 feature squared norm `551.999669233319`，NaN/Inf/divergence 总数均为 0。没有删 seed、缩短 horizon、提高 `1e12` 阈值或加入 clipping。

生产根因修复 checkpoint 当时的本地全套为 112 tests passed；2026-07-18 部署刷新加入检查后为 124 tests passed。生产 smoke 覆盖 5 environments × 7 required conditions × 2 seeds，共 70/70 valid；最大 control update `0.652002648331488`、最大最终 parameter norm `14.5456764908534`，NaN/Inf/divergence 均为 0。smoke 生成 70 个 strided/decision partitions 和 29 张图，不生成重复的 `aggregate_steps.csv`、`aggregate_updates.csv` 或 `aggregate_decisions.csv`。这些是实现与数值有效性证据，不是 4200-run 科学结论。

## 3. compact v2 输出结构

正式 extension profiles 使用 `storage_schema=compact_v2`，每个 run 是一个独立 partition：

- `summary.csv/json`：per-run scalar summary；
- `representation_metrics.csv`：二阶几何、秩、相关、矩误差和 transform state；
- `task_information_metrics.csv` 与 `decision_probe_by_position.csv`：全轨迹及 decision-time probe；
- `strided_trace.npz`：typed、compressed、按 `metrics_stride` 保存的控制/预测/代价诊断；
- `decision_event_trace.npz`：只保存具有外部 correctness 标签的稀疏决策事件；
- `disturbance_event_trace.npz`：只保存 disturbance/recovery event；
- `prediction_feature_summary.npz`：每个 GVF 的在线累计 TD²/cumulant summary；
- `model_state.npz`：最终 controller、predictive bank、eligibility 和 transform state；
- `runtime_validity.json`、`manifest.json`、`config.json`：有效性与复现元数据；
- `diagnostic_samples.npz`：仅当 `diagnostic_full_trace_runs` 明确选择该 run 时保存。

hidden-velocity 每步指标仍逐步计算，但 mean/moments、final window、settling、event-conditioned recovery 和 boundary statistics 在线累计；它在每个 transition 都选择动作，但这些无 correctness 标签的高频动作不会绕过 `metrics_stride`，正式 run 不再保存每一步宽表。

聚合只复制 scalar、representation、task/probe summary。strided/event NPZ 按 run partition 读取以作图，不生成 `aggregate_steps.csv`、`aggregate_updates.csv` 或 `aggregate_decisions.csv`。`trace_partition_inventory.json` 记录分区数和字节数。

## 4. resume 与失败保留

resume 只复用同时满足以下条件的 run：

- required compact files 完整；
- manifest `exit_status=ok` 且 `resume_eligible=true`；
- summary `run_status=valid`；
- `runtime_validity.status=valid`，NaN/Inf/divergence 均为 0；
- config hash、Git commit、result schema version、environment/condition/seed/candidate alpha 一致；
- required production metrics finite 且未超过原始 `1e12` 阈值。

普通 `--resume` 跳过严格有效的 run。对 `lr_tune`，身份、config hash、commit 和 schema 均匹配的 invalid candidate 也作为“已完成的无效调优尝试”复用；它没有有效 summary、不能参与 alpha 选择，但不会无限重跑。`--retry-invalid` 才会先把旧 evidence 移入同一 suite 的 `failed_attempts/.../<UTC timestamp>/`，再明确重跑。fixed、lr_eval、norm_scaled 中的 invalid 继续使阶段 fail closed。

## 5. 依赖与 Matplotlib

依赖仍声明 `matplotlib>=3.7,<4`。boxplot 不再使用 Matplotlib 3.9 才加入的 `tick_labels=`；先调用最低版本兼容的 `Axes.boxplot`，再用 `set_xticks(..., labels=...)` 设置标签。测试会阻止重新引入不兼容关键字。

项目没有外部 dataset、checkpoint 或模型权重；实验环境由代码和 seed 生成。无网络计算节点只需提前把 `requirements.txt`、`requirements-dev.txt` 对应 wheels 放进仓库内 `wheelhouse/`。不创建 venv 或 Conda 环境。

## 6. 资源与磁盘预检

正式运行只使用 CPU process parallelism，BLAS 每进程单线程。预检读取 cgroup quota，而不是宿主机显示的 128 CPUs/1.5 TiB；20-CPU quota 的安全上限为 19 workers，推荐 16。请求超过上限会失败，不会自动降级。

预检同时检查：真实 repository root、全部输出/日志/cache/temp containment、Git branch/commit/clean state、Python imports、cgroup CPU、cgroup memory、free bytes、free inodes、pilot-calibrated 4200-run peak projection 和 10 GiB 安全余量。storage report 必须来自同一 `EXPECTED_COMMIT` 的 clean worktree 所生成的 50 个 valid compact-v2 runs，并覆盖当前四个正式配置；dirty worktree、旧 commit、旧 schema 或错误 run 数的报告不能启动正式运行。预计峰值超过 50 GiB 或 `projected + safety margin > free` 时 fail closed，并报告最大文件类别；绝不改写到其他磁盘。

本地真实 storage pilot 覆盖 5 environments × 10 conditions × 1 seed（50/50 valid）。pilot tree 为 5.363 MiB，实际 full package 为 3.908 MiB，analysis-core 为 2.676 MiB。分类外推只让 compact trace 随 interaction 增长，manifest/model state 按 run 增长，aggregate rows 与 figures 按 stage 增长；full package 使用“不假设 gzip 压缩收益的 source + tar header”上界。4200-run 预计 source 11.312 GiB、full package 11.374 GiB、analysis-core 0.056 GiB，三者同时存在的 peak 为 22.742 GiB；最大类别是 compact traces（11.135 GiB）。因此低于 50 GiB 目标，在约 68 GiB free space 上再留 10 GiB 仍通过。云端仍必须在 clean final commit 上重新运行 pilot，不能直接把提交前本地数字当成远端实测。

## 7. 云端命令（分批复制）

以下每批都从固定仓库路径开始。所有命令在前台运行，不使用 nohup、setsid、tmux、screen、scheduler 或后台 `&`。

如果希望把依赖检查、全部 tests、production smoke、storage pilot、preflight、四个正式阶段、聚合和打包合并为一条前台命令，请使用：

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh \
  --allow-full-run \
  --workers 16 \
  --run-name production-<COMMIT_SHA>-<UTC_TIMESTAMP> \
  --expected-branch <BRANCH> \
  --expected-commit <COMMIT_SHA>
```

先增加 `--dry-run` 检查完整命令展开；dry-run 不安装依赖、不创建正式结果、不启动 remote full。分批诊断或手工恢复时再使用下面 A–H。

### A. 拉取修复分支并核对 commit

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
git fetch origin
git switch codex/fix-streaming-rl-production-root-causes
git pull --ff-only origin codex/fix-streaming-rl-production-root-causes
git branch --show-current
git rev-parse HEAD
git status --short --branch
```

把最终交付回答中的 commit SHA 记为 `EXPECTED_COMMIT`，并确认 `git rev-parse HEAD` 完全相同。

### B. 配置全部仓库内运行目录

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
export REPO_ROOT="$PWD"
export TMPDIR="$REPO_ROOT/.runtime/tmp"
export TMP="$REPO_ROOT/.runtime/tmp"
export TEMP="$REPO_ROOT/.runtime/tmp"
export MPLCONFIGDIR="$REPO_ROOT/.runtime/matplotlib"
export XDG_CACHE_HOME="$REPO_ROOT/.runtime/cache"
export PIP_CACHE_DIR="$REPO_ROOT/.runtime/pip-cache"
export RESULTS_ROOT="$REPO_ROOT/results"
export LOGS_ROOT="$REPO_ROOT/logs"
export ARTIFACTS_ROOT="$REPO_ROOT/artifacts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$RESULTS_ROOT" "$LOGS_ROOT" "$ARTIFACTS_ROOT" "$REPO_ROOT/.local_diagnostics"
```

### C. 使用现有 `/usr/bin/python3` 检查和安装依赖

联网安装：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
/usr/bin/python3 --version
/usr/bin/python3 -c 'import sys; print(sys.executable)'
/usr/bin/python3 -m pip install -r requirements.txt -r requirements-dev.txt
/usr/bin/python3 -m pip install --no-build-isolation --no-deps -e .
/usr/bin/python3 -m pip check
/usr/bin/python3 -c 'import numpy,pandas,matplotlib,pytest,streaming_rl_feature_geometry; print(numpy.__version__, pandas.__version__, matplotlib.__version__)'
```

无网络安装（wheelhouse 必须已经位于仓库内）：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
WHEELHOUSE="$PWD/wheelhouse/py310-linux-x86_64"
/usr/bin/python3 -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.txt -r requirements-dev.txt
/usr/bin/python3 -m pip install --no-index --find-links "$WHEELHOUSE" --no-build-isolation --no-deps -e .
/usr/bin/python3 -m pip check
/usr/bin/python3 -c 'import numpy,pandas,matplotlib,pytest,streaming_rl_feature_geometry'
```

### D. 真实测试、storage pilot 与 production preflight

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
/usr/bin/python3 -m pytest -q
set +e
set -o pipefail
PILOT_NAME="storage-pilot-$(git rev-parse --short=12 HEAD)"
PILOT_LOG="$PWD/logs/${PILOT_NAME}.log"
bash scripts/run_storage_pilot.sh --workers 16 --run-name "$PILOT_NAME" 2>&1 | tee -a "$PILOT_LOG"
PILOT_STATUS=${PIPESTATUS[0]}
echo "PILOT_STATUS=$PILOT_STATUS"
printf '%s\n' "$PILOT_STATUS" > "$PWD/logs/${PILOT_NAME}.status"
```

storage report 位于：

`artifacts/storage_pilot/$PILOT_NAME/storage_projection.json`

查看关键预算：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
/usr/bin/python3 - "$PWD/artifacts/storage_pilot/$PILOT_NAME/storage_projection.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("formal_runs", "projected_result_bytes", "projected_full_package_bytes", "projected_analysis_core_bytes", "projected_peak_bytes", "target_peak_bytes", "required_safe_margin_bytes", "within_50_gib_target", "largest_file_categories"):
    print(f"{key}={r[key]}")
PY
```

### E. fresh 4200-run 正式运行

仅当测试和 storage pilot 状态均为 0、projection 通过时运行。不要复用旧 `RUN_NAME`。

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
set +e
set -o pipefail
EXPECTED_COMMIT="$(git rev-parse HEAD)"
RUN_NAME="production-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
STORAGE_REPORT="$PWD/artifacts/storage_pilot/$PILOT_NAME/storage_projection.json"
LOG_FILE="$PWD/logs/${RUN_NAME}.log"
STATUS_FILE="$PWD/logs/${RUN_NAME}.status"
RL_RUN_CONTEXT=remote bash scripts/run_cross_extension_remote.sh \
  --stage all \
  --allow-full-run \
  --workers 16 \
  --run-name "$RUN_NAME" \
  --storage-report "$STORAGE_REPORT" \
  --expected-branch codex/fix-streaming-rl-production-root-causes \
  --expected-commit "$EXPECTED_COMMIT" 2>&1 | tee -a "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
echo "RUN_STATUS=$RUN_STATUS"
printf '%s\n' "$RUN_STATUS" > "$STATUS_FILE"
```

命令末尾故意没有 `exit "$RUN_STATUS"`，不会主动关闭当前交互 shell。

### F. resume 与 retry invalid/interrupted

严格 resume：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
set +e
set -o pipefail
EXPECTED_COMMIT="$(git rev-parse HEAD)"
STORAGE_REPORT="$PWD/artifacts/storage_pilot/$PILOT_NAME/storage_projection.json"
RESUME_LOG="$PWD/logs/${RUN_NAME}.resume.log"
RL_RUN_CONTEXT=remote bash scripts/run_cross_extension_remote.sh --stage all --allow-full-run --workers 16 --run-name "$RUN_NAME" --storage-report "$STORAGE_REPORT" --expected-branch codex/fix-streaming-rl-production-root-causes --expected-commit "$EXPECTED_COMMIT" --resume 2>&1 | tee -a "$RESUME_LOG"
RESUME_STATUS=${PIPESTATUS[0]}
echo "RESUME_STATUS=$RESUME_STATUS"
printf '%s\n' "$RESUME_STATUS" > "$PWD/logs/${RUN_NAME}.resume.status"
```

保留并重跑 invalid/failed/interrupted：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
set +e
set -o pipefail
EXPECTED_COMMIT="$(git rev-parse HEAD)"
STORAGE_REPORT="$PWD/artifacts/storage_pilot/$PILOT_NAME/storage_projection.json"
RETRY_LOG="$PWD/logs/${RUN_NAME}.retry-invalid.log"
RL_RUN_CONTEXT=remote bash scripts/run_cross_extension_remote.sh --stage all --allow-full-run --workers 16 --run-name "$RUN_NAME" --storage-report "$STORAGE_REPORT" --expected-branch codex/fix-streaming-rl-production-root-causes --expected-commit "$EXPECTED_COMMIT" --resume --retry-invalid 2>&1 | tee -a "$RETRY_LOG"
RETRY_STATUS=${PIPESTATUS[0]}
echo "RETRY_STATUS=$RETRY_STATUS"
printf '%s\n' "$RETRY_STATUS" > "$PWD/logs/${RUN_NAME}.retry-invalid.status"
```

### G. 日志、阶段状态和异常查看

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
tail -n 200 "$PWD/logs/${RUN_NAME}.log"
find "$PWD/logs/cross_extension/$RUN_NAME" -maxdepth 1 -name '*.status.json' -print -exec /usr/bin/python3 -m json.tool {} \;
for stage in fixed-full lr-tune lr-eval norm-scaled; do
  /usr/bin/python3 -m streaming_rl_feature_geometry.remote_deployment inspect-run --run-dir "$PWD/results/cross_extension/$RUN_NAME/$stage" --config "$PWD/results/cross_extension/$RUN_NAME/$stage/config.json"
done
```

只查看具体异常记录，不扫描 trace：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
find "$PWD/results/cross_extension/$RUN_NAME" -path '*/runs/*/runtime_validity.json' -type f -exec /usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); f=r.get("first_failure"); f and print(sys.argv[1], json.dumps(f, ensure_ascii=False))' {} \;
find "$PWD/results/cross_extension/$RUN_NAME" -path '*/failed_attempts/*/manifest.json' -type f -print
```

### H. 最终验证、打包、SHA-256 与磁盘清单

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
for stage in fixed-full lr-tune lr-eval norm-scaled; do
  EXTRA=()
  if [[ "$stage" == "lr-eval" ]]; then EXTRA=(--selected-learning-rates "$PWD/results/cross_extension/$RUN_NAME/lr-tune/selected_learning_rates.csv"); fi
  /usr/bin/python3 -m streaming_rl_feature_geometry.cross_production aggregate --config "$PWD/results/cross_extension/$RUN_NAME/$stage/config.json" --run-dir "$PWD/results/cross_extension/$RUN_NAME/$stage" "${EXTRA[@]}"
done
/usr/bin/python3 -m streaming_rl_feature_geometry.cross_production package --suite-dir "$PWD/results/cross_extension/$RUN_NAME" --run-name "$RUN_NAME" --output-dir "$PWD/artifacts/cross_extension/$RUN_NAME"
find "$PWD/artifacts/cross_extension/$RUN_NAME" -maxdepth 1 -type f -printf '%s %p\n' | sort -n
sha256sum -c "$PWD/artifacts/cross_extension/$RUN_NAME"/*.sha256
du -sh "$PWD/results/cross_extension/$RUN_NAME" "$PWD/artifacts/cross_extension/$RUN_NAME"
df -h "$PWD"
df -Pi "$PWD"
```

analysis-core 包含 scalar/probe summaries、figures、validation reports、manifests、日志摘要和 SHA-256；optional full 包包含 compact per-run partitions 与失败证据。两类包都位于仓库内 `artifacts/`。

## 8. 科学解释限制

数值稳定只证明实现不再因变换统计时间尺度不一致而爆炸，不证明某种 representation 一定改善控制。fixed-alpha 路径仍保留 feature scale/geometry 与统一 α 的真实交互；lr-tune/lr-eval 与 norm-scaled 路径用于区分该 confound。`gaussian_moment` 仍不提供 Gaussian distribution 保证。只有云端 4200-run 全部有效、聚合和 production validation 均通过后，才可给出正式跨环境结论。
