# Cross-extension 云端正式运行指南

## 最简单运行步骤

本指南假设云平台已有仓库、Python 环境、`git`、`tmux`，且上一轮实验曾成功运行。先确认云端仓库没有未提交修改；不要覆盖或清理未知文件。

```bash
cd <已有仓库路径>
git status --short
git fetch origin
git switch codex/hidden-velocity-metrics-lr-production
git pull --ff-only origin codex/hidden-velocity-metrics-lr-production
git rev-parse HEAD
```

`git rev-parse HEAD` 必须与 Codex 最终交付的 commit SHA 完全一致。若 `git status --short` 有输出，先保存并确认这些修改的来源；不要运行 `git reset --hard` 或 `git clean -fd`。

进入 `tmux`，一条主命令完成四阶段、聚合、绘图、报告、双包和 SHA-256：

```bash
tmux new -s cross-extension
export RL_RUN_CONTEXT=remote
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage all \
  --run-name cross-extension-$(git rev-parse --short HEAD) \
  --workers 16 \
  --resume
```

这套实现只使用 CPU 上的 NumPy/Pandas/Matplotlib 线性算法，不使用 GPU，不下载外部数据集，正式运行时不需要联网。对已知云平台的 20 CPU cgroup 配额，推荐 `--workers 16`；无需申请 H200 或 4090。若换到不同资源，请以脚本打印的 safe maximum 为准。

## 正式阶段和 run 数

| stage | config | seeds | 精确 run 数 |
|---|---|---:|---:|
| `fixed-full` | `configs/cross_extension_fixed_full.json` | 0–19 | 1000 |
| `lr-tune` | `configs/cross_extension_lr_tune_full.json` | 100–104 | 1500 |
| `lr-eval` | `configs/cross_extension_lr_eval_full.json` | 0–19 | 1000 |
| `norm-scaled` | `configs/cross_extension_norm_scaled_full.json` | 0–19 | 700 |

`all` 固定按上表顺序运行。`lr-tune` 生成 `selected_learning_rates.csv` 后，`lr-eval` 只读取该文件，不重新选择 alpha。正式 full profile 同时需要 `RL_RUN_CONTEXT=remote` 与内部传入的 `--allow-full-run`；缺少任一条件都会在创建正式结果前退出。

## tmux 操作

创建并进入：

```bash
tmux new -s cross-extension
```

暂离但保持任务运行：按 `Ctrl-b`，松开后按 `d`。

恢复：

```bash
tmux attach -t cross-extension
```

列出会话：

```bash
tmux ls
```

## 日志、进度和失败数

设定与主命令一致的名字：

```bash
RUN_NAME=cross-extension-$(git rev-parse --short HEAD)
SUITE=results/cross_extension/$RUN_NAME
tail -f "$SUITE/logs/fixed-full.log"
```

查看四阶段日志：

```bash
ls -lh "$SUITE/logs"
tail -n 100 "$SUITE/logs/lr-tune.log"
tail -n 100 "$SUITE/logs/lr-eval.log"
tail -n 100 "$SUITE/logs/norm-scaled.log"
```

统计成功、失败和全部 manifest：

```bash
find "$SUITE" -path '*/runs/*/manifest.json' -type f | wc -l
grep -rl '"exit_status": "ok"' "$SUITE"/*/runs --include manifest.json | wc -l
grep -rl '"exit_status": "failed"' "$SUITE"/*/runs --include manifest.json | wc -l
```

每个阶段结束后，launcher 还会打印 `expected_runs`、`completed_runs`、`pending_runs` 和 `existing_incomplete_or_failed_runs`。

## 中断续跑和失败重试

中断后重复完全相同的主命令即可；成功且完整的 seed 会自动跳过，不完整目录会先保存在 `failed_attempts/` 再安全重跑：

```bash
export RL_RUN_CONTEXT=remote
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage all --run-name "$RUN_NAME" --workers 16 --resume
```

只重试明确失败或不完整的项目：

```bash
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage all --run-name "$RUN_NAME" --workers 16 --retry-failed
```

失败 run 的原目录不会静默删除，而会移入对应 stage 的 `failed_attempts/`。日志和失败 manifest 保留用于审计。

## 分阶段和过滤运行

只跑 fixed baseline：

```bash
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage fixed-full --run-name "$RUN_NAME" --workers 16 --resume
```

只跑原 Hidden velocity 或 informative variant：

```bash
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage fixed-full --run-name "$RUN_NAME-hv" --workers 16 --resume \
  --environment hidden_velocity

bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage fixed-full --run-name "$RUN_NAME-hvi" --workers 16 --resume \
  --environment hidden_velocity_informative
```

只跑某个 condition 可追加 `--condition whitened`；环境和 condition 参数均可重复。

先 tuning，再 tuned evaluation：

```bash
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage lr-tune --run-name "$RUN_NAME" --workers 16 --resume

bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage lr-eval --run-name "$RUN_NAME" --workers 16 --resume \
  --selected-learning-rates "$SUITE/lr-tune/selected_learning_rates.csv"
```

只做 dry-run（不会创建 run 或执行实验）：

```bash
bash scripts/run_cross_extension_remote.sh \
  --allow-full-run \
  --stage all --run-name "$RUN_NAME" --workers 16 --resume --dry-run
```

## 只聚合或只打包

以 fixed 阶段为例，聚合脚本会验证完整性、manifest、重复项、stage、有限值与异常更新，再重建 CSV、robust statistics、图、中文摘要和 `analysis_manifest.json`：

```bash
bash scripts/aggregate_cross_extension_remote.sh \
  --config "$SUITE/fixed-full/config.json" \
  --run-dir "$SUITE/fixed-full"
```

对 tuned evaluation 还应提供：

```bash
bash scripts/aggregate_cross_extension_remote.sh \
  --config "$SUITE/lr-eval/config.json" \
  --run-dir "$SUITE/lr-eval" \
  --selected-learning-rates "$SUITE/lr-tune/selected_learning_rates.csv"
```

只打包：

```bash
bash scripts/package_cross_extension_remote.sh \
  --suite-dir "$SUITE" --run-name "$RUN_NAME" \
  --output-dir "$SUITE/packages"
```

## 结果包和 SHA-256

默认包位于：

```bash
ls -lh "$SUITE/packages"/cross-extension-*.tar.gz*
```

优先下载：

- `cross-extension-<run-name>-analysis-core.tar.gz`
- `cross-extension-<run-name>-analysis-core.tar.gz.sha256`

深度排查时再下载：

- `cross-extension-<run-name>-full.tar.gz`
- `cross-extension-<run-name>-full.tar.gz.sha256`

云端或下载到 Linux/macOS 后校验：

```bash
cd <压缩包目录>
sha256sum -c cross-extension-<run-name>-analysis-core.tar.gz.sha256
sha256sum -c cross-extension-<run-name>-full.tar.gz.sha256
```

Windows PowerShell 校验：

```powershell
Get-FileHash .\cross-extension-<run-name>-analysis-core.tar.gz -Algorithm SHA256
Get-Content .\cross-extension-<run-name>-analysis-core.tar.gz.sha256
```

## 从云端下载

可使用云平台网页文件管理器，或在本地终端使用占位符命令：

```bash
scp <user>@<server>:<remote-path>/cross-extension-<run-name>-analysis-core.tar.gz .
scp <user>@<server>:<remote-path>/cross-extension-<run-name>-analysis-core.tar.gz.sha256 .
```

也可断点续传：

```bash
rsync -avP <user>@<server>:<remote-path>/cross-extension-<run-name>-analysis-core.tar.gz* .
```

下载并校验后，把 analysis-core 包和 `.sha256` 上传给 Codex/GPT 做后续结果分析；只有需要逐步排查时再附 full 包。不要把大型正式结果提交到 Git。
