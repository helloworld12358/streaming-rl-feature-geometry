# 远程服务器完整运行指南（核心 T-maze）

> 2026-07-17 更新：本文件保留核心单批次历史入口。当前受支持的 clean `.venv` 与一键完整 suite 请使用 `REMOTE_ONE_CLICK_GUIDE_ZH.md`；新指南覆盖下文的 base-Python 安装说明。

本指南只用于远程 Linux 上的核心 full profiles；不要在本地 Windows 执行 full。跨环境批次见 `docs/CROSS_ENV_REMOTE_RUN_GUIDE_ZH.md`，当前云平台的完整复制粘贴手册见 `docs/CLOUD_PLATFORM_RUNBOOK_ZH.md`。

## 1. 当前环境与硬件结论

- 项目支持 Python >= 3.10；当前云平台使用 Ubuntu 22.04.4 和 `/usr/bin/python3`（Python 3.10.12）。
- 远程 Linux 使用当前 active/base Python，不创建或激活 `.venv`，不创建 Conda，也不默认升级 pip、setuptools 或 wheel。Windows 本地 PowerShell 的 `.venv` 工作流保持不变。
- 容器可见 128 个逻辑 CPU，但 cgroup v2 `cpu.max` 为 `2000000 100000`，实际只有 20 CPU；`memory.max` 为 80 GiB。
- NumPy 使用 OpenBLAS。先把 OMP/OpenBLAS/MKL/NumExpr 线程数都设为 1，再从 16 workers 开始；32、48、128 都不是当前环境的合理默认值。
- 当前没有 tmux、screen 或作业调度器；有 nohup。
- 项目没有外部数据集或模型权重，环境、轨迹和结果由代码生成，运行时不需要联网。
- 当前代码不使用 GPU 或 CUDA。优先选 CPU 环境；平台强制要求 GPU 时选最低成本的 1 × RTX 4090，GPU 空闲是正常现象。

## 2. 更新已有云端仓库

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
git status --short
test -z "$(git status --porcelain)" || { echo "工作区不干净，停止 pull"; exit 1; }
git fetch origin
git checkout codex/cross-environment-representation-priors
git pull --ff-only
git status
git log -1 --oneline
git rev-parse HEAD
```

不要重复 clone、force pull、`git reset --hard` 或删除旧结果。

## 3. 安装当前项目并运行 tests

```bash
python3 --version
python3 -m pip --version
python3 -m pip install -e '.[dev]'
python3 -m pip check
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 -m pytest -q
```

当前依赖已经基本存在；editable install 主要注册本项目包。`scripts/bootstrap_remote.sh` 会使用同一 base Python 依次执行版本检查、editable install、tests、核心 smoke、validation 和 cross smoke。

## 4. 核心 smoke

```bash
SMOKE_NAME="remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/run_experiment.py \
  --config configs/smoke.json \
  --workers 1 \
  --run-name "$SMOKE_NAME"
python3 scripts/validate_results.py \
  "results/smoke/$SMOKE_NAME" \
  --config configs/smoke.json
```

应完成 4/4 runs，validation 返回 0，顶层 manifest 为 `ok`。失败时不得启动 full。

## 5. stationary full

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh \
  --allow-full-run \
  --config configs/full_stationary.json \
  --workers 16 \
  --run-name full-stationary-$(git rev-parse --short=12 HEAD)
```

当前 cgroup 配额为 20 CPU，16 workers 为系统和主进程留出余量。`run_full_remote.sh` 同时读取在线 CPU 数和 `/sys/fs/cgroup/cpu.max`，取较小值，并至少保留一个 CPU。`RL_RUN_CONTEXT=remote` 与 `--allow-full-run` 缺一不可。

## 6. non-stationary full

stationary 完成并验证后再运行：

```bash
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh \
  --allow-full-run \
  --config configs/full_nonstationary.json \
  --workers 16 \
  --run-name full-nonstationary-$(git rev-parse --short=12 HEAD)
```

每次必须使用新的 run name；不得覆盖或删除旧批次。

## 7. 平台预输入命令

平台预输入命令应在前台运行，让平台把实验作为主任务跟踪；最外层不要添加 `&`：

```bash
bash -lc 'set -euo pipefail; cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers 16 --run-name full-stationary-$(git rev-parse --short=12 HEAD)'
```

## 8. CPU 交互终端使用 nohup

只有平台保证终端断开后容器仍然存在时才使用：

```bash
RUN_NAME="full-stationary-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
nohup bash -lc "cd '$PWD'; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers 16 --run-name '$RUN_NAME'" > "results/full_stationary/$RUN_NAME.nohup.log" 2>&1 &
echo $! > "results/full_stationary/$RUN_NAME.pid"
tail -f "results/full_stationary/$RUN_NAME.nohup.log"
```

nohup 只能抵抗终端断开，无法在平台销毁容器后继续运行。

## 9. 日志、进程、磁盘与 manifest

```bash
RUN_NAME="full-stationary-$(git rev-parse --short=12 HEAD)"
tail -n 100 "results/full_stationary/$RUN_NAME/remote_launcher.log"
ps -ef | grep '[r]un_experiment.py'
python3 - "results/full_stationary/$RUN_NAME/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["exit_status"])
PY
df -h .
du -sh results
```

无进程不表示成功；manifest 必须为 `ok`。失败、缺失或中断的目录与日志都必须保留。

## 10. 完整性检查、重新聚合和打包

```bash
RUN_NAME="full-stationary-$(git rev-parse --short=12 HEAD)"
RL_RUN_CONTEXT=remote bash scripts/aggregate_remote.sh \
  --config configs/full_stationary.json \
  --run-dir "results/full_stationary/$RUN_NAME"
ls -lh "results/full_stationary/$RUN_NAME.tar.gz"
sha256sum "results/full_stationary/$RUN_NAME.tar.gz"
```

non-stationary 批次使用 `configs/full_nonstationary.json` 和对应结果目录。聚合脚本先检查全部 condition/seed manifest，只有完整且成功后才重建 CSV/图表并打包。

## 11. 下载到 Windows

退出 SSH 后在本地 PowerShell 执行：

```powershell
scp <REMOTE_USER>@<REMOTE_HOST>:/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry/results/full_stationary/full-stationary-<COMMIT>.tar.gz D:\download\GitHub\
Get-FileHash D:\download\GitHub\full-stationary-<COMMIT>.tar.gz -Algorithm SHA256
```

不要把大型 results、wheel、密码、token 或私钥提交到 Git。真正远程 full 完成前，报告必须持续注明“remote full 尚未运行”。
