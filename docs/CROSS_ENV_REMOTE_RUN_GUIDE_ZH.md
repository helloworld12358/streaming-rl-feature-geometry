# 跨环境 Remote Full 运行指南（Linux / CPU）

本指南只用于远程 Linux。当前 cross-environment full 尚未运行；不得在本地 Windows 伪装 `RL_RUN_CONTEXT=remote` 启动 full。完整的云平台首次部署、离线 wheelhouse、无 tmux 运行和结果下载步骤见 `docs/CLOUD_PLATFORM_RUNBOOK_ZH.md`。

## 1. 当前云平台事实

- Ubuntu 22.04.4 LTS，`/usr/bin/python3` 为 Python 3.10.12。
- 项目支持 Python >= 3.10；远程 Linux 使用当前 active/base Python，不创建或激活 `.venv`，也不升级共享 base 环境的 pip、setuptools 或 wheel。
- `nproc`/`lscpu` 可见 128 个逻辑 CPU，但 `/sys/fs/cgroup/cpu.max` 为 `2000000 100000`，实际配额是 20 CPU。
- `/sys/fs/cgroup/memory.max` 为 `85899345920`，即 80 GiB。
- NumPy 使用 OpenBLAS；每个 worker 的数值库线程必须限制为 1。
- 当前没有 tmux、screen 或 Slurm/PBS/LSF；有 nohup。
- 本项目没有外部数据集、模型权重或运行时联网需求。环境、轨迹和实验数据均由代码生成。
- 当前实现是 NumPy/Pandas/Matplotlib 的线性 streaming RL，不使用 CUDA；H200、4090 或多卡不会加速。优先选 CPU 环境；平台强制要求 GPU 时只选最低成本的 1 × RTX 4090。

## 2. 更新已有仓库并核对分支

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

不要重复 clone，不要运行 `git reset --hard`、force pull 或删除已有结果目录。

## 3. 使用 base Python 安装并测试

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

当前依赖已经基本存在；editable install 主要用于注册本项目包。不要默认升级 pip。

## 4. 先运行 cross smoke

```bash
CROSS_SMOKE_NAME="cross-remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/run_cross_experiment.py \
  --config configs/cross_smoke.json \
  --workers 4 \
  --run-name "$CROSS_SMOKE_NAME"
```

预期完成 20/20 runs，顶层 `manifest.json` 为 `ok`，并在 `results/cross_smoke/$CROSS_SMOKE_NAME/figures/` 生成真实 PNG。smoke 失败时不得启动 full。

## 5. main cross full

先固定线程数。当前 cgroup 只有 20 CPU，推荐从 16 workers 开始，为系统和主进程留余量；不要使用 32、48 或 128 workers。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh \
  --allow-full-run \
  --config configs/cross_full.json \
  --workers 16 \
  --run-name cross-full-$(git rev-parse --short=12 HEAD)
```

`RL_RUN_CONTEXT=remote` 与 `--allow-full-run` 缺一不可。run name 必须唯一，不得覆盖旧目录。

## 6. compact、short-horizon 与 non-stationary 批次

main full 完成并验证后，按既定顺序分别运行：

```bash
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_compact.json --workers 16 --run-name cross-full-compact-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_short_horizon.json --workers 16 --run-name cross-full-short-horizon-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_nonstationary.json --workers 16 --run-name cross-full-nonstationary-$(git rev-parse --short=12 HEAD)
```

non-stationary 配置只覆盖 E1，在 interaction 150000 将 corridor length 从 5 改到 9，并保持 trial-start 改变规则。

## 7. 按环境拆分的四个 full

仅在平台配额或分批管理需要时使用；这些结果必须分别检查，不能与 main full 混合后伪装成一次运行。

```bash
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_tmaze.json --workers 16 --run-name cross-full-tmaze-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_ringworld.json --workers 16 --run-name cross-full-ringworld-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_two_loop.json --workers 16 --run-name cross-full-two-loop-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_hidden_velocity.json --workers 16 --run-name cross-full-hidden-velocity-$(git rev-parse --short=12 HEAD)
```

## 8. 云平台预输入命令与 nohup

平台“预输入命令”任务应让命令在前台成为主进程，不要在最外层添加 `&`：

```bash
bash -lc 'set -euo pipefail; cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-$(git rev-parse --short=12 HEAD)'
```

只有平台保证 SSH 断开后容器仍存活时，交互终端才可使用 nohup：

```bash
RUN_NAME="cross-full-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
nohup bash -lc "cd '$PWD'; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name '$RUN_NAME'" > "results/cross_full/$RUN_NAME.nohup.log" 2>&1 &
echo $! > "results/cross_full/$RUN_NAME.pid"
```

nohup 无法在平台销毁容器后继续运行。

## 9. 日志、manifest 与资源检查

```bash
RUN_NAME="cross-full-$(git rev-parse --short=12 HEAD)"
tail -n 100 "results/cross_full/$RUN_NAME/remote_launcher.log"
tail -f "results/cross_full/$RUN_NAME/remote_launcher.log"
ps -ef | grep '[r]un_cross_experiment.py'
python3 - "results/cross_full/$RUN_NAME/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["exit_status"])
PY
df -h .
du -sh results
```

无进程不等于成功；顶层 manifest 必须为 `ok`。保留所有失败、缺失和中断证据。

## 10. 完整性检查、聚合与打包

```bash
RUN_NAME="cross-full-$(git rev-parse --short=12 HEAD)"
RL_RUN_CONTEXT=remote bash scripts/aggregate_remote.sh \
  --config configs/cross_full.json \
  --run-dir "results/cross_full/$RUN_NAME"
ls -lh "results/cross_full/$RUN_NAME.tar.gz"
sha256sum "results/cross_full/$RUN_NAME.tar.gz"
```

对其他配置替换 config 和 run-dir 后重复。`aggregate_remote.sh` 会逐个检查 environment/condition/seed manifest，失败或缺失时返回非零。

## 11. 下载到 Windows

退出 SSH 后在本地 PowerShell 执行：

```powershell
scp <REMOTE_USER>@<REMOTE_HOST>:/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry/results/cross_full/cross-full-<COMMIT>.tar.gz D:\download\GitHub\
Get-FileHash D:\download\GitHub\cross-full-<COMMIT>.tar.gz -Algorithm SHA256
```

不要把大型 results、wheel 或凭据提交到 Git。完成真实远程执行之前，状态必须始终记录为“remote full 尚未运行”。
