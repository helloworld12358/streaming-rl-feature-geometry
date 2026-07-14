# 云平台一键运行手册（CPU 优先）

本手册对应仓库 `streaming-rl-feature-geometry` 的现有分支 `codex/cross-environment-representation-priors`。它只提供云端操作命令；本地 Windows 不执行 smoke、pilot、diagnostics 或 remote full。当前 remote full 尚未运行。

## 1. 已确认的云平台事实

- CPU 准备环境：Ubuntu 22.04.4 LTS，root 用户，`/usr/bin/python3` 为 Python 3.10.12；不在 venv 或 Conda 中。
- `/usr/local/lib/python3.10/dist-packages` 可写；GitHub 和 PyPI 可访问。
- 已有 numpy 1.24.4、pandas 2.2.1、matplotlib 3.8.4、pytest 8.1.1、setuptools 68.2.2，`pip check` 无冲突。
- 项目位于 GPFS 共享存储：
  `/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry`。
- `nproc`/`lscpu` 可见 128 个逻辑 CPU，但 `/sys/fs/cgroup/cpu.max` 为 `2000000 100000`，实际 CPU quota 是 20。
- `/sys/fs/cgroup/memory.max` 为 `85899345920`，即 80 GiB；不能按主机表面的 1.5 TiB 内存规划。
- NumPy 使用 OpenBLAS；OMP/OpenBLAS/MKL/NumExpr 线程变量原本均未设置。
- 没有 tmux、screen 或 Slurm/PBS/LSF；有 nohup。
- CPU 准备环境可以联网；GPU 运行环境执行期间不能联网；GPFS 共享文件可同时访问。
- 本项目是 NumPy/Pandas/Matplotlib 的非深度、线性、严格 streaming RL；没有 PyTorch、TensorFlow、JAX、CuPy 或 CUDA 后端。
- 没有外部数据集、模型权重或预训练资源。环境、轨迹和实验数据由代码生成，运行阶段不需要联网。

结论：优先选 CPU 环境。若平台强制必须选择 GPU，选择最低成本的 1 × RTX 4090；不要选择 H200 或 2/4/8 卡。GPU 空闲是预期行为，速度由 CPU quota、workers 和 OpenBLAS 线程决定。

## 2. 首次 clone

只在父目录中尚不存在仓库时使用：

```bash
set -euo pipefail
BASE=/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007
cd "$BASE"
test ! -e streaming-rl-feature-geometry || { echo "仓库目录已存在；不要重复 clone，请使用下一节的 pull 流程"; exit 1; }
git clone https://github.com/helloworld12358/streaming-rl-feature-geometry.git
cd streaming-rl-feature-geometry
git fetch origin
git checkout codex/cross-environment-representation-priors
git pull --ff-only
git status
git log -1 --oneline
git rev-parse HEAD
```

把 `git rev-parse HEAD` 与 Codex 最终 handoff 给出的 commit SHA 对照。

## 3. 已经 clone 后拉取 Codex 新 commit

每次 Codex push 后使用。先检查工作区；若非空就停止并报告，不得覆盖、stash、reset 或删除结果。

```bash
set -euo pipefail
REPO=/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
cd "$REPO"
git status --short
test -z "$(git status --porcelain)" || { echo "工作区不干净，停止 pull 并先人工核对"; exit 1; }
git fetch origin
git checkout codex/cross-environment-representation-priors
git pull --ff-only
git status
git log -1 --oneline
git rev-parse HEAD
```

不要重复 clone，不要 `git reset --hard`，不要 force pull，也不要删除本地 results。

## 4. CPU 准备环境安装

远程 Linux 使用当前 active/base Python。项目支持 Python >= 3.10；当前环境是 Python 3.10.12。不要创建或激活 venv，不要创建 Conda，不要安装新 Python，不要默认升级 pip/setuptools/wheel。

```bash
set -euo pipefail
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
python3 --version
python3 -m pip --version
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"需要 Python >=3.10，当前为 {sys.version.split()[0]}")
print("Python 版本通过：", sys.version.split()[0])
PY
python3 -m pip install -e '.[dev]'
python3 -m pip check
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 -m pytest -q
```

依赖已经基本存在；editable install 主要用于注册当前源码包。也可在云端运行 `bash scripts/bootstrap_remote.sh`，它会使用同一 base Python 完成安装、tests、核心 smoke、validation 和 cross smoke。

## 5. 外部数据与联网

本项目没有外部数据集，不下载模型权重。环境、轨迹、观测和实验结果全部由代码生成。完成 clone 和依赖准备后，smoke/full 运行本身不需要联网。不得下载 CUDA、PyTorch、JAX 或其他深度学习包。

## 6. 核心 smoke

```bash
set -euo pipefail
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
SMOKE_NAME="cloud-core-smoke-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/run_experiment.py \
  --config configs/smoke.json \
  --workers 1 \
  --run-name "$SMOKE_NAME"
python3 scripts/validate_results.py \
  "results/smoke/$SMOKE_NAME" \
  --config configs/smoke.json
echo "核心 smoke 结果：results/smoke/$SMOKE_NAME"
```

预期是 4/4 runs，validation 返回 0。run name 带时间戳，不能复用。

## 7. cross smoke

```bash
set -euo pipefail
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
CROSS_SMOKE_NAME="cloud-cross-smoke-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/run_cross_experiment.py \
  --config configs/cross_smoke.json \
  --workers 4 \
  --run-name "$CROSS_SMOKE_NAME"
echo "cross smoke 结果：results/cross_smoke/$CROSS_SMOKE_NAME"
```

预期是 20/20 runs、顶层 manifest 为 `ok`，并生成真实 figures。tests、核心 smoke、validation 和 cross smoke 全部成功后才能启动 full。

## 8. 当前 CPU 环境的 main full

当前 20-CPU quota 推荐 `--workers 16`，为系统、主进程和 I/O 留余量。不得用 128 workers；32 或 48 也不适合作为当前环境默认值。

```bash
set -euo pipefail
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
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

`RL_RUN_CONTEXT=remote` 与 `--allow-full-run` 缺一不可。main full 只能在 smoke 成功后运行；run name 不得复用。

## 9. 其他批次

按既定顺序分别执行，不能更改配置定义或混合结果目录：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_compact.json --workers 16 --run-name cross-full-compact-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_short_horizon.json --workers 16 --run-name cross-full-short-horizon-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_nonstationary.json --workers 16 --run-name cross-full-nonstationary-$(git rev-parse --short=12 HEAD)
```

四个按环境拆分的配置：

```bash
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_tmaze.json --workers 16 --run-name cross-full-tmaze-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_ringworld.json --workers 16 --run-name cross-full-ringworld-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_two_loop.json --workers 16 --run-name cross-full-two-loop-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_hidden_velocity.json --workers 16 --run-name cross-full-hidden-velocity-$(git rev-parse --short=12 HEAD)
```

核心 T-maze full：

```bash
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers 16 --run-name full-stationary-$(git rev-parse --short=12 HEAD)
RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/full_nonstationary.json --workers 16 --run-name full-nonstationary-$(git rev-parse --short=12 HEAD)
```

聚合示例：

```bash
RL_RUN_CONTEXT=remote bash scripts/aggregate_remote.sh \
  --config configs/cross_full.json \
  --run-dir results/cross_full/cross-full-$(git rev-parse --short=12 HEAD)
```

## 10. 没有 tmux、screen 或调度器时

### 10.1 云平台“预输入命令”任务

让命令在前台运行并成为平台跟踪的主任务，最外层不要加 `&`：

```bash
bash -lc 'set -euo pipefail; cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-$(git rev-parse --short=12 HEAD)'
```

### 10.2 CPU 交互终端

只有平台保证 SSH 断开后容器仍存活时才使用 nohup：

```bash
set -euo pipefail
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
RUN_NAME="cross-full-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results/cross_full
nohup bash -lc "cd '$PWD'; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name '$RUN_NAME'" > "results/cross_full/$RUN_NAME.nohup.log" 2>&1 &
echo $! > "results/cross_full/$RUN_NAME.pid"
echo "PID=$! 日志=results/cross_full/$RUN_NAME.nohup.log"
```

nohup 不能阻止云平台销毁容器；容器被销毁后进程一定停止。

## 11. GPU 环境选择

当前项目不使用 GPU。H200、RTX 4090 和多卡都不会加速 NumPy 多进程实验。优先选择 CPU 环境；如果平台 UI 强制要求 GPU 实例，选择 1 × RTX 4090，不选择 H200 或 2/4/8 卡。GPU 利用率接近 0 属于预期行为。

GPU 任务仍应关注实际 CPU quota。启动后可只读检查：

```bash
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/memory.max
python3 --version
```

## 12. GPU 无法联网时的离线 wheelhouse 备选

仅当 GPU base Python 缺少依赖时使用。wheelhouse 必须与 GPU 环境的 Python ABI、Linux 架构匹配；先在 GPU 环境只读查看 `python3 --version`，再用兼容的联网 CPU 环境准备。不要新建 Python 环境。

### 12.1 在联网 CPU 环境准备共享 wheelhouse

wheelhouse 放在仓库外的共享父目录，因此不会被 Git 提交：

```bash
set -euo pipefail
BASE=/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007
WHEELHOUSE="$BASE/.wheelhouse/py310-linux-x86_64"
mkdir -p "$WHEELHOUSE"
python3 -m pip download --dest "$WHEELHOUSE" \
  'setuptools>=68' \
  wheel \
  'numpy>=1.24,<3' \
  'pandas>=2.0,<4' \
  'matplotlib>=3.7,<4' \
  'pytest>=7,<10'
find "$WHEELHOUSE" -maxdepth 1 -type f -printf '%f\n' | sort
```

只下载 pyproject.toml 声明的依赖范围、setuptools、wheel 及其传递依赖；不要下载 CUDA、PyTorch、JAX 或深度学习包。

### 12.2 GPU 离线环境安装到当前 base Python

```bash
set -euo pipefail
BASE=/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007
REPO="$BASE/streaming-rl-feature-geometry"
WHEELHOUSE="$BASE/.wheelhouse/py310-linux-x86_64"
cd "$REPO"
python3 -m pip install --no-index --find-links "$WHEELHOUSE" setuptools wheel
python3 -m pip install --no-index --find-links "$WHEELHOUSE" --no-build-isolation -e '.[dev]'
python3 - <<'PY'
import matplotlib, numpy, pandas, pytest
import streaming_rl_feature_geometry
print("offline imports ok")
PY
```

所有安装都使用 `--no-index` 和 `--find-links`，不会访问 PyPI。import 检查成功后再启动 smoke 或 full。

## 13. 日志、进程、manifest、磁盘与结果

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
RUN_NAME="cross-full-$(git rev-parse --short=12 HEAD)"
tail -n 100 "results/cross_full/$RUN_NAME/remote_launcher.log"
tail -f "results/cross_full/$RUN_NAME/remote_launcher.log"
ps -ef | grep '[r]un_cross_experiment.py'
python3 - "results/cross_full/$RUN_NAME/manifest.json" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
print(path, payload.get("exit_status"))
PY
df -h .
du -sh results
```

无进程不代表成功；manifest 必须为 `ok`。失败、缺失、中断和负面结果必须保留。

## 14. 聚合、打包、SHA-256 与下载

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
RUN_NAME="cross-full-$(git rev-parse --short=12 HEAD)"
RL_RUN_CONTEXT=remote bash scripts/aggregate_remote.sh \
  --config configs/cross_full.json \
  --run-dir "results/cross_full/$RUN_NAME"
ls -lh "results/cross_full/$RUN_NAME.tar.gz"
sha256sum "results/cross_full/$RUN_NAME.tar.gz"
```

退出 SSH，在 Windows PowerShell 下载并核对：

```powershell
scp <REMOTE_USER>@<REMOTE_HOST>:/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry/results/cross_full/cross-full-<COMMIT>.tar.gz D:\download\GitHub\
Get-FileHash D:\download\GitHub\cross-full-<COMMIT>.tar.gz -Algorithm SHA256
```

不要把大型 results 或 wheel 文件提交到 Git。

---

## A. CLOUD_CPU_PREP_COMMANDS

已有仓库的 pull、base 安装、tests、核心 smoke、validation 和 cross smoke：

```bash
set -euo pipefail
REPO=/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
cd "$REPO"
git status --short
test -z "$(git status --porcelain)" || { echo "工作区不干净，停止"; exit 1; }
git fetch origin
git checkout codex/cross-environment-representation-priors
git pull --ff-only
git log -1 --oneline
git rev-parse HEAD
python3 --version
python3 -m pip --version
python3 -m pip install -e '.[dev]'
python3 -m pip check
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 -m pytest -q
SMOKE_NAME="cloud-core-smoke-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/run_experiment.py --config configs/smoke.json --workers 1 --run-name "$SMOKE_NAME"
python3 scripts/validate_results.py "results/smoke/$SMOKE_NAME" --config configs/smoke.json
CROSS_SMOKE_NAME="cloud-cross-smoke-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/run_cross_experiment.py --config configs/cross_smoke.json --workers 4 --run-name "$CROSS_SMOKE_NAME"
echo "CORE_SMOKE=results/smoke/$SMOKE_NAME"
echo "CROSS_SMOKE=results/cross_smoke/$CROSS_SMOKE_NAME"
```

## B. CLOUD_CPU_FULL_COMMAND

当前 cgroup 20 CPU、workers 16、OpenBLAS 单线程的 main full：

```bash
bash -lc 'set -euo pipefail; cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-$(git rev-parse --short=12 HEAD)'
```

## C. CLOUD_GPU_PRESET_COMMAND

仅在平台强制要求 GPU 实例时选择 1 × RTX 4090。命令前台运行，不使用 CUDA，不访问网络；它首先使用共享代码和已存在的 base 依赖：

```bash
bash -lc 'set -euo pipefail; cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry; python3 -c "import numpy,pandas,matplotlib,streaming_rl_feature_geometry"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-$(git rev-parse --short=12 HEAD)'
```

若 import 失败，先由 CPU 环境按第 12.1 节准备 wheelhouse，再把 GPU 预输入命令改为下面这一块；`--no-index` 保证不联网：

```bash
bash -lc 'set -euo pipefail; BASE=/inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007; cd "$BASE/streaming-rl-feature-geometry"; WHEELHOUSE="$BASE/.wheelhouse/py310-linux-x86_64"; python3 -m pip install --no-index --find-links "$WHEELHOUSE" setuptools wheel; python3 -m pip install --no-index --find-links "$WHEELHOUSE" --no-build-isolation -e ".[dev]"; python3 -c "import numpy,pandas,matplotlib,pytest,streaming_rl_feature_geometry"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; RL_RUN_CONTEXT=remote bash scripts/run_full_remote.sh --allow-full-run --config configs/cross_full.json --workers 16 --run-name cross-full-$(git rev-parse --short=12 HEAD)'
```

## D. WEB_GPT_HANDOFF

把 Codex 最终响应中的实际值与以下命令输出一起交给网页端 ChatGPT：

```text
分支：codex/cross-environment-representation-priors
Codex 新 commit：<CODEX_FINAL_COMMIT_SHA>
云端 git rev-parse HEAD：<CLOUD_HEAD_SHA，必须等于上面的 SHA>
本地 pytest：<CODEX_FINAL_TEST_RESULT>
本地未运行：smoke、pilot、diagnostics、remote full
云端需先运行：tests、核心 smoke、validation、cross smoke
远程 full 状态：尚未运行
需要分析的日志：
  results/smoke/<CORE_SMOKE_RUN>/manifest.json
  results/cross_smoke/<CROSS_SMOKE_RUN>/manifest.json
  results/cross_full/<FULL_RUN>/remote_launcher.log
  results/cross_full/<FULL_RUN>/manifest.json
  results/cross_full/<FULL_RUN>/runs/**/manifest.json
修改文件：以 Codex 最终响应和 git show --stat <CODEX_FINAL_COMMIT_SHA> 为准
```

云端核对命令：

```bash
cd /inspire/hdd/project/reinforcement-learning-course/weiyuqi-CZXS25110007/streaming-rl-feature-geometry
git branch --show-current
git rev-parse HEAD
git status --short
git show --stat --oneline HEAD
```
