# 跨环境 Remote Full 完整运行指南（Linux / CPU）

本指南只用于远程 Linux 服务器。当前 remote full **尚未运行**，不得在本地 Windows 电脑伪装 `RL_RUN_CONTEXT=remote` 执行。尖括号均为占位符；密码、token、SSH 私钥不得写入仓库、脚本、结果包或聊天。

## 1. 从 Windows 登录服务器

在 PowerShell 或 Windows Terminal 中：

```powershell
ssh <REMOTE_USER>@<REMOTE_HOST>
```

首次连接只在管理员确认主机指纹后输入 `yes`。认证失败时联系管理员，不要打印私钥。

## 2. 准备项目目录并 clone

```bash
mkdir -p <REMOTE_PROJECT_DIR>
cd <REMOTE_PROJECT_DIR>
git clone https://github.com/helloworld12358/streaming-rl-feature-geometry.git
cd streaming-rl-feature-geometry
git remote -v
```

若仓库为 private，使用管理员认可的 Git credential、`gh auth login --web` 或只读 deploy key。不要把 token 放入 URL。服务器上的 clone 不改变本地仓库；不要 `git init` 或修改 `origin`。

## 3. Checkout 报告给出的精确 commit

```bash
git fetch origin codex/cross-environment-representation-priors
git checkout codex/cross-environment-representation-priors
git pull --ff-only
git checkout <EXTENSION_COMMIT_SHA>
git status --short
git rev-parse HEAD
```

`git rev-parse HEAD` 必须等于 `<EXTENSION_COMMIT_SHA>`；full 应从干净 commit 运行。detached HEAD 对只运行实验是正常的。

## 4. 检查 Python 3.11 与 CPU

```bash
python3 --version
nproc
```

必须是 Python 3.11.x。项目不假设 GPU；`workers` 从 4 开始，并至少保留一个 CPU 给系统。

## 5. 创建 venv 和安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

也可执行 `scripts/bootstrap_remote.sh` 自动完成 Python 检查、venv、安装、全量 tests、核心 smoke 和跨环境 smoke。脚本使用 `set -euo pipefail`，任何失败都会停止。

## 6. 手动运行全量 tests

```bash
source .venv/bin/activate
python -m pytest -q
```

必须退出 0；不得带着失败测试启动 full。

## 7. 手动运行 cross smoke

```bash
SMOKE_NAME="cross-remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/run_cross_experiment.py \
  --config configs/cross_smoke.json \
  --workers 1 \
  --run-name "$SMOKE_NAME"
```

应得到 20/20 runs、顶层 `manifest.json` 为 `ok`，并在 `results/cross_smoke/$SMOKE_NAME/figures/` 看到真实 PNG。

## 8. 用 tmux 保护长任务

```bash
tmux new -s cross-full
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
source .venv/bin/activate
```

离开但不中止：按 `Ctrl+B`，松开，再按 `D`。重新进入：`tmux attach -t cross-full`。也可使用服务器认可的作业调度器。

## 9. 运行 stationary cross full

```bash
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh \
  --allow-full-run \
  --config configs/cross_full.json \
  --workers <WORKERS> \
  --run-name cross-full-<EXTENSION_COMMIT_SHA>
```

环境变量与命令行 flag 缺一不可。主配置为 20 seeds、E1-E4、task-agnostic 核心条件与 task-matched prior。不要在同一结果名上重跑。

## 10. 运行有限 bank/horizon full

主 full 完成并验证后，再分别执行：

```bash
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_compact.json --workers <WORKERS> --run-name cross-full-compact-<EXTENSION_COMMIT_SHA>
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_short_horizon.json --workers <WORKERS> --run-name cross-full-short-horizon-<EXTENSION_COMMIT_SHA>
```

compact 仅比较 raw/matched；short horizon 仅比较 E1/E2 的 raw/matched，不是完整笛卡尔积。

如果机器配额或排队系统要求按环境拆分，四份配置与主矩阵使用完全相同的环境内条件、seed 和预算，可分别运行：

```bash
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_tmaze.json --workers <WORKERS> --run-name cross-full-tmaze-<EXTENSION_COMMIT_SHA>
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_ringworld.json --workers <WORKERS> --run-name cross-full-ringworld-<EXTENSION_COMMIT_SHA>
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_two_loop.json --workers <WORKERS> --run-name cross-full-two-loop-<EXTENSION_COMMIT_SHA>
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/cross_full_hidden_velocity.json --workers <WORKERS> --run-name cross-full-hidden-velocity-<EXTENSION_COMMIT_SHA>
```

拆分配置生成彼此隔离的结果目录，不能与 `cross_full.json` 的输出混合后伪装成一次完整运行；应分别完成 manifest 检查和聚合，再在报告中明确其来源。

## 11. 运行 E1 non-stationary full

stationary full 验证后，新建独立 tmux：

```bash
tmux new -s cross-nonstationary
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
source .venv/bin/activate
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh \
  --allow-full-run \
  --config configs/cross_full_nonstationary.json \
  --workers <WORKERS> \
  --run-name cross-full-nonstationary-<EXTENSION_COMMIT_SHA>
```

该配置仅 E1，在 interaction 150000 将 corridor length 从 5 改到 9；环境仍强制只在 trial start 改变。结果记录 pre/final-post accuracy、recovery time，并从真实数据生成 adaptation figure。

## 12. 查看日志和进程

启动日志先保存在 profile 父目录，runner 创建隔离目录后也会复制进去：

```bash
tail -n 100 results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>/remote_launcher.log
tail -f results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>/remote_launcher.log
ps -ef | grep '[r]un_cross_experiment.py'
```

`Ctrl+C` 只退出 `tail -f`。无进程不等于成功；必须检查 manifest。

## 13. 检查磁盘和 manifest

```bash
df -h .
du -sh results
python - <<'PY'
import json
from pathlib import Path
p = Path('results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>/manifest.json')
print(json.loads(p.read_text())['exit_status'])
PY
```

磁盘不足时停止启动新批次并联系管理员，不删除失败证据或未知文件。顶层状态必须是 `ok`。

## 14. 完整性检查、重新聚合和画图

```bash
RL_RUN_CONTEXT=remote scripts/aggregate_remote.sh \
  --config configs/cross_full.json \
  --run-dir results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>
```

脚本逐一检查 environment/condition/seed manifest；缺失或失败会列出并返回非零。只有完整后才重建 CSV/图表并打包。对其他 full 配置替换 config 与 run-dir 后重复。

## 15. 检查图表和结果路径

```bash
find results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>/figures -maxdepth 1 -name '*.png' -ls
realpath results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>
git rev-parse HEAD
git status --short
```

图必须来自保存的 CSV/NPZ，不手工制作占位图。记录 commit、result path、run name、workers、机器信息和起止时间。

## 16. 打包与 SHA-256

`aggregate_remote.sh` 自动生成相邻 `.tar.gz`：

```bash
ls -lh results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>.tar.gz
sha256sum results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>.tar.gz
```

保存 SHA-256，下载后核对。

## 17. scp 到 Windows

退出 SSH，回到本地 PowerShell：

```powershell
scp <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry/results/cross_full/cross-full-<EXTENSION_COMMIT_SHA>.tar.gz D:\download\GitHub\
Get-FileHash D:\download\GitHub\cross-full-<EXTENSION_COMMIT_SHA>.tar.gz -Algorithm SHA256
```

不要把大型原始结果提交到 Git。

## 18. 常见错误

- `Permission denied` / `Repository not found`：SSH 或仓库授权问题；核对 URL/账号，禁止打印 credential。
- `Python 3.11 is required` / `No module named venv`：请管理员安装正确 Python/venv，不破坏系统 Python。
- `Refusing full run` / `Cross-environment full profile blocked`：缺少 remote 环境变量或显式 flag；两者必须同时存在，且只能在真实 remote 使用。
- `Requested N workers but safe maximum is M`：减小 workers，至少保留一个 CPU。
- `FileExistsError`：run name 已存在；保留旧目录，换新名字。
- `Missing seeds/runs` / `Failed seeds/runs`：查看对应 manifest 与 launcher log；保留失败批次，修复后用新 run name。
- `No space left on device`：停止新任务并联系管理员，不随意删除结果。
- `Killed` / exit 137：通常为内存或调度器终止；减小 workers，新目录重跑。
- SSH 中断：用 tmux 重新连接；不要仅凭终端断开判断任务失败。
- manifest 长期 `running` 且无进程：视为中断/不完整，禁止手改为 `ok`。

## 最终清单

精确 commit 已记录；全量 tests 与 cross smoke 通过；每个 full 顶层 manifest 为 `ok`；逐 run 完整性检查返回 0；图表来自真实数据；结果包 SHA-256 已核对；失败/缺失 seed 未被隐藏。完成真实远程执行前，报告状态必须始终写为“remote full 尚未运行”。
