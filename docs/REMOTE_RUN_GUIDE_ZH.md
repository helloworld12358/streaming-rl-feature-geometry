# 远程服务器完整运行指南（零 Linux 基础版）

本指南只准备远程 full experiment；不要在本地电脑执行 full。命令中的尖括号内容必须替换，例如 `<REMOTE_USER>`。不要把密码、token 或 SSH 私钥写入仓库、命令历史截图或结果包。

## 1. 在 Windows 打开终端

打开 PowerShell 或 Windows Terminal。后续“本地命令”都在这里执行；登录服务器以后，提示符会改变，之后的命令是在 Linux 服务器执行。

## 2. SSH 登录

```powershell
ssh <REMOTE_USER>@<REMOTE_HOST>
```

第一次连接会询问主机指纹。只有在管理员确认指纹正确后输入 `yes`。密码只在 SSH 提示中输入，不要发给 Codex，也不要写进文件。

## 3. 查看当前目录

```bash
pwd
ls -la
```

`pwd` 应输出服务器路径；`ls -la` 显示该路径内容。

## 4. 创建项目父目录

```bash
mkdir -p <REMOTE_PROJECT_DIR>
cd <REMOTE_PROJECT_DIR>
pwd
```

请让 `<REMOTE_PROJECT_DIR>` 指向一个新建或确认属于本项目的目录，不要覆盖未知目录。

## 5. Clone 已有 GitHub 仓库

```bash
git clone https://github.com/helloworld12358/streaming-rl-feature-geometry.git
cd streaming-rl-feature-geometry
git remote -v
```

不要执行 `git init`，不要创建新仓库，不要修改 `origin`。

## 6. 安全处理 GitHub 凭据

公开读取通常不需要登录。若仓库以后变为 private，优先使用服务器管理员认可的 Git Credential Manager、GitHub CLI device/browser login 或只读 deploy key。不要把 token 放入 URL，不要把 token 写入脚本或 `.env`，不要复制私钥内容。可用 GitHub CLI 时：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth status
```

## 7. Checkout 精确 commit

先同步远程分支，再 checkout 本报告提供的精确 SHA：

```bash
git fetch origin codex/predictive-feature-properties
git checkout codex/predictive-feature-properties
git pull --ff-only
git checkout <COMMIT_SHA>
git status
git rev-parse HEAD
```

detached HEAD 对只运行实验是正常的。`git rev-parse HEAD` 必须等于 `<COMMIT_SHA>`。

## 8. 检查 Python

```bash
python3 --version
```

必须是 Python 3.11.x。若不是，请联系服务器管理员安装；不要破坏系统 Python。

## 9. 创建仓库内虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

成功后提示符通常出现 `(.venv)`。

## 10. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

也可以让自动脚本执行第 8–12 步：

```bash
scripts/bootstrap_remote.sh
```

## 11. 运行完整 tests

```bash
python -m pytest -q
```

必须退出为 0。任何 failed/error 都要先解决，不要继续 full。

## 12. 运行 smoke

```bash
SMOKE_NAME="remote-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/run_experiment.py --config configs/smoke.json --workers 1 --run-name "$SMOKE_NAME"
python scripts/validate_results.py "results/smoke/$SMOKE_NAME" --config configs/smoke.json
```

最后应显示 4 个 expected runs，顶层 manifest 为 `ok`。

## 13. 启动 tmux

长实验必须放在 tmux，SSH 断线后才能继续：

```bash
tmux new -s rl-full
```

若提示没有 tmux，请联系管理员安装，或使用服务器认可的作业调度器。

## 14. 在 tmux 中运行 stationary full

重新进入项目并激活环境，然后执行：

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
source .venv/bin/activate
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_stationary.json --workers <WORKERS> --run-name full-stationary-<COMMIT_SHA>
```

`<WORKERS>` 建议从 4 开始。脚本拒绝占满全部 CPU。不要删除两个安全条件中的任意一个。

## 15. 安全离开 tmux

按键顺序：先按 `Ctrl+B`，松开，再按 `D`。这叫 detach，不会停止实验。不要在 tmux 内按 `Ctrl+C`，除非确实要中止。

## 16. 重新进入 tmux

重新 SSH 登录后：

```bash
tmux ls
tmux attach -t rl-full
```

## 17. 查看日志

不进入 tmux 也可以查看：

```bash
tail -n 100 results/full_stationary/full-stationary-<COMMIT_SHA>/remote_launcher.log
```

持续跟踪使用：

```bash
tail -f results/full_stationary/full-stationary-<COMMIT_SHA>/remote_launcher.log
```

按 `Ctrl+C` 只退出 `tail`，不会停止实验。

## 18. 检查进程

```bash
ps -ef | grep '[r]un_experiment.py'
```

有输出表示 runner 仍在；无输出时检查 manifest 和日志，不能只凭无进程判断成功。

## 19. 检查磁盘

```bash
df -h .
du -sh results
```

若磁盘接近满，停止启动新实验并联系管理员；不要删除不明文件或已完成的失败证据。

## 20. 运行 non-stationary full

stationary 完成并验证后，建议新建另一个 tmux：

```bash
tmux new -s rl-nonstationary
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
source .venv/bin/activate
RL_RUN_CONTEXT=remote scripts/run_full_remote.sh --allow-full-run --config configs/full_nonstationary.json --workers <WORKERS> --run-name full-nonstationary-<COMMIT_SHA>
```

该配置运行 300,000 interactions，并在 interaction 150,000 将 corridor length 从 5 改为 9 一次。

## 21. 聚合和完整性检查

runner 正常完成时已聚合；仍应独立验证/重建：

```bash
RL_RUN_CONTEXT=remote scripts/aggregate_remote.sh --config configs/full_stationary.json --run-dir results/full_stationary/full-stationary-<COMMIT_SHA>
RL_RUN_CONTEXT=remote scripts/aggregate_remote.sh --config configs/full_nonstationary.json --run-dir results/full_nonstationary/full-nonstationary-<COMMIT_SHA>
```

脚本逐一检查 condition/seed manifest，列出缺失或失败项，并在存在问题时返回非零，不会把失败 seed 当成功。

## 22. 生成图表

上一步会从保存的 CSV 重新生成 `figures/`。可检查：

```bash
find results/full_stationary/full-stationary-<COMMIT_SHA>/figures -maxdepth 1 -type f -name '*.png' -ls
```

不要手工制作与 CSV 无关的占位图。

## 23. 打包结果

`aggregate_remote.sh` 自动生成相邻的 `.tar.gz`。确认：

```bash
ls -lh results/full_stationary/full-stationary-<COMMIT_SHA>.tar.gz
ls -lh results/full_nonstationary/full-nonstationary-<COMMIT_SHA>.tar.gz
sha256sum results/full_*/*.tar.gz
```

保存终端输出中的 SHA-256，用于下载后核对。

## 24. 用 scp 下载到 Windows

先退出 SSH 回到本地 PowerShell，再执行：

```powershell
scp <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry/results/full_stationary/full-stationary-<COMMIT_SHA>.tar.gz D:\download\GitHub\
scp <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry/results/full_nonstationary/full-nonstationary-<COMMIT_SHA>.tar.gz D:\download\GitHub\
```

下载后在 Windows 核对：

```powershell
Get-FileHash D:\download\GitHub\full-stationary-<COMMIT_SHA>.tar.gz -Algorithm SHA256
```

## 25. 记录 commit SHA

服务器内执行并把输出记入实验记录：

```bash
git rev-parse HEAD
git status --short
```

full 应从干净、精确 commit 运行；若 dirty，manifest 会记录，必须解释。

## 26. 记录 result path

```bash
realpath results/full_stationary/full-stationary-<COMMIT_SHA>
realpath results/full_nonstationary/full-nonstationary-<COMMIT_SHA>
```

同时保存 run name、config、workers、开始/结束时间和机器信息。

## 27. 常见报错与准确含义

- `Permission denied (publickey)`：SSH 身份验证失败；联系管理员配置访问，不能索要或打印私钥。
- `Repository not found`：URL 错、仓库 private 且未授权，或账号无权限；先核对 URL 和安全登录状态。
- `Python 3.11 is required`：服务器 Python 版本不符；安装/加载 3.11 后重新创建 `.venv`。
- `No module named venv`：系统缺少 Python venv 组件；让管理员安装对应 3.11 包。
- `Full profile blocked`：缺少 `RL_RUN_CONTEXT=remote` 或 `--allow-full-run`；两者必须同时存在。
- `Requested N workers but safe maximum is M`：要求的 workers 会占满 CPU；改成不大于 M。
- `FileExistsError`：run name 已存在；这是防覆盖保护。换新 run name，不要删除旧结果。
- `Missing seeds/runs`：一个或多个 condition/seed 目录或 manifest 不存在；查看 launcher log，不要直接聚合。
- `Failed seeds/runs`：对应 manifest 不是 `ok`；保留目录和日志，修复后用新 run name 重跑。
- `No space left on device`：磁盘已满；停止新任务并联系管理员清理已确认可移除的非项目文件。
- `Killed` 或 exit 137：常见于内存/调度器强制终止；减少 workers 后使用新 run name。
- `Recv failure: Connection was reset`：GitHub 连接中断；先测试 `git ls-remote`，可单次使用 `git -c http.version=HTTP/1.1 ...`，不要关闭 SSL 验证或永久修改代理。
- tmux `no server running`：没有活跃会话；检查进程、manifest 和日志，任务可能已结束或未启动。
- manifest 长期 `running` 但无进程：运行被外部中断；将该批次视为失败/不完整，不手工改成成功。

## 最终检查清单

tests 和 smoke 通过；精确 commit 已记录；两个 full 顶层 manifest 为 `ok`；每批 20 seeds × 10 conditions 完整；聚合验证返回 0；图来自 CSV；结果包 SHA-256 已核对。远程 full 在真正执行前必须一直报告为“尚未运行”。
