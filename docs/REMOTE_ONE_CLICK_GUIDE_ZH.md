# 远程一键运行完整教程

> **历史证据，禁止按本文旧命令执行。** 旧版 venv/tmux/nohup 工作流已被生产根因修复流程取代。请只执行 [`PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`](PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md) 中的前台、仓库内、`/usr/bin/python3` 命令。

本教程假设你不熟悉 Linux、Git、Python venv 或 tmux。命令中的尖括号内容必须替换；不要把密码、token、真实服务器 IP 写进仓库。当前项目使用 CPU，不使用 GPU 训练。

## 1. 从 Windows 登录服务器

在 Windows 打开 PowerShell：

```powershell
ssh <REMOTE_USER>@<REMOTE_HOST>
```

登录后确认位置和资源：

```bash
pwd
nproc
lscpu | grep -E 'Model name|CPU\(s\)|Core|Thread'
free -h
df -h .
nvidia-smi || true
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
```

`nvidia-smi` 不存在或 GPU 空闲都不影响本项目。算法是 NumPy 线性 streaming RL，正确加速方式是多个独立 CPU 进程。

## 2. 第一次把私有仓库放到服务器

创建父目录：

```bash
mkdir -p <REMOTE_PROJECT_DIR>
cd <REMOTE_PROJECT_DIR>
```

推荐用 GitHub SSH key。先在服务器创建 key，显示公钥：

```bash
umask 077
ssh-keygen -t ed25519 -C "<GITHUB_USERNAME>-remote"
cat ~/.ssh/id_ed25519.pub
```

把公钥添加到 GitHub 的 SSH keys 后测试：

```bash
ssh -T git@github.com
```

只在目录中尚无仓库时 clone：

```bash
git clone git@github.com:helloworld12358/streaming-rl-feature-geometry.git
cd streaming-rl-feature-geometry
```

不要把 PAT 写在 clone URL、shell history、`.env` 或 config 中。若组织只能用 HTTPS，使用系统 credential helper 的交互提示，不要把 token 直接写进命令。

## 3. 已经 clone 时更新，不要再次 clone

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
git status --short
test -z "$(git status --porcelain)" || { echo "工作树非空，请先人工核对，禁止 reset/clean"; exit 1; }
git remote -v
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
git status
git log -1 --oneline
test "$(git rev-parse HEAD)" = "<COMMIT_SHA>" || { echo "commit 不匹配，停止"; exit 1; }
```

这里完成了 checkout 精确 branch，并用 SHA 检查精确 commit。不要执行 `git reset --hard`、`git clean -fd` 或 force pull。

## 4. 检查 Python 与依赖网络

```bash
python3 --version
python3 -m pip --version
```

需要 Python >= 3.10。项目没有数据集或模型下载；联网只用于 clone 和 pip/wheel。

若运行节点能联网，直接进入下一节。若运行节点不能联网，先在可联网、同 Linux/Python 版本的准备节点执行：

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
mkdir -p wheelhouse
python3 -m pip download -r requirements-dev.txt -d wheelhouse
python3 -m pip download 'setuptools>=68' wheel -d wheelhouse
```

确保计算节点能看到同一共享目录。不要把 wheelhouse 提交 Git。

## 5. 先单独 bootstrap（可选检查）

联网路径：

```bash
bash scripts/bootstrap_remote.sh \
  --workers <WORKERS> \
  --run-name bootstrap-<COMMIT_SHA>
```

离线 wheelhouse 路径：

```bash
bash scripts/bootstrap_remote.sh \
  --wheelhouse "$PWD/wheelhouse" \
  --workers <WORKERS> \
  --run-name bootstrap-<COMMIT_SHA>
```

脚本创建 `.venv`、安装 requirements、安装当前包、运行全部 pytest 和三个 smoke。成功时会打印 `BOOTSTRAP_COMPLETE=true`。smoke 会验证 manifests、NaN/Inf、聚合和真实 figures。

检查 smoke：

```bash
find results/remote_smoke -maxdepth 3 -name manifest.json -print
find results/remote_smoke -path '*/figures/*.png' -print | head
```

## 6. workers 怎么选

先看：

```bash
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
```

如果 cgroup quota 是 20 CPU，用 `<WORKERS>=16`；如果确认是 128 CPU，先用 64。脚本会至少留一个 CPU，并拒绝超过安全上限。每个 worker 内部的 BLAS 线程自动限制为 1。

## 7. 最少操作：一条命令启动全部流程

如果还没有执行 bootstrap，联网节点使用：

```bash
RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh \
  --allow-full-run \
  --workers <WORKERS> \
  --run-name <RUN_NAME>
```

无网络计算节点使用：

```bash
RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh \
  --allow-full-run \
  --wheelhouse "$PWD/wheelhouse" \
  --workers <WORKERS> \
  --run-name <RUN_NAME>
```

它会按顺序完成环境检查、venv、依赖、tests、smoke，再启动 core、cross-environment、production extension 和 non-stationary formal suite，最后聚合和打包。`RL_RUN_CONTEXT=remote` 与 `--allow-full-run` 缺一不可。不要复用 `<RUN_NAME>`。

## 8. tmux、nohup 和云平台任务

脚本发现 tmux 时创建 detached session，并打印：

```bash
tmux ls
tmux attach -t srl-<RUN_NAME>
```

在 tmux 内按 `Ctrl+B`，再按 `D`，即可安全离开；任务继续。重新进入仍用 `tmux attach`。

当前服务器没有 tmux 时，脚本自动使用 nohup，打印 PID 与日志。查看：

```bash
cat "results/remote/background/<RUN_NAME>.pid"
tail -f "results/remote/background/<RUN_NAME>.log"
```

nohup 只能防止 SSH 断开，不能防止云平台销毁容器。若平台提供“任务/作业命令”，更稳妥的做法是在平台作业里加 `--foreground`：

```bash
RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh \
  --allow-full-run --foreground \
  --workers <WORKERS> --run-name <RUN_NAME>
```

## 9. 查看进度和资源

```bash
tail -f "results/remote/background/<RUN_NAME>.log"
ps -ef | grep '[p]ython'
top
free -h
df -h .
nvidia-smi || true
```

GPU 使用率接近 0 是正确状态；manifest 会记录 GPU 清单和 `gpu_backend_used=none`。

## 10. 判断是否结束

```bash
cat "results/remote/<RUN_NAME>/suite_manifest.json"
cat "results/remote/<RUN_NAME>/remote_aggregation_manifest.json"
ls -lh "results/remote/<RUN_NAME>/packages"
sha256sum -c "results/remote/<RUN_NAME>/packages/"*.sha256
```

只有 `suite_manifest.json` 的 `exit_status` 为 `ok`、aggregation 的 `ok` 为 true、SHA 校验通过，才算完成。失败进程、缺失 seed 或 incomplete manifest 不会被计入成功。

## 11. 检测失败和缺失 seed

```bash
cat "results/remote/<RUN_NAME>/remote_missing_runs.csv"
cat "results/remote/<RUN_NAME>/remote_failed_runs.csv"
find "results/remote/<RUN_NAME>" -name manifest.json -print0 | xargs -0 grep -l '"exit_status": "failed"' || true
```

失败证据会保留。不要删除失败目录，也不要修改 seeds 后把新旧结果混入同一路径；修复后使用新 `<RUN_NAME>`。

## 12. 手动重新聚合、画图和打包

```bash
PYTHON_BIN=.venv/bin/python bash scripts/aggregate_remote.sh \
  --suite-dir "results/remote/<RUN_NAME>"
PYTHON_BIN=.venv/bin/python bash scripts/package_remote_results.sh \
  --suite-dir "results/remote/<RUN_NAME>"
```

聚合会重新检查预期 seed、manifest、单行成功 summary 和有限数值，生成统一 CSV、真实 batch completion 图和中文摘要。打包只收 aggregates、figures、configs、manifests 和日志摘要，排除巨大逐 step 流，并生成 SHA-256。

## 13. 从 Windows 下载

在 Windows PowerShell 执行：

```powershell
scp -r <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry/results/remote/<RUN_NAME>/packages <LOCAL_DOWNLOAD_DIR>
```

同时保存：

```bash
git rev-parse HEAD
realpath "results/remote/<RUN_NAME>"
```

把 commit SHA、branch、run name 和 result path 一起记录。

## 14. 下次更新代码

先确认结果目录已保存且 Git 工作树为空：

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
git status --short
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
git log -1 --oneline
git rev-parse HEAD
```

不要复用旧 run name；新 commit 用新 run name。

## 15. 常见错误与最小处理

| 错误 | 准确含义 | 最小处理 |
|---|---|---|
| `Permission denied (publickey)` | SSH key 未被 GitHub/服务器接受 | 检查 `~/.ssh` 权限，把公钥加到正确 GitHub 账号，再跑 `ssh -T git@github.com` |
| `Repository not found` | 仓库地址错误或账号无私库权限 | 核对 origin 和账号；不要把 token 写入 URL |
| `Connection reset` | SSH/GitHub/PyPI 网络被重置 | 在可联网准备节点重试 clone/wheel 下载；不要在 formal run 中联网 |
| `Python version mismatch` | Python < 3.10 | 选择带 Python >=3.10 的镜像；不要修改代码来绕过版本检查 |
| `ModuleNotFoundError` | venv 未安装依赖或未使用 `.venv/bin/python` | 重新执行 bootstrap，检查 `.venv/bin/python -m pip check` |
| `CUDA unavailable` | GPU/CUDA 不可见 | 本项目不使用 CUDA；继续 CPU 流程 |
| GPU 空闲 | 当前没有 GPU backend | 正常；查看 CPU 使用率，不要安装深度框架 |
| `No space left on device` | GPFS/容器磁盘不足 | `df -h`，下载并归档旧结果；不要删除未知 run |
| `tmux: command not found` | 镜像没装 tmux | one-click 会用 nohup；或使用平台前台作业 `--foreground` |
| `session not found` | tmux session 名错误或已结束 | `tmux ls`；再查看 suite manifest 和日志 |
| `missing seed` | 预期 run 目录/manifest/summary 不完整 | 查看 missing CSV 和该批次日志；保留证据，用新 run name 重跑 |
| full-run guard failure | 缺少环境变量或显式 flag | 同时使用 `RL_RUN_CONTEXT=remote` 和 `--allow-full-run` |
| `Refusing ... dirty worktree` | 源码 checkout 有未提交文件 | `git status` 核对；提交合法改动或换干净 checkout，禁止 reset/clean |
| `Refusing to overwrite` | run name 已存在 | 生成新的 `<RUN_NAME>`；不要覆盖或删除旧结果 |
| worker 超过 safe maximum | cgroup 实际 CPU 少于表面 CPU | 按错误中的 safe maximum 调小 `--workers` |
| SHA 校验失败 | 包损坏或文件被替换 | 重新打包/下载，不要使用损坏结果 |

如错误不在表中，保存 exact command、完整日志、`git rev-parse HEAD`、`git status` 和 result path，再请求分析；不要先删除现场。
