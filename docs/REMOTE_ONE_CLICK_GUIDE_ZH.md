# 远程前台一键运行完整教程

本教程面向不熟悉 Linux、Git、Python、服务器任务和结果下载的用户。当前项目是严格 streaming、线性、NumPy 型强化学习；正式实验使用 CPU 多进程，不使用 GPU 训练框架，也不需要外部数据集。

生产运行固定使用服务器现有 `/usr/bin/python3`，所有 temp/cache/results/logs/artifacts 都放在仓库内。正式命令在前台执行；SSH 终端或云平台作业必须保持存活。不要使用 `nohup`、`setsid`、后台 `&` 或把密码/token 写进脚本。

命令中的以下占位符必须替换：

```text
<REMOTE_USER> <REMOTE_HOST> <REMOTE_PROJECT_DIR> <GITHUB_USERNAME>
<BRANCH> <COMMIT_SHA> <WORKERS> <RUN_NAME> <LOCAL_DOWNLOAD_DIR>
```

## 1. Windows 登录服务器

在 Windows 打开 PowerShell：

```powershell
ssh <REMOTE_USER>@<REMOTE_HOST>
```

登录后查看当前目录与资源：

```bash
pwd
nproc
lscpu | grep -E 'Model name|CPU\(s\)|Core|Thread'
free -h
df -h .
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
nvidia-smi || true
```

`nvidia-smi` 不存在或 GPU 利用率为 0 都不是错误。本项目单个 agent 的 transition 必须因果顺序执行，正确的并行单位是 environment × condition × seed。

## 2. 第一次安全获取私有仓库

创建项目父目录：

```bash
mkdir -p <REMOTE_PROJECT_DIR>
cd <REMOTE_PROJECT_DIR>
```

推荐使用 GitHub SSH key：

```bash
umask 077
ssh-keygen -t ed25519 -C "<GITHUB_USERNAME>-remote"
cat ~/.ssh/id_ed25519.pub
```

把公钥添加到 GitHub 账号的 SSH keys 后：

```bash
ssh -T git@github.com
git clone git@github.com:helloworld12358/streaming-rl-feature-geometry.git
cd streaming-rl-feature-geometry
```

不要把 PAT 放进 clone URL、shell history、`.env`、JSON config 或仓库。若必须使用 HTTPS，请使用系统 credential helper 的交互提示。

## 3. 已有仓库时更新，不要重新 clone

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
git status --short --branch
test -z "$(git status --porcelain)" || { echo "工作树非空，请人工核对；禁止 reset/clean"; exit 1; }
git remote -v
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
git rev-parse HEAD
test "$(git rev-parse HEAD)" = "<COMMIT_SHA>" || { echo "commit 不匹配"; exit 1; }
```

origin 应为：

```text
https://github.com/helloworld12358/streaming-rl-feature-geometry.git
```

不要执行 `git reset --hard`、`git clean -fd`、force pull 或 force push。

## 4. Python 与依赖

```bash
/usr/bin/python3 --version
/usr/bin/python3 -m pip --version
```

要求 Python >= 3.10。`pyproject.toml` 是主依赖定义；`requirements.txt` 是运行依赖；`requirements-dev.txt` 增加 pytest。

联网安装由 one-click 内部的 bootstrap 完成。若正式节点不能联网，先在同 Linux、同 Python 版本的联网节点准备 wheelhouse：

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
mkdir -p wheelhouse/py310-linux-x86_64
python3 -m pip download -r requirements-dev.txt -d wheelhouse/py310-linux-x86_64
python3 -m pip download 'setuptools>=68' wheel -d wheelhouse/py310-linux-x86_64
```

然后在无网络节点可先手工执行：

```bash
WHEELHOUSE="$PWD/wheelhouse/py310-linux-x86_64"
/usr/bin/python3 -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.txt -r requirements-dev.txt
/usr/bin/python3 -m pip install --no-index --find-links "$WHEELHOUSE" --no-build-isolation --no-deps -e .
/usr/bin/python3 -m pip check
```

wheelhouse 不得提交 Git。

也可以让 one-click 把同一目录传给 bootstrap；此时运行时保持离线：

```bash
WHEELHOUSE="$PWD/wheelhouse/py310-linux-x86_64"
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh --allow-full-run \
  --wheelhouse "$WHEELHOUSE" \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

## 5. 选择 workers

脚本读取 cgroup quota，并至少保留一个 CPU。`--workers` 是实际并行进程数；显式写了 `--workers 16` 后，即使把平台环境扩大到 64 或 128 核，脚本也仍只开 16 个 worker，不会自动放大。要利用新增核心，只需调整这一处参数，其余终端指令不变。推荐：

- quota 20：`<WORKERS>=16`；
- quota 32：先用 24；
- quota 64：先用 48；
- 真正 quota 128：先用 64。

不要只看宿主机 `lscpu` 的 128 CPUs；如果 cgroup 只有 20，设置 64 或 128 会被脚本拒绝。

## 6. 先做完全无副作用 dry-run

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh \
  --allow-full-run \
  --workers <WORKERS> \
  --run-name <RUN_NAME> \
  --expected-branch <BRANCH> \
  --expected-commit <COMMIT_SHA> \
  --dry-run
```

成功时会打印 bootstrap、storage pilot 以及 fixed-full、lr-tune、lr-eval、norm-scaled 四阶段的 exact commands，并以 `DRY_RUN_ONLY=true` 结束。dry-run 不启动 formal experiment。

## 7. 一条命令执行完整生产流程

确认 dry-run、commit、磁盘和 workers 后执行：

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh \
  --allow-full-run \
  --workers <WORKERS> \
  --run-name <RUN_NAME> \
  --expected-branch <BRANCH> \
  --expected-commit <COMMIT_SHA>
```

one-click 会依次执行：

1. Python/依赖检查与安装；
2. 完整 pytest；
3. 70-run production smoke；
4. 50-run storage pilot；
5. cgroup、Git、磁盘、inode、路径 containment preflight；
6. fixed-full 1000 attempts；
7. lr-tune 1500 attempts（真实 invalid candidate 被保留，不伪装成 valid）；
8. lr-eval 1000 runs；
9. norm-scaled 700 runs；
10. 聚合、验证、打包与 SHA-256。

完整正式规模为 4200 attempts/runs。不要复用 `<RUN_NAME>`。

## 8. 为什么保持前台

当前云平台生产约束要求 `/usr/bin/python3` 和前台 `tee`/状态链路，避免 SSH 后台进程、tmux 或容器生命周期造成“进程仍跑但任务状态丢失”。因此 one-click 不自动创建 tmux/nohup。

如果平台提供“作业命令”，把第 7 节的整条前台命令原样放入平台作业；否则保持 SSH 窗口打开。不要自行在末尾加 `&`。

## 9. 查看进度与资源

另开一个 SSH 窗口：

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
ps -ef | grep '[p]ython'
top
free -h
df -h .
nvidia-smi || true
find "logs/cross_extension/<RUN_NAME>" -maxdepth 1 -name '*.status.json' -print
tail -n 200 "logs/cross_extension/<RUN_NAME>/lr-tune.log"
```

GPU 空闲是算法类型决定的。CPU 利用率和 stage status 才是主要检查项。

## 10. 判断是否结束

```bash
cat "logs/cross_extension/<RUN_NAME>/fixed-full.status.json"
cat "logs/cross_extension/<RUN_NAME>/lr-tune.status.json"
cat "logs/cross_extension/<RUN_NAME>/lr-eval.status.json"
cat "logs/cross_extension/<RUN_NAME>/norm-scaled.status.json"
find "artifacts/cross_extension/<RUN_NAME>" -maxdepth 2 -type f -print
sha256sum -c "artifacts/cross_extension/<RUN_NAME>"/*.sha256
```

只有四阶段状态均为 0、validation 通过且 SHA 校验通过，才算完成。`lr-tune` 的 invalid candidates 是调优证据；总 attempted identities 仍必须完整覆盖 1500。

## 11. 查看失败与 invalid evidence

```bash
find "results/cross_extension/<RUN_NAME>" -path '*/runtime_validity.json' -type f -exec \
  /usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); f=r.get("first_failure"); f and print(sys.argv[1], json.dumps(f, ensure_ascii=False))' {} \;
find "results/cross_extension/<RUN_NAME>" -name invalid_tuning_candidates.csv -print
find "results/cross_extension/<RUN_NAME>" -path '*/failed_attempts/*/manifest.json' -print
```

不要删除失败目录，不要修改 seed/grid 后混入同一路径。

## 12. Resume 与 retry-invalid

普通 resume 会复用严格匹配的 valid run，也会把同 commit 下已保存的 invalid lr-tune candidate 视为已完成的无效尝试：

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh --allow-full-run --skip-bootstrap --resume \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --storage-report "$PWD/artifacts/storage_pilot/<PILOT_NAME>/storage_projection.json" \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

明确保留旧 invalid evidence 并重跑：

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh --allow-full-run --skip-bootstrap --retry-invalid \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --storage-report "$PWD/artifacts/storage_pilot/<PILOT_NAME>/storage_projection.json" \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

commit/config/schema 不匹配时不会复用旧结果。

## 13. 可选：完整 9-batch suite

第 7 节的 one-click 是当前推荐的 4200-run production extension 流程。如果还要在同一次远程任务中执行 core、cross-environment、production extension 和 non-stationary 共 9 个顶层批次，先完成 storage pilot，再执行：

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/run_remote_suite.sh --allow-full-run \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --storage-report "$PWD/artifacts/storage_pilot/<PILOT_NAME>/storage_projection.json" \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

该命令同样保持前台，并会在任一 batch 失败时停止、保留证据。不要与 4200-run one-click 使用相同 `<RUN_NAME>`。

## 14. 手工聚合与打包

通常 stage launcher 已自动完成。需要复核时：

```bash
for stage in fixed-full lr-tune lr-eval norm-scaled; do
  EXTRA=()
  if [[ "$stage" == "lr-eval" ]]; then
    EXTRA=(--selected-learning-rates "$PWD/results/cross_extension/<RUN_NAME>/lr-tune/selected_learning_rates.csv")
  fi
  /usr/bin/python3 -m streaming_rl_feature_geometry.cross_production aggregate \
    --config "$PWD/results/cross_extension/<RUN_NAME>/$stage/config.json" \
    --run-dir "$PWD/results/cross_extension/<RUN_NAME>/$stage" "${EXTRA[@]}"
done
/usr/bin/python3 -m streaming_rl_feature_geometry.cross_production package \
  --suite-dir "$PWD/results/cross_extension/<RUN_NAME>" \
  --run-name <RUN_NAME> \
  --output-dir "$PWD/artifacts/cross_extension/<RUN_NAME>"
```

## 15. 从 Windows 下载结果

在 Windows PowerShell：

```powershell
scp -r <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry/artifacts/cross_extension/<RUN_NAME> <LOCAL_DOWNLOAD_DIR>
```

同时保存：

```bash
git branch --show-current
git rev-parse HEAD
realpath "results/cross_extension/<RUN_NAME>"
realpath "artifacts/cross_extension/<RUN_NAME>"
```

## 16. 下次更新代码

```bash
cd <REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry
git status --short --branch
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
git rev-parse HEAD
```

新 commit 使用新 run name，并重新运行 storage pilot；不要为了复用旧结果放宽 commit 检查。

## 17. 常见错误与最小处理

| 错误 | 准确含义 | 最小处理 |
|---|---|---|
| `Permission denied (publickey)` | SSH key 未被服务器/GitHub 接受 | 检查 key 权限，把公钥添加到正确账号，重新 `ssh -T git@github.com` |
| `Repository not found` | URL 错误或账号无私库权限 | 核对 origin 与账号；不要把 token 放入 URL |
| `Connection reset` | SSH/GitHub/PyPI 网络中断 | 在联网准备节点重试 clone/wheel 下载；formal run 本身不联网 |
| Python version mismatch | `/usr/bin/python3` < 3.10 | 更换正确镜像/节点，不修改研究代码绕过 |
| `ModuleNotFoundError` | 系统 Python 未装依赖或 editable package | 重跑 bootstrap 或离线 wheelhouse 安装 |
| `CUDA unavailable` | GPU/CUDA 不可见 | 本项目不使用 CUDA，继续 CPU 流程 |
| GPU 空闲 | 没有 GPU backend | 正常，查看 CPU 和 stage log |
| `No space left on device` | 磁盘或 inode 不足 | 查看 `df -h`、`df -Pi`，归档旧结果；不要删除未知 run |
| tmux session not found | 本流程未创建 tmux session | 使用云平台作业或保持前台终端，不要猜 session 名 |
| missing seed/identity | 预期 run 或 evidence 不完整 | 查看 stage log、manifest、invalid CSV；保留现场后 resume/retry |
| full-run guard failure | 缺环境变量或 flag | 同时设置 `RL_RUN_CONTEXT=remote` 和 `--allow-full-run` |
| dirty worktree | checkout 有生产相关修改 | `git status` 人工核对；用干净 checkout，禁止 reset/clean |
| commit mismatch | storage/result 来自其他 commit | checkout 最终 SHA，重新测试和 storage pilot |
| worker exceeds safe maximum | cgroup quota 小于表面 CPU | 按错误中的 safe maximum 调小 workers |
| SHA 校验失败 | 包损坏或被替换 | 重新打包/下载，不使用损坏文件 |

## 18. 用户真正只需做的最少步骤

1. SSH 登录并进入已有仓库；
2. `git fetch`、切到 `<BRANCH>`、核对 `<COMMIT_SHA>`；
3. 先执行第 6 节 dry-run；
4. 执行第 7 节一条前台 one-click 命令；
5. 完成后用第 14 节 `scp` 下载 artifacts。

remote full 不会在本地电脑执行；只有远程 Linux、双重 guard、storage pilot 与 preflight 全部通过后才会启动。
