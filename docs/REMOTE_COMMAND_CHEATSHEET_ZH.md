# 远程命令速查

> **历史证据，禁止按本文旧命令执行。** 当前可复制命令以 [`PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`](PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md) 为准；旧版 tmux/venv 命令不再受支持。

以下命令均在仓库根目录执行。

## 第一次：一键安装、测试、smoke、后台启动 full suite

```bash
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
test "$(git rev-parse HEAD)" = "<COMMIT_SHA>"
RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh \
  --allow-full-run \
  --workers <WORKERS> \
  --run-name <RUN_NAME>
```

## 查看状态

```bash
tail -f "results/remote/background/<RUN_NAME>.log"
ps -ef | grep '[p]ython'
top
df -h .
cat "results/remote/<RUN_NAME>/suite_manifest.json"
```

有 tmux 时：

```bash
tmux ls
tmux attach -t srl-<RUN_NAME>
```

## 重新聚合和打包

```bash
PYTHON_BIN=.venv/bin/python bash scripts/aggregate_remote.sh \
  --suite-dir "results/remote/<RUN_NAME>"
PYTHON_BIN=.venv/bin/python bash scripts/package_remote_results.sh \
  --suite-dir "results/remote/<RUN_NAME>"
```

## 下载到 Windows PowerShell

```powershell
scp -r <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/results/remote/<RUN_NAME>/packages <LOCAL_DOWNLOAD_DIR>
```

formal full 必须同时有 `RL_RUN_CONTEXT=remote` 和 `--allow-full-run`。不要复用 `<RUN_NAME>`，不要提交服务器账号、IP、密码或 token。
