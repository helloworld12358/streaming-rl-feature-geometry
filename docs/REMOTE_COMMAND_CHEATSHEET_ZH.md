# 远程命令速查

以下命令均在远程 Linux 仓库根目录执行。正式流程使用 CPU、多进程、`/usr/bin/python3` 和前台运行。

## 更新并核对交付 commit

```bash
git status --short --branch
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
test "$(git rev-parse HEAD)" = "<COMMIT_SHA>"
```

## 无副作用 dry-run

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh --allow-full-run --dry-run \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

## 一条命令：tests、smoke、storage pilot、4200-run formal、聚合和打包

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh --allow-full-run \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

命令保持前台；不要加 `&`、nohup 或 setsid。

## 状态与资源

```bash
ps -ef | grep '[p]ython'
top
df -h .
df -Pi .
find "logs/cross_extension/<RUN_NAME>" -name '*.status.json' -print
tail -n 200 "logs/cross_extension/<RUN_NAME>/lr-tune.log"
```

GPU 空闲是预期行为。

## Resume / retry invalid

```bash
RL_RUN_CONTEXT=remote PYTHON_BIN=/usr/bin/python3 \
bash scripts/remote_one_click.sh --allow-full-run --skip-bootstrap --resume \
  --workers <WORKERS> --run-name <RUN_NAME> \
  --storage-report "$PWD/artifacts/storage_pilot/<PILOT_NAME>/storage_projection.json" \
  --expected-branch <BRANCH> --expected-commit <COMMIT_SHA>
```

将 `--resume` 替换为 `--retry-invalid` 可归档旧 invalid evidence 后明确重跑。

## 下载到 Windows

```powershell
scp -r <REMOTE_USER>@<REMOTE_HOST>:<REMOTE_PROJECT_DIR>/streaming-rl-feature-geometry/artifacts/cross_extension/<RUN_NAME> <LOCAL_DOWNLOAD_DIR>
```

完整说明见 `docs/REMOTE_ONE_CLICK_GUIDE_ZH.md` 和 `docs/PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`。formal full 必须同时具备 `RL_RUN_CONTEXT=remote` 与 `--allow-full-run`。
