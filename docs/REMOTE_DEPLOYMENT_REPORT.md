# Remote Deployment Implementation Report

> Historical implementation record. The current production workflow is documented in [`PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`](PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md); its `/usr/bin/python3`, foreground-only, repository-contained constraints supersede the earlier venv/background workflow below.

## 2026-07-18 发布刷新

本轮在 `codex/fix-streaming-rl-production-root-causes` 的已推送 checkpoint `a638ac62a6a9b1727f0dbe60bb1fdb9e0478da86` 上继续整理，没有重新 clone、切换历史或删除旧证据。当前 one-click 在前台依次执行 bootstrap、70-run production smoke、50-run storage pilot、4200-run 四阶段 production extension、聚合和打包；`--dry-run` 只展开命令，不安装依赖或启动正式实验。离线节点可把仓库内 wheelhouse 用 `--wheelhouse` 传入 one-click。

本轮实际本地验证：

- 新建仓库内忽略目录 `.runtime/release-clean-env-20260718-132842`，安装 requirements、editable package 成功，`pip check` 为 `No broken requirements found`；Python 3.11.9、NumPy 2.4.6、pandas 3.0.3、Matplotlib 3.11.1、pytest 9.1.1；
- 干净环境安装后的首次全套 `124 passed in 123.76s`；收紧 lr-tune invalid identity 后的 focused suite `36 passed in 44.16s`，全部 Bash dry-run 与 9-batch 参数合约补齐后的最终全套 `124 passed in 96.47s`；
- `configs/cross_extension_smoke.json` 实际完成并验证 `70/70`，manifest=`ok`，CPU process-level、4 workers、GPU backend=`none`；第一次未设置仓库内 runtime 环境变量的探测被 guard 正确拒绝，未启动实验；
- 37 个 JSON config 全部解析；tree-sitter-bash 对 14 个 shell 脚本均无语法错误；`git diff --check`、compileall 和敏感信息扫描通过；tracked 文件中无 token、AWS key、private key、密码赋值或 IPv4 地址，无大于 5 MiB 文件；
- 当前 Windows 没有可用 Linux Bash/shellcheck，所以真正的 `bash -n`、one-click dry-run 和 9-batch suite dry-run 已加入 Ubuntu CI，推送后以该 CI 结果为发布证据；
- formal 4200-run remote full 未在本地执行。本地新增 smoke 位于被 `.gitignore` 排除的 `results/bootstrap_smoke/deployment-refresh-20260718`，不提交原始结果。

`--workers` 是实际 CPU 进程数：扩大云实例核数但仍显式传入 `--workers 16` 时，程序仍只启动 16 个 worker。要利用新增核心，只需按 cgroup quota 调整该参数；GPU 型号和卡数不会改善当前线性 NumPy 工作负载。

## 实现范围

部署层只编排既有严格 streaming、非深度、线性实验，不修改科研变量。正式 suite 包含：

1. core stationary、cue-only 和 short-horizon；
2. cross main、compact-bank 和 short-horizon；
3. production extension 的 fixed、LR tune、LR eval 和 norm-scaled 四阶段；
4. core/cross non-stationary；
5. 聚合、缺失/失败检测、真实完成图和带 SHA-256 的分析包。

默认计划共 9 个顶层批次。可用 `--stages` 或 `--skip-stage` 关闭阶段；E5 明确保持 deferred。

## 可复现记录

core/cross 单 run manifest 现包含 commit、branch、dirty、exact command、config hash、seed、condition/environment、predictive bank、horizons、interaction budget、Python/dependencies、hostname/OS、CPU、GPU inventory、start/end、exit status 和 result path。GPU backend 固定记录为 `none`。

suite 另外保存顶层 host/suite manifest 和每批状态。目录存在时拒绝覆盖；失败目录和日志保留。聚合要求 manifest=`ok` 且 summary 恰好一行、run_status=`ok`，否则列入 missing/failed CSV 并返回非零。

## 依赖与资源

- 主依赖：`pyproject.toml`。
- 运行快捷文件：`requirements.txt`。
- 测试快捷文件：`requirements-dev.txt`。
- 运行环境：Linux Python >=3.10 的 `.venv`。
- 并行：CPU process-level；BLAS 每进程 1 thread。
- GPU：仅检测与记录，不执行 GPU 计算。
- 外部数据：无；运行阶段不联网。

## 正式 guard

`run_remote_full.sh`、`run_remote_suite.sh`、`remote_one_click.sh` 和 production extension launcher 都要求：

```text
RL_RUN_CONTEXT=remote
--allow-full-run
```

缺一即退出 2，不创建 formal result。local 只允许 tests、smoke、pilot 和 dry-run。

## 验证状态

部署代码完成后的实际证据：

- 最终当前 `.venv` 全套回归：`101 passed in 45.52s`。
- 仓库外全新 Python 3.11 venv：依次安装两个 requirements、editable package，`pip check` 为 `No broken requirements found`；NumPy 2.4.6、pandas 3.0.3、Matplotlib 3.11.0、pytest 9.1.1；`101 passed in 54.37s`。
- clean venv 核心 smoke：4/4，validation 通过，路径 `results/clean_install_smoke/clean-install-smoke-20260717`。
- 新部署 smoke：core 4/4、cross 20/20、extension 40/40；统一汇总 3 batches、0 missing、0 failed，路径 `results/remote_smoke/deployment-smoke-20260717`。
- smoke analysis package 的 SHA-256 复核通过；最新验证包为 `a568859a2a41467acae9b15a363f58e152e21d54f695ed4f00dde1b7a6cae436`。
- 13 个 Bash 脚本全部通过 `bash -n` 且含 `set -euo pipefail`；本机没有 shellcheck，因此按规范记录为 unavailable，不阻塞。
- core full、suite full、production extension 的缺环境变量/缺 flag 四个 guard probe 均返回 2，且未创建 formal 目录。
- 最外层 `remote_one_click.sh` 到 9-batch suite 的端到端 dry-run 通过，计划共 5900 runs：core 360、cross 1040、production extension 4200、non-stationary 300。
- 36 个 JSON configs 解析通过；新增 missing-seed、result-isolation、package inventory/SHA 测试均在 101 tests 内通过。
- credential-like secret scan、no-deep dependency scan、README/extension guard 命令一致性和 `git diff --check` 均通过。
- `docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md` 与附件逐字节 SHA-256 均为 `8b23db2815f7e7a95dc37e6aaf70d1a931a5cc00aa9338ae08c6e6ecdf3c3ff4`。

remote full 没有在本地执行；上述 formal 数量只来自 config 解析与 dry-run。

## 入口

```bash
RL_RUN_CONTEXT=remote bash scripts/remote_one_click.sh \
  --allow-full-run --workers <WORKERS> --run-name <RUN_NAME>
```

完整教程：`docs/REMOTE_ONE_CLICK_GUIDE_ZH.md`。速查：`docs/REMOTE_COMMAND_CHEATSHEET_ZH.md`。

## 发布状态

验证后的部署实现已发布到 `origin/codex/remote-deployment`，使用普通 fast-forward push；没有 force push，没有自动 merge。按用户最新要求没有创建 PR。
