# Remote Deployment Implementation Report

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
