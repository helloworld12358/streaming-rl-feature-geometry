# Remote Deployment Reconciliation

## 2026-07-18 当前 Goal 重新核对

本轮从已完成并推送的生产根因修复继续执行，不重新 clone、不切换分支、不重写历史：

- 起始分支：`codex/fix-streaming-rl-production-root-causes`；
- 起始 commit：`a638ac62a6a9b1727f0dbe60bb1fdb9e0478da86`；
- 起始 upstream：`origin/codex/fix-streaming-rl-production-root-causes`，ahead/behind=`0/0`；
- origin：`https://github.com/helloworld12358/streaming-rl-feature-geometry.git`；
- tracked worktree 与 index 均干净；唯一未跟踪目录 `paper_from_existing_results/` 保持原样且继续排除；
- 当前基线全套：122 tests passed；本地核心 smoke 4/4；
- 当前仓库保留 tracked 小型 pilot 汇总 `docs/pilot_results/hidden_velocity_informative_pilot_summary.csv`；历史大结果已按先前清理 Goal 从工作区移除，不能把文档中的旧路径误称为当前仍存在；
- formal remote full 仍未在本地运行。

本轮审计确认依赖入口、remote configs、CPU 资源策略、full-run guard 和大部分部署脚本仍存在，同时发现三个需要修复的部署一致性缺口：

1. `remote_one_click.sh` 已退化为只转发参数的兼容包装器，不能执行文档承诺的 bootstrap、storage pilot 和最少点击流程；
2. `run_remote_suite.sh --dry-run` 会把 `--dry-run` 传给尚未支持该参数的 production extension launcher；
3. 通用 remote inspection 仍把 lr-tune 已完整保存的 invalid candidate 当成 missing run，与当前 `valid summaries + invalid attempts = expected attempts` 语义不一致。

本轮采用的修复原则：

- 保留最新 4200-run production extension、`/usr/bin/python3`、仓库内 runtime 路径、storage pilot 和前台运行约束；
- one-click 在前台顺序执行 bootstrap、测试、production smoke、storage pilot、四阶段 formal、聚合与打包，并提供真正无副作用的 dry-run；
- lr-tune inspection 接受严格匹配且证据完整的 invalid attempt，但非 lr-tune invalid、未知异常和缺失 evidence 仍 fail closed；
- CPU process-level parallelism 继续作为唯一正式后端，GPU 只检测并记录，`gpu_backend_used=none`；
- release 验证在新的临时 venv 中检查 requirements 与 tests；目标云端正式运行仍按已验证的 `/usr/bin/python3` 约束执行；
- `gh` 在当前 Windows 环境不可用时不阻塞普通 Git push；Draft PR 仅在认证能力真实可用时创建，否则返回准确的手动 URL。

## 2026-07-18 当前验证证据

- 干净 Python 3.11.9 venv 完成 requirements、editable install 和 `pip check`；版本为 NumPy 2.4.6、pandas 3.0.3、Matplotlib 3.11.1、pytest 9.1.1；
- 干净环境安装后的首次全套 `124 passed in 123.76s`；收紧 lr-tune invalid identity 后 focused suite `36 passed in 44.16s`，全部 Bash dry-run 与 9-batch 参数合约补齐后的最终全套 `124 passed in 96.47s`；
- production extension smoke `70/70`、manifest=`ok`、CPU process-level、GPU backend=`none`；
- 37 个 JSON config、14 个 shell 脚本的本地语法解析、compileall、`git diff --check`、敏感信息与大文件扫描均通过；
- Windows 本机没有可用 Linux Bash/shellcheck；Ubuntu CI 已增加全部 shell 的 `bash -n`、one-click dry-run 和 9-batch suite dry-run，必须在 push 后核对；
- 未在本地运行 formal remote full；`.runtime/` 和 smoke 结果保持 ignored；`paper_from_existing_results/` 仍未读取、未修改、未纳入任务 diff。

本节是当前 Goal 的起始与决策记录；下面较早的 `codex/remote-deployment` 内容保留为历史，不代表当前 branch/commit 或当前测试计数。

## 起始状态

- 本地仓库：`D:\download\GitHub\streaming-rl-feature-geometry`
- 延后任务起始分支：`codex/hidden-velocity-metrics-lr-production`
- 延后任务起始本地 commit：`a9f7f1f10e410f19fa01cbb4d90590604280f298`
- 同树远端 commit：`f76788b2917e0dc217132676e0f7530f3380563c`
- origin：`https://github.com/helloworld12358/streaming-rl-feature-geometry.git`
- 起始 Git 状态：研究实现已提交；存在与本任务无关的未跟踪目录 `paper_from_existing_results/`，保持原样且不纳入部署 commit。
- 主 Goal 附件归档已重新核对，并用 `.gitattributes` 保证 `docs/COMPLETE_OVERNIGHT_MASTER_GOAL.md` 保持原始 LF 字节；源和目标 SHA-256 均为 `8b23db2815f7e7a95dc37e6aaf70d1a931a5cc00aa9338ae08c6e6ecdf3c3ff4`。

没有重新 clone，没有执行 `git reset --hard`、`git clean -fd`、force push 或历史重写，也没有删除既有结果。

## 已有研究证据

- 全套测试在延后任务开始前为 95 passed；部署新增测试会在最终报告中给出重新运行后的精确结果。
- 核心 smoke：4/4，`results/smoke/overnight-extension-regression-20260717`。
- 跨环境 smoke：20/20，`results/local_validation_20260717/legacy-cross-smoke`。
- production extension smoke：40/40，`results/local_validation_20260717/extension-smoke`。
- informative hidden-velocity 有限 pilot：20 个 per-run summary 已生成，但历史顶层 manifest 因旧版聚合读取空 decision CSV 报 `EmptyDataError`，状态为 `failed`；路径 `results/local_validation_20260717/informative-design-pilot`，失败证据与 mixed 结果均保留。
- LR tune/eval 与 norm-scaled 有限 smoke 均完成；formal remote full 未在本地运行。

上述结果目录由 `.gitignore` 排除，未混入新的正式远程结果。

## 依赖审计

`pyproject.toml` 是项目依赖的主定义。运行时实际导入为 NumPy、pandas 和 Matplotlib；pytest 仅用于开发与验证。仓库不使用 SciPy、外部数据集、模型权重、PyTorch、TensorFlow、JAX、CuPy、Ray、W&B 或 MLflow。

部署层保留 `requirements.txt` 作为运行依赖入口，并新增 `requirements-dev.txt`。未生成平台绑定的 freeze/lock 文件，因为 Windows 的精确二进制版本不应伪装成 Linux 锁；各 run manifest 会保存实际依赖版本和配置哈希。

## 原远程能力与缺口

起始仓库只有单批次 `bootstrap_remote.sh`、`run_full_remote.sh` 和 `aggregate_remote.sh`，没有完整的 clean-venv、统一 smoke、suite 调度、分析包 SHA、one-click 后台入口、部署配置、缺失 seed 汇总或完整中文教程。

本任务补齐：

- `requirements-dev.txt`；
- `scripts/run_remote_smoke.sh`、`run_remote_full.sh`、`run_remote_suite.sh`、`package_remote_results.sh`、`remote_one_click.sh` 和共享资源检查；
- 五个 `configs/remote_*.json` 计划；
- 缺失/失败 run 检测、统一聚合、真实完成图、日志摘要、分析包及 SHA-256；
- clean venv、联网与离线 wheelhouse 安装路径；
- 中文一键教程、命令速查、资源策略和部署报告。

旧名称 `scripts/run_full_remote.sh` 保留为兼容包装器，不重命名或删除历史入口。

## CPU/GPU 决策

最终采用 CPU 进程级并行。并行单位是 environment × representation × bank/horizon × seed；每个 NumPy/BLAS 进程限制为单线程。脚本读取在线 CPU 和 cgroup quota，至少保留一个 CPU，并拒绝超过安全上限的 `--workers`。

不实现 GPU backend。原因是 agent、predictor 和 transform 都是小型线性在线递推，每步严格顺序依赖，当前无可验证的 CUDA 数值后端；增加深度学习框架会改变依赖与研究范围。`nvidia-smi` 只记录硬件清单，manifest 明确写入 `gpu_backend_used=none`。

用户给出的 128 逻辑 CPU、1.5 TiB RAM、379 GiB 可用磁盘的 CPU 环境适合本项目。若云平台强制选 GPU 实例，选最低成本的 1 × RTX 4090；H200 或多卡不会给当前代码带来真实收益。运行阶段不需要联网，因为环境和轨迹全部由代码生成。

## 验证完成状态

- clean venv requirements 安装、editable install、`pip check`、101 tests 和 4/4 smoke 通过；
- 当前环境 101 tests 通过；
- 新 remote smoke 三批共 64/64，0 missing、0 failed；
- 9-batch/5900-run formal plan dry-run 通过；
- Bash、guards、result isolation、missing seed、package SHA、secret scan 与 diff check 通过；
- 本地没有运行 formal remote full。

## GitHub 状态

主 Goal 分支已通过 GitHub 连接器发布，远端比基线领先 4 commits，且远端树与本地树一致。因为该远端分支与本地保留了“同树、不同 commit 元数据”的平行历史，延后任务采用附件明确允许的独立分支 `codex/remote-deployment`，避免 non-fast-forward 或 force push。

`codex/remote-deployment` 已通过普通 `git push -u origin codex/remote-deployment` 成功创建并设置 tracking；原 Goal 分支保持不动。按用户最新要求不创建 PR，PR URL 为 N/A。远端仓库仍为 `https://github.com/helloworld12358/streaming-rl-feature-geometry.git`。
