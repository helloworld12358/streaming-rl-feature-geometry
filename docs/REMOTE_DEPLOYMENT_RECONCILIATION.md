# Remote Deployment Reconciliation

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
