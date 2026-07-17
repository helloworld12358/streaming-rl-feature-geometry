# Remote Resource Strategy

## 结论

选择 CPU 环境，不为本项目购买 H200 或多卡。当前 allocation 虽显示 128 逻辑 CPU 和 1.5 TiB host RAM，但真实 cgroup quota 是 20 CPUs、生产内存预算 80 GiB，仓库文件系统可用约 68 GiB。因此使用 `--workers 16`，硬上限 19；不得在本 allocation 提高到 32、64 或 128。

如果平台必须选择 GPU，选最低成本的 1 × RTX 4090。GPU 利用率为 0 是预期行为，不是错误。

## 原因

- 每个 transition 必须按因果顺序处理一次，单个 agent 内部不能分片并行。
- agent、GVF、transform 和 controller 都是 NumPy 的小型线性递推。
- 当前没有 CuPy/CUDA 后端，也没有证明 GPU 数值一致性或加速收益。
- 为占用 GPU 而引入 PyTorch/JAX 会违反 non-deep 和最小依赖约束。

## 正确的并行层级

并行任务是独立的 environment × condition × seed × bank/horizon。每个 worker 使用一个进程，并设置：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

所有正式入口同时检查在线 CPU 数和 cgroup v2 quota，安全上限为 `max(1, available_cpus - 1)`；用户显式提供的 `--workers` 超过上限时脚本失败，不会偷偷降级或占满宿主机。

## 内存、磁盘和网络

- 每个 run 有独立目录、seed、config、manifest、日志和退出状态。
- 结果目录不覆盖；复用 run name 会失败。
- formal suite 使用 compact v2 typed NPZ partitions，不生成重复的巨型 trace aggregate CSV。高频 continuing-control 动作不会被误当成稀疏决策事件；hidden-velocity 逐步统计在线累计，轨迹严格遵守 `metrics_stride`。必须先用实际 storage pilot 校准，4200-run source + aggregate + full/analysis-core package 的预计峰值目标不超过 50 GiB，并另留 10 GiB 安全余量。
- 没有外部数据集。clone、pip/wheel 下载完成后，smoke 和 full 运行不需要联网。
- 正式分析包排除逐 step 原始流，只包含 aggregates、figures、configs、manifests、日志摘要和 SHA-256。

本地 50-run storage pilot 的分类外推结果是：4200-run source 11.312 GiB、无压缩收益假设下的 full package 上界 11.374 GiB、analysis-core 0.056 GiB、同时存在的 peak 22.742 GiB；compact traces 是最大类别（11.135 GiB）。这通过 50 GiB 目标与额外 10 GiB free-space margin，但远端正式运行前仍须在 clean final commit 和目标文件系统重跑 pilot/preflight。

## 推荐值

| 可用 CPU quota | 推荐起始 workers |
|---:|---:|
| 20 | 16 |
| 32 | 24（仅适用于真实 cgroup quota 为 32 的其他 allocation） |
| 64 | 48（仅适用于真实 cgroup quota 为 64 的其他 allocation） |
| 128 | 64（仅适用于真实 cgroup quota 为 128 的其他 allocation） |

不要根据 `lscpu` 的宿主机数字直接设置 128；先看：

```bash
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
free -h
df -h .
```
