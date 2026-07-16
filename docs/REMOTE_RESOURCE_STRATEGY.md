# Remote Resource Strategy

## 结论

选择高 CPU 配额环境，不为本项目购买 H200 或多卡。用户提供的 128 逻辑 CPU、1.5 TiB RAM、379 GiB 可用磁盘环境是合适起点。先用 `--workers 16`；确认 `/sys/fs/cgroup/cpu.max` 确实允许更多 CPU 后，才逐步提高到 32 或 64。

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
- formal suite 预计包含大量 CSV，至少预留 100 GiB；脚本每次启动打印 `df -h .`。
- 没有外部数据集。clone、pip/wheel 下载完成后，smoke 和 full 运行不需要联网。
- 正式分析包排除逐 step 原始流，只包含 aggregates、figures、configs、manifests、日志摘要和 SHA-256。

## 推荐值

| 可用 CPU quota | 推荐起始 workers |
|---:|---:|
| 20 | 16 |
| 32 | 24 |
| 64 | 48 |
| 128 | 64（确认 I/O 后再提高） |

不要根据 `lscpu` 的宿主机数字直接设置 128；先看：

```bash
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
free -h
df -h .
```
