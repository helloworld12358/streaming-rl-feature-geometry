# 依赖安装指南

> 当前云端固定使用现有 `/usr/bin/python3`，pip/cache/temp 均位于仓库内；完整生产命令见 [`PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md`](PRODUCTION_ROOT_CAUSE_FIX_AND_CLOUD_RUN_ZH.md)。

## 三个依赖入口分别是什么

- `pyproject.toml`：项目的主依赖与打包定义。`pip install -e .` 会按它安装当前源码包。
- `requirements.txt`：服务器运行依赖快捷入口，仅含 NumPy、pandas、Matplotlib。
- `requirements-dev.txt`：先引用运行依赖，再加入 pytest，用于 tests、smoke 和部署验收。

当前源码没有使用 SciPy，也不需要 torch、TensorFlow、JAX、CUDA 包、Ray、W&B 或 MLflow。项目没有外部数据集、预训练模型或模型权重。

## 发布前 clean-install 验证

每次部署变更在新的临时 venv 中验证 requirements、editable install、`pip check` 和 pytest；这个 venv 只用于证明依赖声明完整，不改变目标云端正式运行必须使用 `/usr/bin/python3` 的约束。Linux 示例：

```bash
python3 -m venv .runtime/release-clean-env
.runtime/release-clean-env/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.runtime/release-clean-env/bin/python -m pip install --no-build-isolation --no-deps -e .
.runtime/release-clean-env/bin/python -m pip check
.runtime/release-clean-env/bin/python -m pytest -q
```

`.runtime/` 已被 Git 忽略，不提交环境或 cache。

## 联网 Linux 服务器

在仓库根目录执行：

```bash
bash scripts/bootstrap_remote.sh --workers <WORKERS> --run-name bootstrap-<COMMIT_SHA>
```

脚本不会创建 venv 或 Conda，也不会切换 Python；它按顺序执行：

```bash
/usr/bin/python3 -m pip install -r requirements.txt -r requirements-dev.txt
/usr/bin/python3 -m pip install --no-build-isolation --no-deps -e .
/usr/bin/python3 -m pip check
/usr/bin/python3 -m pytest -q
```

随后运行 `configs/cross_extension_smoke.json` 的 70-run 生产 smoke。若共享系统 Python 不允许安装，应在平台提供的可写 CPU 环境完成安装；不要临时改用无法在正式节点复现的解释器。

## GPU 计算节点不能联网时

本项目应优先直接在 CPU 环境运行。如果必须把代码送到无网络计算节点，先在同操作系统、同 Python 主次版本且可联网的 Linux 节点下载 wheel：

```bash
mkdir -p wheelhouse
python3 -m pip download -r requirements-dev.txt -d wheelhouse
python3 -m pip download 'setuptools>=68' wheel -d wheelhouse
```

将仓库和 `wheelhouse/` 放在计算节点可见的共享存储，再执行：

```bash
bash scripts/bootstrap_remote.sh \
  --wheelhouse "$PWD/wheelhouse" \
  --workers <WORKERS> \
  --run-name bootstrap-<COMMIT_SHA>
```

完整 one-click 入口也接受 `--wheelhouse "$PWD/wheelhouse"`，并把它原样交给 bootstrap；无需先联网安装后再启动正式流程。

此路径使用 `--no-index --find-links`，运行时不会访问网络。wheelhouse 不提交 Git。

## 为什么没有平台绑定 lock

`pyproject.toml` 与 requirements 使用受控版本范围，正式 run manifest 保存实际 Python 和依赖版本。直接提交由 Windows 生成的精确 freeze 会把 Windows wheel 选择误当成 Linux 可复现环境，因此本仓库没有添加平台绑定 `requirements-lock.txt`。需要归档服务器精确环境时执行：

```bash
/usr/bin/python3 -m pip freeze > "results/remote/<RUN_NAME>/pip-freeze.txt"
```

该文件随结果保存，不作为跨平台主依赖源。
