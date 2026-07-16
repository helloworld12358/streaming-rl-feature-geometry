# 依赖安装指南

## 三个依赖入口分别是什么

- `pyproject.toml`：项目的主依赖与打包定义。`pip install -e .` 会按它安装当前源码包。
- `requirements.txt`：服务器运行依赖快捷入口，仅含 NumPy、pandas、Matplotlib。
- `requirements-dev.txt`：先引用运行依赖，再加入 pytest，用于 tests、smoke 和部署验收。

当前源码没有使用 SciPy，也不需要 torch、TensorFlow、JAX、CUDA 包、Ray、W&B 或 MLflow。项目没有外部数据集、预训练模型或模型权重。

## 联网 Linux 服务器

在仓库根目录执行：

```bash
bash scripts/bootstrap_remote.sh --workers <WORKERS> --run-name bootstrap-<COMMIT_SHA>
```

脚本会创建 `.venv`，升级其中的 pip，按顺序执行：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

随后运行三个 smoke 批次。不要使用系统 Python 的 root/site-packages 代替 clean venv。

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

此路径使用 `--no-index --find-links`，运行时不会访问网络。wheelhouse 不提交 Git。

## 为什么没有平台绑定 lock

`pyproject.toml` 与 requirements 使用受控版本范围，正式 run manifest 保存实际 Python 和依赖版本。直接提交由 Windows 生成的精确 freeze 会把 Windows wheel 选择误当成 Linux 可复现环境，因此本仓库没有添加平台绑定 `requirements-lock.txt`。需要归档服务器精确环境时执行：

```bash
.venv/bin/python -m pip freeze > "results/remote/<RUN_NAME>/pip-freeze.txt"
```

该文件随结果保存，不作为跨平台主依赖源。
