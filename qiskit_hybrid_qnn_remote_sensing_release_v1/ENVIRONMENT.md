# Environment

## 目标

当前仓库统一为**一个虚拟环境**，避免旧项目阶段的多环境冲突。

- 唯一环境目录：`.venv`
- 唯一依赖文件：`requirements.txt`

## 初始化

```bash
bash scripts/bootstrap_env.sh
```

或手动：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 核验

```bash
.venv/bin/python - <<'PY'
import torch, qiskit, qiskit_aer, qiskit_machine_learning
from qiskit_machine_learning.connectors import TorchConnector
print("ok")
PY
```

## 使用约定

- 运行所有脚本时，优先使用 `.venv/bin/python`
- 不再使用 `venv/`、`.venvs/`、`requirements-hybrid.txt`、`requirements-dev.txt`

