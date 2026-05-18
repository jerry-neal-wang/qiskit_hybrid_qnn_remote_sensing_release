#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

pick_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  echo "Error: neither python3.11 nor python3 is available." >&2
  return 1
}

PYTHON_BIN="$(pick_python)"

echo "Bootstrapping unified environment"
echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Venv: $VENV_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Running import check"
"$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

required = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "cv2",
    "sklearn",
    "torch",
    "qiskit",
    "qiskit_aer",
    "qiskit_machine_learning",
    "pytest",
]
missing = []

for name in required:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        print(f"[ok] {name} version={version}")
    except Exception as exc:
        missing.append(name)
        print(f"[missing] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

try:
    from qiskit_machine_learning.connectors import TorchConnector  # noqa: F401
    print("[ok] TorchConnector import")
except Exception as exc:
    missing.append("TorchConnector")
    print(f"[missing] TorchConnector: {type(exc).__name__}: {exc}", file=sys.stderr)

if missing:
    raise SystemExit(f"Environment import check failed for: {', '.join(missing)}")
PY

echo "Unified environment ready: $VENV_DIR"
