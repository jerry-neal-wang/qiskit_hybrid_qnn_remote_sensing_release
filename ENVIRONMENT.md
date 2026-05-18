# Environment

Recommended interpreter: Python 3.11.

The current project uses one virtual environment and one requirements file:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The helper script runs the same setup and checks the key Qiskit imports:

```bash
bash scripts/bootstrap_env.sh
```

Expected core packages are pinned in `requirements.txt`, including PyTorch, Qiskit, Qiskit Aer, Qiskit Machine Learning, NumPy, SciPy, pandas, matplotlib, OpenCV, scikit-learn, and pytest.

All commands in `REPRODUCIBILITY.md` assume they are run from the release root with `.venv/bin/python`.
