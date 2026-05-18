"""Random seed helpers.

All training, inference, and ablation scripts call :func:`set_seed` so that
sample selection, NumPy operations, and Torch behavior stay aligned across
reruns. The helper intentionally centralizes deterministic flags instead of
letting each script configure them differently.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and torch seeds when available."""
    # ``PYTHONHASHSEED`` makes dict/set hashing deterministic across processes,
    # which helps keep serialized outputs and some iteration orders stable.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # For this project reproducibility matters more than squeezing out the
        # last bit of CuDNN throughput.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
