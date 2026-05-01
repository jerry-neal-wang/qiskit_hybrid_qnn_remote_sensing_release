"""Model definitions for DIOR low-shot hybrid QNN experiments.

The exported classes deliberately mirror the three model tiers used in the
project:

- ``CNNBackbone``: reusable convolutional feature extractor
- ``CNNClassifier``: fully classical baseline
- ``HybridFusionQNNClassifier`` / ``HybridQNNClassifier``: fusion-style CNN + QNN hybrid
- ``HybridFusionControlClassifier`` / ``HybridControlClassifier``: matched classical control branch
- ``QNNHead``: isolated quantum layer building block
"""

from .cnn_backbone import CNNBackbone, CNNClassifier
from .hybrid_model import (
    HybridControlClassifier,
    HybridFusionControlClassifier,
    HybridFusionQNNClassifier,
    HybridQNNClassifier,
)
from .qnn_head import QNNHead, qnn_dependencies_available

__all__ = [
    "CNNBackbone",
    "CNNClassifier",
    "HybridControlClassifier",
    "HybridFusionControlClassifier",
    "HybridFusionQNNClassifier",
    "HybridQNNClassifier",
    "QNNHead",
    "qnn_dependencies_available",
]
