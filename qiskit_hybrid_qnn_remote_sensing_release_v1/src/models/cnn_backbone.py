"""Lightweight CNN backbone used by recognition mainline experiments.

The classical model is intentionally compact because the dataset consists of
cropped ROIs rather than full high-resolution remote-sensing scenes. The same
backbone is also reused by the hybrid model so architectural differences remain
localized to the classification head.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn

    TORCH_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

try:
    from torchvision import models as tv_models
    from torchvision.models import ResNet18_Weights

    TORCHVISION_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    tv_models = None  # type: ignore[assignment]
    ResNet18_Weights = None  # type: ignore[assignment]
    TORCHVISION_IMPORT_ERROR = exc


if nn is not None:

    class CNNBackbone(nn.Module):
        """Configurable feature extractor for ROI classification experiments."""

        def __init__(
            self,
            in_channels: int = 3,
            feature_dim: int = 64,
            dropout: float = 0.2,
            backbone_type: str = "small_cnn",
            pretrained_backbone: bool = False,
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.backbone_type = str(backbone_type).strip().lower()

            if self.backbone_type == "small_cnn":
                # The original compact CNN remains the default because it is
                # fast and keeps existing experiments backward-compatible.
                self.features = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                projection_input_dim = 64
            elif self.backbone_type == "resnet18":
                if tv_models is None or ResNet18_Weights is None:
                    raise ImportError(
                        "torchvision is required for backbone_type='resnet18'."
                    ) from TORCHVISION_IMPORT_ERROR
                if int(in_channels) != 3:
                    raise ValueError("backbone_type='resnet18' requires in_channels=3.")
                weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
                backbone = tv_models.resnet18(weights=weights)
                self.features = nn.Sequential(*list(backbone.children())[:-1])
                projection_input_dim = 512
            else:
                raise ValueError(
                    "backbone_type must be either 'small_cnn' or 'resnet18'."
                )

            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(projection_input_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        def forward(self, inputs):
            x = self.features(inputs)
            return self.projection(x)


    class CNNClassifier(nn.Module):
        """Baseline CNN classifier with a linear head."""

        def __init__(
            self,
            in_channels: int = 3,
            feature_dim: int = 64,
            num_classes: int = 2,
            dropout: float = 0.2,
            backbone_type: str = "small_cnn",
            pretrained_backbone: bool = False,
        ) -> None:
            super().__init__()
            self.backbone = CNNBackbone(
                in_channels=in_channels,
                feature_dim=feature_dim,
                dropout=dropout,
                backbone_type=backbone_type,
                pretrained_backbone=pretrained_backbone,
            )
            self.classifier = nn.Linear(feature_dim, num_classes)

        def forward(self, inputs):
            features = self.backbone(inputs)
            return self.classifier(features)

else:

    class CNNBackbone:  # type: ignore[no-redef]
        """Import-time placeholder when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for CNNBackbone.") from TORCH_IMPORT_ERROR


    class CNNClassifier:  # type: ignore[no-redef]
        """Import-time placeholder when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for CNNClassifier.") from TORCH_IMPORT_ERROR
