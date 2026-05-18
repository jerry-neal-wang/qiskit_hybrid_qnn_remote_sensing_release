"""Hybrid model combining a CNN backbone with a QNN fusion head."""

from __future__ import annotations

try:
    import torch
    from torch import nn

    TORCH_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

from .cnn_backbone import CNNBackbone
from .qnn_head import QNNHead


if nn is not None:

    class ClassicalControlHead(nn.Module):
        """Small bounded classical head used as a control for the QNN branch."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
        ) -> None:
            super().__init__()
            self.projection = nn.Linear(int(input_dim), int(output_dim))
            self.activation = nn.Tanh()

        def forward(self, inputs):
            return self.activation(self.projection(inputs))

    class HybridFusionQNNClassifier(nn.Module):
        """CNN backbone with classical and quantum branches fused at the head."""

        def __init__(
            self,
            in_channels: int = 3,
            backbone_feature_dim: int = 64,
            backbone_type: str = "small_cnn",
            pretrained_backbone: bool = False,
            num_qubits: int = 6,
            qnn_output_dim: int = 4,
            feature_map_reps: int = 1,
            ansatz_reps: int = 1,
            circuit_type: str = "zz_real_amplitudes",
            num_classes: int = 2,
            dropout: float = 0.2,
            fusion_mode: str = "concat",
            classical_head_weight: float = 1.0,
            qnn_head_weight: float = 1.0,
            observable_mode: str = "auto",
            max_zz_observables: int | None = None,
        ) -> None:
            super().__init__()
            self.backbone_feature_dim = int(backbone_feature_dim)
            self.num_qubits = int(num_qubits)
            self.qnn_output_dim = int(qnn_output_dim)
            self.num_classes = int(num_classes)
            self.fusion_mode = str(fusion_mode).strip().lower()
            self.classical_head_weight = float(classical_head_weight)
            self.qnn_head_weight = float(qnn_head_weight)
            if self.fusion_mode not in {"concat", "add"}:
                raise ValueError("fusion_mode must be either 'concat' or 'add'.")

            self.backbone = CNNBackbone(
                in_channels=in_channels,
                feature_dim=self.backbone_feature_dim,
                dropout=dropout,
                backbone_type=backbone_type,
                pretrained_backbone=pretrained_backbone,
            )
            self.classical_head = nn.Linear(self.backbone_feature_dim, self.num_classes)
            self.pre_qnn_projector = nn.Sequential(
                nn.Linear(self.backbone_feature_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, self.num_qubits),
            )
            self.qnn_head = QNNHead(
                input_dim=self.num_qubits,
                num_qubits=self.num_qubits,
                qnn_output_dim=self.qnn_output_dim,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
                circuit_type=circuit_type,
                observable_mode=observable_mode,
                max_zz_observables=max_zz_observables,
            )
            self.concat_fusion = nn.Linear(
                self.backbone_feature_dim + self.qnn_output_dim,
                self.num_classes,
            )
            self.additive_qnn_projection = nn.Linear(self.qnn_output_dim, self.num_classes)

        def encode_backbone(self, inputs):
            return self.backbone(inputs)

        def forward_with_details(self, inputs) -> dict[str, torch.Tensor]:
            features = self.encode_backbone(inputs)
            classical_logits = self.classical_head(features)
            qnn_inputs = self.pre_qnn_projector(features)
            qnn_outputs = self.qnn_head(qnn_inputs)

            if self.fusion_mode == "concat":
                fusion_inputs = torch.cat(
                    [features, self.qnn_head_weight * qnn_outputs],
                    dim=1,
                )
                fusion_logits = self.concat_fusion(fusion_inputs)
                final_logits = fusion_logits + self.classical_head_weight * classical_logits
            else:
                qnn_logits = self.additive_qnn_projection(qnn_outputs)
                fusion_logits = qnn_logits
                final_logits = (
                    self.classical_head_weight * classical_logits
                    + self.qnn_head_weight * qnn_logits
                )

            return {
                "features": features,
                "classical_logits": classical_logits,
                "qnn_inputs": qnn_inputs,
                "qnn_outputs": qnn_outputs,
                "fusion_logits": fusion_logits,
                "logits": final_logits,
            }

        def forward(self, inputs):
            return self.forward_with_details(inputs)["logits"]


    class HybridQNNClassifier(HybridFusionQNNClassifier):
        """Backward-compatible alias for the stronger fusion hybrid model."""
        pass


    class HybridFusionControlClassifier(nn.Module):
        """CNN backbone with a matched classical control branch fused at the head."""

        def __init__(
            self,
            in_channels: int = 3,
            backbone_feature_dim: int = 64,
            backbone_type: str = "small_cnn",
            pretrained_backbone: bool = False,
            branch_input_dim: int = 6,
            branch_output_dim: int = 4,
            num_classes: int = 2,
            dropout: float = 0.2,
            fusion_mode: str = "concat",
            classical_head_weight: float = 1.0,
            control_head_weight: float = 1.0,
        ) -> None:
            super().__init__()
            self.backbone_feature_dim = int(backbone_feature_dim)
            self.branch_input_dim = int(branch_input_dim)
            self.branch_output_dim = int(branch_output_dim)
            self.num_classes = int(num_classes)
            self.fusion_mode = str(fusion_mode).strip().lower()
            self.classical_head_weight = float(classical_head_weight)
            self.control_head_weight = float(control_head_weight)
            if self.fusion_mode not in {"concat", "add"}:
                raise ValueError("fusion_mode must be either 'concat' or 'add'.")

            self.backbone = CNNBackbone(
                in_channels=in_channels,
                feature_dim=self.backbone_feature_dim,
                dropout=dropout,
                backbone_type=backbone_type,
                pretrained_backbone=pretrained_backbone,
            )
            self.classical_head = nn.Linear(self.backbone_feature_dim, self.num_classes)
            self.pre_control_projector = nn.Sequential(
                nn.Linear(self.backbone_feature_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, self.branch_input_dim),
            )
            self.control_head = ClassicalControlHead(
                input_dim=self.branch_input_dim,
                output_dim=self.branch_output_dim,
            )
            self.concat_fusion = nn.Linear(
                self.backbone_feature_dim + self.branch_output_dim,
                self.num_classes,
            )
            self.additive_control_projection = nn.Linear(self.branch_output_dim, self.num_classes)

        def encode_backbone(self, inputs):
            return self.backbone(inputs)

        def forward_with_details(self, inputs) -> dict[str, torch.Tensor]:
            features = self.encode_backbone(inputs)
            classical_logits = self.classical_head(features)
            control_inputs = self.pre_control_projector(features)
            control_outputs = self.control_head(control_inputs)

            if self.fusion_mode == "concat":
                fusion_inputs = torch.cat(
                    [features, self.control_head_weight * control_outputs],
                    dim=1,
                )
                fusion_logits = self.concat_fusion(fusion_inputs)
                final_logits = fusion_logits + self.classical_head_weight * classical_logits
            else:
                control_logits = self.additive_control_projection(control_outputs)
                fusion_logits = control_logits
                final_logits = (
                    self.classical_head_weight * classical_logits
                    + self.control_head_weight * control_logits
                )

            return {
                "features": features,
                "classical_logits": classical_logits,
                "control_inputs": control_inputs,
                "control_outputs": control_outputs,
                "fusion_logits": fusion_logits,
                "logits": final_logits,
            }

        def forward(self, inputs):
            return self.forward_with_details(inputs)["logits"]


    class HybridControlClassifier(HybridFusionControlClassifier):
        """Backward-compatible alias for the classical control fusion model."""
        pass

else:

    class HybridFusionQNNClassifier:  # type: ignore[no-redef]
        """Import-time placeholder when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for HybridFusionQNNClassifier.") from TORCH_IMPORT_ERROR


    class HybridQNNClassifier(HybridFusionQNNClassifier):  # type: ignore[no-redef]
        """Import-time placeholder when torch is unavailable."""


    class HybridFusionControlClassifier:  # type: ignore[no-redef]
        """Import-time placeholder when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for HybridFusionControlClassifier.") from TORCH_IMPORT_ERROR


    class HybridControlClassifier(HybridFusionControlClassifier):  # type: ignore[no-redef]
        """Import-time placeholder when torch is unavailable."""
