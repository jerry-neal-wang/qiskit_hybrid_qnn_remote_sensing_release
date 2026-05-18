"""TorchConnector-based QNN classification head.

This module isolates the dependency-heavy part of the hybrid model. The rest of
the repository can therefore import classical components without requiring
``qiskit-machine-learning`` to be installed.
"""

from __future__ import annotations

import importlib.util
import math

try:
    import torch
    from torch import nn

    TORCH_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    QNN_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    QuantumCircuit = None  # type: ignore[assignment]
    ParameterVector = None  # type: ignore[assignment]
    RealAmplitudes = None  # type: ignore[assignment]
    ZZFeatureMap = None  # type: ignore[assignment]
    SparsePauliOp = None  # type: ignore[assignment]
    TorchConnector = None  # type: ignore[assignment]
    EstimatorQNN = None  # type: ignore[assignment]
    QNN_IMPORT_ERROR = exc


def qnn_dependencies_available() -> bool:
    """Return whether torch and qiskit-machine-learning are both available."""
    return TORCH_IMPORT_ERROR is None and QNN_IMPORT_ERROR is None


def get_qnn_dependency_report() -> dict[str, object]:
    """Describe the local Hybrid QNN dependency state for CLI error messages."""
    qiskit_ml_spec = importlib.util.find_spec("qiskit_machine_learning")
    missing_packages: list[str] = []
    missing_symbols: list[str] = []

    if TORCH_IMPORT_ERROR is not None:
        missing_packages.append("torch")
    if qiskit_ml_spec is None:
        missing_packages.append("qiskit-machine-learning")
    if qiskit_ml_spec is not None and TorchConnector is None:
        missing_symbols.append("TorchConnector")
    if qiskit_ml_spec is not None and EstimatorQNN is None:
        missing_symbols.append("EstimatorQNN")

    # The CLI uses these canned commands directly in error messages so users can
    # quickly recover a broken Hybrid environment.
    install_commands: list[str] = []
    if "torch" in missing_packages:
        install_commands.append("python3 -m pip install torch")
    if "qiskit-machine-learning" in missing_packages:
        install_commands.append("python3 -m pip install qiskit-machine-learning")
        install_commands.append("python3 -m pip install -r requirements.txt")
    if not install_commands:
        install_commands.append("python3 -m pip install -r requirements.txt")

    return {
        "torch_available": TORCH_IMPORT_ERROR is None,
        "qiskit_machine_learning_available": qiskit_ml_spec is not None,
        "torchconnector_available": TorchConnector is not None,
        "estimator_qnn_available": EstimatorQNN is not None,
        "missing_packages": missing_packages,
        "missing_symbols": missing_symbols,
        "import_error": None if QNN_IMPORT_ERROR is None else f"{type(QNN_IMPORT_ERROR).__name__}: {QNN_IMPORT_ERROR}",
        "install_commands": install_commands,
    }


if nn is not None:

    def _build_pauli_label(
        num_qubits: int,
        active_indices: tuple[int, ...],
    ) -> str:
        return "".join(
            "Z" if qubit_index in active_indices else "I"
            for qubit_index in range(num_qubits)
        )


    def _build_observables(
        *,
        num_qubits: int,
        qnn_output_dim: int,
        observable_mode: str,
        max_zz_observables: int | None,
    ) -> list:
        """Build a compact observable bank for EstimatorQNN outputs."""
        observable_mode = str(observable_mode).strip().lower()
        if qnn_output_dim < 1:
            raise ValueError("qnn_output_dim must be at least 1.")

        z_observables = [
            SparsePauliOp.from_list([(_build_pauli_label(num_qubits, (qubit_index,)), 1.0)])
            for qubit_index in range(num_qubits)
        ]
        zz_candidates = [
            SparsePauliOp.from_list([(_build_pauli_label(num_qubits, (left, right)), 1.0)])
            for left in range(num_qubits)
            for right in range(left + 1, num_qubits)
        ]

        if observable_mode == "auto":
            if qnn_output_dim == 1:
                observable_mode = "single_z"
            elif qnn_output_dim <= num_qubits:
                observable_mode = "multi_z"
            else:
                observable_mode = "z_plus_zz"

        if observable_mode == "single_z":
            return [z_observables[0]]

        if observable_mode == "multi_z":
            if qnn_output_dim > len(z_observables):
                raise ValueError(
                    "multi_z observable mode cannot exceed the number of qubits. "
                    f"Got qnn_output_dim={qnn_output_dim}, num_qubits={num_qubits}."
                )
            return z_observables[:qnn_output_dim]

        if observable_mode != "z_plus_zz":
            raise ValueError(
                "Unsupported observable_mode. Expected one of: auto, single_z, multi_z, z_plus_zz."
            )

        observables = list(z_observables[: min(qnn_output_dim, len(z_observables))])
        remaining = qnn_output_dim - len(observables)
        if remaining <= 0:
            return observables

        zz_limit = remaining if max_zz_observables is None else min(remaining, max_zz_observables)
        observables.extend(zz_candidates[:zz_limit])
        if len(observables) < qnn_output_dim:
            raise ValueError(
                "Unable to build enough observables for the requested qnn_output_dim. "
                f"Requested {qnn_output_dim}, built {len(observables)}."
            )
        return observables


    def _apply_ring_entanglement(circuit, num_qubits: int) -> None:
        """Apply a simple nearest-neighbor ring entanglement pattern."""
        if num_qubits <= 1:
            return
        for qubit_index in range(num_qubits - 1):
            circuit.cx(qubit_index, qubit_index + 1)
        circuit.cx(num_qubits - 1, 0)


    def _build_reupload_circuit(
        *,
        num_qubits: int,
        feature_map_reps: int,
        ansatz_reps: int,
        with_entanglement: bool,
    ):
        """Build a lightweight data-reupload circuit with optional entanglement."""
        circuit = QuantumCircuit(num_qubits)
        input_params = ParameterVector("x", length=num_qubits)
        weight_params = ParameterVector("theta", length=max(1, ansatz_reps) * num_qubits)

        for _ in range(max(1, feature_map_reps)):
            for qubit_index in range(num_qubits):
                circuit.ry(input_params[qubit_index], qubit_index)
            if with_entanglement:
                _apply_ring_entanglement(circuit, num_qubits)

        weight_index = 0
        for _ in range(max(1, ansatz_reps)):
            for qubit_index in range(num_qubits):
                circuit.ry(weight_params[weight_index], qubit_index)
                weight_index += 1
            if with_entanglement:
                _apply_ring_entanglement(circuit, num_qubits)

        return circuit, list(input_params), list(weight_params)


    def _build_qnn_circuit(
        *,
        circuit_type: str,
        num_qubits: int,
        feature_map_reps: int,
        ansatz_reps: int,
    ):
        """Build the selected QNN circuit and return its input and weight params."""
        normalized_type = str(circuit_type).strip().lower()

        if normalized_type == "zz_real_amplitudes":
            feature_map = ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=feature_map_reps,
            )
            ansatz = RealAmplitudes(
                num_qubits=num_qubits,
                reps=ansatz_reps,
            )
            circuit = QuantumCircuit(num_qubits)
            circuit.compose(feature_map, inplace=True)
            circuit.compose(ansatz, inplace=True)
            return circuit, list(feature_map.parameters), list(ansatz.parameters)

        if normalized_type == "no_entanglement":
            return _build_reupload_circuit(
                num_qubits=num_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
                with_entanglement=False,
            )

        if normalized_type == "ry_ring":
            return _build_reupload_circuit(
                num_qubits=num_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
                with_entanglement=True,
            )

        raise ValueError(
            "Unsupported circuit_type. Expected one of: "
            "zz_real_amplitudes, no_entanglement, ry_ring."
        )

    class QNNHead(nn.Module):
        """TorchConnector-backed QNN feature extractor."""

        def __init__(
            self,
            input_dim: int | None = None,
            num_qubits: int = 4,
            qnn_output_dim: int = 4,
            feature_map_reps: int = 1,
            ansatz_reps: int = 1,
            circuit_type: str = "zz_real_amplitudes",
            observable_mode: str = "auto",
            max_zz_observables: int | None = None,
            input_scale: float = math.pi / 2.0,
        ) -> None:
            if TORCH_IMPORT_ERROR is not None:
                raise ImportError("PyTorch is required for QNNHead.") from TORCH_IMPORT_ERROR
            if QNN_IMPORT_ERROR is not None:
                raise ImportError(
                    "qiskit-machine-learning is required for QNNHead."
                ) from QNN_IMPORT_ERROR

            super().__init__()
            self.num_qubits = int(num_qubits)
            self.input_dim = self.num_qubits if input_dim is None else int(input_dim)
            self.qnn_output_dim = int(qnn_output_dim)
            self.feature_map_reps = int(feature_map_reps)
            self.ansatz_reps = int(ansatz_reps)
            self.circuit_type = str(circuit_type).strip().lower()
            self.observable_mode = str(observable_mode).strip().lower()
            self.max_zz_observables = max_zz_observables
            self.input_scale = float(input_scale)
            self.input_projection = (
                nn.Identity()
                if self.input_dim == self.num_qubits
                else nn.Linear(self.input_dim, self.num_qubits)
            )

            circuit, input_params, weight_params = _build_qnn_circuit(
                circuit_type=self.circuit_type,
                num_qubits=self.num_qubits,
                feature_map_reps=self.feature_map_reps,
                ansatz_reps=self.ansatz_reps,
            )
            observables = _build_observables(
                num_qubits=self.num_qubits,
                qnn_output_dim=self.qnn_output_dim,
                observable_mode=self.observable_mode,
                max_zz_observables=self.max_zz_observables,
            )
            estimator_qnn = EstimatorQNN(
                circuit=circuit,
                observables=observables,
                input_params=input_params,
                weight_params=weight_params,
                input_gradients=True,
            )

            self.quantum_layer = TorchConnector(estimator_qnn)

        def forward(self, features):
            qnn_inputs = self.input_projection(features)
            qnn_inputs = torch.tanh(qnn_inputs) * self.input_scale
            qnn_outputs = self.quantum_layer(qnn_inputs)
            batch_size = int(qnn_inputs.shape[0]) if qnn_inputs.ndim > 1 else 1
            return qnn_outputs.reshape(batch_size, self.qnn_output_dim)

else:

    class QNNHead:  # type: ignore[no-redef]
        """Import-time placeholder when required dependencies are missing."""

        def __init__(self, *args, **kwargs) -> None:
            if TORCH_IMPORT_ERROR is not None:
                raise ImportError("PyTorch is required for QNNHead.") from TORCH_IMPORT_ERROR
            raise ImportError(
                "qiskit-machine-learning is required for QNNHead."
            ) from QNN_IMPORT_ERROR
