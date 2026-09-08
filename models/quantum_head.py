"""
models/quantum_head.py (PennyLane version)

VQC stage, following "Quantum Stress Monitoring" Section II-D / Fig. 5(b):
  - 2 qubits
  - Quantum Projection: Linear(128, 2)
  - Feature map using H, RZ, RY and ZZ-style entanglement
  - RealAmplitudes-equivalent ansatz with reps=1
  - 4 trainable ansatz parameters
  - Pauli-Z expectation measurements
  - Post-quantum classifier

NOTE ON BATCHING:
PennyLane's TorchLayer automatic batching does not reliably handle QNodes
that return multiple measurements in this setup. Therefore, forward()
processes each sample individually and stacks the results.
"""

import torch
import torch.nn as nn
import pennylane as qml


N_QUBITS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, theta):

    # Feature map
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    qml.RZ(2 * inputs[0], wires=0)
    qml.RY(2 * inputs[0], wires=0)

    qml.RZ(2 * inputs[1], wires=1)
    qml.RY(2 * inputs[1], wires=1)

    phi_01 = (torch.pi - inputs[0]) * (torch.pi - inputs[1])

    qml.CNOT(wires=[0, 1])
    qml.RZ(2 * phi_01, wires=1)
    qml.CNOT(wires=[0, 1])

    # RealAmplitudes-equivalent ansatz
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)

    qml.CNOT(wires=[0, 1])

    qml.RY(theta[2], wires=0)
    qml.RY(theta[3], wires=1)

    # Measurements
    return [
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
    ]


def build_quantum_layer() -> qml.qnn.TorchLayer:
    weight_shapes = {"theta": 4}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumStressHead(nn.Module):

    def __init__(
        self,
        in_dim: int = 128,
        post_quantum_dim: int = 16,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Classical projection: 128-dim deep-learning feature -> 2 quantum inputs
        self.quantum_projection = nn.Linear(in_dim, N_QUBITS)

        # Quantum layer
        self.quantum_layer = build_quantum_layer()

        # Classical post-quantum processing
        self.post_quantum = nn.Sequential(
            nn.Linear(N_QUBITS, post_quantum_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final classifier
        self.classifier = nn.Linear(post_quantum_dim, n_classes)

    def forward(self, pooled: torch.Tensor):
        # pooled: (B, 128)
        latent = self.quantum_projection(pooled)

        # Restrict quantum inputs to [0, pi]
        latent = torch.pi * torch.sigmoid(latent)

        # Explicit per-sample quantum processing
        # (TorchLayer batching is unreliable with multi-measurement QNodes)
        quantum_outs = []
        for i in range(latent.shape[0]):
            quantum_outs.append(self.quantum_layer(latent[i]))

        quantum_out = torch.stack(quantum_outs, dim=0)  # (B, 2)

        # Post-quantum processing
        post = self.post_quantum(quantum_out)

        # Final classification
        logits = self.classifier(post)

        return logits, quantum_out, latent


if __name__ == "__main__":
    model = QuantumStressHead()

    dummy = torch.randn(4, 128)

    logits, quantum_out, latent = model(dummy)

    print(f"Logits shape: {logits.shape}")
    print(f"Quantum output shape: {quantum_out.shape}")
    print(f"Latent (pre-quantum) shape: {latent.shape}")
    print(f"Latent range: [{latent.min().item():.3f}, {latent.max().item():.3f}]")

    n_ansatz_params = sum(p.numel() for p in model.quantum_layer.parameters())
    print(f"Trainable ansatz parameters: {n_ansatz_params}")
