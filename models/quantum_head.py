"""
models/quantum_head.py (PennyLane version)

VQC stage, following "Quantum Stress Monitoring" Section II-D / Fig. 5(b):
  - 2 qubits (kept at 2 for tractability/interpretability per the paper's
    own justification -- our upstream feature vector is 128-dim, not 2-D,
    so "Quantum Projection" = Linear(128, 2) plays the role their CNN's
    own FC-compression layer played before their quantum layer)
  - Feature map built manually to match Fig. 5(b)'s gate sequence:
        H (both wires)
        -> RZ(2*x_j) then RY(2*x_j) per wire  [single-Pauli Z and Y terms,
           phi_{j}(x) = x_j]
        -> CNOT(0,1) -> RZ(2*phi_01) on wire 1 -> CNOT(0,1)  [the ZZ
           entangling term, phi_01(x) = (pi - x0)(pi - x1)]
    This is a from-scratch reconstruction of Qiskit's PauliFeatureMap
    (paulis=['Z','Y','ZZ'], reps=1) rather than a built-in template --
    PennyLane doesn't ship an equivalent drop-in. It reproduces the same
    gate roles and data-mapping functions described in the paper; if
    gate-level unitary equivalence to Qiskit's exact convention matters
    for your writeup, worth a footnote rather than assuming byte-identical.
  - RealAmplitudes-equivalent ansatz, reps=1: RY(theta0) wire0, RY(theta1)
    wire1, CNOT(0,1), RY(theta2) wire0, RY(theta3) wire1 -- 4 params,
    matching Fig. 5(b)'s 2-rotation-layer/1-entangler structure
  - Measurement: Pauli-Z expectation on each qubit (2 outputs)
  - Post-quantum: Linear(2, 16) -> Dropout(0.3) -> Linear(16, 2)

REQUIRES: pennylane
    pip install pennylane --break-system-packages
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS = 2
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, theta):
    # inputs: (2,) -- the two latent features x0, x1 (already bounded to [0, pi])
    # theta: (4,) -- trainable ansatz parameters

    # -- Feature map U_Phi(x), d=1, paulis=['Z','Y','ZZ'] --
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

    # -- RealAmplitudes ansatz, reps=1 --
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[2], wires=0)
    qml.RY(theta[3], wires=1)

    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]


def build_quantum_layer() -> qml.qnn.TorchLayer:
    weight_shapes = {"theta": 4}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumStressHead(nn.Module):
    def __init__(self, in_dim: int = 128, post_quantum_dim: int = 16,
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.quantum_projection = nn.Linear(in_dim, N_QUBITS)
        self.quantum_layer = build_quantum_layer()  # trainable ansatz weights live here

        self.post_quantum = nn.Sequential(
            nn.Linear(N_QUBITS, post_quantum_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(post_quantum_dim, n_classes)

    def forward(self, pooled: torch.Tensor):
        # pooled: (B, in_dim) -- frozen Bi-LSTM+attention output
        latent = self.quantum_projection(pooled)          # (B, 2)
        latent = torch.pi * torch.sigmoid(latent)          # bound to [0, pi]

        quantum_out = self.quantum_layer(latent)             # (B, 2) -- TorchLayer batches automatically

        post = self.post_quantum(quantum_out)                # (B, 16)
        logits = self.classifier(post)                        # (B, 2)
        return logits, quantum_out, latent


if __name__ == "__main__":
    model = QuantumStressHead()
    dummy = torch.randn(4, 128)
    logits, quantum_out, latent = model(dummy)
    print(f"Logits shape: {logits.shape}")              # expect (4, 2)
    print(f"Quantum output shape: {quantum_out.shape}")  # expect (4, 2)
    print(f"Latent (pre-quantum) shape: {latent.shape}") # expect (4, 2)
    print(f"Latent range: [{latent.min().item():.3f}, {latent.max().item():.3f}]")  # expect within [0, pi]
    n_ansatz_params = sum(p.numel() for p in model.quantum_layer.parameters())
    print(f"Trainable ansatz parameters: {n_ansatz_params}")  # expect 4