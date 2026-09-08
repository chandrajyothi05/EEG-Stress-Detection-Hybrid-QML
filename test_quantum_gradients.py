import torch
import torch.nn as nn

from models.quantum_head import QuantumStressHead

torch.manual_seed(42)

model = QuantumStressHead(in_dim=128)
model.train()

# Simulate the 128-d output from Bi-LSTM + Attention
pooled = torch.randn(4, 128, requires_grad=True)

labels = torch.randint(0, 2, (4,))

logits, quantum_out, latent = model(pooled)

loss = nn.CrossEntropyLoss()(logits, labels)

print("Logits:", logits.shape)
print("Quantum output:", quantum_out.shape)
print("Latent:", latent.shape)
print("Loss:", loss.item())

loss.backward()

print("\nGradient check:")

for name, p in model.quantum_layer.named_parameters():
    print(
        name,
        "grad_exists =", p.grad is not None,
        "grad_mean =", (
            p.grad.abs().mean().item()
            if p.grad is not None else None
        )
    )