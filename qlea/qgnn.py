"""Q-CABS quantum graph residual networks.

Images and time series are converted to a small graph of learned local regions
or temporal segments.  A classical message-passing encoder supplies compact
node states; a local quantum Ising layer encodes graph-edge correlations with
data-dependent RZZ interactions.  The matched classical control replaces only
the quantum edge layer.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    import pennylane as qml
except ImportError as error:  # pragma: no cover
    raise ImportError("Q-CABS QGNN requires PennyLane.") from error


class RingMessagePassing(nn.Module):
    def __init__(self, node_dim: int) -> None:
        super().__init__()
        self.self_map = nn.Linear(node_dim, node_dim)
        self.neighbor_map = nn.Linear(node_dim, node_dim, bias=False)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        # Ring adjacency gives a common local graph for image regions and
        # temporal windows while retaining a constant NISQ qubit topology.
        neighbors = 0.5 * (torch.roll(nodes, 1, dims=1) + torch.roll(nodes, -1, dims=1))
        return self.norm(nodes + torch.nn.functional.gelu(self.self_map(nodes) + self.neighbor_map(neighbors)))


class QuantumGraphEdgeLayer(nn.Module):
    def __init__(self, n_nodes: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        device = qml.device("default.qubit", wires=n_nodes, shots=None)
        self.weights = nn.Parameter(0.05 * torch.randn(n_layers, n_nodes, 3))

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(angles, weights):
            for node in range(n_nodes):
                qml.RY(angles[node], wires=node)
                qml.RZ(angles[node] * angles[node] / torch.pi, wires=node)
            # Each graph edge is an input-dependent Ising Hamiltonian term.
            for node in range(n_nodes):
                right = (node + 1) % n_nodes
                qml.IsingZZ(angles[node] * angles[right], wires=[node, right])
            for layer in range(n_layers):
                for node in range(n_nodes):
                    qml.Rot(*weights[layer, node], wires=node)
                for node in range(n_nodes):
                    qml.CNOT(wires=[node, (node + 1) % n_nodes])
            return tuple(
                [qml.expval(qml.PauliZ(node)) for node in range(n_nodes)]
                + [qml.expval(qml.PauliZ(node) @ qml.PauliZ((node + 1) % n_nodes)) for node in range(n_nodes)]
            )

        self.circuit = circuit

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim == 1:
            return torch.stack(self.circuit(angles, self.weights)).reshape(-1)
        return torch.stack([torch.stack(self.circuit(row, self.weights)).reshape(-1) for row in angles], dim=0)


class QCABSQuantumGraphClassifier(nn.Module):
    """High-capacity classical graph encoder with a quantum graph residual."""

    def __init__(self, input_dim: int, n_classes: int, *, n_nodes: int = 4, node_dim: int = 48, quantum_layers: int = 2) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.node_embed = nn.Sequential(nn.Linear(input_dim, n_nodes * node_dim), nn.GELU())
        self.message_1 = RingMessagePassing(node_dim)
        self.message_2 = RingMessagePassing(node_dim)
        self.to_angles = nn.Sequential(nn.Linear(node_dim, 1), nn.Tanh())
        self.quantum_edges = QuantumGraphEdgeLayer(n_nodes=n_nodes, n_layers=quantum_layers)
        self.head = nn.Sequential(
            nn.Linear(node_dim + 2 * n_nodes, node_dim), nn.GELU(), nn.Dropout(0.10), nn.Linear(node_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = self.node_embed(x).reshape(x.shape[0], self.n_nodes, -1)
        nodes = self.message_2(self.message_1(nodes))
        angles = torch.pi * self.to_angles(nodes).squeeze(-1)
        quantum = self.quantum_edges(angles).to(nodes.dtype)
        return self.head(torch.cat([nodes.mean(dim=1), quantum], dim=-1))


class QCABSClassicalGraphClassifier(nn.Module):
    """Matched graph control with the quantum residual replaced by a small MLP."""

    def __init__(self, input_dim: int, n_classes: int, *, n_nodes: int = 4, node_dim: int = 48) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.node_embed = nn.Sequential(nn.Linear(input_dim, n_nodes * node_dim), nn.GELU())
        self.message_1 = RingMessagePassing(node_dim)
        self.message_2 = RingMessagePassing(node_dim)
        self.to_angles = nn.Sequential(nn.Linear(node_dim, 1), nn.Tanh())
        self.classical_edges = nn.Sequential(nn.Linear(n_nodes, 2 * n_nodes), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(node_dim + 2 * n_nodes, node_dim), nn.GELU(), nn.Dropout(0.10), nn.Linear(node_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = self.node_embed(x).reshape(x.shape[0], self.n_nodes, -1)
        nodes = self.message_2(self.message_1(nodes))
        angles = torch.pi * self.to_angles(nodes).squeeze(-1)
        return self.head(torch.cat([nodes.mean(dim=1), self.classical_edges(angles)], dim=-1))
