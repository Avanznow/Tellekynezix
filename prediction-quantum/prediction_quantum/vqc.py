"""The variational quantum classifier: angle embedding -> StronglyEntanglingLayers -> <Z_i> -> linear head.

    AngleEmbedding(RY)  ->  [ StronglyEntanglingLayers (1 layer) -> optional depolarising noise ] x n_layers
        ->  expval(PauliZ) on every qubit  ->  nn.Linear(n_qubits, 6)

The ansatz is applied one layer at a time with the same CNOT ranges qml.StronglyEntanglingLayers uses when all
layers are given at once (range of layer l = (l mod (n-1)) + 1), so the circuit is identical to a single
StronglyEntanglingLayers call (tested) and a noise channel can be inserted after every layer. Simulation is on
default.qubit (state vector) with the torch interface and backprop differentiation; the noise study rebuilds the
same circuit on default.mixed (density matrix) with qml.DepolarizingChannel after the embedding and after each
layer, inference only.

Circuit evaluations. qml.Tracker on the device counts "executions" per input (one circuit run per window, the
hardware-equivalent count) and "simulations" (batched state-vector simulations actually performed, thanks to
parameter broadcasting). Backprop gradients cost no extra circuit runs on a simulator; on hardware the
parameter-shift rule would need 2 extra runs per trainable circuit parameter and window, see
parameter_shift_evaluations().
"""
import pennylane as qml
import torch
from torch import nn

N_CLASSES = 6


def entangling_ranges(n_qubits, n_layers):
    """CNOT ranges StronglyEntanglingLayers uses by default when all layers are applied in one call."""
    if n_qubits < 2:
        return [0] * n_layers
    return [(layer % (n_qubits - 1)) + 1 for layer in range(n_layers)]


def build_circuit(n_qubits, n_layers, device_name="default.qubit", noise_p=0.0, interface="torch", diff_method="backprop"):
    """QNode (inputs (batch, n_qubits), weights (n_layers, n_qubits, 3)) -> list of n_qubits <Z> expectations."""
    dev = qml.device(device_name, wires=n_qubits)
    ranges = entangling_ranges(n_qubits, n_layers)
    noise_p = float(noise_p)
    if noise_p > 0 and device_name == "default.qubit":
        raise ValueError("default.qubit cannot simulate channels; use default.mixed for noise_p > 0")

    def noise():
        if noise_p > 0:
            for w in range(n_qubits):
                qml.DepolarizingChannel(noise_p, wires=w)

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        noise()
        for layer in range(n_layers):
            qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(n_qubits), ranges=[ranges[layer]])
            noise()
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


class VQCClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes=N_CLASSES, seed=0, device_name="default.qubit", noise_p=0.0):
        super().__init__()
        self.n_qubits, self.n_layers, self.n_classes = int(n_qubits), int(n_layers), int(n_classes)
        self.device_name, self.noise_p = device_name, float(noise_p)
        self.circuit = build_circuit(self.n_qubits, self.n_layers, device_name, noise_p)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        torch.manual_seed(seed)
        self.qlayer = qml.qnn.TorchLayer(self.circuit, {"weights": shape})   # uniform(0, 2 pi) init
        self.head = nn.Linear(self.n_qubits, self.n_classes)
        self.windows_evaluated = 0

    def expectations(self, x):
        """(batch, n_qubits) angles -> (batch, n_qubits) Pauli-Z expectations."""
        if x.shape[-1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} angles per window, got {tuple(x.shape)}")
        self.windows_evaluated += int(x.shape[0]) if x.dim() > 1 else 1
        return self.qlayer(x)

    def forward(self, x):
        return self.head(self.expectations(x))

    @property
    def quantum_device(self):
        return self.circuit.device

    def tracker(self):
        return qml.Tracker(self.quantum_device)

    def n_quantum_parameters(self):
        return int(self.qlayer.weights.numel())

    def n_parameters(self):
        return int(sum(p.numel() for p in self.parameters()))

    def config(self):
        return dict(n_qubits=self.n_qubits, n_layers=self.n_layers, n_classes=self.n_classes, device=self.device_name,
                    noise_p=self.noise_p, n_quantum_parameters=self.n_quantum_parameters(), n_parameters=self.n_parameters())


def parameter_shift_evaluations(n_windows, n_quantum_parameters):
    """Circuit runs a parameter-shift gradient would need on hardware for n_windows: one forward plus two per
    trainable circuit parameter (shots not counted; every expectation would need ~1e3 shots on a real device)."""
    return int(n_windows) * (1 + 2 * int(n_quantum_parameters))
