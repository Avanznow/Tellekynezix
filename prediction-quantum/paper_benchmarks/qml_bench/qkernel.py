"""Quantum kernel support-vector classifier (fidelity kernel), simulated exactly.

Feature map (Havlicek et al., 2019, "ZZ" map on a ring, `reps` repetitions), for angles x in [0, pi]^n:
    U(x) = [ ZZ(x) . RZ(2x) . H^{(x)n} ]^reps,   ZZ(x) = prod_{(i,j) in ring} exp(-i (pi - x_i)(pi - x_j) Z_i Z_j)
Kernel:  k(x, x') = |<0| U(x)^dagger U(x') |0>|^2 = |<psi(x)|psi(x')>|^2
The statevector of every window is simulated once with PennyLane (default.qubit, parameter broadcasting); the
Gram matrices are then exact inner products. On hardware each kernel entry would be estimated from repeated runs
of U(x)^dagger U(x') (counted in `hardware_kernel_evaluations`). The SVC uses the precomputed kernel, C = 1.
"""
import time

import numpy as np
import pennylane as qml
from sklearn.svm import SVC


def ring_pairs(n):
    """Nearest-neighbour (i, i+1 mod n) qubit pairs around a ring; with two qubits the ring would list the same
    edge twice, so that case gets the single pair (0, 1)."""
    return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]


def feature_map(x, n, reps=2):
    """Queue the ZZ feature map from the module docstring on the active tape.

    x is one angle vector of shape (n,) or a (batch, n) array; the `...` index keeps the batch axis so PennyLane
    can broadcast every window through the same circuit.
    """
    for _ in range(reps):
        for w in range(n):
            qml.Hadamard(wires=w)
            qml.RZ(2.0 * x[..., w], wires=w)
        for i, j in ring_pairs(n):
            # CNOT-RZ-CNOT realises exp(-i phi Z_i Z_j); the RZ angle is doubled because RZ(t) = exp(-i t Z / 2).
            qml.CNOT(wires=[i, j])
            qml.RZ(2.0 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
            qml.CNOT(wires=[i, j])


def make_state_fn(n, reps=2):
    """QNode returning the 2**n statevector |psi(x)> of one window, or of a broadcast batch of windows."""
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="numpy")
    def state(x):
        feature_map(x, n, reps)
        return qml.state()
    return state


def states(X, n, reps=2, batch=1024):
    """Simulate |psi(x)> for every row of X (angles, shape (N, n)); returns a complex (N, 2**n) array.

    Windows go through the QNode in chunks of `batch` so parameter broadcasting does not blow up memory.
    """
    fn = make_state_fn(n, reps)
    out = [np.asarray(fn(np.asarray(X[i:i + batch], np.float64))) for i in range(0, len(X), batch)]
    # A 1-D input makes the state function return a bare (2**n,) vector; restore the batch axis so chunks stack.
    S = np.concatenate([o if o.ndim == 2 else o[None] for o in out])
    return S


def gram(SA, SB):
    """Fidelity kernel |<psi_a|psi_b>|^2 between two statevector arrays, as a real (len(SA), len(SB)) matrix."""
    return np.abs(SA.conj() @ SB.T) ** 2


class QuantumKernelSVC:
    """SVC on the exact fidelity kernel; `n_qubits` must equal the number of angle features per window.

    Training needs the full N x N Gram matrix, which is why `models.run_model` caps the training set
    (`qsvm.max_train`). `seed` is only stored for interface parity with the other models: the simulation is exact,
    so there is nothing random here.
    """
    def __init__(self, n_qubits, reps=2, C=1.0, seed=0):
        self.n, self.reps, self.C, self.seed = int(n_qubits), int(reps), float(C), seed

    def fit(self, A, y):
        """Simulate the training states once, build the N x N Gram matrix and fit the precomputed-kernel SVC.

        A is (N, n_qubits) angles; the training states are kept because `predict` needs the test-vs-train kernel.
        """
        t0 = time.perf_counter()
        self.S_train_ = states(A, self.n, self.reps)
        K = gram(self.S_train_, self.S_train_)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        self.fit_seconds_ = time.perf_counter() - t0
        self.n_train_ = len(y)
        return self

    def predict(self, A):
        """Class labels for new windows from their (len(A), n_train) kernel against the stored training states."""
        S = states(A, self.n, self.reps)
        return self.svc_.predict(gram(S, self.S_train_))

    def hardware_kernel_evaluations(self, n_test):
        """Distinct kernel entries a device would estimate (training upper triangle + test-vs-train), before shots."""
        n = self.n_train_
        return int(n * (n - 1) // 2 + n_test * n)

    def n_parameters(self):
        """Trained parameters: dual coefficients and intercepts of the SVC (the feature map has none)."""
        return int(self.svc_.dual_coef_.size + self.svc_.intercept_.size)
