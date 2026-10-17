"""Unit tests of the circuit, the encoder, the windowing and the split (no EEG data needed)."""
import numpy as np
import pennylane as qml
import pytest
import torch

from prediction_quantum import data as D
from prediction_quantum.encoding import FeatureEncoder
from prediction_quantum.noise import noisy_model
from prediction_quantum.train import train_classifier
from prediction_quantum.vqc import VQCClassifier, build_circuit, entangling_ranges, parameter_shift_evaluations


@pytest.mark.parametrize("n_qubits", [4, 6])
def test_circuit_output_dimension_equals_n_qubits(n_qubits):
    model = VQCClassifier(n_qubits, n_layers=2, seed=0)
    z = model.expectations(torch.rand(5, n_qubits) * np.pi)
    assert z.shape == (5, n_qubits)
    assert torch.all(z <= 1.0 + 1e-6) and torch.all(z >= -1.0 - 1e-6)
    assert model.windows_evaluated == 5


def test_head_output_has_six_classes():
    model = VQCClassifier(4, n_layers=1, seed=0)
    logits = model(torch.rand(3, 4))
    assert logits.shape == (3, 6)
    assert model.n_quantum_parameters() == 1 * 4 * 3
    assert model.n_parameters() == 12 + 4 * 6 + 6


def test_circuit_rejects_wrong_input_width():
    model = VQCClassifier(4, n_layers=1)
    with pytest.raises(ValueError):
        model(torch.rand(2, 5))


def test_encoder_rejects_wrong_feature_dimension():
    rng = np.random.default_rng(0)
    enc = FeatureEncoder(4).fit(rng.normal(size=(60, 570)))
    with pytest.raises(ValueError):
        enc.transform(rng.normal(size=(2, 569)))
    with pytest.raises(ValueError):
        enc.transform(rng.normal(size=(2, 571)))
    A = enc.transform(rng.normal(size=(7, 570)))
    assert A.shape == (7, 4) and A.dtype == np.float32
    assert A.min() >= 0.0 and A.max() <= np.pi + 1e-6


def test_encoder_needs_fit_first():
    with pytest.raises(RuntimeError):
        FeatureEncoder(4).transform(np.zeros((1, 570)))


def test_layered_ansatz_equals_single_strongly_entangling_call():
    n, L = 5, 4
    w = torch.rand(L, n, 3)
    x = torch.rand(3, n)
    layered = build_circuit(n, L)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch")
    def single(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    assert entangling_ranges(n, L) == [1, 2, 3, 4]
    assert torch.allclose(torch.stack(layered(x, w)), torch.stack(single(x, w)), atol=1e-6)


def test_noisy_circuit_at_p0_matches_noiseless_and_noise_shrinks_expectations():
    n, L = 4, 2
    clean = VQCClassifier(n, L, seed=3).eval()
    x = torch.rand(6, n) * np.pi
    with torch.no_grad():
        z_clean = clean.expectations(x)
        z_p0 = noisy_model(clean.state_dict(), n, L, 0.0).expectations(x)
        z_p = noisy_model(clean.state_dict(), n, L, 0.2).expectations(x)
    assert torch.allclose(z_clean, z_p0, atol=1e-5)
    assert z_p.abs().mean() < z_clean.abs().mean()
    with pytest.raises(ValueError):
        build_circuit(n, L, device_name="default.qubit", noise_p=0.1)


def test_window_rule_matches_notebook():
    assert D.window_starts(300).tolist() == [0, 64, 128, 172]
    assert D.window_starts(128).tolist() == [0]
    assert D.window_starts(100).tolist() == [0]          # one shorter window
    W = D.cut_windows(np.random.default_rng(0).normal(size=(300, 15)))
    assert W.shape == (4, 15, 128)
    assert D.cut_windows(np.zeros((7, 15))).shape[0] == 0


def test_split_keeps_recordings_apart_and_subsample_caps_classes():
    rng = np.random.default_rng(0)
    groups = np.repeat([f"rec{i}" for i in range(60)], 20)
    y = np.repeat(rng.integers(0, 6, size=60), 20)
    part = D.make_split(y, groups, seed=0, test_fraction=0.2, val_fraction=0.1)
    recs = {p: set(groups[part == p]) for p in ("train", "val", "test")}
    assert all(len(r) for r in recs.values())
    assert not (recs["train"] & recs["test"]) and not (recs["train"] & recs["val"]) and not (recs["val"] & recs["test"])
    assert len(recs["test"]) == 12
    keep = D.subsample_per_class(y, 50, seed=0)
    assert all((y[keep] == c).sum() <= 50 for c in range(6))
    assert np.array_equal(keep, np.sort(keep))


def test_training_loop_logs_every_epoch_and_counts_evaluations():
    n = 4
    rng = np.random.default_rng(0)
    X, y = rng.uniform(0, np.pi, size=(64, n)).astype(np.float32), rng.integers(0, 6, size=64)
    model = VQCClassifier(n, 1, seed=0)
    with model.tracker() as tr:
        hist, best, timing = train_classifier(model, X, y, X[:16], y[:16], epochs=2, lr=0.05, batch_size=32, seed=0)
    assert list(hist.columns[:5]) == ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"] and len(hist) == 2
    assert best["state_dict"] is not None and 1 <= best["epoch"] <= 2
    assert timing["windows_forwarded"] == 128
    assert tr.totals["executions"] == 128 + 2 * 16            # training windows plus the two validation passes
    assert parameter_shift_evaluations(128, 12) == 128 * 25
