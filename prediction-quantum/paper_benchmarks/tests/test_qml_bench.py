"""Unit and smoke tests (run from benchmarks/qml: ~/qml_env/bin/python -m pytest -q -p no:cacheprovider tests)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qml_bench import paths  # noqa: E402

paths.setup()
from qml_bench.metrics import scores  # noqa: E402
from qml_bench.qkernel import QuantumKernelSVC, gram, states  # noqa: E402

CACHE = list(paths.FEATURE_CACHE_DIR.glob("features_*.npz"))


def test_states_normalized_and_kernel_psd():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, np.pi, size=(20, 4))
    S = states(A, 4)
    assert S.shape == (20, 16)
    assert np.allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-10)
    K = gram(S, S)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10) and np.allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    assert (K <= 1 + 1e-12).all() and (K >= 0).all()


def test_feature_map_is_entangling():
    """The kernel must not factorize into a product of single-qubit kernels (otherwise it is classically trivial)."""
    rng = np.random.default_rng(1)
    A = rng.uniform(0, np.pi, size=(2, 4))
    S = states(A, 4)
    psi = S[0].reshape(2, 8)
    assert np.linalg.svd(psi, compute_uv=False)[1] > 1e-3   # Schmidt rank > 1 across qubit 0 | rest


def test_qsvm_learns_a_separable_toy_problem():
    rng = np.random.default_rng(2)
    A0 = rng.normal(0.6, 0.15, size=(40, 3))
    A1 = rng.normal(2.4, 0.15, size=(40, 3))
    A = np.clip(np.vstack([A0, A1]), 0, np.pi)
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    m = QuantumKernelSVC(3).fit(A[::2], y[::2])
    assert (m.predict(A[1::2]) == y[1::2]).mean() > 0.9


def test_scores_definitions():
    y = np.array([0, 1, 2, 3, 4, 5, 0, 1])
    s = scores(y, y)
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0 and s["cohen_kappa"] == 1.0


@pytest.mark.skipif(not CACHE, reason="feature cache not available")
def test_protocols_have_no_leakage():
    from qml_bench.data import load_windows, subsample_per_class
    from qml_bench.protocols import check_no_leakage, participant_folds, recording_folds
    W = load_windows()
    keep = subsample_per_class(W["y"], 2000, 0)
    assert len(keep) == 11698
    folds = participant_folds(keep, W["y"], W["groups"], W["participants"])
    covered = np.concatenate([f["test"] for f in folds])
    assert np.array_equal(np.sort(covered), keep)          # every window tested exactly once
    for f in folds + recording_folds():
        assert check_no_leakage(f, W["groups"], W["participants"])


@pytest.mark.skipif(not CACHE, reason="feature cache not available")
def test_smoke_run(tmp_path):
    from qml_bench import run
    cfg = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    run.main(["--config", str(cfg), "--out", str(tmp_path)])
    import pandas as pd
    r = pd.read_csv(tmp_path / "runs.csv")
    assert set(r.model) >= {"qsvm", "vqc", "logreg"} and (r.accuracy.between(0, 1)).all()
    assert (tmp_path / "REPORT.md").exists()
