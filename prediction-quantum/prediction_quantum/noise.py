"""Sensitivity of a trained circuit to simulated depolarising noise (inference only).

The trained parameters are loaded into the same circuit built on default.mixed with qml.DepolarizingChannel(p)
on every qubit after the embedding and after each entangling layer; the test windows are classified again for
every p. Density-matrix simulation scales as 4^n, so the sweep is limited to n_qubits <= noise.max_qubits.
"""
import time

import numpy as np
import pandas as pd
import torch

from .metrics import classification_scores
from .vqc import VQCClassifier


def noisy_model(state_dict, n_qubits, n_layers, p, n_classes=6):
    model = VQCClassifier(n_qubits, n_layers, n_classes=n_classes, device_name="default.mixed", noise_p=p)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def expectations_and_predictions(model, X_angles, batch_size=256):
    """(<Z> expectations (n, n_qubits), predicted class (n,)) of a circuit model on encoded windows."""
    zs, preds = [], []
    for i in range(0, len(X_angles), batch_size):
        z = model.expectations(torch.as_tensor(np.asarray(X_angles[i:i + batch_size]), dtype=torch.float32))
        zs.append(z)
        preds.append(model.head(z).argmax(1))
    return torch.cat(zs).numpy(), torch.cat(preds).numpy()


def noise_sweep(state_dict, n_qubits, n_layers, X_angles, y, probabilities, batch_size=256, log=None):
    """One row per p: accuracy, macro-F1, balanced accuracy, the mean |<Z>| over the test windows (depolarising
    noise contracts every expectation towards 0) and the share of windows whose predicted class differs from p = 0."""
    rows, pred_clean = [], None
    for p in probabilities:
        model = noisy_model(state_dict, n_qubits, n_layers, float(p))
        t0 = time.perf_counter()
        z, pred = expectations_and_predictions(model, X_angles, batch_size)
        seconds = time.perf_counter() - t0
        if pred_clean is None:
            pred_clean = pred
        row = dict(noise_p=float(p), **classification_scores(y, pred), mean_abs_expectation=float(np.abs(z).mean()),
                   changed_predictions=float((pred != pred_clean).mean()), n_windows=int(len(y)), seconds=seconds)
        rows.append(row)
        if log:
            log(f"  p={p:<5}: accuracy {row['accuracy']:.4f} macro-F1 {row['macro_f1']:.4f} mean|<Z>| {row['mean_abs_expectation']:.3f} "
                f"changed {row['changed_predictions']:.1%} ({seconds:.1f} s)")
    return pd.DataFrame(rows)


def chance_level(y):
    counts = np.bincount(np.asarray(y))
    return float(counts.max() / counts.sum())
