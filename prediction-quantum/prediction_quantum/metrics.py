"""Metric definitions shared by every model.

    accuracy            share of test windows whose predicted command is the true one (sklearn accuracy_score)
    macro_f1            unweighted mean over the six commands of the per-command F1 (f1_score, average="macro")
    balanced_accuracy   mean over the six commands of the per-command recall (balanced_accuracy_score)
    train_seconds       wall time of fitting (torch models: forward + backward + optimiser steps over all epochs)
    latency_ms          inference time for ONE window (batch of 1), median of repeated calls after warm-up
    batched_us_per_window   inference time per window when the whole test set is passed in batches
    circuit_evaluations     circuit runs counted by qml.Tracker (one per window per forward pass)
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)))


def median_latency_ms(fn, repeats=50, warmup=5):
    """Median (and 90th percentile) wall time of fn() in milliseconds over `repeats` calls after `warmup` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1000.0
    return float(np.median(times)), float(np.percentile(times, 90))
