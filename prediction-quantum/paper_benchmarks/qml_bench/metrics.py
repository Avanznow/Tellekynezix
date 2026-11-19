"""Metrics shared by every model (definitions as in the paper and in prediction_quantum/metrics.py)."""
import time

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

N_CLASSES = 6


def scores(y_true, y_pred):
    """Metric dict for one set of predictions (integer labels 0..N_CLASSES-1).

    Values are plain floats and lists (lists nested under per_class and confusion), so run.py can write them to CSV
    and JSON as is; the macro averages run over all six classes, including any absent from this fold.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Fix the label set so a class absent from this fold still gets a zero entry instead of shifting the others.
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(N_CLASSES), zero_division=0)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                macro_precision=float(p.mean()), macro_recall=float(r.mean()), macro_f1=float(f.mean()),
                cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
                # share of the most common test class = accuracy of always predicting it; report.py quotes its fold
                # average next to the 1/6 chance level
                majority_share=float(np.bincount(y_true, minlength=N_CLASSES).max() / len(y_true)),
                per_class=dict(precision=p.tolist(), recall=r.tolist(), f1=f.tolist(), support=s.tolist()),
                confusion=confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)).tolist())


def median_latency_ms(fn, repeats=50, warmup=5):
    """Time fn() repeatedly and return (median, 90th percentile) in milliseconds.

    The warm-up calls are not timed, so first-call costs (compilation, cache fills) stay out of the timed runs.
    """
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    # perf_counter gives seconds; the latency_ms / latency_p90_ms columns and the report table are in milliseconds
    t = np.asarray(t) * 1000.0
    return float(np.median(t)), float(np.percentile(t, 90))
