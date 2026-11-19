"""Shared preprocessing helpers for the GaussianNB notebooks (PyTorch, TensorFlow, JAX).

ParallelPowerTransformer is a drop-in replacement for sklearn's PowerTransformer.
sklearn fits one exponent (lambda) per feature with a scalar optimiser, one feature
after another, which dominates the runtime on raw EEG windows (800-2,000 features).
Each feature's lambda and standardisation are independent of every other feature,
so fitting groups of columns in separate processes gives identical results.
"""

import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted

# Below this many values (samples x features) process start-up costs more than it saves.
_MIN_PARALLEL_SIZE = 2_000_000


def _fit_columns(X_cols, method, standardize):
    """Fit a plain PowerTransformer on one block of columns.

    Module-level rather than a method so joblib can send it to the worker processes by name.
    """
    return PowerTransformer(method=method, standardize=standardize).fit(X_cols)


class ParallelPowerTransformer(TransformerMixin, BaseEstimator):
    """PowerTransformer whose per-feature fits run in parallel. Results match PowerTransformer exactly."""

    def __init__(self, method="yeo-johnson", standardize=True, n_jobs=-1):
        """method and standardize go straight to PowerTransformer; n_jobs=-1 (or None) means one chunk per CPU."""
        self.method = method
        self.standardize = standardize
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit one PowerTransformer per chunk of columns; X is (n_samples, n_features), y is ignored.

        Inputs below _MIN_PARALLEL_SIZE values are fitted in-process to skip the process start-up cost.
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        n_workers = os.cpu_count() if self.n_jobs in (None, -1) else int(self.n_jobs)
        n_chunks = 1 if X.size < _MIN_PARALLEL_SIZE else max(1, min(n_workers, n_features))
        # array_split rather than split: n_features rarely divides evenly, so the last chunks are a column shorter.
        self.column_chunks_ = np.array_split(np.arange(n_features), n_chunks)
        if n_chunks == 1:
            self.transformers_ = [_fit_columns(X, self.method, self.standardize)]
        else:
            # One process per chunk; each worker only receives its own column slice, not the whole of X.
            self.transformers_ = Parallel(n_jobs=n_chunks)(
                delayed(_fit_columns)(X[:, cols], self.method, self.standardize) for cols in self.column_chunks_
            )
        # Same attribute name as sklearn's PowerTransformer.
        self.lambdas_ = np.concatenate([t.lambdas_ for t in self.transformers_])
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """Apply the fitted per-column transforms and stitch the chunks back together in column order."""
        check_is_fitted(self, "transformers_")
        X = np.asarray(X)
        if len(self.transformers_) == 1:
            # Single chunk: hand X over whole and skip the hstack copy.
            return self.transformers_[0].transform(X)
        return np.hstack([t.transform(X[:, cols]) for t, cols in zip(self.transformers_, self.column_chunks_)])

    def inverse_transform(self, X):
        """Undo transform(): map each chunk back to the original feature scale and reassemble the columns."""
        check_is_fitted(self, "transformers_")
        X = np.asarray(X)
        return np.hstack([t.inverse_transform(X[:, cols]) for t, cols in zip(self.transformers_, self.column_chunks_)])
