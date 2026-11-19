"""
Gaussian Naive Bayes classifier implementation using JAX.

This module provides a JAX-based GaussianNB implementation for EEG movement
classification. The model is trained offline and can be saved/loaded for
inference on new data.

Usage:
    # Training
    model = GaussianNBJAX(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Saving/Loading
    save_model('path/to/model.pkl', model, metadata_dict)
    model, meta = load_model('path/to/model.pkl')
"""

from typing import Any, Dict, Tuple
import pickle

import numpy as np
import jax
import jax.numpy as jnp


class GaussianNBJAX:
    """Gaussian Naive Bayes classifier using JAX for efficient computation.
    
    Attributes:
        var_smoothing: Regularization parameter for variance estimates.
        class_prior_: Prior probability of each class.
        theta_: Mean feature vectors per class.
        var_: Variance estimates per feature per class.
        n_classes_: Number of classes.
        n_features_: Number of features.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """Initialize GaussianNBJAX.
        
        Args:
            var_smoothing: Smoothing value added to variance estimates to prevent
                          singularity (default: 1e-9).
        """
        self.var_smoothing = float(var_smoothing)
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Gaussian Naive Bayes model.
        
        Args:
            X: Training feature matrix (n_samples, n_features).
            y: Training labels (n_samples,), assumed 0..C-1.
        
        Returns:
            self
        """
        X_j = jnp.asarray(X, dtype=jnp.float32)
        y_j = jnp.asarray(y, dtype=jnp.int32)

        n_samples = X_j.shape[0]
        n_features = X_j.shape[1]

        num_classes = int(jnp.max(y_j) + 1)
        counts = jnp.bincount(y_j, length=num_classes)
        # Floor the counts at 1 so a label that never appears in y does not divide by zero below.
        counts_f = jnp.maximum(counts.astype(jnp.float32), 1.0)

        # Per-class sums via one-hot matmul: (C,N) @ (N,D) -> (C,D), without
        # materialising a masked (C,N,D) copy of X for every class.
        one_hot = jax.nn.one_hot(y_j, num_classes, dtype=jnp.float32).T
        hp = jax.lax.Precision.HIGHEST
        sums = jnp.matmul(one_hot, X_j, precision=hp)
        sums2 = jnp.matmul(one_hot, X_j * X_j, precision=hp)

        means = sums / counts_f[:, None]
        # Population variance (divide by N, as sklearn does) via E[x^2] - E[x]^2 in one pass. The subtraction
        # cancels badly when a feature's mean is large next to its spread, hence the HIGHEST-precision sums above.
        vars_ = (sums2 / counts_f[:, None]) - jnp.square(means)
        # A floor, not sklearn's additive epsilon (var_smoothing * max feature variance): a constant feature gets
        # var_smoothing instead of 0, which would divide by zero at predict time. Same convention as the TF sibling.
        vars_ = jnp.maximum(vars_, self.var_smoothing)

        priors = counts.astype(jnp.float32) / float(n_samples)

        # Host-side numpy copies under sklearn's attribute names; save_model pickles these as they are.
        self.theta_ = np.asarray(means, dtype=np.float32)
        self.var_ = np.asarray(vars_, dtype=np.float32)
        self.class_prior_ = np.asarray(priors, dtype=np.float32)
        self.n_classes_ = int(num_classes)
        self.n_features_ = int(n_features)
        return self

    # staticmethod so jax.jit sees only arrays: a plain Python self is not a valid traced argument.
    @staticmethod
    @jax.jit
    def _predict_log_proba_jit(X: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray,
                                log_prior: jnp.ndarray) -> jnp.ndarray:
        """JAX-compiled log-probability computation (internal).
        
        Args:
            X: Test feature matrix (n_samples, n_features).
            mu: Class means (n_classes, n_features).
            var: Class variances (n_classes, n_features).
            log_prior: Log class priors (n_classes,).
        
        Returns:
            Log-probabilities (n_samples, n_classes).
        """
        # Log normalising constant of each class's diagonal Gaussian, shape (C,); it does not depend on X.
        const_term = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * var), axis=1)
        # (N,1,D) - (1,C,D) -> (N,C,D): every window against every class in one broadcast.
        diff = X[:, None, :] - mu[None, :, :]
        quad = -0.5 * jnp.sum((diff * diff) / (var[None, :, :]), axis=2)
        log_lik = quad + const_term[None, :]
        return log_lik + log_prior[None, :]

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute log-probabilities for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features).
        
        Returns:
            Log-probabilities (n_samples, n_classes).
        """
        X_j = jnp.asarray(X, dtype=jnp.float32)
        mu = jnp.asarray(self.theta_, dtype=jnp.float32)
        var = jnp.asarray(self.var_, dtype=jnp.float32)
        # 1e-12 keeps log(0) out of the scores for a class with zero prior (same guard as the TF sibling).
        log_prior = jnp.log(jnp.asarray(self.class_prior_, dtype=jnp.float32) + 1e-12)
        out = self._predict_log_proba_jit(X_j, mu, var, log_prior)
        return np.asarray(out)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features).
        
        Returns:
            Class probabilities (n_samples, n_classes), row-wise normalized.
        """
        logp = self.predict_log_proba(X)
        # Shift by the row max before exp: the largest term becomes exp(0), so no float32 overflow and no
        # all-zero row in the normalisation below.
        m = np.max(logp, axis=1, keepdims=True)
        p = np.exp(logp - m)
        p /= np.sum(p, axis=1, keepdims=True)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features).
        
        Returns:
            Predicted class labels (n_samples,).
        """
        logp = self.predict_log_proba(X)
        return np.argmax(logp, axis=1).astype(np.int32)


def save_model(path: str, model: GaussianNBJAX, meta: Dict[str, Any]) -> None:
    """Save trained model to disk.
    
    Args:
        path: Output file path (e.g., 'path/to/model.pkl').
        model: Fitted GaussianNBJAX instance.
        meta: Optional metadata dictionary (e.g., accuracy, class names).
    """
    payload: Dict[str, Any] = {
        'model': {
            'var_smoothing': float(model.var_smoothing),
            'class_prior_': model.class_prior_,
            'theta_': model.theta_,
            'var_': model.var_,
            'n_classes_': model.n_classes_,
            'n_features_': model.n_features_,
        },
        'meta': meta or {},
        'format': 'GaussianNBJAX-pickle-v1'
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> Tuple[GaussianNBJAX, Dict[str, Any]]:
    """Load trained model from disk.
    
    Args:
        path: Path to pickle file (e.g., 'path/to/model.pkl').
    
    Returns:
        Tuple of (fitted GaussianNBJAX, metadata dictionary).
    """
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    # A pickle without var_smoothing loads with the constructor default (1e-9).
    m = GaussianNBJAX(var_smoothing=payload['model'].get('var_smoothing', 1e-9))
    m.class_prior_ = np.array(payload['model']['class_prior_'], dtype=np.float32)
    m.theta_ = np.array(payload['model']['theta_'], dtype=np.float32)
    m.var_ = np.array(payload['model']['var_'], dtype=np.float32)
    m.n_classes_ = int(payload['model']['n_classes_'])
    m.n_features_ = int(payload['model']['n_features_'])
    return m, payload.get('meta', {})
