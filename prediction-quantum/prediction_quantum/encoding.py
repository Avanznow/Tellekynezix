"""Classical pre-processing in front of the circuit: standardise, PCA to n_qubits, map to rotation angles.

Encoding choice. Angle embedding needs one feature per qubit, so the 570 window features are reduced to n_qubits
principal components (StandardScaler + PCA fitted on the training windows only). Each component is mapped to an
RY rotation angle in [0, pi]: the training 1st and 99th percentiles of the component are sent to 0 and pi and
values beyond them are clipped, so the embedding never wraps around (RY(theta) with theta in [0, pi] is injective
in <Z> = cos theta). The classical PCA baselines receive exactly the same n_qubits-dimensional angle vectors, so
"same input" holds literally. Amplitude encoding (log2(570) ~ 10 qubits, state preparation cost) and data
re-uploading were not used; see BENCHMARK.md, Limitations.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEncoder:
    def __init__(self, n_qubits, angle_range=(0.0, np.pi), clip_percentiles=(1.0, 99.0), random_state=0):
        self.n_qubits = int(n_qubits)
        self.angle_range = (float(angle_range[0]), float(angle_range[1]))
        self.clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-d array (windows, features), got shape {X.shape}")
        if X.shape[1] < self.n_qubits:
            raise ValueError(f"cannot take {self.n_qubits} components from {X.shape[1]} features")
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=self.n_qubits, svd_solver="full", random_state=self.random_state).fit(Z)
        S = self.pca_.transform(Z)
        self.lo_, self.hi_ = np.percentile(S, self.clip_percentiles, axis=0)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_.copy()
        return self

    def _check(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("FeatureEncoder is not fitted")
        X = np.asarray(X, np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features per window, got shape {X.shape}")
        return X

    def scores(self, X):
        """Standardised PCA scores (windows, n_qubits), unbounded."""
        X = self._check(X)
        return self.pca_.transform(self.scaler_.transform(X))

    def transform(self, X):
        """Rotation angles (windows, n_qubits) float32 in angle_range: what the circuit and the PCA baselines see."""
        u = np.clip((self.scores(X) - self.lo_) / (self.hi_ - self.lo_), 0.0, 1.0)
        lo, hi = self.angle_range
        return (lo + (hi - lo) * u).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def describe(self):
        return dict(n_qubits=self.n_qubits, n_features_in=int(self.n_features_in_), angle_range=list(self.angle_range),
                    clip_percentiles=list(self.clip_percentiles),
                    explained_variance_ratio=[round(float(v), 4) for v in self.explained_variance_ratio_],
                    explained_variance_total=round(float(self.explained_variance_ratio_.sum()), 4))
