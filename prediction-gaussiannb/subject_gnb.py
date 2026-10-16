"""Subject-specific Gaussian Naive Bayes on top of the repo's PyTorch GaussianNB.

    scaled features Z (n, d)  ->  per-participant linear projection (LDA, fitted on that participant's training windows)
                              ->  per-participant TorchGaussianNB            (fitted on the same windows)
    with a GLOBAL projection + GaussianNB (fitted on everyone's training windows) as the fallback for participants
    without enough labelled training data or without every command. backend="torch" uses this repo's PyTorch
    GaussianNB (pytorch/gaussiannb_model.py); backend="tensorflow" the TensorFlow GaussianNBTF
    (tensorflow/gaussiannb_model.py via tensorflow/subject_gnb_tf.py); the chain is otherwise identical.

Why: one GaussianNB for everyone reaches ~49 % window accuracy on Data_clean (6 commands, chance 16.7 %); the same
chain fitted per participant reaches ~81 % (per recording ~88 %), because the discriminative directions differ from
person to person. The price is that every user needs their own labelled recordings (see the notebook's learning
curve). Nothing here uses test labels or recording order: a participant's model sees only that participant's
training windows, and participant identity only selects which model answers.
"""
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from gaussiannb_model import GaussianNB as TorchGaussianNB


def tensorflow_backend():
    """prediction-gaussiannb/tensorflow/subject_gnb_tf (lazy: imports TensorFlow). The tensorflow folder is appended
    to sys.path, never prepended: `gaussiannb_model` must keep resolving to the PyTorch model of pytorch/."""
    import importlib, sys
    from pathlib import Path
    tf_dir = str(Path(__file__).resolve().parent / "tensorflow")
    if tf_dir not in sys.path:
        sys.path.append(tf_dir)
    return importlib.import_module("subject_gnb_tf")


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


class ProjectedTorchGaussianNB:
    """One linear projection (LDA fitted on train, used as a transform only) followed by one TorchGaussianNB.

    Works on the classes PRESENT in its training labels (a participant may have recorded only some commands):
    the projection has min(n_components, n_present - 1) dimensions, the TorchGaussianNB has n_present classes, and
    predict_log_proba returns all n_classes columns with log(absent_floor) for the classes it never saw.
    """

    def __init__(self, n_components=5, shrinkage="auto", var_smoothing=1e-9, absent_floor=1e-6):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.var_smoothing = var_smoothing
        self.absent_floor = absent_floor

    def fit(self, Z, y, n_classes):
        Z = np.asarray(Z, np.float64); y = np.asarray(y, np.int64)
        self.n_classes_ = n_classes
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        y_local = np.searchsorted(self.classes_, y)
        lda = LinearDiscriminantAnalysis(n_components=min(self.n_components, len(self.classes_) - 1),
                                         solver="eigen", shrinkage=self.shrinkage).fit(Z, y_local)
        # keep only the projection (sklearn's eigen-solver transform is Z @ scalings_[:, :k]); the fitted LDA also
        # holds two d x d matrices that are not needed and would make a saved model 50x larger
        self.projection_ = np.ascontiguousarray(lda.scalings_[:, :lda._max_components])
        P = self.transform(Z)
        self.nb_ = TorchGaussianNB(num_features=P.shape[1], num_classes=len(self.classes_))
        self.nb_.fit(torch.from_numpy(P.astype(np.float32)), torch.from_numpy(y_local))
        # sklearn-style variance floor: var_smoothing * largest feature variance (the repo model adds 1e-9 only)
        with torch.no_grad():
            eps = self.var_smoothing * float(P.var(axis=0).max())
            self.nb_.variances += eps
        self.n_train_ = len(y)
        return self

    def transform(self, Z):
        return np.asarray(Z, np.float64) @ self.projection_

    def log_joint(self, Z):
        P = self.transform(Z).astype(np.float32)
        with torch.no_grad():
            return self.nb_._joint_log_likelihood(torch.from_numpy(P)).cpu().numpy().astype(np.float64)

    def predict_log_proba(self, Z):
        local = _log_softmax(self.log_joint(Z))
        if len(self.classes_) == self.n_classes_:
            return local
        out = np.full((len(local), self.n_classes_), np.log(self.absent_floor))
        out[:, self.classes_] = local
        return out


class SubjectSpecificGaussianNB:
    """Per-participant ProjectedTorchGaussianNB with a global one as fallback.

    Parameters
    ----------
    n_components, shrinkage, var_smoothing : passed to every ProjectedTorchGaussianNB (global and per participant).
    min_windows : a participant gets an own model only with at least this many training windows and >= 2 classes
                  (commands the participant never recorded get a near-zero posterior from that model).
    n_classes   : number of commands (6).
    """

    def __init__(self, n_components=5, shrinkage="auto", var_smoothing=1e-9, min_windows=60, n_classes=6, absent_floor=1e-6,
                 backend="torch"):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.var_smoothing = var_smoothing
        self.min_windows = min_windows
        self.n_classes = n_classes
        self.absent_floor = absent_floor
        if backend not in ("torch", "tensorflow"):
            raise ValueError("backend must be 'torch' or 'tensorflow'")
        self.backend = backend

    def _new(self):
        """A fresh projected GaussianNB of the chosen framework (same interface, same chain)."""
        if self.backend == "tensorflow":
            return tensorflow_backend().ProjectedGaussianNBTF(self.n_components, self.shrinkage, self.var_smoothing, self.absent_floor)
        return ProjectedTorchGaussianNB(self.n_components, self.shrinkage, self.var_smoothing, self.absent_floor)

    def fit(self, Z, y, participants):
        Z = np.asarray(Z, np.float64); y = np.asarray(y, np.int64); participants = np.asarray(participants)
        self.global_ = self._new().fit(Z, y, self.n_classes)
        self.models_ = {}
        self.skipped_ = {}
        for q in np.unique(participants):
            m = participants == q
            self.add_participant(Z[m], y[m], q)
        return self

    def add_participant(self, Z_q, y_q, participant):
        """Fit (or refit) one participant's model from that participant's labelled windows; falls back when too few."""
        Z_q = np.asarray(Z_q, np.float64); y_q = np.asarray(y_q, np.int64)
        classes = np.unique(y_q)
        if len(y_q) < self.min_windows or len(classes) < 2:
            self.skipped_[str(participant)] = f"{len(y_q)} windows, {len(classes)} classes -> global fallback"
            self.models_.pop(str(participant), None)
            return None
        self.models_[str(participant)] = self._new().fit(Z_q, y_q, self.n_classes)
        self.skipped_.pop(str(participant), None)
        return self.models_[str(participant)]

    def has_model(self, participant):
        return str(participant) in self.models_

    def predict_log_proba(self, Z, participants):
        Z = np.asarray(Z, np.float64); participants = np.asarray(participants)
        out = self.global_.predict_log_proba(Z)
        for q in np.unique(participants):
            if str(q) in self.models_:
                m = participants == q
                out[m] = self.models_[str(q)].predict_log_proba(Z[m])
        return out

    def predict_proba(self, Z, participants):
        return np.exp(self.predict_log_proba(Z, participants))

    def predict(self, Z, participants):
        return self.predict_log_proba(Z, participants).argmax(axis=1)

    def summary(self):
        return dict(participants_with_model=sorted(self.models_), fallback=self.skipped_,
                    n_components=self.n_components, shrinkage=self.shrinkage, min_windows=self.min_windows,
                    backend=getattr(self, "backend", "torch"))
