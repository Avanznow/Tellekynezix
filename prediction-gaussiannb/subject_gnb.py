"""Subject-specific Gaussian Naive Bayes on top of the repo's PyTorch GaussianNB.

    scaled features Z (n, d)  ->  per-participant linear projection (LDA, fitted on that participant's training windows)
                              ->  per-participant TorchGaussianNB            (fitted on the same windows)
    with a GLOBAL projection + GaussianNB (fitted on everyone's training windows) as the fallback for participants
    without enough labelled training data or without every command. backend="torch" uses this repo's PyTorch
    GaussianNB (pytorch/gaussiannb_model.py); backend="tensorflow" the TensorFlow GaussianNBTF
    (tensorflow/gaussiannb_model.py via tensorflow/subject_gnb_tf.py); backend="jax" the GaussianNBJAX
    (jax/algorithm_gnb_jax.py via jax/subject_gnb_jax.py, JAX pinned to the CPU); the chain is otherwise identical.

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


def _require_library(name, display):
    """Raise ImportError unless the library `name` is really installed.

    This folder holds sub-folders called jax/ and tensorflow/, and it is on sys.path whenever the chains are used.
    Without the library, `import jax` would silently return an empty namespace package made of that folder and the
    backend would fail later with an AttributeError; this turns it into the ImportError a missing library deserves.
    """
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ImportError(f"{display} is not installed in this environment (the {name}/ folder of prediction-gaussiannb "
                          f"is not the library); install it or choose the PyTorch framework")


def tensorflow_backend():
    """prediction-gaussiannb/tensorflow/subject_gnb_tf (lazy: imports TensorFlow). The tensorflow folder is appended
    to sys.path, never prepended: `gaussiannb_model` must keep resolving to the PyTorch model of pytorch/."""
    import importlib, sys
    from pathlib import Path
    _require_library("tensorflow", "TensorFlow")
    tf_dir = str(Path(__file__).resolve().parent / "tensorflow")
    if tf_dir not in sys.path:
        sys.path.append(tf_dir)
    return importlib.import_module("subject_gnb_tf")


def jax_backend():
    """prediction-gaussiannb/jax/subject_gnb_jax (lazy: imports JAX, pinned to the CPU there). The jax folder is
    appended to sys.path like the tensorflow one (it holds no module that shadows anything the live path imports)."""
    import importlib, sys
    from pathlib import Path
    _require_library("jax", "JAX")
    jax_dir = str(Path(__file__).resolve().parent / "jax")
    if jax_dir not in sys.path:
        sys.path.append(jax_dir)
    return importlib.import_module("subject_gnb_jax")


BACKENDS = ("torch", "tensorflow", "jax")


def framework_backend(backend):
    """The backend module of a non-PyTorch framework: ProjectedGaussianNB, load_pickle, subject_models_from_pickle."""
    if backend == "tensorflow":
        return tensorflow_backend()
    if backend == "jax":
        return jax_backend()
    raise ValueError(f"no framework backend module for {backend!r} (the PyTorch model lives in this module)")


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    # subtract each row's max before exp() so large-magnitude log joints neither overflow nor underflow to log(0)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


class ProjectedTorchGaussianNB:
    """One linear projection (LDA fitted on train, used as a transform only) followed by one TorchGaussianNB.

    Works on the classes PRESENT in its training labels (a participant may have recorded only some commands):
    the projection has min(n_components, n_present - 1) dimensions, the TorchGaussianNB has n_present classes, and
    predict_log_proba returns all n_classes columns with log(absent_floor) for the classes it never saw.
    """

    def __init__(self, n_components=5, shrinkage="auto", var_smoothing=1e-9, absent_floor=1e-6):
        """n_components and shrinkage go to the LDA, var_smoothing scales the GaussianNB variance floor, and
        absent_floor is the posterior a command this model never trained on gets (small, never zero)."""
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.var_smoothing = var_smoothing
        self.absent_floor = absent_floor

    def fit(self, Z, y, n_classes):
        """Fit the LDA, then the GaussianNB on the projected windows. Z is (n, d) scaled features, y the command id
        per window, n_classes the full command count (6) so a model that saw fewer commands still knows how wide
        the log-proba it returns has to be."""
        Z = np.asarray(Z, np.float64); y = np.asarray(y, np.int64)
        self.n_classes_ = n_classes
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        # relabel to 0..n_present-1: TorchGaussianNB.fit masks y == c for c in range(num_classes)
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
        """Z (n, d) -> (n, k) on the k = min(n_components, n_present - 1) LDA directions kept in fit."""
        return np.asarray(Z, np.float64) @ self.projection_

    def log_joint(self, Z):
        """Log prior + log likelihood per present class, shape (n, n_present). The Torch model computes in float32;
        the result is returned as float64 so _log_softmax runs in double precision."""
        P = self.transform(Z).astype(np.float32)
        with torch.no_grad():
            return self.nb_._joint_log_likelihood(torch.from_numpy(P)).cpu().numpy().astype(np.float64)

    def predict_log_proba(self, Z):
        """Normalised log posteriors with all n_classes_ columns. Commands absent from training get log(absent_floor)
        rather than -inf, so exp() of a row is small but nonzero everywhere."""
        local = _log_softmax(self.log_joint(Z))
        if len(self.classes_) == self.n_classes_:
            return local
        out = np.full((len(local), self.n_classes_), np.log(self.absent_floor))
        # classes_ are the original command ids: scatter the local columns back to their global positions
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
        """absent_floor goes to every projected model (the posterior of a command a participant never recorded);
        backend picks which framework's projected GaussianNB is fitted, see the module docstring."""
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.var_smoothing = var_smoothing
        self.min_windows = min_windows
        self.n_classes = n_classes
        self.absent_floor = absent_floor
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend

    def _new(self):
        """A fresh projected GaussianNB of the chosen framework (same interface, same chain)."""
        if self.backend != "torch":
            return framework_backend(self.backend).ProjectedGaussianNB(self.n_components, self.shrinkage, self.var_smoothing, self.absent_floor)
        return ProjectedTorchGaussianNB(self.n_components, self.shrinkage, self.var_smoothing, self.absent_floor)

    def fit(self, Z, y, participants):
        """Fit the global model on all n windows, then one model per participant id. participants is a length-n
        array of ids; models are keyed by str(id), so an int id here and the same id as a string at predict time
        hit the same model."""
        Z = np.asarray(Z, np.float64); y = np.asarray(y, np.int64); participants = np.asarray(participants)
        # the global model answers for anyone skipped below and for participant ids never seen at predict time
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
            # a refit with too little data must also drop a model fitted earlier, not keep the stale one
            self.models_.pop(str(participant), None)
            return None
        self.models_[str(participant)] = self._new().fit(Z_q, y_q, self.n_classes)
        self.skipped_.pop(str(participant), None)
        return self.models_[str(participant)]

    def has_model(self, participant):
        """True when this participant answers with an own model rather than the global fallback."""
        return str(participant) in self.models_

    def predict_log_proba(self, Z, participants):
        """Log posteriors (n, n_classes): the global model's answer, with the rows of every participant that has an
        own model overwritten by that model. An id never seen in fit simply keeps the global rows."""
        Z = np.asarray(Z, np.float64); participants = np.asarray(participants)
        out = self.global_.predict_log_proba(Z)
        for q in np.unique(participants):
            if str(q) in self.models_:
                m = participants == q
                out[m] = self.models_[str(q)].predict_log_proba(Z[m])
        return out

    def predict_proba(self, Z, participants):
        """Posteriors (n, n_classes); a row sums to 1 plus absent_floor for each command its model never saw."""
        return np.exp(self.predict_log_proba(Z, participants))

    def predict(self, Z, participants):
        """Predicted command id (0 .. n_classes - 1) per window."""
        return self.predict_log_proba(Z, participants).argmax(axis=1)

    def summary(self):
        """Who got an own model, who fell back and why, plus the settings. The getattr lets this work on objects
        pickled before the backend attribute existed: those are PyTorch models."""
        return dict(participants_with_model=sorted(self.models_), fallback=self.skipped_,
                    n_components=self.n_components, shrinkage=self.shrinkage, min_windows=self.min_windows,
                    backend=getattr(self, "backend", "torch"))
