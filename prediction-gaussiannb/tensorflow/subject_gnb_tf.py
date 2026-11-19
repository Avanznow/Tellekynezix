"""TensorFlow backend for the subject-specific GaussianNB chain.

Same interface as prediction-gaussiannb/subject_gnb.ProjectedTorchGaussianNB, with the TensorFlow GaussianNBTF of this
folder as the decision model: one compact LDA projection (fitted on train, used as a transform) followed by one
GaussianNBTF, over the classes present in the training labels. Also loads the pickles written by
gaussiannb_train.ipynb (gaussiannb_tf_improved_global.pkl, gaussiannb_tf_subject_specific.pkl), whose GaussianNBTF
objects were pickled as ``gaussiannb_model.GaussianNBTF``: that module name is the PyTorch model when the live path
is loaded, so the pickles are read with a resolver that maps the name to the TensorFlow class.

TensorFlow is imported lazily and pinned to the CPU (the GaussianNB is tiny).
"""
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

TF_MODEL_DIR = Path(__file__).resolve().parent
GNB_DIR = TF_MODEL_DIR.parent
# subject_gnb.py lives one folder up; put it on the path so the lazy import in subject_models_from_pickle resolves
if str(GNB_DIR) not in sys.path:
    sys.path.insert(0, str(GNB_DIR))

TF_GLOBAL_FILE = "gaussiannb_tf_improved_global.pkl"
TF_SUBJECT_FILE = "gaussiannb_tf_subject_specific.pkl"
_tf_model_module = None


def tf_model_module():
    """The TensorFlow GaussianNB module of this folder under its own name (gaussiannb_model_tf), so it never collides
    with prediction-gaussiannb/pytorch/gaussiannb_model.py."""
    global _tf_model_module
    if _tf_model_module is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        # hiding the GPU only works before TF has initialised its devices; after that the call raises and we move on
        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError):
            pass
        spec = importlib.util.spec_from_file_location("gaussiannb_model_tf", TF_MODEL_DIR / "gaussiannb_model.py")
        module = importlib.util.module_from_spec(spec)
        # importlib's recipe for loading a file by path: register the module under its name before executing it
        sys.modules["gaussiannb_model_tf"] = module
        spec.loader.exec_module(module)
        _tf_model_module = module
    return _tf_model_module


def GaussianNBTF(*args, **kwargs):
    """Build a GaussianNBTF with the class's own arguments; TensorFlow is imported on the first call."""
    return tf_model_module().GaussianNBTF(*args, **kwargs)


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    # subtract each row's max before exp() so large log joints cannot overflow; it cancels out of the result
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


class ProjectedGaussianNBTF:
    """LDA projection (train-only transform) + GaussianNBTF, over the classes present in the training labels.

    Mirrors subject_gnb.ProjectedTorchGaussianNB: fit(Z, y, n_classes), transform(Z), log_joint(Z),
    predict_log_proba(Z) -> (n, n_classes) with log(absent_floor) for classes the model never saw. The GaussianNBTF
    keeps its own conventions (population variance floored at var_smoothing), as in gaussiannb_train.ipynb.
    """

    def __init__(self, n_components=5, shrinkage="auto", var_smoothing=1e-9, absent_floor=1e-6):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.var_smoothing = var_smoothing
        self.absent_floor = absent_floor

    def fit(self, Z, y, n_classes):
        """Fit the projection and the GaussianNBTF on Z (n_windows, n_features) with integer labels y taken from a
        command set of size n_classes. The TF model only sees the classes present in y (kept in classes_); the
        missing ones are padded in at prediction time."""
        Z = np.asarray(Z, np.float64); y = np.asarray(y, np.int64)
        self.n_classes_ = int(n_classes)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        # relabel to 0..k-1 over the classes present: GaussianNBTF sizes itself from max(y) + 1, so a gap in the
        # labels would add an empty class
        y_local = np.searchsorted(self.classes_, y)
        lda = LinearDiscriminantAnalysis(n_components=min(self.n_components, len(self.classes_) - 1),
                                         solver="eigen", shrinkage=self.shrinkage).fit(Z, y_local)
        # the eigen solver keeps every eigenvector in scalings_; the first _max_components (at most k - 1) columns
        # are exactly what lda.transform uses, so from here on the projection is a plain matmul
        self.projection_ = np.ascontiguousarray(lda.scalings_[:, :lda._max_components])
        self.nb_ = GaussianNBTF(var_smoothing=self.var_smoothing).fit(self.transform(Z).astype(np.float32), y_local.astype(np.int32))
        self.n_train_ = len(y)
        return self

    @classmethod
    def from_parts(cls, projection, nb, classes, n_classes, n_train=0, absent_floor=1e-6):
        """Wrap an already fitted GaussianNBTF (e.g. from the notebook's pickle) with its projection and class set."""
        model = cls(n_components=projection.shape[1], absent_floor=absent_floor)
        model.projection_ = np.ascontiguousarray(projection, dtype=np.float64)
        model.classes_ = np.asarray(classes, np.int64)
        model.n_classes_ = int(n_classes)
        model.n_train_ = int(n_train)
        model.nb_ = nb
        model.var_smoothing = float(getattr(nb, "var_smoothing", 1e-9))
        return model

    def transform(self, Z):
        """Project standardised features (n, n_features) onto the min(n_components, k - 1) LDA axes kept in fit."""
        return np.asarray(Z, np.float64) @ self.projection_

    def log_joint(self, Z):
        """Unnormalised log p(class, z) per window, (n, len(classes_)) in float64. GaussianNBTF.predict_log_proba,
        despite its name, returns log-likelihood + log-prior without the softmax; predict_log_proba below adds it."""
        return np.asarray(self.nb_.predict_log_proba(self.transform(Z).astype(np.float32)), np.float64)

    def predict_log_proba(self, Z):
        """Normalised log posteriors as (n, n_classes): the softmax over the classes seen in training, with
        log(absent_floor) in the columns this model never saw so its rows drop straight into the global model's."""
        local = _log_softmax(self.log_joint(Z))
        if len(self.classes_) == self.n_classes_:
            return local
        out = np.full((len(local), self.n_classes_), np.log(self.absent_floor))
        # classes_ is sorted, so it doubles as the column index of each local class in the full command set
        out[:, self.classes_] = local
        return out

    # arrays <-> model (the same keys the PyTorch live path stores in model.npz)
    def to_arrays(self):
        """Everything needed to rebuild the model as plain numpy arrays under the model.npz keys: projection
        (n_features, n_components), priors (k,), means and variances (k, n_components) in the TF model's float32."""
        nb = self.nb_
        return dict(projection=np.asarray(self.projection_, np.float64), classes=np.asarray(self.classes_, np.int64),
                    n_classes=np.int64(self.n_classes_), n_train=np.int64(getattr(self, "n_train_", 0)),
                    absent_floor=np.float64(self.absent_floor), priors=np.asarray(nb.class_prior_, np.float32),
                    means=np.asarray(nb.theta_, np.float32), variances=np.asarray(nb.var_, np.float32))

    @classmethod
    def from_arrays(cls, z):
        """Inverse of to_arrays: rebuild the GaussianNBTF from its arrays (no TensorFlow fit) and wrap it."""
        # a fitted GaussianNBTF is nothing but these attributes; theta_ is (k, n_components), which gives both sizes
        nb = GaussianNBTF(var_smoothing=1e-9)
        nb.class_prior_ = np.asarray(z["priors"], np.float32)
        nb.theta_ = np.asarray(z["means"], np.float32)
        nb.var_ = np.asarray(z["variances"], np.float32)
        nb.n_classes_, nb.n_features_ = int(nb.theta_.shape[0]), int(nb.theta_.shape[1])
        return cls.from_parts(np.asarray(z["projection"], np.float64), nb, np.asarray(z["classes"], np.int64),
                              int(z["n_classes"]), int(z["n_train"]), float(z["absent_floor"]))


ProjectedGaussianNB = ProjectedGaussianNBTF      # the name the live path uses for every framework backend


class _Resolver(pickle.Unpickler):
    """Resolves the notebook's ``gaussiannb_model.GaussianNBTF`` to the TensorFlow class."""

    def find_class(self, module, name):
        """Send GaussianNBTF to the TensorFlow class whichever module the notebook pickled it from (its own import
        name, the name this file loads it under, or __main__ when it was defined inline); anything else is normal."""
        if name == "GaussianNBTF" and module in ("gaussiannb_model", "gaussiannb_model_tf", "__main__"):
            return tf_model_module().GaussianNBTF
        return super().find_class(module, name)


def load_tf_pickle(path):
    """Unpickle one of the notebook's chain files (global or subject-specific) with the GaussianNBTF name resolver."""
    with open(path, "rb") as f:
        return _Resolver(f).load()


# callers of subject_gnb.framework_backend() expect load_pickle, ProjectedGaussianNB and subject_models_from_pickle here
load_pickle = load_tf_pickle


def subject_models_from_pickle(payload, n_classes=6):
    """The notebook's subject-specific chain dict -> (SubjectSpecificGaussianNB(backend='tensorflow'), scaler).
    The returned wrapper has global_, models_ (one ProjectedGaussianNBTF per training participant) and skipped_."""
    from subject_gnb import SubjectSpecificGaussianNB
    chain = payload["chain"]
    if chain.get("kind") != "subject":
        raise ValueError("not a subject-specific chain pickle")
    wrapper = SubjectSpecificGaussianNB(n_components=chain["global_projection"].shape[1], shrinkage="auto",
                                        n_classes=n_classes, backend="tensorflow")
    # the notebook fits the global model on the raw command ids (no relabelling), so its columns are 0..n_classes-1;
    # each subject's class set is stored in the pickle
    wrapper.global_ = ProjectedGaussianNBTF.from_parts(chain["global_projection"], chain["global_model"],
                                                       np.arange(n_classes), n_classes)
    wrapper.models_ = {str(q): ProjectedGaussianNBTF.from_parts(s["projection"], s["model"], s["classes"], n_classes)
                       for q, s in chain["subjects"].items()}
    # "fallback" lists the participants the notebook left on the global model (too few windows or commands)
    wrapper.skipped_ = {str(q): str(v) for q, v in chain.get("fallback", {}).items()}
    return wrapper, chain["scaler"]
