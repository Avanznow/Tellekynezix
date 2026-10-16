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
        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError):
            pass
        spec = importlib.util.spec_from_file_location("gaussiannb_model_tf", TF_MODEL_DIR / "gaussiannb_model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["gaussiannb_model_tf"] = module
        spec.loader.exec_module(module)
        _tf_model_module = module
    return _tf_model_module


def GaussianNBTF(*args, **kwargs):
    return tf_model_module().GaussianNBTF(*args, **kwargs)


def _log_softmax(x):
    x = np.asarray(x, np.float64)
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
        Z = np.asarray(Z, np.float64); y = np.asarray(y, np.int64)
        self.n_classes_ = int(n_classes)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        y_local = np.searchsorted(self.classes_, y)
        lda = LinearDiscriminantAnalysis(n_components=min(self.n_components, len(self.classes_) - 1),
                                         solver="eigen", shrinkage=self.shrinkage).fit(Z, y_local)
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
        return np.asarray(Z, np.float64) @ self.projection_

    def log_joint(self, Z):
        return np.asarray(self.nb_.predict_log_proba(self.transform(Z).astype(np.float32)), np.float64)

    def predict_log_proba(self, Z):
        local = _log_softmax(self.log_joint(Z))
        if len(self.classes_) == self.n_classes_:
            return local
        out = np.full((len(local), self.n_classes_), np.log(self.absent_floor))
        out[:, self.classes_] = local
        return out

    # arrays <-> model (the same keys the PyTorch live path stores in model.npz)
    def to_arrays(self):
        nb = self.nb_
        return dict(projection=np.asarray(self.projection_, np.float64), classes=np.asarray(self.classes_, np.int64),
                    n_classes=np.int64(self.n_classes_), n_train=np.int64(getattr(self, "n_train_", 0)),
                    absent_floor=np.float64(self.absent_floor), priors=np.asarray(nb.class_prior_, np.float32),
                    means=np.asarray(nb.theta_, np.float32), variances=np.asarray(nb.var_, np.float32))

    @classmethod
    def from_arrays(cls, z):
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
        if name == "GaussianNBTF" and module in ("gaussiannb_model", "gaussiannb_model_tf", "__main__"):
            return tf_model_module().GaussianNBTF
        return super().find_class(module, name)


def load_tf_pickle(path):
    with open(path, "rb") as f:
        return _Resolver(f).load()


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
    wrapper.global_ = ProjectedGaussianNBTF.from_parts(chain["global_projection"], chain["global_model"],
                                                       np.arange(n_classes), n_classes)
    wrapper.models_ = {str(q): ProjectedGaussianNBTF.from_parts(s["projection"], s["model"], s["classes"], n_classes)
                       for q, s in chain["subjects"].items()}
    wrapper.skipped_ = {str(q): str(v) for q, v in chain.get("fallback", {}).items()}
    return wrapper, chain["scaler"]
