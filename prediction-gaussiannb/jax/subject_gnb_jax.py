"""JAX backend for the subject-specific GaussianNB chain.

Same interface as prediction-gaussiannb/subject_gnb.ProjectedTorchGaussianNB and tensorflow/subject_gnb_tf.ProjectedGaussianNBTF,
with the GaussianNBJAX of this folder (algorithm_gnb_jax.py) as the decision model: one compact LDA projection (fitted
on train, used as a transform) followed by one GaussianNBJAX, over the classes present in the training labels. Also
loads the pickles written by instructions_log_gnb_jax.ipynb (gaussiannb_jax_improved_global.pkl,
gaussiannb_jax_subject_specific.pkl), whose models were pickled as ``algorithm_gnb_jax.GaussianNBJAX``: the module is
loaded from this folder under that name (no other module of the repo is called algorithm_gnb_jax) before any pickle
is read, so the pickles resolve without this folder on sys.path.

JAX is imported lazily and pinned to the CPU: the GaussianNB has 6 classes and <= 5 projected features, and on the GPU
every new input shape costs a ~2.5 s XLA compile plus a device copy, which made the GPU 2-3x slower than the CPU for
this model (see README.md in this folder). JAX_PLATFORMS is set before the jax package is imported; when something imported
jax earlier, jax.config.update("jax_platforms", "cpu") still works as long as no computation has run yet.
"""
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

JAX_MODEL_DIR = Path(__file__).resolve().parent
GNB_DIR = JAX_MODEL_DIR.parent
if str(GNB_DIR) not in sys.path:              # the pickles also hold improved_gnb_features.ParticipantScaler
    sys.path.insert(0, str(GNB_DIR))

JAX_GLOBAL_FILE = "gaussiannb_jax_improved_global.pkl"
JAX_SUBJECT_FILE = "gaussiannb_jax_subject_specific.pkl"
JAX_PLATFORM = "cpu"
_jax_model_module = None


def pin_jax_to_cpu():
    """Import jax with the CPU as its platform (environment variable before the import, config update after it) and
    return the module. Prints a note when another backend was already initialised in this process."""
    os.environ.setdefault("JAX_PLATFORMS", JAX_PLATFORM)
    import jax
    try:
        jax.config.update("jax_platforms", JAX_PLATFORM)
    except Exception:  # noqa: BLE001  (an initialised backend cannot be changed any more; reported below)
        pass
    if jax.default_backend() != JAX_PLATFORM:
        print(f"[GaussianNB JAX] JAX already initialised its {jax.default_backend()} backend in this process; "
              f"the GaussianNB runs there (set JAX_PLATFORMS=cpu before importing jax to use the CPU)")
    return jax


def jax_model_module():
    """algorithm_gnb_jax of this folder (GaussianNBJAX), imported once with JAX pinned to the CPU. An already imported
    algorithm_gnb_jax (the notebook's folder on sys.path) is reused, so there is only ever one GaussianNBJAX class."""
    global _jax_model_module
    if _jax_model_module is None:
        pin_jax_to_cpu()
        module = sys.modules.get("algorithm_gnb_jax")
        if module is None:
            spec = importlib.util.spec_from_file_location("algorithm_gnb_jax", JAX_MODEL_DIR / "algorithm_gnb_jax.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["algorithm_gnb_jax"] = module
            spec.loader.exec_module(module)
        _jax_model_module = module
    return _jax_model_module


def GaussianNBJAX(*args, **kwargs):
    return jax_model_module().GaussianNBJAX(*args, **kwargs)


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


class ProjectedGaussianNBJAX:
    """LDA projection (train-only transform) + GaussianNBJAX, over the classes present in the training labels.

    Mirrors subject_gnb.ProjectedTorchGaussianNB: fit(Z, y, n_classes), transform(Z), log_joint(Z),
    predict_log_proba(Z) -> (n, n_classes) with log(absent_floor) for classes the model never saw. The GaussianNBJAX
    keeps its own conventions (population variance floored at var_smoothing, float32), as in the notebook.
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
        self.nb_ = GaussianNBJAX(var_smoothing=self.var_smoothing).fit(self.transform(Z).astype(np.float32), y_local.astype(np.int32))
        self.n_train_ = len(y)
        return self

    @classmethod
    def from_parts(cls, projection, nb, classes, n_classes, n_train=0, absent_floor=1e-6):
        """Wrap an already fitted GaussianNBJAX (e.g. from the notebook's pickle) with its projection and class set."""
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

    # arrays <-> model (the same keys the PyTorch and TensorFlow live paths store in model.npz)
    def to_arrays(self):
        nb = self.nb_
        return dict(projection=np.asarray(self.projection_, np.float64), classes=np.asarray(self.classes_, np.int64),
                    n_classes=np.int64(self.n_classes_), n_train=np.int64(getattr(self, "n_train_", 0)),
                    absent_floor=np.float64(self.absent_floor), priors=np.asarray(nb.class_prior_, np.float32),
                    means=np.asarray(nb.theta_, np.float32), variances=np.asarray(nb.var_, np.float32))

    @classmethod
    def from_arrays(cls, z):
        nb = GaussianNBJAX(var_smoothing=1e-9)
        nb.class_prior_ = np.asarray(z["priors"], np.float32)
        nb.theta_ = np.asarray(z["means"], np.float32)
        nb.var_ = np.asarray(z["variances"], np.float32)
        nb.n_classes_, nb.n_features_ = int(nb.theta_.shape[0]), int(nb.theta_.shape[1])
        return cls.from_parts(np.asarray(z["projection"], np.float64), nb, np.asarray(z["classes"], np.int64),
                              int(z["n_classes"]), int(z["n_train"]), float(z["absent_floor"]))


ProjectedGaussianNB = ProjectedGaussianNBJAX     # the name the live path uses for every framework backend


class _Resolver(pickle.Unpickler):
    """Resolves the notebook's ``algorithm_gnb_jax.GaussianNBJAX`` to the class of jax_model_module() (JAX pinned to
    the CPU before the module is imported)."""

    def find_class(self, module, name):
        if name == "GaussianNBJAX" and module in ("algorithm_gnb_jax", "__main__"):
            return jax_model_module().GaussianNBJAX
        return super().find_class(module, name)


def load_jax_pickle(path):
    with open(path, "rb") as f:
        return _Resolver(f).load()


load_pickle = load_jax_pickle


def subject_models_from_pickle(payload, n_classes=6):
    """The notebook's subject-specific chain dict -> (SubjectSpecificGaussianNB(backend='jax'), scaler).
    The returned wrapper has global_, models_ (one ProjectedGaussianNBJAX per training participant) and skipped_."""
    from subject_gnb import SubjectSpecificGaussianNB
    chain = payload["chain"]
    if chain.get("kind") != "subject":
        raise ValueError("not a subject-specific chain pickle")
    wrapper = SubjectSpecificGaussianNB(n_components=chain["global_projection"].shape[1], shrinkage="auto",
                                        n_classes=n_classes, backend="jax")
    wrapper.global_ = ProjectedGaussianNBJAX.from_parts(chain["global_projection"], chain["global_model"],
                                                        np.arange(n_classes), n_classes)
    wrapper.models_ = {str(q): ProjectedGaussianNBJAX.from_parts(s["projection"], s["model"], s["classes"], n_classes)
                       for q, s in chain["subjects"].items()}
    wrapper.skipped_ = {str(q): str(v) for q, v in chain.get("fallback", {}).items()}
    return wrapper, chain["scaler"]
