"""Classical baselines trained on the same windows and the same split as the circuit.

    gnb_pca     sklearn GaussianNB on the n_qubits angle vectors the circuit receives (same input dimension)
    gnb_full    sklearn GaussianNB on all 570 standardised features (the project's GaussianNB decoder family)
    logreg_pca  multinomial logistic regression on the angle vectors: the fairest match, a linear model on the
                circuit's input, like the circuit's own linear head on <Z>
    mlp_pca     torch MLP, one hidden layer of 64 ReLU units, on the angle vectors; same optimiser, epochs and
                batch size as the circuit, trained with the same loop (train.py)
    mlp_full    the same MLP on all 570 standardised features, as an unconstrained classical reference
"""
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from torch import nn

N_CLASSES = 6


class MLPClassifier(nn.Module):
    """One hidden ReLU layer on the circuit's input, trained by the circuit's own loop (train.py) so the comparison
    is fair. `n_in` is the input width (n_qubits angles or all 570 features); `seed` pins the initial weights so a
    run is reproducible (load_trained() rebuilds the network and then overwrites them from params.pt)."""
    def __init__(self, n_in, hidden=64, n_classes=N_CLASSES, seed=0):
        super().__init__()
        # seed before the layers exist: nn.Linear draws its initial weights from the global RNG at construction
        torch.manual_seed(seed)
        self.n_in, self.hidden, self.n_classes = int(n_in), int(hidden), int(n_classes)
        self.net = nn.Sequential(nn.Linear(self.n_in, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.n_classes))

    def forward(self, x):
        """Raw logits, shape (batch, n_classes); train.py applies cross-entropy to these, so no softmax here."""
        return self.net(x)

    def n_parameters(self):
        """Trainable parameter count, the "parameters" column next to the circuit's in the benchmark table."""
        return int(sum(p.numel() for p in self.parameters()))

    def config(self):
        """The hyperparameters and size as a plain dict, the same shape as VQCClassifier.config(); nothing in the
        package calls it yet (run_single() reads n_parameters() directly)."""
        return dict(n_in=self.n_in, hidden=self.hidden, n_classes=self.n_classes, n_parameters=self.n_parameters())


def make_sklearn_model(kind, seed=0, max_iter=5000):
    """The sklearn baseline named by `kind`, 'gnb' or 'logreg'. GaussianNB has nothing to tune; the logistic
    regression keeps sklearn's default C=1.0 but gets far more than the default 100 lbfgs iterations so the
    multinomial fit converges instead of stopping early with a warning."""
    if kind == "gnb":
        return GaussianNB()
    if kind == "logreg":
        return LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed)
    raise ValueError(f"unknown sklearn baseline {kind!r}")


def fit_sklearn(model, Xtr, ytr):
    """-> seconds of fit()."""
    t0 = time.perf_counter()
    # inputs arrive as float32 from runs.py; fit in float64, the same cast runs.py applies before predict()
    model.fit(np.asarray(Xtr, np.float64), np.asarray(ytr))
    return time.perf_counter() - t0


def sklearn_n_parameters(model):
    """Number of fitted values, comparable to n_parameters() of the torch models. GaussianNB stores per-class means
    and variances, each (n_classes, n_features), plus the n_classes priors; logistic regression stores a weight
    matrix of that shape plus one intercept per class."""
    if isinstance(model, GaussianNB):
        return int(model.theta_.size + model.var_.size + model.class_prior_.size)
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    raise TypeError(type(model))
