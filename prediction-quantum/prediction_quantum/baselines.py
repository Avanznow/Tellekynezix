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
    def __init__(self, n_in, hidden=64, n_classes=N_CLASSES, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_in, self.hidden, self.n_classes = int(n_in), int(hidden), int(n_classes)
        self.net = nn.Sequential(nn.Linear(self.n_in, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.n_classes))

    def forward(self, x):
        return self.net(x)

    def n_parameters(self):
        return int(sum(p.numel() for p in self.parameters()))

    def config(self):
        return dict(n_in=self.n_in, hidden=self.hidden, n_classes=self.n_classes, n_parameters=self.n_parameters())


def make_sklearn_model(kind, seed=0, max_iter=5000):
    if kind == "gnb":
        return GaussianNB()
    if kind == "logreg":
        return LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed)
    raise ValueError(f"unknown sklearn baseline {kind!r}")


def fit_sklearn(model, Xtr, ytr):
    """-> seconds of fit()."""
    t0 = time.perf_counter()
    model.fit(np.asarray(Xtr, np.float64), np.asarray(ytr))
    return time.perf_counter() - t0


def sklearn_n_parameters(model):
    if isinstance(model, GaussianNB):
        return int(model.theta_.size + model.var_.size + model.class_prior_.size)
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    raise TypeError(type(model))
