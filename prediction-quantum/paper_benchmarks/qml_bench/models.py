"""Every model behind one interface: fit(train, val) -> predict(test) -> latency of one window.

Quantum (simulated):  qsvm      quantum kernel SVC on the n PCA angles (training subsample)
                      vqc       the project's variational quantum classifier (prediction_quantum/vqc.py)
Classical, same n angles:  logreg, gnb, svm_rbf (same training subsample as qsvm), mlp (same loop as vqc)
Classical, 570 features:   gnb_full, lda_gnb_full (global shrinkage LDA to 5 + GaussianNB), mlp_full
"""
import time

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from .metrics import median_latency_ms
from .qkernel import QuantumKernelSVC

QUANTUM = {"qsvm", "vqc"}
ANGLE_INPUT = {"qsvm", "vqc", "logreg", "gnb", "svm_rbf", "mlp"}


def stratified_subsample(y, n, seed):
    """Sorted indices of about n windows drawn without replacement with the class shares of y (all of them when n
    is None or not below len(y)). The same seed gives the same rows, which is how qsvm and svm_rbf share them."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    per = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    take = []
    for c, idx in per.items():
        # each class keeps its share of the full set; rounding per class can leave the total a few off n
        k = int(round(n * len(idx) / len(y)))
        take.append(rng.choice(idx, min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(take))


def _sk_params(m):
    """Fitted numbers a sklearn model carries into prediction (None for kinds we do not count)."""
    # a pipeline is counted by its last step, i.e. the GaussianNB after LDA; the projection is left out
    m = m.steps[-1][1] if hasattr(m, "steps") else m
    if isinstance(m, GaussianNB):
        return int(m.theta_.size + m.var_.size + m.class_prior_.size)
    if isinstance(m, LogisticRegression):
        return int(m.coef_.size + m.intercept_.size)
    if isinstance(m, SVC):
        return int(m.dual_coef_.size + m.intercept_.size + m.support_vectors_.size)
    return None


def run_model(kind, cfg, Xtr, ytr, Xva, yva, Xte, seed, n_qubits=None, n_layers=None):
    """-> dict(pred, fit_seconds, latency_ms, latency_p90_ms, n_parameters, extra)."""
    # imported here, not at module level: this module never calls paths.setup(), so prediction_quantum is on
    # sys.path only once the caller (run.py, its loky workers, the tests) has run it
    from prediction_quantum.baselines import MLPClassifier
    from prediction_quantum.train import predict as torch_predict, train_classifier
    from prediction_quantum.vqc import VQCClassifier, parameter_shift_evaluations
    # set in every worker so the torch models (vqc, mlp, mlp_full) are timed with one thread budget; it has no
    # effect on the sklearn fits
    torch.set_num_threads(int(cfg["compute"]["torch_threads"]))
    rep, warm = cfg["latency"]["repeats"], cfg["latency"]["warmup"]
    extra = {}
    if kind in ("logreg", "gnb", "gnb_full", "lda_gnb_full", "svm_rbf"):
        sub = np.arange(len(ytr))
        # svm_rbf trains on the same stratified rows as qsvm, so the two kernels are compared on equal footing
        if kind == "svm_rbf":
            sub = stratified_subsample(ytr, cfg["qsvm"]["max_train"], seed)
        model = {"logreg": lambda: LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
                 "gnb": GaussianNB, "gnb_full": GaussianNB,
                 "lda_gnb_full": lambda: make_pipeline(LinearDiscriminantAnalysis(n_components=5, solver="eigen",
                                                                                    shrinkage="auto"), GaussianNB()),
                 "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale")}[kind]()
        t0 = time.perf_counter()
        model.fit(np.asarray(Xtr[sub], np.float64), ytr[sub])
        fit_s = time.perf_counter() - t0
        pred = model.predict(np.asarray(Xte, np.float64))
        # latency is measured on a single window, the unit the online decoder works with
        x1 = np.asarray(Xte[:1], np.float64)
        lat, p90 = median_latency_ms(lambda: model.predict(x1), rep, warm)
        extra["n_train_used"] = int(len(sub))
        return dict(pred=pred, fit_seconds=fit_s, latency_ms=lat, latency_p90_ms=p90, n_parameters=_sk_params(model),
                    extra=extra)
    if kind == "qsvm":
        sub = stratified_subsample(ytr, cfg["qsvm"]["max_train"], seed)
        model = QuantumKernelSVC(n_qubits, reps=cfg["qsvm"]["reps"], C=cfg["qsvm"]["C"], seed=seed).fit(Xtr[sub], ytr[sub])
        t0 = time.perf_counter()
        pred = model.predict(Xte)
        extra.update(n_train_used=int(len(sub)), predict_seconds=time.perf_counter() - t0,
                     hardware_kernel_evaluations=model.hardware_kernel_evaluations(len(Xte)),
                     n_support=int(model.svc_.support_.size), feature_map_reps=int(cfg["qsvm"]["reps"]))
        x1 = Xte[:1]
        # fewer repeats here: one predict simulates a statevector and a kernel row against every training window
        lat, p90 = median_latency_ms(lambda: model.predict(x1), max(10, rep // 5), warm)
        return dict(pred=pred, fit_seconds=model.fit_seconds_, latency_ms=lat, latency_p90_ms=p90,
                    n_parameters=model.n_parameters(), extra=extra)
    # gradient-trained torch models: vqc, mlp, mlp_full
    if kind == "vqc":
        model = VQCClassifier(n_qubits, n_layers, seed=seed)
        tc = cfg["vqc"]
    else:
        model = MLPClassifier(Xtr.shape[1], hidden=cfg["mlp"]["hidden"], seed=seed)
        tc = cfg["mlp"]
    hist, best, timing = train_classifier(model, Xtr, ytr, Xva, yva, epochs=tc["epochs"], lr=tc["lr"],
                                          batch_size=tc["batch_size"], seed=seed)
    # restore the epoch with the best validation accuracy rather than the last one
    model.load_state_dict(best["state_dict"])
    pred = torch_predict(model, Xte, batch_size=256)
    # the tensor is built once, outside the timed call, so only the forward pass is clocked
    x1 = torch.as_tensor(np.asarray(Xte[:1], np.float32))

    def one():
        with torch.no_grad():
            model(x1)
    lat, p90 = median_latency_ms(one, rep, warm)
    extra.update(best_epoch=int(best["epoch"]), best_val_accuracy=float(best["val_acc"]),
                 final_train_loss=float(hist["train_loss"].iloc[-1]), final_val_loss=float(hist["val_loss"].iloc[-1]),
                 windows_forwarded=int(timing["windows_forwarded"]))
    if kind == "vqc":
        extra["n_circuit_parameters"] = model.n_quantum_parameters()
        # training used backprop on the simulator; this is what a parameter-shift gradient would cost on hardware
        extra["parameter_shift_equivalent"] = parameter_shift_evaluations(timing["windows_forwarded"],
                                                                          model.n_quantum_parameters())
    return dict(pred=pred, fit_seconds=timing["train_seconds"], latency_ms=lat, latency_p90_ms=p90,
                n_parameters=model.n_parameters(), extra=extra)
