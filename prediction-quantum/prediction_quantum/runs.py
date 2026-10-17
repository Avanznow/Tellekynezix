"""One benchmark run: a model on one split, written to results/runs/<run_id>/ (config.yaml, metrics.json, history.csv,
params.pt or model.joblib). `python -m prediction_quantum.runs results/runs/<run_id>/config.yaml` re-trains that run
from its own config into results/runs_repro/<run_id>/ and prints the metrics, so every run is reproducible on its own.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from . import data as D
from .baselines import MLPClassifier, fit_sklearn, make_sklearn_model, sklearn_n_parameters
from .metrics import classification_scores
from .train import predict, train_classifier
from .vqc import VQCClassifier, parameter_shift_evaluations

MODELS = ("vqc", "mlp_pca", "gnb_pca", "logreg_pca", "mlp_full", "gnb_full")


def versions():
    import pennylane
    import scipy
    import sklearn
    return dict(python=str(platform.python_version()), torch=str(torch.__version__), pennylane=str(pennylane.__version__),
                numpy=str(np.__version__), scipy=str(scipy.__version__), scikit_learn=str(sklearn.__version__), pandas=str(pd.__version__))


def label_of(spec):
    m, n, L = spec["model"], spec.get("n_qubits"), spec.get("n_layers")
    return {"vqc": f"VQC {n}q x {L} layer{'s' if L != 1 else ''}", "mlp_pca": f"MLP-64 (PCA-{n})", "gnb_pca": f"GaussianNB (PCA-{n})",
            "logreg_pca": f"Logistic regression (PCA-{n})", "mlp_full": "MLP-64 (570 features)",
            "gnb_full": "GaussianNB (570 features)"}[m]


def family_of(model):
    return {"vqc": "VQC", "mlp_pca": "MLP", "mlp_full": "MLP", "gnb_pca": "GaussianNB", "gnb_full": "GaussianNB",
            "logreg_pca": "Logistic regression"}[model]


def input_of(spec):
    return "570 standardised features" if spec["model"].endswith("_full") else f"PCA-{spec['n_qubits']} angles"


def run_specs(cfg):
    specs = []
    for seed in cfg["split"]["seeds"]:
        for n in cfg["encoding"]["n_qubits"]:
            for L in cfg["vqc"]["n_layers"]:
                specs.append(dict(run_id=f"vqc_q{n}_l{L}_s{seed}", model="vqc", n_qubits=n, n_layers=L, seed=seed))
            for m in ("mlp_pca", "gnb_pca", "logreg_pca"):
                specs.append(dict(run_id=f"{m}{n}_s{seed}", model=m, n_qubits=n, n_layers=None, seed=seed))
        for m in ("mlp_full", "gnb_full"):
            specs.append(dict(run_id=f"{m}_s{seed}", model=m, n_qubits=None, n_layers=None, seed=seed))
    return specs


def run_config(spec, cfg, out_dir):
    """Everything needed to reproduce the run, written next to its results."""
    seed, n = spec["seed"], spec.get("n_qubits")
    rc = dict(run_id=spec["run_id"], model=spec["model"], label=label_of(spec), family=family_of(spec["model"]),
              input=input_of(spec), n_qubits=n, n_layers=spec.get("n_layers"), seed=seed,
              split_csv=f"split_seed{seed}.csv", output_dir=str(out_dir),
              features=dict(cache_dir=cfg["data"]["cache_dir"], data_dir=cfg["data"]["data_dir"], cache_key=D.feature_cache_key(),
                            spec=D.feature_spec()),
              torch_threads=cfg["compute"]["torch_threads"], versions=versions())
    if spec["model"] == "vqc":
        rc["training"] = dict(cfg["vqc"], optimizer="Adam", loss="cross-entropy", checkpoint="best validation accuracy")
        rc["encoder"] = f"encoders/encoder_q{n}_s{seed}.joblib"
        rc["circuit"] = dict(embedding="AngleEmbedding(RY)", ansatz="StronglyEntanglingLayers", measurement="PauliZ on every qubit",
                             head="Linear(n_qubits, 6)", device="default.qubit", interface="torch", diff_method="backprop")
    elif spec["model"].startswith("mlp"):
        rc["training"] = dict(cfg["mlp"], optimizer="Adam", loss="cross-entropy", checkpoint="best validation accuracy")
        rc["encoder"] = f"encoders/encoder_q{n}_s{seed}.joblib" if n else f"encoders/scaler_full_s{seed}.joblib"
    else:
        rc["training"] = dict(estimator={"gnb_pca": "sklearn GaussianNB", "gnb_full": "sklearn GaussianNB",
                                         "logreg_pca": "sklearn LogisticRegression(C=1, lbfgs, max_iter=5000)"}[spec["model"]])
        rc["encoder"] = f"encoders/encoder_q{n}_s{seed}.joblib" if n else f"encoders/scaler_full_s{seed}.joblib"
    return rc


def model_inputs(rc, out_dir, cfg_data):
    """(train, val, test) arrays of the model's input and labels, read from the split CSV and the saved encoder."""
    feats = D.build_features(cfg_data["data_dir"], cfg_data["cache_dir"], verbose=False)
    split = D.load_split(Path(out_dir) / rc["split_csv"])
    enc = joblib.load(Path(out_dir) / rc["encoder"])
    X, y = feats["X"], feats["y"]
    parts = {}
    for p in ("train", "val", "test"):
        idx = split[p]
        A = enc.transform(X[idx]).astype(np.float32)
        parts[p] = (A, y[idx])
    return parts


def run_single(rc, out_dir, cfg_data, run_dir=None, force=False, log=None):
    """Train / fit one model from its run config; returns the metrics dict (also written to metrics.json)."""
    out_dir = Path(out_dir)
    run_dir = Path(run_dir) if run_dir else out_dir / "runs" / rc["run_id"]
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(rc["torch_threads"]))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(rc, sort_keys=False))
    parts = model_inputs(rc, out_dir, cfg_data)
    (Atr, ytr), (Ava, yva), (Ate, yte) = parts["train"], parts["val"], parts["test"]
    say = (lambda m: log(f"[{rc['run_id']}] {m}")) if log else (lambda m: None)
    m = dict(run_id=rc["run_id"], model=rc["model"], label=rc["label"], family=rc["family"], input=rc["input"],
             n_qubits=rc["n_qubits"], n_layers=rc["n_layers"], seed=rc["seed"], input_dim=int(Atr.shape[1]),
             n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)))
    model_kind = rc["model"]
    if model_kind in ("vqc", "mlp_pca", "mlp_full"):
        tr_cfg = rc["training"]
        if model_kind == "vqc":
            model = VQCClassifier(rc["n_qubits"], rc["n_layers"], seed=rc["seed"])
            m["n_quantum_parameters"] = model.n_quantum_parameters()
        else:
            model = MLPClassifier(Atr.shape[1], hidden=tr_cfg["hidden"], seed=rc["seed"])
        m["n_parameters"] = model.n_parameters()
        tracker = model.tracker() if model_kind == "vqc" else None
        if tracker:
            tracker.__enter__()
        history, best, timing = train_classifier(model, Atr, ytr, Ava, yva, epochs=tr_cfg["epochs"], lr=tr_cfg["lr"],
                                                 batch_size=tr_cfg["batch_size"], seed=rc["seed"], log=say)
        if tracker:
            tracker.__exit__(None, None, None)
            m["circuit_evaluations_training"] = int(tracker.totals.get("executions", 0))     # one per window per forward pass
            m["simulations_training"] = int(tracker.totals.get("simulations", 0))            # batched state-vector runs
            m["circuit_evaluations_training_train_only"] = int(timing["windows_forwarded"])
            m["parameter_shift_equivalent_evaluations"] = parameter_shift_evaluations(timing["windows_forwarded"], m["n_quantum_parameters"])
        model.load_state_dict(best["state_dict"])
        torch.save(best["state_dict"], run_dir / "params.pt")
        history.to_csv(run_dir / "history.csv", index=False)
        pred = predict(model, Ate)
        m.update(train_seconds=timing["train_seconds"], wall_seconds=timing["wall_seconds"], epochs=timing["epochs"], steps=timing["steps"],
                 best_epoch=best["epoch"], best_val_acc=best["val_acc"], final_train_loss=float(history.train_loss.iloc[-1]),
                 final_val_loss=float(history.val_loss.iloc[-1]))
    else:
        model = make_sklearn_model("gnb" if model_kind.startswith("gnb") else "logreg", seed=rc["seed"])
        seconds = fit_sklearn(model, Atr, ytr)
        joblib.dump(model, run_dir / "model.joblib")
        pred = model.predict(Ate.astype(np.float64))
        m.update(n_parameters=sklearn_n_parameters(model), train_seconds=seconds, wall_seconds=seconds)
    m.update(classification_scores(yte, pred))
    np.save(run_dir / "test_predictions.npy", pred.astype(np.int16))
    m["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics_path.write_text(json.dumps(m, indent=2))
    say(f"test accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}, train {m['train_seconds']:.1f} s")
    return m


def load_trained(rc, out_dir, run_dir=None):
    """The trained model of a finished run (torch models carry their saved parameters)."""
    run_dir = Path(run_dir) if run_dir else Path(out_dir) / "runs" / rc["run_id"]
    if rc["model"] == "vqc":
        model = VQCClassifier(rc["n_qubits"], rc["n_layers"], seed=rc["seed"])
        model.load_state_dict(torch.load(run_dir / "params.pt"))
        return model.eval()
    if rc["model"].startswith("mlp"):
        state = torch.load(run_dir / "params.pt")
        n_in = state["net.0.weight"].shape[1]
        model = MLPClassifier(n_in, hidden=rc["training"]["hidden"], seed=rc["seed"])
        model.load_state_dict(state)
        return model.eval()
    return joblib.load(run_dir / "model.joblib")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-train one benchmark run from its config.yaml")
    ap.add_argument("run_config")
    ap.add_argument("--out", default=None, help="run directory to write (default results/runs_repro/<run_id>)")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args(argv)
    rc = yaml.safe_load(Path(a.run_config).read_text())
    out_dir = Path(rc["output_dir"])
    if not out_dir.is_absolute():
        out_dir = D.PROJECT_DIR / out_dir
    cfg_data = dict(data_dir=a.data_dir or rc["features"]["data_dir"], cache_dir=rc["features"]["cache_dir"])
    if not Path(cfg_data["cache_dir"]).is_absolute():
        cfg_data["cache_dir"] = str(D.PROJECT_DIR / cfg_data["cache_dir"])
    run_dir = Path(a.out) if a.out else out_dir / "runs_repro" / rc["run_id"]
    m = run_single(rc, out_dir, cfg_data, run_dir=run_dir, force=True, log=print)
    print(json.dumps({k: m[k] for k in ("run_id", "accuracy", "macro_f1", "balanced_accuracy", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
