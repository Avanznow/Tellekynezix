"""Run the QML benchmark: every model x every fold of both protocols, on identical windows and inputs.

    cd prediction-quantum/paper_benchmarks
    ~/qml_env/bin/python -m qml_bench.run --config configs/default.yaml [--out ../results/paper/qml] [--only qsvm,vqc]

Writes runs.csv, per_class.csv, confusions.json, predictions.npz, folds.csv, environment.json to the output
folder, then REPORT.md and figures (qml_bench.report). Outputs go to prediction-quantum/results/paper/qml.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from . import paths

paths.setup()

from .data import CLASSES, load_windows, subsample_per_class  # noqa: E402
from .metrics import scores  # noqa: E402
from .models import ANGLE_INPUT, QUANTUM, run_model  # noqa: E402
from .protocols import check_no_leakage, participant_folds, recording_folds  # noqa: E402


def say(msg):
    """Timestamped progress line, flushed so it shows up immediately when stdout is a log file."""
    print(f"[qml {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_folds(cfg, W):
    """Return (keep, folds): the fixed-seed per-class subsample of window indices and the folds of both protocols
    restricted to it, each fold a dict(protocol, fold, train, val, test) of indices into W["X"]. Every fold is
    checked for recording leakage (and participant leakage in the participant protocol) before it is handed back."""
    keep = subsample_per_class(W["y"], cfg["data"]["max_windows_per_class"], cfg["data"]["subsample_seed"])
    pc = cfg["protocols"]["participant"]
    folds = participant_folds(keep, W["y"], W["groups"], W["participants"], pc["n_folds"], pc["val_fraction"])
    # the split CSVs index the full cache and already hold the original benchmark's subsample (the same draw with
    # the default config, so this filter changes nothing); it keeps them inside `keep` if that config changes
    kset = set(keep.tolist())
    for f in recording_folds(cfg["protocols"]["recording"]["seeds"]):
        for p in ("train", "val", "test"):
            f[p] = np.array([i for i in f[p] if i in kset], dtype=int)
        folds.append(f)
    for f in folds:
        check_no_leakage(f, W["groups"], W["participants"])
    return keep, folds


def encode(cfg, W, f):
    """Fit the two input representations on the training part of fold f and transform all three parts.
    Returns (ang, full, pca_variance): ang[p] are (windows, n_qubits) RY angles for the quantum and PCA models,
    full[p] the (windows, 570) standardized features for the *_full models, and the variance share the PCA keeps."""
    from prediction_quantum.encoding import FeatureEncoder
    from sklearn.preprocessing import StandardScaler
    X, n = W["X"], cfg["encoding"]["n_qubits"]
    # both fitted on the training part only; the encoder is StandardScaler + PCA to n_qubits, then the scores are
    # clipped at the training 1st/99th percentiles and mapped to [0, pi]
    enc = FeatureEncoder(n).fit(X[f["train"]])
    sc = StandardScaler().fit(X[f["train"]])
    ang = {p: enc.transform(X[f[p]]) for p in ("train", "val", "test")}
    full = {p: sc.transform(X[f[p]]).astype(np.float32) for p in ("train", "val", "test")}
    return ang, full, float(enc.explained_variance_ratio_.sum())


def job(kind, cfg, f, ang, full, y, seed):
    """One (model, fold) run inside a loky worker. Picks the angle or full-feature input for `kind`, fits, predicts
    the test part and returns (kind, protocol, fold, result) with the wall time of run_model (fit, predict and
    the latency measurement) added."""
    # loky workers are separate interpreters; setup() is idempotent and cheap, so it is repeated here instead of
    # relying on the module-level call having run in the worker
    paths.setup()
    inp = ang if kind in ANGLE_INPUT else full
    t0 = time.perf_counter()
    r = run_model(kind, cfg, inp["train"], y[f["train"]], inp["val"], y[f["val"]], inp["test"], seed,
                  n_qubits=cfg["encoding"]["n_qubits"], n_layers=cfg["vqc"]["n_layers"])
    r["wall_seconds"] = time.perf_counter() - t0
    return kind, f["protocol"], f["fold"], r


def main(argv=None):
    """Load the cached windows, build the folds, fit the encoders once per fold, run every (model, fold) pair in
    a process pool and write the result tables, then hand the output folder to qml_bench.report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=str(paths.RESULTS))
    ap.add_argument("--only", default=None, help="comma-separated subset of models")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = a.only.split(",") if a.only else cfg["models"]
    W = load_windows()
    keep, folds = build_folds(cfg, W)
    y = W["y"]
    say(f"{len(keep)} windows, {len(folds)} folds, models {models}")
    rows_f, encs = [], {}
    # encoders are fitted once per fold and shared by every model of that fold, so "same input" holds literally
    for f in folds:
        ang, full, var = encode(cfg, W, f)
        encs[(f["protocol"], f["fold"])] = (ang, full)
        for p in ("train", "val", "test"):
            rows_f.append(dict(protocol=f["protocol"], fold=f["fold"], part=p, windows=len(f[p]),
                               recordings=len(set(W["groups"][f[p]])), participants=len(set(W["participants"][f[p]])),
                               test_participants=";".join(sorted(set(W["participants"][f[p]]))) if p == "test" else "",
                               pca_variance=var))
    pd.DataFrame(rows_f).to_csv(out / "folds.csv", index=False)
    # slowest first so the pool stays busy
    order = {"vqc": 0, "qsvm": 1, "svm_rbf": 2, "mlp_full": 3, "mlp": 4}
    tasks = sorted([(m, f) for f in folds for m in models], key=lambda t: order.get(t[0], 9))
    # the fold number doubles as the model seed, so a (model, fold) run is reproducible; folds of the two
    # protocols with the same number share a seed
    res = Parallel(n_jobs=cfg["compute"]["n_jobs"], backend="loky", verbose=0)(
        delayed(job)(m, cfg, f, *encs[(f["protocol"], f["fold"])], y, int(f["fold"])) for m, f in tasks)
    runs, per_class, conf, preds = [], [], {}, {}
    fmap = {(f["protocol"], f["fold"]): f for f in folds}
    for kind, prot, fold, r in res:
        f = fmap[(prot, fold)]
        s = scores(y[f["test"]], r["pred"])
        runs.append(dict(protocol=prot, fold=fold, model=kind, family="quantum (simulated)" if kind in QUANTUM else "classical",
                         input=f"{cfg['encoding']['n_qubits']} PCA angles" if kind in ANGLE_INPUT else "570 standardized features",
                         **{k: v for k, v in s.items() if k not in ("per_class", "confusion")},
                         fit_seconds=r["fit_seconds"], wall_seconds=r["wall_seconds"], latency_ms=r["latency_ms"],
                         latency_p90_ms=r["latency_p90_ms"], n_parameters=r["n_parameters"], n_train=len(f["train"]),
                         n_test=len(f["test"]), **{f"x_{k}": v for k, v in r["extra"].items()}))
        for c, name in enumerate(CLASSES):
            per_class.append(dict(protocol=prot, fold=fold, model=kind, cls=name,
                                  **{k: s["per_class"][k][c] for k in ("precision", "recall", "f1", "support")}))
        conf[f"{prot}|{fold}|{kind}"] = s["confusion"]
        preds[f"{prot}|{fold}|{kind}"] = np.asarray(r["pred"], np.int8)
        # identical for every model of the fold, so rewriting these two per model is harmless
        preds[f"{prot}|{fold}|y_true"] = y[f["test"]].astype(np.int8)
        preds[f"{prot}|{fold}|window_index"] = f["test"].astype(np.int32)
    pd.DataFrame(runs).sort_values(["protocol", "model", "fold"]).to_csv(out / "runs.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    (out / "confusions.json").write_text(json.dumps(dict(labels=list(CLASSES), confusions=conf)))
    np.savez_compressed(out / "predictions.npz", **preds)
    # library versions are stored next to the config so a results folder says exactly what produced it
    import pennylane, sklearn, torch
    (out / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), pennylane=pennylane.__version__, torch=torch.__version__,
        scikit_learn=sklearn.__version__, numpy=np.__version__, platform=platform.platform(),
        cpu=platform.processor(), config=cfg, feature_cache=W["cache"], n_windows=int(len(keep)),
        generated=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    say("runs written; building report")
    from . import report
    report.main([str(out)])


if __name__ == "__main__":
    main()
