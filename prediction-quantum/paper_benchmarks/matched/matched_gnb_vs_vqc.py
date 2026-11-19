#!/usr/bin/env python3
"""Matched benchmark: the best GaussianNB chains against the best variational quantum classifier on IDENTICAL windows
and IDENTICAL recording-level splits (revision checklist, section 6).

Why. The paper's best GNB model (subject-specific chain, 0.841 +/- 0.016) was measured on six GroupShuffleSplit
hold-outs of all 12,089 windows (prediction-gaussiannb/defense/make_defense_figures.py, project_models_benchmark),
while the VQC (prediction-quantum, branch qml-benchmark-52) was measured on 11,698 windows (<= 2000 per command,
subsample seed 0) split into train / val / test by recording (seeds 0, 1, 2). This script puts both on the quantum
benchmark's own split files (prediction-quantum/results/split_seed{0,1,2}.csv), window by window.

What it does (everything read-only on the repository; outputs only under --out):
  1. alignment proof: the quantum feature cache, the GNB improved (570) and subject (435) caches, the split CSVs and,
     when Data_clean is present, a re-run of the verified loader that recovers (recording, segment, start sample) of
     every window and recomputes its 570 features;
  2. per seed and per fit variant ("train" = the VQC's own fitting part; "train+val" = sensitivity variant):
       a. subject-specific chain (ParticipantScaler -> SubjectSpecificGaussianNB, LDA <= 5 dims + TorchGaussianNB,
          global fallback), PyTorch backend, window-level argmax;
       b. improved global chain refitted exactly as the notebook fits it (ParticipantScaler -> LDA 5 -> TorchGaussianNB
          on a class-balanced oversample of 80 % of the training windows, thresholds calibrated on the other 20 %),
          window-level argmax (as chain_predictions) and, secondary, the deployed threshold rule;
       c. sklearn GaussianNB on the 570 standardised features (sanity check: must reproduce the quantum benchmark's
          "GaussianNB (570 features)" runs);
  3. VQC 6 qubits x 4 layers: the stored test predictions and a re-evaluation of the saved params.pt on the same test
     windows, in the PennyLane environment (worker mode of this script, run with --qml-python), inference only;
  4. metrics, per-class metrics, confusion matrices, fit time, per-window latency, exact McNemar tests;
  5. fidelity checks of the re-implemented chains against the repository's own recorded results.

    /home/prasndhdl/ai_env/bin/python matched_gnb_vs_vqc.py [--repo REPO] [--out OUT] [--qml-python PY]
        [--skip-positions] [--skip-fidelity]

The worker mode (--vqc-worker) is started by the main mode with --qml-python; it is not meant to be run by hand.
"""
import os
import sys

# never write byte-code caches into the repository (also inherited by joblib / subprocess workers)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

import argparse  
import hashlib  
import json  
import platform  
import subprocess 
import time  
import warnings 
from pathlib import Path  

import numpy as np  
import pandas as pd  

warnings.filterwarnings("ignore")

DEFAULT_REPO = str(Path(__file__).resolve().parents[3])                # repository root
DEFAULT_OUT = str(Path(__file__).resolve().parents[2] / "results" / "paper" / "matched")
DEFAULT_QML_PY = "/home/prasndhdl/qml_env/bin/python"
CLASSES = ["backward", "forward", "landing", "left", "right", "takeoff"]   # eeg_data.LABELS (LabelEncoder order)
SEEDS = (0, 1, 2)
BEST_Q, BEST_L = 6, 4                     # best circuit of benchmark_summary.csv (VQC 6q x 4 layers)
TORCH_THREADS = 2                         # compute.torch_threads of prediction-quantum/results/config_used.yaml
LAT_REPEATS, LAT_WARMUP = 50, 5           # latency.repeats / latency.warmup of the same config
MIN_WINDOWS = 60                          # subject_gnb_inference.MIN_WINDOWS (pytorch/subject_gnb_inference.py line 96)

M_SUB = "GNB subject-specific chain (PyTorch)"
M_IMP = "GNB improved global chain (PyTorch, argmax)"
M_IMP_T = "GNB improved global chain (PyTorch, threshold rule)"
M_SKL = "sklearn GaussianNB (570 standardised features)"
M_VQC = f"VQC {BEST_Q}q x {BEST_L} layers"
V_TRAIN, V_TRAINVAL, V_VQC = "train", "train+val", "train (val for checkpoint selection)"


def say(msg):
    """Progress line with a wall-clock stamp; flushed so worker and main mode lines interleave in order."""
    print(f"[matched {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_paths(repo):
    """The directories this script reads, keyed by short name; gnb_cache is where the PyTorch GNB notebook left its
    improved_windows_*.npz / subject_windows_*.npz caches (CACHE_DIR of make_defense_figures.py)."""
    repo = Path(repo)
    q = repo / "prediction-quantum"
    return dict(repo=repo, gnb=repo / "prediction-gaussiannb", gnb_torch=repo / "prediction-gaussiannb" / "pytorch",
                quantum=q, qres=q / "results", data_dir=repo / "Data_clean",
                gnb_cache=repo / "predictions_local" / "feature_cache")


def quantum_cache_path(P):
    """The feature cache the quantum benchmark used: results/dataset.json -> feature_cache (features_<key>.npz)."""
    info = json.loads((P["qres"] / "dataset.json").read_text())
    npz = P["qres"] / "cache" / Path(info["feature_cache"]).name
    return npz, npz.with_suffix(".json"), info


# =============================================================================================== VQC worker (qml_env)
def vqc_worker(a):
    """Inference only: the saved VQC 6q x 4 layers parameters of seeds 0-2 on their own test windows."""
    import joblib
    import torch
    P = repo_paths(a.repo)
    sys.path.insert(0, str(P["quantum"]))
    # the package's own code, imported read-only (prediction_quantum/vqc.py, train.py, metrics.py, encoding.py)
    from prediction_quantum.metrics import median_latency_ms
    from prediction_quantum.train import logits_of, predict
    from prediction_quantum.vqc import VQCClassifier
    torch.set_num_threads(TORCH_THREADS)
    npz, _, _ = quantum_cache_path(P)
    z = np.load(npz, allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(np.int64)
    bench = pd.read_csv(P["qres"] / "benchmark_results.csv")
    out, report = {}, dict(runs=[])
    for seed in SEEDS:
        run_id = f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}"
        run_dir = P["qres"] / "runs" / run_id
        split = pd.read_csv(P["qres"] / f"split_seed{seed}.csv")
        # window_index addresses rows of the quantum cache (X); kept in CSV order, the main mode checks it row by row
        test = split.loc[split.part == "test", "window_index"].to_numpy()
        enc = joblib.load(P["qres"] / "encoders" / f"encoder_q{BEST_Q}_s{seed}.joblib")
        Ate = enc.transform(X[test]).astype(np.float32)
        # runs.load_trained (prediction_quantum/runs.py lines 162-165) for a VQC run, reproduced
        model = VQCClassifier(BEST_Q, BEST_L, seed=seed)
        model.load_state_dict(torch.load(run_dir / "params.pt"))
        model.eval()
        logits = logits_of(model, Ate).numpy()                    # run_single: predict(model, Ate), batch 512
        pred = logits.argmax(1)
        stored = np.load(run_dir / "test_predictions.npy").astype(np.int64)
        row = bench[bench.run_id == run_id].iloc[0]
        metrics_json = json.loads((run_dir / "metrics.json").read_text())
        acc = float((pred == y[test]).mean())
        acc_stored = float((stored == y[test]).mean())
        # latency exactly as phase_latency (run_benchmark.py): model(x1) on one encoded window, then the encoder
        x1 = torch.as_tensor(Ate[:1])

        def one():
            with torch.no_grad():
                model(x1)
        lat, p90 = median_latency_ms(one, LAT_REPEATS, LAT_WARMUP)
        Xte1 = X[test][:1]

        def full():
            with torch.no_grad():
                model(torch.as_tensor(enc.transform(Xte1).astype(np.float32))).argmax(1)
        lat_full, p90_full = median_latency_ms(full, LAT_REPEATS, LAT_WARMUP)
        # batched throughput: one untimed batch to warm up, then the whole test set; microseconds per window
        predict(model, Ate[:256])
        t0 = time.perf_counter()
        predict(model, Ate, batch_size=256)
        batched_us = (time.perf_counter() - t0) / len(test) * 1e6
        out[f"s{seed}_window_index"] = test
        out[f"s{seed}_pred"] = pred
        out[f"s{seed}_logits"] = logits
        out[f"s{seed}_stored_pred"] = stored
        report["runs"].append(dict(
            seed=seed, run_id=run_id, n_test=int(len(test)), n_train=int(row.n_train), n_val=int(row.n_val),
            accuracy_recomputed=acc, accuracy_stored_predictions=acc_stored, accuracy_benchmark_csv=float(row.accuracy),
            accuracy_metrics_json=float(metrics_json["accuracy"]), abs_diff_recomputed_vs_csv=abs(acc - float(row.accuracy)),
            predictions_identical_to_stored=bool(np.array_equal(pred, stored)), n_pred_differ_from_stored=int((pred != stored).sum()),
            train_seconds_benchmark_csv=float(row.train_seconds), best_epoch=int(row.best_epoch), best_val_acc=float(row.best_val_acc),
            latency_ms_benchmark_csv=float(row.latency_ms), encoder_latency_ms_benchmark_csv=float(row.encoder_latency_ms),
            latency_ms=lat, latency_p90_ms=p90, latency_full_ms=lat_full, latency_full_p90_ms=p90_full,
            batched_us_per_window=batched_us, n_parameters=int(row.n_parameters), n_quantum_parameters=int(row.n_quantum_parameters),
            params_sha256=sha256(run_dir / "params.pt"), encoder_sha256=sha256(P["qres"] / "encoders" / f"encoder_q{BEST_Q}_s{seed}.joblib"),
            stored_predictions_sha256=sha256(run_dir / "test_predictions.npy")))
        print(json.dumps({k: report["runs"][-1][k] for k in ("run_id", "accuracy_recomputed", "accuracy_benchmark_csv",
                                                              "predictions_identical_to_stored", "latency_ms")}), flush=True)
    import pennylane
    import scipy
    import sklearn
    report["versions"] = dict(python=platform.python_version(), executable=sys.executable, numpy=np.__version__,
                              scipy=scipy.__version__, scikit_learn=sklearn.__version__, pandas=pd.__version__,
                              torch=torch.__version__, pennylane=pennylane.__version__, joblib=joblib.__version__,
                              torch_threads=torch.get_num_threads())
    out_dir = Path(a.out)
    np.savez_compressed(out_dir / "matched_vqc_test_predictions.npz", **out)
    (out_dir / "matched_vqc_worker.json").write_text(json.dumps(report, indent=2))


# =============================================================================================== main (ai_env)
def setup_gnb_imports(P):
    """Side-effect-free GNB building blocks only (no file writes on import): improved_gnb_features (ParticipantScaler,
    participant_of, extract_features), subject_gnb (SubjectSpecificGaussianNB), pytorch/gaussiannb_model (TorchGaussianNB).
    make_defense_figures.py is NOT imported: its module level creates a temporary directory (TMP_USERS, line 84) and
    imports the live inference modules; the few functions needed are copied below with their source lines."""
    for p in (P["repo"], P["gnb"], P["gnb_torch"], P["quantum"]):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# ---- verbatim copies from prediction-gaussiannb/defense/make_defense_figures.py 
# latest() and load_caches(): lines 149-167, CACHE_DIR turned into an argument (the only change)
# (no docstrings added to latest, load_caches and _scores: their bodies are meant to stay copies of the defense script)
# latest: the newest file matching the pattern by mtime; the suffix is a hash of the cache spec, so a cache built with
# a changed spec lands next to the old file instead of replacing it
def latest(cache_dir, pattern):
    files = sorted(Path(cache_dir).glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {pattern} in {cache_dir}")
    return files[-1]


# load_caches: the 570-feature (improved) and 435-feature (subject) caches must list the same windows in the same
# order, one contiguous block per recording; parts = participant id of every window, derived from its recording path
def load_caches(cache_dir):
    from improved_gnb_features import participant_of
    zi = np.load(latest(cache_dir, "improved_windows_*.npz"), allow_pickle=True)
    zs = np.load(latest(cache_dir, "subject_windows_*.npz"), allow_pickle=True)
    groups = zs["groups"].astype(str)
    y = zs["y"].astype(np.int64)
    if not (np.array_equal(zi["groups"].astype(str), groups) and np.array_equal(zi["y"].astype(np.int64), y)):
        raise RuntimeError("the improved and subject feature caches do not hold the same windows")
    runs = 1 + int(np.sum(groups[1:] != groups[:-1]))
    if runs != len(np.unique(groups)):
        raise RuntimeError("windows of a recording are not contiguous in the cache")
    return dict(X_imp=zi["X"].astype(np.float32), X_sub=zs["X"].astype(np.float32), y=y, groups=groups,
                parts=np.array([participant_of(g) for g in groups]))


# subject_chain_logp(): lines 1225-1232, verbatim (min_windows default S.MIN_WINDOWS = 60)
def subject_chain_logp(Xtr, ytr, ptr, Xte, pte, n_components=5, min_windows=MIN_WINDOWS):
    """The notebook's _subject_chain: ParticipantScaler on the training windows, then SubjectSpecificGaussianNB
    (own LDA + own GaussianNB for every participant with >= min_windows training windows and >= 2 commands, the
    global projection + GaussianNB as the fallback). Returns (log-posteriors of Xte, the fitted models)."""
    from improved_gnb_features import ParticipantScaler
    from subject_gnb import SubjectSpecificGaussianNB
    sc = ParticipantScaler().fit(Xtr, ptr)
    model = SubjectSpecificGaussianNB(n_components=n_components, shrinkage="auto", min_windows=min_windows).fit(
        sc.transform(Xtr, ptr), ytr, ptr)
    return model.predict_log_proba(sc.transform(Xte, pte), pte), model


# _scores(): lines 708-712, verbatim (used for the fidelity check against defense_project_models_benchmark_runs.csv)
def _scores(yt, yp):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    labels = list(range(len(CLASSES)))
    return dict(Accuracy=float((yp == yt).mean()),
                **{"Macro-F1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))},
                **{"Balanced accuracy": float(balanced_accuracy_score(yt, yp))})
# ---- end of the verbatim copies -----------------------------------------------------------------------------------


def fit_subject_chain(Xtr, ytr, ptr, n_components=5, min_windows=MIN_WINDOWS):
    """The two fitting steps of subject_chain_logp (above), returning the scaler too (needed for the latency and the
    per-window path); predictions are asserted identical to subject_chain_logp in run_seed()."""
    from improved_gnb_features import ParticipantScaler
    from subject_gnb import SubjectSpecificGaussianNB
    sc = ParticipantScaler().fit(Xtr, ptr)
    model = SubjectSpecificGaussianNB(n_components=n_components, shrinkage="auto", min_windows=min_windows).fit(
        sc.transform(Xtr, ptr), ytr, ptr)
    return sc, model


def _log_softmax(x):
    """improved_gnb_inference._log_softmax (pytorch/improved_gnb_inference.py lines 121-124)."""
    x = np.asarray(x, np.float64)
    # subtract the row maximum first so exp() cannot overflow on large joint log-likelihoods
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def predict_with_thresholds(probabilities, thresholds):
    """The notebook's decision rule (cell 11) = improved_gnb_inference.predict_with_thresholds (lines 115-118)."""
    # a class threshold of 0 would divide by zero; the deployed module clips at 1e-6 as well
    safe = np.clip(np.asarray(thresholds, np.float64), 1e-6, None)
    return np.argmax(np.asarray(probabilities, np.float64) / safe[np.newaxis, :], axis=1)


class ImprovedGlobalChain:
    """The improved global chain refitted exactly as the PyTorch notebook fits it
    (prediction-gaussiannb/pytorch/GaussianNB_EEG_Classification_Notebook.ipynb, HEAD f178d5e):
      cell 8 : ParticipantScaler().fit(train) -> Pipeline([("lda", LDA(n_components=5, solver="eigen",
               shrinkage="auto"))]).fit_transform(Z_train, y_train).astype(float32)
      cell 10: train_test_split(arange(n_train), test_size=0.2, random_state=42, stratify=y_train) -> fit / calib;
               class-balanced oversampling of the fit part (np.random.default_rng(42), every class to the largest
               class count, then shuffled) -> TorchGaussianNB(5, 6).fit; one-vs-rest thresholds on the calib part
               (grid linspace(0.20, 0.80, 13), best F1 per class, default 0.50)
    and evaluated as the deployed chain (pytorch/improved_gnb_inference.py, ImprovedGaussianNBChain._projected /
    log_posteriors, lines 291-325): log_softmax of the TorchGaussianNB joint log-likelihood of
    prep.transform(scaler.transform(X, participant)).astype(float32); window decision = argmax (chain_predictions /
    confusion_and_per_class 'argmax') or argmax(probability / threshold) (the deployed threshold rule)."""

    def fit(self, X_train, y_train, parts_train):
        """Fit scaler, LDA, TorchGaussianNB and the per-class thresholds on the 570-feature training windows
        (labels 0-5, participant id per window); seed 42 everywhere, as in the notebook, so the fit is repeatable."""
        import torch
        from gaussiannb_model import GaussianNB as TorchGaussianNB
        from improved_gnb_features import ParticipantScaler
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        # ---- cell 8
        self.scaler = ParticipantScaler().fit(X_train, parts_train)
        Z_train = self.scaler.transform(X_train, parts_train)
        self.prep = Pipeline([("lda", LinearDiscriminantAnalysis(n_components=5, solver="eigen", shrinkage="auto"))])
        X_train_ready = self.prep.fit_transform(Z_train, y_train).astype(np.float32)
        # ---- cell 10
        # 80 % of the training windows fit the GaussianNB, the other 20 % only calibrate the thresholds
        train_indices = np.arange(len(X_train_ready))
        fit_idx, calib_idx = train_test_split(train_indices, test_size=0.2, random_state=42, stratify=y_train)
        X_fit, y_fit = X_train_ready[fit_idx], y_train[fit_idx]
        X_calib, y_calib = X_train_ready[calib_idx], y_train[calib_idx]
        # every class is drawn with replacement up to the largest class count, so the GaussianNB learns uniform priors
        rng = np.random.default_rng(42)
        classes, class_counts = np.unique(y_fit, return_counts=True)
        target_count = int(class_counts.max())
        balanced_parts = []
        for cls, cls_count in zip(classes, class_counts):
            cls_idx = np.where(y_fit == cls)[0]
            sampled = rng.choice(cls_idx, size=target_count, replace=True)
            balanced_parts.append(sampled)
        balanced_idx = np.concatenate(balanced_parts)
        rng.shuffle(balanced_idx)
        X_fit_bal, y_fit_bal = X_fit[balanced_idx], y_fit[balanced_idx]
        num_features = X_train_ready.shape[1]
        num_classes = len(np.unique(y_train))
        self.model = TorchGaussianNB(num_features=num_features, num_classes=num_classes)
        self.model.fit(torch.from_numpy(X_fit_bal.astype(np.float32)), torch.from_numpy(y_fit_bal.astype(np.int64)))
        with torch.no_grad():
            calib_probs = self.model.predict_proba(torch.from_numpy(X_calib.astype(np.float32))).cpu().numpy()
        # one-vs-rest threshold per class on the calibration part: 13 steps of 0.05, best F1 wins, strict '>' keeps
        # the lowest threshold on a tie (best_f1 starts at -1, so the 0.50 default is always replaced by a grid value)
        threshold_grid = np.linspace(0.20, 0.80, 13)
        class_thresholds = np.ones(num_classes, dtype=np.float32)
        for cls in range(num_classes):
            y_true_bin = (y_calib == cls).astype(np.int32)
            best_thr, best_f1 = 0.50, -1.0
            for thr in threshold_grid:
                y_pred_bin = (calib_probs[:, cls] >= thr).astype(np.int32)
                score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                if score > best_f1:
                    best_f1, best_thr = score, float(thr)
            class_thresholds[cls] = best_thr
        self.thresholds = class_thresholds
        self.n_fit_balanced = int(len(y_fit_bal))
        return self

    def projected(self, X, parts):
        """570 features -> participant z-score -> 5 LDA dimensions, float32 and C-contiguous for torch.from_numpy."""
        return np.ascontiguousarray(self.prep.transform(self.scaler.transform(X, parts)).astype(np.float32))

    def joint_log_likelihood(self, P):
        """log p(x | class) + log prior of every class for projected windows P (n, 5), as float64 (n, 6)."""
        import torch
        with torch.no_grad():
            return self.model._joint_log_likelihood(torch.from_numpy(P)).cpu().numpy().astype(np.float64)

    def log_posteriors(self, X, parts):
        """Normalised log-posteriors (n, 6) of raw 570-feature windows; argmax of this is the window decision."""
        return _log_softmax(self.joint_log_likelihood(self.projected(X, parts)))


# ----------------------------------------------------------------------------------------------- metrics and tests
def all_metrics(y, pred):
    """Window-level metrics of one model on one test split: (summary dict, per-class list of dicts, 6 x 6 confusion
    matrix with true commands in rows). Macro precision / recall are plain means over the six classes."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    # labels pinned to all six commands so a class missing from a test split still gets its (zero) row and column
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    m = dict(accuracy=float(accuracy_score(y, pred)),
             macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
             balanced_accuracy=float(balanced_accuracy_score(y, pred)),
             macro_precision=float(np.mean(p)), macro_recall=float(np.mean(r)),
             cohen_kappa=float(cohen_kappa_score(y, pred, labels=labels)))
    per_class = [dict(cls=c, precision=float(p[k]), recall=float(r[k]), f1=float(f[k]), support=int(s[k]))
                 for k, c in enumerate(CLASSES)]
    return m, per_class, confusion_matrix(y, pred, labels=labels)


def mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) two-sided McNemar test on paired per-window correctness. b = only A right, c = only B right."""
    from scipy.special import logsumexp
    from scipy.stats import binom, binomtest
    ca, cb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = b + c
    if n == 0:
        return dict(b=b, c=c, n_discordant=0, p_value=1.0, log10_p=0.0, p_binomtest=1.0, p_underflow=False)
    # two-sided exact p = twice the lower tail of Binomial(n, 1/2) at min(b, c), capped at 1
    k = min(b, c)
    p = min(1.0, 2.0 * float(binom.cdf(k, n, 0.5)))
    # log10 of the same p-value from the log-pmf terms (exact, no underflow when p < 1e-308)
    log_cdf = float(logsumexp(binom.logpmf(np.arange(k + 1), n, 0.5)))
    log10_p = min(0.0, (np.log(2.0) + log_cdf) / np.log(10.0))
    return dict(b=b, c=c, n_discordant=n, p_value=p, log10_p=float(log10_p), p_binomtest=float(binomtest(k, n, 0.5).pvalue),
                p_underflow=bool(p == 0.0))


def median_latency(fn):
    """prediction_quantum/metrics.py median_latency_ms (imported read-only, the quantum benchmark's own definition)."""
    from prediction_quantum.metrics import median_latency_ms
    return median_latency_ms(fn, LAT_REPEATS, LAT_WARMUP)


# ----------------------------------------------------------------------------------------------- 1. alignment proof
def window_starts(n_rows, size=128, stride=64):
    """prediction_quantum/data.py lines 68-74 (= improved_gnb_inference.window_starts with tail=True), verbatim body."""
    starts = list(range(0, max(1, n_rows - size + 1), stride))
    if not starts or starts[-1] != max(0, n_rows - size):
        starts.append(max(0, n_rows - size))
    return np.asarray(starts)


def subsample_per_class(y, max_per_class, seed, n_classes=6):
    """prediction_quantum/data.py lines 151-160, verbatim body."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, int(max_per_class), replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def make_split(y, groups, seed, test_fraction=0.2, val_fraction=0.1):
    """prediction_quantum/data.py lines 163-176, verbatim body."""
    from sklearn.model_selection import GroupShuffleSplit
    idx = np.arange(len(y))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(idx, y, groups))
    rel_val = val_fraction / (1.0 - test_fraction)
    tr2, va = next(GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed).split(tr, y[tr], groups[tr]))
    part = np.empty(len(y), dtype=object)
    part[tr[tr2]], part[tr[va]], part[te] = "train", "val", "test"
    return part.astype(str)


def ordinal_in_recording(groups):
    """Position of every window inside its recording (0, 1, 2, ...) in cache order; recordings are contiguous."""
    # first window of every recording: index 0 plus every position where the group label changes
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(groups)]
    out = np.empty(len(groups), np.int64)
    for a, b in zip(starts, ends):
        out[a:b] = np.arange(b - a)
    return out


def reconstruct_positions(P, Xq, gq):
    """Re-run the verified loader on Data_clean (read-only) with the quantum benchmark's window rule
    (prediction_quantum/data.py build_features lines 113-127) and return, per window in cache order, the recording,
    the segment index inside the recording, the start sample inside the filtered segment and the segment length,
    plus the max abs difference between the recomputed and the cached 570 features."""
    from eeg_data import LoadConfig, load_recordings
    from improved_gnb_features import extract_features
    cfg = LoadConfig()
    t0 = time.perf_counter()
    recordings, skipped = load_recordings(P["data_dir"], cfg, n_jobs=-1, verbose=False)
    rec, seg, start, seglen, feats = [], [], [], [], []
    n_nonfinite = 0
    for r in recordings:
        rel = str(Path(r["path"]).relative_to(P["data_dir"])) if Path(r["path"]).is_absolute() else r["path"]
        for k, s in enumerate(r["segments"]):
            s = np.asarray(s, np.float64)
            n_rows = s.shape[0]
            if n_rows < 8:
                continue
            st = window_starts(n_rows)
            # a segment shorter than 128 samples gives exactly one window covering the whole segment
            win_len = min(128, n_rows)
            from numpy.lib.stride_tricks import sliding_window_view
            # W is (n_windows, channels, win_len): sliding_window_view puts the window axis last (as cut_windows in data.py)
            W = sliding_window_view(s, win_len, axis=0)[st]
            if not np.isfinite(W).all():
                n_nonfinite += int((~np.isfinite(W)).sum())
                W = np.nan_to_num(W)
            feats.append(extract_features(W, fs=cfg.sfreq))
            rec.extend([rel] * len(st))
            seg.extend([k] * len(st))
            start.extend(st.tolist())
            seglen.extend([n_rows] * len(st))
    Xr = np.concatenate(feats).astype(np.float32)
    rec = np.asarray(rec, dtype=str)
    res = dict(seconds=round(time.perf_counter() - t0, 1), n_recordings_loaded=len(recordings), n_skipped=len(skipped),
               n_windows_reconstructed=int(len(rec)), nonfinite_samples_zeroed=n_nonfinite)
    same_len = len(rec) == len(gq)
    res["same_number_of_windows"] = bool(same_len)
    if same_len:
        res["recording_sequence_identical"] = bool(np.array_equal(rec, gq))
        d = np.abs(Xr.astype(np.float64) - Xq.astype(np.float64))
        res["recomputed_vs_cached_570_max_abs_diff"] = float(d.max())
        res["recomputed_vs_cached_570_exactly_equal"] = bool(np.array_equal(Xr, Xq))
        res["recomputed_vs_cached_rows_with_any_diff"] = int((d > 0).any(1).sum())
        # 1e-12 floor: an exact zero in the cache would otherwise give inf or nan
        denom = np.maximum(np.abs(Xq.astype(np.float64)), 1e-12)
        res["recomputed_vs_cached_570_max_rel_diff"] = float((d / denom).max())
    return res, dict(recording=rec, segment=np.asarray(seg), start=np.asarray(start), segment_len=np.asarray(seglen))


def alignment(P, data, skip_positions=False):
    """Prove, numerically, that window_index of the quantum split CSVs addresses the same windows as the GNB caches."""
    from improved_gnb_features import participant_of
    npz, js, info = quantum_cache_path(P)
    zq = np.load(npz, allow_pickle=True)
    Xq, yq, gq, pq = zq["X"].astype(np.float32), zq["y"].astype(np.int64), zq["groups"].astype(str), zq["participants"].astype(str)
    meta = json.loads(js.read_text())
    # the GNB caches store absolute recording paths, the quantum cache paths relative to Data_clean; strip the prefix
    prefix = str(P["data_dir"]) + "/"
    gi = data["groups"]
    A = dict(quantum_cache=str(npz), quantum_cache_meta=str(js), quantum_cache_shape=list(Xq.shape),
             improved_cache=str(latest(P["gnb_cache"], "improved_windows_*.npz")),
             subject_cache=str(latest(P["gnb_cache"], "subject_windows_*.npz")),
             improved_cache_shape=list(data["X_imp"].shape), subject_cache_shape=list(data["X_sub"].shape),
             quantum_meta_n_windows=meta.get("n_windows"), quantum_meta_n_recordings=meta.get("n_recordings"),
             quantum_meta_n_participants=meta.get("n_participants"), quantum_meta_feature_version=meta["spec"]["feature_version"])
    A["gnb_groups_all_under_data_dir"] = bool(all(g.startswith(prefix) for g in gi))
    gi_rel = np.array([g[len(prefix):] if g.startswith(prefix) else g for g in gi])
    # ---- 1. whole-cache key matching by (recording, position inside the recording)
    oq, oi = ordinal_in_recording(gq), ordinal_in_recording(gi_rel)
    A["quantum_recordings_contiguous"] = bool(1 + int(np.sum(gq[1:] != gq[:-1])) == len(np.unique(gq)))
    A["gnb_recordings_contiguous"] = bool(1 + int(np.sum(gi_rel[1:] != gi_rel[:-1])) == len(np.unique(gi_rel)))
    key_to_gnb = {(g, int(o)): j for j, (g, o) in enumerate(zip(gi_rel, oi))}
    A["gnb_keys_unique"] = bool(len(key_to_gnb) == len(gi_rel))
    # q_to_gnb[i] = row of the GNB caches holding quantum window i (-1 if unmatched); used by every later lookup
    q_to_gnb = np.array([key_to_gnb.get((g, int(o)), -1) for g, o in zip(gq, oq)])
    A["cache_windows_aligned_by_recording_and_position"] = int((q_to_gnb >= 0).sum())
    A["cache_alignment_is_bijection"] = bool((q_to_gnb >= 0).all() and len(np.unique(q_to_gnb)) == len(gq) == len(gi_rel))
    A["cache_alignment_is_identity"] = bool(np.array_equal(q_to_gnb, np.arange(len(gq))))
    d = np.abs(Xq.astype(np.float64) - data["X_imp"][q_to_gnb].astype(np.float64))
    A["cache_570_max_abs_diff_all_12089"] = float(d.max())
    A["cache_570_bitwise_equal_all_12089"] = bool(np.array_equal(Xq, data["X_imp"][q_to_gnb]))
    A["cache_labels_equal"] = bool(np.array_equal(yq, data["y"][q_to_gnb]))
    A["cache_participants_equal"] = bool(np.array_equal(pq, data["parts"][q_to_gnb]))
    A["cache_participant_of_quantum_groups_equal"] = bool(np.array_equal(pq, np.array([participant_of(g) for g in gq])))
    # subject cache = the same windows: equal groups / labels (load_caches) and equal shared log-covariance columns
    shared = 240   # log-covariance 120 + log-covariance of the differenced signal 120, first in both feature sets
    ds = np.abs(data["X_sub"][:, :shared].astype(np.float64) - data["X_imp"][:, :shared].astype(np.float64))
    A["subject_vs_improved_first_240_columns_max_abs_diff"] = float(ds.max())
    # ---- 2. independent matching by the exact bytes of the 570 feature values
    # SHA-1 of the raw float32 bytes of each row; must give the same mapping as the (recording, position) keys above
    hi = {}
    for j, row in enumerate(data["X_imp"]):
        hi.setdefault(hashlib.sha1(row.tobytes()).hexdigest(), []).append(j)
    A["gnb_feature_rows_unique"] = bool(all(len(v) == 1 for v in hi.values()))
    by_hash = np.array([hi.get(hashlib.sha1(row.tobytes()).hexdigest(), [-1])[0] for row in Xq])
    A["feature_hash_match_found"] = int((by_hash >= 0).sum())
    A["feature_hash_match_equals_position_match"] = bool(np.array_equal(by_hash, q_to_gnb))
    # ---- 3. loader re-run: (recording, segment, start sample) of every window
    positions = None
    if not skip_positions and P["data_dir"].is_dir():
        res, positions = reconstruct_positions(P, Xq, gq)
        A["loader_reconstruction"] = res
    else:
        A["loader_reconstruction"] = "skipped (--skip-positions or Data_clean missing)"
    # ---- 4. the split CSVs
    A["subsample_reproduced"] = None
    keep = subsample_per_class(yq, 2000, 0)
    A["splits"] = {}
    for seed in SEEDS:
        df = pd.read_csv(P["qres"] / f"split_seed{seed}.csv")
        wi = df["window_index"].to_numpy()
        # j = GNB cache row of every window listed in the split file
        j = q_to_gnb[wi]
        s = dict(rows=int(len(df)), unique_window_index=bool(len(np.unique(wi)) == len(wi)),
                 window_index_in_range=bool(((wi >= 0) & (wi < len(gq))).all()),
                 window_index_sorted=bool(np.all(np.diff(wi) > 0)),
                 equals_subsample_per_class_2000_seed0=bool(np.array_equal(np.sort(wi), keep)),
                 recording_matches_quantum_cache=int((df["recording"].to_numpy().astype(str) == gq[wi]).sum()),
                 participant_matches_quantum_cache=int((df["participant"].to_numpy().astype(str) == pq[wi]).sum()),
                 label_matches_quantum_cache=int((df["label"].to_numpy().astype(str) == np.array(CLASSES)[yq[wi]]).sum()),
                 aligned_to_gnb_uniquely=int(((j >= 0) & (len(np.unique(j)) == len(j))).sum()),
                 recording_matches_gnb_cache=int((df["recording"].to_numpy().astype(str) == gi_rel[j]).sum()),
                 participant_matches_gnb_cache=int((df["participant"].to_numpy().astype(str) == data["parts"][j]).sum()),
                 label_matches_gnb_cache=int((df["label"].to_numpy().astype(str) == np.array(CLASSES)[data["y"][j]]).sum()),
                 features_570_max_abs_diff=float(np.abs(Xq[wi].astype(np.float64) - data["X_imp"][j].astype(np.float64)).max()),
                 features_570_bitwise_equal=bool(np.array_equal(Xq[wi], data["X_imp"][j])),
                 parts={p: int((df.part == p).sum()) for p in ("train", "val", "test")},
                 recordings_per_part={p: int(df.loc[df.part == p, "recording"].nunique()) for p in ("train", "val", "test")},
                 participants_per_part={p: int(df.loc[df.part == p, "participant"].nunique()) for p in ("train", "val", "test")},
                 train_test_recordings_overlap=int(len(set(df.loc[df.part == "train", "recording"]) & set(df.loc[df.part == "test", "recording"]))),
                 val_test_recordings_overlap=int(len(set(df.loc[df.part == "val", "recording"]) & set(df.loc[df.part == "test", "recording"]))),
                 train_val_recordings_overlap=int(len(set(df.loc[df.part == "train", "recording"]) & set(df.loc[df.part == "val", "recording"]))),
                 test_participants_absent_from_train=sorted(set(df.loc[df.part == "test", "participant"]) - set(df.loc[df.part == "train", "participant"])))
        part = make_split(yq[keep], gq[keep], seed, 0.2, 0.1)
        s["part_column_reproduced_by_make_split"] = bool(np.array_equal(np.sort(wi), keep) and np.array_equal(part, df["part"].to_numpy().astype(str)))
        if positions is not None:
            s["first_test_window_position"] = dict(zip(("recording", "segment", "start_sample_in_segment"),
                                                       (str(positions["recording"][wi[df.part.to_numpy() == "test"][0]]),
                                                        int(positions["segment"][wi[df.part.to_numpy() == "test"][0]]),
                                                        int(positions["start"][wi[df.part.to_numpy() == "test"][0]]))))
        A["splits"][f"seed{seed}"] = s
    A["subsample_reproduced"] = bool(all(A["splits"][f"seed{s}"]["equals_subsample_per_class_2000_seed0"] for s in SEEDS))
    # the hard requirement: all 11,698 windows of every split file found in the GNB caches, bit-identical features,
    # same label / participant / recording; the loader re-run above is informative only and does not gate the run
    ok = (A["cache_alignment_is_bijection"] and A["cache_labels_equal"] and A["cache_participants_equal"]
          and A["feature_hash_match_equals_position_match"]
          and all(A["splits"][f"seed{s}"]["aligned_to_gnb_uniquely"] == A["splits"][f"seed{s}"]["rows"] == 11698
                  and A["splits"][f"seed{s}"]["features_570_max_abs_diff"] == 0.0
                  and A["splits"][f"seed{s}"]["label_matches_gnb_cache"] == 11698
                  and A["splits"][f"seed{s}"]["participant_matches_gnb_cache"] == 11698
                  and A["splits"][f"seed{s}"]["recording_matches_gnb_cache"] == 11698 for s in SEEDS))
    A["ALIGNMENT_PROVEN"] = bool(ok)
    return A, dict(Xq=Xq, yq=yq, gq=gq, pq=pq, q_to_gnb=q_to_gnb, positions=positions)


# ----------------------------------------------------------------------------------------------- 2. models per seed
def latency_rows(name, decision_fn, full_fn, batched_fn, n_test):
    """Three latencies of one model, as the VQC worker measures them: the decision step alone on one prepared window,
    the full single-window path from raw features, and the batched cost per window over the whole test set."""
    lat, p90 = median_latency(decision_fn)
    lat_f, p90_f = median_latency(full_fn)
    # untimed warm-up call, then the whole test set once
    batched_fn()
    t0 = time.perf_counter()
    batched_fn()
    return dict(latency_ms=lat, latency_p90_ms=p90, latency_full_ms=lat_f, latency_full_p90_ms=p90_f,
                batched_us_per_window=(time.perf_counter() - t0) / n_test * 1e6)


def timed_fits(fn, repeats):
    """Fit `repeats` times (the fits are deterministic) and return (last result, median seconds, all seconds): the
    machine is a shared desktop, so the median of a few wall-clock fits is reported instead of a single one."""
    times, res = [], None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        res = fn()
        times.append(time.perf_counter() - t0)
    return res, float(np.median(times)), [round(t, 4) for t in times]


def run_seed(P, data, AL, seed, variant, fit_repeats=3):
    """Fit and evaluate the three GNB chains on one split file (seed) and one fit variant (train / train+val).
    Returns rows (four dicts: the improved chain is scored with argmax and with the threshold rule; timings,
    latencies, chain details), preds per row, extra sanity flags,
    and the test labels / window indices / participants, all in the split file's test order."""
    import torch
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(P["qres"] / f"split_seed{seed}.csv")
    fit_parts = ["train"] if variant == V_TRAIN else ["train", "val"]
    # fit windows sorted by window_index (the split CSVs are written sorted, so this is the benchmark's own train
    # order); test kept in CSV order
    wi_fit = np.sort(df.loc[df.part.isin(fit_parts), "window_index"].to_numpy())
    wi_te = df.loc[df.part == "test", "window_index"].to_numpy()
    # quantum window_index -> GNB cache row; the alignment proof records whether this is the identity but does not
    # require it, so go through the map
    tr, te = AL["q_to_gnb"][wi_fit], AL["q_to_gnb"][wi_te]
    y, parts = data["y"], data["parts"]
    ytr, yte, ptr, pte = y[tr], y[te], parts[tr], parts[te]
    out = dict(rows=[], preds={}, extra={})
    base = dict(seed=seed, variant=variant, n_train=int(len(tr)), n_test=int(len(te)),
                n_train_recordings=int(len(np.unique(data["groups"][tr]))), n_test_recordings=int(len(np.unique(data["groups"][te]))),
                n_test_participants=int(len(np.unique(pte))))

    # ---- a. subject-specific chain (435 subject features)
    Xs_tr, Xs_te = data["X_sub"][tr], data["X_sub"][te]
    (sc, model), fit_s, fit_all = timed_fits(lambda: fit_subject_chain(Xs_tr, ytr, ptr), fit_repeats)
    logp = model.predict_log_proba(sc.transform(Xs_te, pte), pte)
    pred = np.asarray(logp).argmax(1)
    if variant == V_TRAIN:
        logp_v, _ = subject_chain_logp(Xs_tr, ytr, ptr, Xs_te, pte)
        out["extra"]["subject_chain_logp_verbatim_identical"] = bool(np.array_equal(np.asarray(logp_v).argmax(1), pred))
    # decision-step latency: the first test participant's own model (or the global fallback) on one projected window
    q0 = str(pte[0])
    m0 = model.models_.get(q0, model.global_)
    Z1 = sc.transform(Xs_te[:1], pte[:1])
    P1 = torch.from_numpy(m0.transform(Z1).astype(np.float32))

    def dec():
        with torch.no_grad():
            m0.nb_._joint_log_likelihood(P1).argmax(1)
    x1 = Xs_te[:1]
    lat = latency_rows(M_SUB, dec, lambda: model.predict_log_proba(sc.transform(x1, pte[:1]), pte[:1]).argmax(1),
                       lambda: model.predict_log_proba(sc.transform(Xs_te, pte), pte).argmax(1), len(te))
    summ = model.summary()
    out["rows"].append(dict(base, model=M_SUB, input="435 subject features -> participant z-score -> own LDA (<=5) -> TorchGaussianNB",
                            fit_s=fit_s, fit_s_all=" ".join(map(str, fit_all)), **lat, latency_input=f"projected {m0.projection_.shape[1]}-dim window of participant {q0} (own model: {q0 in model.models_})",
                            n_participants_own_model=len(summ["participants_with_model"]),
                            fallback_participants=";".join(f"{k} ({v})" for k, v in summ["fallback"].items()),
                            test_windows_on_fallback=int(np.sum(~np.isin(pte, summ["participants_with_model"])))))
    out["preds"][M_SUB] = pred

    # ---- b. improved global chain (570 improved features), refitted as the notebook does
    Xi_tr, Xi_te = data["X_imp"][tr], data["X_imp"][te]
    chain, fit_s, fit_all = timed_fits(lambda: ImprovedGlobalChain().fit(Xi_tr, ytr, ptr), fit_repeats)
    logp_i = chain.log_posteriors(Xi_te, pte)
    pred_i = logp_i.argmax(1)
    # the deployed rule works on probabilities, hence exp of the log-posteriors
    pred_it = predict_with_thresholds(np.exp(logp_i), chain.thresholds)
    Pi1 = torch.from_numpy(chain.projected(Xi_te[:1], pte[:1]))

    def dec_i():
        with torch.no_grad():
            chain.model._joint_log_likelihood(Pi1).argmax(1)
    xi1 = Xi_te[:1]
    lat = latency_rows(M_IMP, dec_i, lambda: chain.log_posteriors(xi1, pte[:1]).argmax(1),
                       lambda: chain.log_posteriors(Xi_te, pte).argmax(1), len(te))
    lat_t = latency_rows(M_IMP_T, dec_i, lambda: predict_with_thresholds(np.exp(chain.log_posteriors(xi1, pte[:1])), chain.thresholds),
                         lambda: predict_with_thresholds(np.exp(chain.log_posteriors(Xi_te, pte)), chain.thresholds), len(te))
    common = dict(fit_s=fit_s, fit_s_all=" ".join(map(str, fit_all)), n_participants_own_model=len(chain.scaler.known_participants()),
                  fallback_participants="", test_windows_on_fallback=int(np.sum(~np.isin(pte, chain.scaler.known_participants()))),
                  latency_input="projected 5-dim window (global LDA)", thresholds=" ".join(f"{t:.2f}" for t in chain.thresholds))
    out["rows"].append(dict(base, model=M_IMP, input="570 improved features -> participant z-score -> global LDA (5) -> TorchGaussianNB (oversampled fit)", **common, **lat))
    out["rows"].append(dict(base, model=M_IMP_T, input="as above, argmax(probability / calibrated threshold)", **common, **lat_t))
    out["preds"][M_IMP], out["preds"][M_IMP_T] = pred_i, pred_it

    # ---- c. sklearn GaussianNB on the 570 standardised features (the quantum benchmark's gnb_full)
    Xq_tr, Xq_te = AL["Xq"][wi_fit], AL["Xq"][wi_te]
    # the benchmark fits the scaler in phase_data and the GaussianNB in baselines; t1 separates the two so the
    # model-only fit time can be reported next to the whole-pipeline one
    def fit_skl():
        t0 = time.perf_counter()
        scaler = StandardScaler().fit(Xq_tr)                                        # run_benchmark.phase_data
        Atr = scaler.transform(Xq_tr).astype(np.float32)                            # runs.model_inputs
        t1 = time.perf_counter()
        gnb = GaussianNB().fit(np.asarray(Atr, np.float64), ytr)                    # baselines.fit_sklearn
        return scaler, gnb, time.perf_counter() - t1
    (scaler, gnb, _), fit_s, fit_all = timed_fits(fit_skl, fit_repeats)
    fit_model_only = float(np.median([fit_skl()[2] for _ in range(max(1, fit_repeats))]))
    # float32 then float64 on purpose: runs.model_inputs casts to float32 before sklearn sees the data, and the
    # rounding has to be the same for the predictions to be bit-identical to the stored gnb_full ones
    Ate = scaler.transform(Xq_te).astype(np.float32)
    pred_g = gnb.predict(Ate.astype(np.float64))
    a1 = Ate[:1].astype(np.float64)
    xq1 = Xq_te[:1]
    lat = latency_rows(M_SKL, lambda: gnb.predict(a1), lambda: gnb.predict(scaler.transform(xq1).astype(np.float32).astype(np.float64)),
                       lambda: gnb.predict(scaler.transform(Xq_te).astype(np.float32).astype(np.float64)), len(te))
    out["rows"].append(dict(base, model=M_SKL, input="570 improved features -> global StandardScaler -> sklearn GaussianNB",
                            fit_s=fit_s, fit_s_all=" ".join(map(str, fit_all)), fit_s_model_only=fit_model_only, **lat, latency_input="standardised 570-dim window",
                            n_participants_own_model=0, fallback_participants="", test_windows_on_fallback=0))
    out["preds"][M_SKL] = pred_g
    if variant == V_TRAIN:
        stored = np.load(P["qres"] / "runs" / f"gnb_full_s{seed}" / "test_predictions.npy").astype(np.int64)
        out["extra"]["gnb_full_predictions_identical_to_stored"] = bool(np.array_equal(pred_g, stored))
        out["extra"]["gnb_full_n_pred_differ_from_stored"] = int((pred_g != stored).sum())
    out["y_test"], out["wi_test"], out["test_parts"] = yte, wi_te, pte
    return out


# ----------------------------------------------------------------------------------------------- 5. fidelity checks
def fidelity_checks(P, data):
    """(i) the copied subject chain reproduces the six recorded hold-outs of project_models_benchmark (all 12,089
    windows, GroupShuffleSplit 20 %, seeds 0-5) in defense_project_models_benchmark_runs.csv; (ii) the refitted improved
    chain, on the notebook's own split (StratifiedGroupKFold 5, shuffle, seed 42, first fold), reproduces the saved
    PyTorch artifacts (thresholds, GaussianNB statistics, LDA projection, test accuracy, window predictions)."""
    import joblib
    import torch
    from gaussiannb_model import GaussianNB as TorchGaussianNB
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    F = {}
    rec = pd.read_csv(P["gnb"] / "defense" / "defense_project_models_benchmark_runs.csv")
    rec = rec[rec.Model == "GaussianNB subject-specific chain (PyTorch)"].set_index("Seed")
    Xs, y, groups, parts = data["X_sub"], data["y"], data["groups"], data["parts"]
    rows = []
    for seed in range(6):     # project_models_benchmark lines 892-902, PyTorch backend
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(Xs, y, groups))
        logp, _ = subject_chain_logp(Xs[tr], y[tr], parts[tr], Xs[te], parts[te])
        s = _scores(y[te], np.asarray(logp).argmax(1))
        r = rec.loc[seed]
        rows.append(dict(seed=seed, n_train=int(len(tr)), n_test=int(len(te)), accuracy=s["Accuracy"], recorded=float(r["Accuracy"]),
                         abs_diff=abs(s["Accuracy"] - float(r["Accuracy"])), macro_f1=s["Macro-F1"], recorded_macro_f1=float(r["Macro-F1"]),
                         n_matches_recorded=bool(int(r["n_train"]) == len(tr) and int(r["n_test"]) == len(te))))
    acc = np.array([r["accuracy"] for r in rows])
    F["subject_chain_six_holdouts"] = dict(runs=rows, mean=float(acc.mean()), sd=float(acc.std(ddof=1)),
                                           recorded_mean=float(rec["Accuracy"].mean()), recorded_sd=float(rec["Accuracy"].std(ddof=1)),
                                           max_abs_diff=float(max(r["abs_diff"] for r in rows)))
    # (ii) improved chain on the notebook split (defense notebook_split, lines 170-178)
    X = data["X_imp"]
    # integer group codes, as the notebook passes them
    codes = np.unique(groups, return_inverse=True)[1]
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=codes))
    chain = ImprovedGlobalChain().fit(X[tr], y[tr], parts[tr])
    logp = chain.log_posteriors(X[te], parts[te])
    pred_arg, pred_thr = logp.argmax(1), predict_with_thresholds(np.exp(logp), chain.thresholds)
    gt = P["gnb_torch"]
    ck = torch.load(gt / "torch_gaussiannb_improved_features.pth", map_location="cpu", weights_only=False)
    prep_art = joblib.load(gt / "torch_gaussiannb_preprocessing.joblib")
    sc_art = joblib.load(gt / "torch_gaussiannb_participant_scaler.joblib")
    nb_art = TorchGaussianNB(num_features=int(ck["num_features"]), num_classes=len(ck["label_encoder_classes"]))
    nb_art.load_state_dict(ck["model_state_dict"])
    P_art = np.ascontiguousarray(prep_art.transform(sc_art.transform(X[te], parts[te])).astype(np.float32))
    with torch.no_grad():
        jll_art = nb_art._joint_log_likelihood(torch.from_numpy(P_art)).numpy().astype(np.float64)
    logp_art = _log_softmax(jll_art)
    thr_art = np.asarray(ck["class_thresholds"], np.float64)
    pred_art_arg, pred_art_thr = logp_art.argmax(1), predict_with_thresholds(np.exp(logp_art), thr_art)
    # saved state minus the is_fitted flag: what is left is means, variances and class_priors
    sd_art = {k: v.numpy() for k, v in ck["model_state_dict"].items() if k != "is_fitted"}
    F["improved_chain_notebook_split"] = dict(
        n_train=int(len(tr)), n_test=int(len(te)),
        refit_test_accuracy_threshold_rule=float((pred_thr == y[te]).mean()),
        artifact_checkpoint_test_accuracy=float(ck["test_accuracy"]),
        artifact_recomputed_test_accuracy_threshold_rule=float((pred_art_thr == y[te]).mean()),
        refit_test_accuracy_argmax=float((pred_arg == y[te]).mean()),
        artifact_recomputed_test_accuracy_argmax=float((pred_art_arg == y[te]).mean()),
        thresholds_refit=chain.thresholds.tolist(), thresholds_artifact=thr_art.tolist(),
        thresholds_equal=bool(np.allclose(chain.thresholds, thr_art)),
        nb_means_max_abs_diff=float(np.abs(chain.model.means.numpy() - sd_art["means"]).max()),
        nb_variances_max_abs_diff=float(np.abs(chain.model.variances.numpy() - sd_art["variances"]).max()),
        nb_priors_max_abs_diff=float(np.abs(chain.model.class_priors.numpy() - sd_art["class_priors"]).max()),
        lda_scalings_max_abs_diff=float(np.abs(chain.prep.named_steps["lda"].scalings_ - prep_art.named_steps["lda"].scalings_).max()),
        projected_test_max_abs_diff=float(np.abs(chain.projected(X[te], parts[te]).astype(np.float64) - P_art.astype(np.float64)).max()),
        argmax_predictions_identical_to_artifact=bool(np.array_equal(pred_arg, pred_art_arg)),
        threshold_predictions_identical_to_artifact=bool(np.array_equal(pred_thr, pred_art_thr)),
        n_argmax_predictions_differ=int((pred_arg != pred_art_arg).sum()))
    return F


# ----------------------------------------------------------------------------------------------- environment
def git(P, *args):
    """Stdout of a read-only git command in the repository, or an 'ERROR ...' string instead of raising (the
    environment record must still be written when git or a branch is missing)."""
    try:
        return subprocess.run(["git", "-C", str(P["repo"]), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e}"


def environment(P, a, vqc_versions, files):
    """Provenance record for matched_environment.json: machine, both interpreters' package versions, git state and,
    for every (label, path) in files, its SHA-256 and whether it equals the blob at HEAD."""
    import joblib
    import scipy
    import sklearn
    import torch
    env = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S %z"), command=" ".join([sys.executable] + sys.argv),
               script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__).resolve()),
               platform=platform.platform(), cpu=_cpu(), logical_cpus=os.cpu_count(), torch_threads=TORCH_THREADS,
               load_average_at_start=getattr(a, "load_at_start", None),
               load_average_at_end=list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
               fit_repeats=a.fit_repeats,
               latency=dict(repeats=LAT_REPEATS, warmup=LAT_WARMUP, definition="median of single-window calls (prediction_quantum/metrics.py median_latency_ms)"),
               gnb_python=dict(executable=sys.executable, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                               scikit_learn=sklearn.__version__, pandas=pd.__version__, torch=torch.__version__, joblib=joblib.__version__),
               vqc_python=vqc_versions)
    env["git"] = dict(repo=str(P["repo"]), HEAD=git(P, "rev-parse", "HEAD"), HEAD_branch=git(P, "rev-parse", "--abbrev-ref", "HEAD"),
                      **{f"{b}": git(P, "log", "-1", "--format=%H %ci %s", b) for b in ("gnb-improved-pipeline", "qml-benchmark-52")},
                      note="read-only: rev-parse / log / hash-object (no -w); the working tree has one user-modified notebook "
                           "(prediction-gaussiannb/pytorch/gaussiannb_train.ipynb) and one untracked file "
                           "(prediction-quantum/results/vqcmean.png), neither used here")
    rows = []
    for label, p in files:
        p = Path(p)
        rel = str(p.relative_to(P["repo"])) if str(p).startswith(str(P["repo"])) else str(p)
        r = dict(label=label, path=str(p), exists=p.exists())
        if p.exists():
            r["sha256"], r["bytes"] = sha256(p), p.stat().st_size
            blob_head = git(P, "rev-parse", f"HEAD:{rel}")
            r["git_tracked_at_HEAD"] = not blob_head.startswith("ERROR")
            if r["git_tracked_at_HEAD"]:
                r["git_blob_HEAD"] = blob_head
                # hash-object without -w only computes the blob id; equal ids mean the working file is unmodified
                r["working_file_equals_HEAD_blob"] = git(P, "hash-object", str(p)) == blob_head
        rows.append(r)
    env["files"] = rows
    return env


def _cpu():
    try:
        import re
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        return m.group(1).strip() if m else platform.processor()
    except OSError:
        return platform.processor()


# ----------------------------------------------------------------------------------------------- main
def main(a):
    """The whole benchmark in the ai_env interpreter: alignment proof, VQC worker subprocess, GNB models per seed and
    variant, metrics / McNemar / fidelity, then the matched_*.csv / .json files under --out."""
    import torch
    a.load_at_start = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    P = repo_paths(a.repo)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the repository is read-only for this script; outputs have to go to a directory outside it
    if str(out_dir.resolve()).startswith(str(P["repo"].resolve())):
        raise SystemExit("refusing to write inside the repository")
    setup_gnb_imports(P)
    torch.set_num_threads(TORCH_THREADS)

    say("loading the GNB caches (defense load_caches) and proving the alignment")
    data = load_caches(P["gnb_cache"])
    A, AL = alignment(P, data, skip_positions=a.skip_positions)
    pos = AL["positions"]
    if pos is not None:
        pd.DataFrame(dict(window_index=np.arange(len(pos["recording"])), recording=pos["recording"], segment=pos["segment"],
                          start_sample_in_segment=pos["start"], segment_samples=pos["segment_len"],
                          gnb_cache_row=AL["q_to_gnb"])).to_csv(out_dir / "matched_window_positions.csv.gz", index=False)
    (out_dir / "matched_alignment.json").write_text(json.dumps(A, indent=2, default=str))
    say(f"alignment proven: {A['ALIGNMENT_PROVEN']}")
    if not A["ALIGNMENT_PROVEN"]:
        raise SystemExit("ALIGNMENT NOT PROVEN - stopping; see matched_alignment.json")

    say("VQC worker (PennyLane environment): stored predictions + re-evaluation of params.pt")
    # PennyLane lives in qml_env, not ai_env, so the script re-runs itself there in worker mode and reads the two
    # files it leaves under --out (predictions .npz, report .json)
    env_w = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([a.qml_python, str(Path(__file__).resolve()), "--vqc-worker", "--repo", str(P["repo"]), "--out", str(out_dir)],
                   check=True, cwd=str(out_dir), env=env_w)
    W = json.loads((out_dir / "matched_vqc_worker.json").read_text())
    Z = np.load(out_dir / "matched_vqc_test_predictions.npz")

    runs, per_class, confusions, mcn, preds_long, extras = [], [], {}, [], [], {}
    for seed in SEEDS:
        wr = next(r for r in W["runs"] if r["seed"] == seed)
        seed_preds = {}
        for variant in (V_TRAIN, V_TRAINVAL):
            say(f"seed {seed}, fit on {variant}")
            o = run_seed(P, data, AL, seed, variant, a.fit_repeats)
            extras[f"seed{seed}_{variant}"] = o["extra"]
            if not np.array_equal(Z[f"s{seed}_window_index"], o["wi_test"]):
                raise SystemExit("VQC test windows differ from the GNB test windows")
            yte = o["y_test"]
            vqc_pred = Z[f"s{seed}_pred"].astype(np.int64)
            for r in o["rows"]:
                m, pc, cm = all_metrics(yte, o["preds"][r["model"]])
                runs.append(dict(r, **m))
                per_class += [dict(seed=seed, model=r["model"], variant=variant, **x) for x in pc]
                confusions.setdefault(f"seed{seed}", {})[f"{r['model']} | {variant}"] = cm.tolist()
            # paired tests on the same test windows: every GNB model against the VQC, then subject vs improved chain
            for name in (M_SUB, M_IMP, M_IMP_T, M_SKL):
                t = mcnemar_exact(o["preds"][name] == yte, vqc_pred == yte)
                mcn.append(dict(seed=seed, variant=variant, model_a=name, model_b=M_VQC, n_test=int(len(yte)),
                                acc_a=float((o["preds"][name] == yte).mean()), acc_b=float((vqc_pred == yte).mean()), **t))
            t = mcnemar_exact(o["preds"][M_SUB] == yte, o["preds"][M_IMP] == yte)
            mcn.append(dict(seed=seed, variant=variant, model_a=M_SUB, model_b=M_IMP, n_test=int(len(yte)),
                            acc_a=float((o["preds"][M_SUB] == yte).mean()), acc_b=float((o["preds"][M_IMP] == yte).mean()), **t))
            seed_preds[variant] = o
        # VQC rows (same predictions for both GNB fit variants: the circuit is not retrained)
        o = seed_preds[V_TRAIN]
        yte = o["y_test"]
        for which, key in (("re-evaluated params.pt", f"s{seed}_pred"), ("stored test_predictions.npy", f"s{seed}_stored_pred")):
            m, pc, cm = all_metrics(yte, Z[key].astype(np.int64))
            label = V_VQC if which.startswith("re-") else V_VQC + " [stored predictions]"
            runs.append(dict(seed=seed, variant=label, model=M_VQC, n_train=wr["n_train"], n_test=wr["n_test"],
                             n_train_recordings=o["rows"][0]["n_train_recordings"], n_test_recordings=o["rows"][0]["n_test_recordings"],
                             n_test_participants=o["rows"][0]["n_test_participants"],
                             input=f"570 improved features -> StandardScaler -> PCA-{BEST_Q} -> RY angles in [0, pi] -> circuit -> Linear({BEST_Q}, 6)",
                             fit_s=wr["train_seconds_benchmark_csv"], latency_ms=wr["latency_ms"], latency_p90_ms=wr["latency_p90_ms"],
                             latency_full_ms=wr["latency_full_ms"], latency_full_p90_ms=wr["latency_full_p90_ms"],
                             batched_us_per_window=wr["batched_us_per_window"], latency_ms_benchmark_csv=wr["latency_ms_benchmark_csv"],
                             latency_input=f"encoded {BEST_Q}-angle window", n_val_for_checkpoint=wr["n_val"], best_epoch=wr["best_epoch"],
                             n_parameters=wr["n_parameters"], **m))
            per_class += [dict(seed=seed, model=M_VQC, variant=label, **x) for x in pc]
            confusions[f"seed{seed}"][f"{M_VQC} | {label}"] = cm.tolist()
        # one row per test window with every model's predicted command (matched_test_predictions.csv.gz)
        long = pd.DataFrame(dict(seed=seed, window_index=o["wi_test"], participant=o["test_parts"], label=np.array(CLASSES)[yte]))
        for variant in (V_TRAIN, V_TRAINVAL):
            for name, pr in seed_preds[variant]["preds"].items():
                long[f"{name} | {variant}"] = np.array(CLASSES)[pr]
        long[f"{M_VQC} | {V_VQC}"] = np.array(CLASSES)[Z[f"s{seed}_pred"].astype(np.int64)]
        preds_long.append(long)

    say("fidelity checks against the repository's recorded results")
    F = fidelity_checks(P, data) if not a.skip_fidelity else "skipped"

    runs_df = pd.DataFrame(runs)
    lead = ["seed", "model", "variant", "accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "cohen_kappa",
            "n_train", "n_test", "fit_s", "latency_ms"]
    runs_df = runs_df[lead + [c for c in runs_df.columns if c not in lead]]
    runs_df.to_csv(out_dir / "matched_benchmark_runs.csv", index=False)
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "cohen_kappa", "fit_s", "latency_ms",
               "latency_full_ms", "batched_us_per_window", "n_train", "n_test"]
    # mean and sample sd over the three seeds; sort=False keeps the models in order of first appearance
    g = runs_df.groupby(["model", "variant"], sort=False)
    summ = g.size().rename("n_seeds").reset_index()
    for c in metrics:
        summ[f"{c}_mean"] = g[c].mean().values
        summ[f"{c}_sd"] = g[c].std(ddof=1).values
    summ.to_csv(out_dir / "matched_benchmark_summary.csv", index=False)
    pc = pd.DataFrame(per_class)
    pc.to_csv(out_dir / "matched_per_class.csv", index=False)
    pcs = pc.groupby(["model", "variant", "cls"], sort=False).agg(precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
                                                                    recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                                                                    f1_mean=("f1", "mean"), f1_sd=("f1", "std"), support_total=("support", "sum")).reset_index()
    pcs.to_csv(out_dir / "matched_per_class_summary.csv", index=False)
    total = {}
    for s, d in confusions.items():
        for k, cm in d.items():
            total[k] = (np.array(total.get(k, np.zeros((6, 6), int))) + np.array(cm)).tolist()
    (out_dir / "matched_confusions.json").write_text(json.dumps(dict(labels=CLASSES, rows="true command", columns="predicted command",
                                                                     per_seed=confusions, summed_over_seeds=total), indent=1))
    pd.DataFrame(mcn).to_csv(out_dir / "matched_mcnemar.csv", index=False)
    pd.concat(preds_long, ignore_index=True).to_csv(out_dir / "matched_test_predictions.csv.gz", index=False)

    # sanity checks against the quantum benchmark CSV
    # the sklearn GaussianNB (train variant) and the VQC must reproduce benchmark_results.csv to 1e-6: same windows,
    # same code path, so anything larger means the split files or the caches are not the ones the benchmark used
    bench = pd.read_csv(P["qres"] / "benchmark_results.csv")
    sanity = []
    for seed in SEEDS:
        for rid, model, variant in ((f"gnb_full_s{seed}", M_SKL, V_TRAIN), (f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}", M_VQC, V_VQC)):
            br = bench[bench.run_id == rid].iloc[0]
            mr = runs_df[(runs_df.seed == seed) & (runs_df.model == model) & (runs_df.variant == variant)].iloc[0]
            sanity.append(dict(seed=seed, run_id=rid, model=model, accuracy_matched=float(mr.accuracy), accuracy_benchmark_csv=float(br.accuracy),
                               macro_f1_matched=float(mr.macro_f1), macro_f1_benchmark_csv=float(br.macro_f1),
                               balanced_accuracy_matched=float(mr.balanced_accuracy), balanced_accuracy_benchmark_csv=float(br.balanced_accuracy),
                               n_test_matched=int(mr.n_test), n_test_benchmark_csv=int(br.n_test), n_train_matched=int(mr.n_train), n_train_benchmark_csv=int(br.n_train),
                               max_abs_diff=float(max(abs(mr.accuracy - br.accuracy), abs(mr.macro_f1 - br.macro_f1), abs(mr.balanced_accuracy - br.balanced_accuracy))),
                               passed=bool(max(abs(mr.accuracy - br.accuracy), abs(mr.macro_f1 - br.macro_f1), abs(mr.balanced_accuracy - br.balanced_accuracy)) < 1e-6
                                           and int(mr.n_test) == int(br.n_test) and int(mr.n_train) == int(br.n_train))))
    sanity_df = pd.DataFrame(sanity)
    sanity_df.to_csv(out_dir / "matched_sanity_vs_quantum_benchmark.csv", index=False)
    checks = dict(sanity_vs_quantum_benchmark=sanity, extras=extras, vqc_worker=W["runs"], fidelity=F,
                  all_sanity_passed=bool(sanity_df.passed.all()))
    (out_dir / "matched_checks.json").write_text(json.dumps(checks, indent=2, default=str))

    files = [("quantum feature cache", A["quantum_cache"]), ("quantum feature cache meta", A["quantum_cache_meta"]),
             ("GNB improved cache (570)", A["improved_cache"]), ("GNB subject cache (435)", A["subject_cache"]),
             ("quantum benchmark_results.csv", P["qres"] / "benchmark_results.csv"), ("quantum dataset.json", P["qres"] / "dataset.json"),
             ("quantum config_used.yaml", P["qres"] / "config_used.yaml"),
             ("defense_project_models_benchmark_runs.csv", P["gnb"] / "defense" / "defense_project_models_benchmark_runs.csv"),
             ("make_defense_figures.py", P["gnb"] / "defense" / "make_defense_figures.py"), ("subject_gnb.py", P["gnb"] / "subject_gnb.py"),
             ("improved_gnb_features.py", P["gnb"] / "improved_gnb_features.py"), ("gaussiannb_model.py", P["gnb_torch"] / "gaussiannb_model.py"),
             ("GaussianNB_EEG_Classification_Notebook.ipynb", P["gnb_torch"] / "GaussianNB_EEG_Classification_Notebook.ipynb"),
             ("improved artifact .pth", P["gnb_torch"] / "torch_gaussiannb_improved_features.pth"),
             ("improved artifact preprocessing", P["gnb_torch"] / "torch_gaussiannb_preprocessing.joblib"),
             ("improved artifact participant scaler", P["gnb_torch"] / "torch_gaussiannb_participant_scaler.joblib"),
             ("prediction_quantum/data.py", P["quantum"] / "prediction_quantum" / "data.py"),
             ("prediction_quantum/vqc.py", P["quantum"] / "prediction_quantum" / "vqc.py"),
             ("prediction_quantum/encoding.py", P["quantum"] / "prediction_quantum" / "encoding.py"),
             ("prediction_quantum/metrics.py", P["quantum"] / "prediction_quantum" / "metrics.py")]
    for seed in SEEDS:
        files += [(f"split_seed{seed}.csv", P["qres"] / f"split_seed{seed}.csv"),
                  (f"encoder_q{BEST_Q}_s{seed}.joblib", P["qres"] / "encoders" / f"encoder_q{BEST_Q}_s{seed}.joblib"),
                  (f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}/params.pt", P["qres"] / "runs" / f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}" / "params.pt"),
                  (f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}/test_predictions.npy", P["qres"] / "runs" / f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}" / "test_predictions.npy"),
                  (f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}/metrics.json", P["qres"] / "runs" / f"vqc_q{BEST_Q}_l{BEST_L}_s{seed}" / "metrics.json"),
                  (f"gnb_full_s{seed}/test_predictions.npy", P["qres"] / "runs" / f"gnb_full_s{seed}" / "test_predictions.npy")]
    env = environment(P, a, W["versions"], files)
    (out_dir / "matched_environment.json").write_text(json.dumps(env, indent=2, default=str))
    say("done: " + ", ".join(sorted(p.name for p in out_dir.glob("matched_*"))))
    show = summ[summ.variant.isin([V_TRAIN, V_VQC, V_TRAINVAL])][["model", "variant", "n_seeds", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "latency_ms_mean"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--qml-python", default=DEFAULT_QML_PY, help="interpreter with PennyLane for the VQC re-evaluation")
    ap.add_argument("--skip-positions", action="store_true", help="do not re-run the loader to recover window positions")
    ap.add_argument("--skip-fidelity", action="store_true", help="skip the fidelity checks against the recorded GNB results")
    ap.add_argument("--fit-repeats", type=int, default=3, help="wall-clock fits per GNB model; the median is reported (default 3)")
    ap.add_argument("--vqc-worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.vqc_worker:
        vqc_worker(args)
    else:
        main(args)
