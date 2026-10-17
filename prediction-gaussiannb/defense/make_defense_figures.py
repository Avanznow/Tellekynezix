"""Defense figures and tables for the GaussianNB chains (PyTorch, TensorFlow, JAX).

Everything is recomputed from the saved artifacts and the notebook's feature caches, on the notebook's own split
(cell 7: one fold of a stratified group split by recording, seed 42), so no notebook has to be re-run:

  1. confusion matrices and per-class metrics of the improved global chain and the subject-specific chain;
  2. per-participant accuracy of the subject-specific chain (bar chart of the notebook's CSV);
  3. vote-gate operating points: coverage against accuracy of accepted decisions for k-of-5 agreement;
  4. confound controls recomputed on the cache: time-only classifier, label permutation, temporal 60/40 split,
     separability of the two halves of one command block, adjacent-block error share, held-out participants;
  5. framework equivalence: PyTorch, TensorFlow and JAX decisions on the same windows, from the same arrays and
     separately fitted, plus the three notebooks' own protocol numbers and window geometry;
  6. latency: the decision models alone (fit, first call, steady state; CPU and GPU where present) and the two live
     chains end to end per capture, split into signal-to-features and features-to-decision;
  7. more confusion matrices of the PyTorch chains: per recording, at the deployed vote gate (with the withheld
     column), the progression original -> improved -> subject-specific, held-out participants, one per participant;
  8. figures: one participant's own LDA against the global LDA, a decision timeline, the permutation null, posterior
     confidence and vote agreement of right against wrong decisions, accuracy against the time distance to the
     training recordings, the learning curve per participant, LDA loadings per channel, the dataset's windows;
  9. tables: dataset counts, information transfer rate, intervals and tests, the subject chain's ablation, model
     complexity and artifact sizes, one-vs-rest average precision and ROC AUC of the deployed chains.

    /home/prasndhdl/ai_env/bin/python prediction-gaussiannb/defense/make_defense_figures.py [--skip-latency] [--skip-controls]
        [--skip-permutation] [--permutations N] [--skip-ablation] [--skip-learning-curve] [--skip-heldout] [--skip-dataset-load]

Needs Data_clean (one recording is read for the latency capture), predictions_local/feature_cache/ (the notebook's
improved_windows_*.npz and subject_windows_*.npz), the PyTorch artifacts in ../pytorch/ and, for the framework parts,
the TensorFlow and JAX pickles. Outputs go next to this script.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
GNB_DIR = HERE.parent
PYTORCH_DIR = GNB_DIR / "pytorch"
REPO_ROOT = GNB_DIR.parent
for p in (REPO_ROOT, GNB_DIR, PYTORCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import improved_gnb_inference as I  # noqa: E402
import subject_gnb_inference as S  # noqa: E402
from fast_preprocessing import ParallelPowerTransformer  # noqa: E402
from improved_gnb_features import ParticipantScaler, participant_of  # noqa: E402
from signal_quality.acquisition import read_openbci  # noqa: E402
from subject_gnb import SubjectSpecificGaussianNB  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.naive_bayes import GaussianNB as SkGaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler, label_binarize  # noqa: E402

CACHE_DIR = REPO_ROOT / "predictions_local" / "feature_cache"
# an empty user store for the chains, so the app's predictions_local/gnb_users* folders are never read or written;
# removed when the interpreter exits
TMP_USERS = tempfile.TemporaryDirectory(prefix="gnb_users_defense_")
COMMANDS = list(S.COMMANDS)
FS = 125.0
BACKENDS = ("torch", "tensorflow", "jax")
BACKEND_LABEL = {"torch": "PyTorch", "tensorflow": "TensorFlow", "jax": "JAX"}

# ------------------------------------------------------------------------------------------------------ chart style
# Categorical slots 1-3 of the reference palette (validated all-pairs), one-hue sequential ramp, text tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("gnb_blue", SEQ)
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})
CHANCE = 1.0 / len(COMMANDS)


def say(msg):
    print(f"[defense] {msg}", flush=True)


def value_of(obj, name):
    """Attribute that may be a property or a method on the chain classes."""
    v = getattr(obj, name)
    return v() if callable(v) else v


def write_csv(df, name):
    df.to_csv(HERE / name, index=False)
    say(f"wrote {name}")


def json_default(o):
    """numpy scalars and arrays, pandas missing values and anything else the summary may hold."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is pd.NA or (isinstance(o, float) and np.isnan(o)):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and not pd.isna(v):
                cells.append(floatfmt.format(v))
            else:
                cells.append("" if (not isinstance(v, str) and pd.isna(v)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------ data
def latest(pattern):
    files = sorted(CACHE_DIR.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {pattern} in {CACHE_DIR}")
    return files[-1]


def load_caches():
    zi = np.load(latest("improved_windows_*.npz"), allow_pickle=True)
    zs = np.load(latest("subject_windows_*.npz"), allow_pickle=True)
    groups = zs["groups"].astype(str)
    y = zs["y"].astype(np.int64)
    if not (np.array_equal(zi["groups"].astype(str), groups) and np.array_equal(zi["y"].astype(np.int64), y)):
        raise RuntimeError("the improved and subject feature caches do not hold the same windows")
    runs = 1 + int(np.sum(groups[1:] != groups[:-1]))
    if runs != len(np.unique(groups)):
        raise RuntimeError("windows of a recording are not contiguous in the cache")
    return dict(X_imp=zi["X"].astype(np.float32), X_sub=zs["X"].astype(np.float32), y=y, groups=groups,
                parts=np.array([participant_of(g) for g in groups]))


def notebook_split(X, y, groups):
    """Cell 7 of the notebook, verbatim: one fold of a stratified group split by recording, seed 42."""
    codes = np.unique(groups, return_inverse=True)[1]
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=codes))
    train = np.zeros(len(y), bool)
    train[tr] = True
    test = np.zeros(len(y), bool)
    test[te] = True
    return train, test


def window_times(groups):
    """Approximate acquisition time of every window: the recording's file timestamp plus index x stride / fs."""
    t = np.zeros(len(groups))
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(groups)]
    file_time = {}
    for a, b in zip(starts, ends):
        m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", groups[a])
        t0 = time.mktime(time.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")) if m else 0.0
        file_time[groups[a]] = t0
        t[a:b] = t0 + np.arange(b - a) * S.WINDOW_STRIDE / FS
    return t, file_time


# ------------------------------------------------------------------------------------------------------ 1. confusion
def chain_predictions(data, train, test):
    chain_imp = I.ImprovedGaussianNBChain()
    chain_sub = S.SubjectSpecificGaussianNBChain(users_dir=TMP_USERS.name)
    n = len(data["y"])
    logp_imp = np.full((n, len(COMMANDS)), np.nan)
    logp_sub = np.full((n, len(COMMANDS)), np.nan)
    for q in np.unique(data["parts"][test]):
        m = test & (data["parts"] == q)
        logp_imp[m] = chain_imp.log_posteriors(data["X_imp"][m], q)
        logp_sub[m] = chain_sub.log_posteriors(data["X_sub"][m], q)
    return chain_imp, chain_sub, logp_imp, logp_sub


def plot_cm(ax, cm, title, xlabels=None, ylabels=None, compact=False, show_x=True, show_y=True):
    """Row-normalised confusion matrix, count and share in every cell (share only when compact, for small
    multiples). Labels default to the six commands; xlabels may add a column (the gate's 'uncertain')."""
    xlabels = list(COMMANDS) if xlabels is None else list(xlabels)
    ylabels = list(COMMANDS) if ylabels is None else list(ylabels)
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(row, cmap=CMAP, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{row[i, j]:.0%}" if compact else f"{cm[i, j]}\n{row[i, j]:.0%}", ha="center", va="center",
                    fontsize=6 if compact else 8, color="white" if row[i, j] > 0.55 else INK)
    if compact:
        ax.set_xticks(range(cm.shape[1]), [SHORT.get(l, l) for l in xlabels] if show_x else [""] * cm.shape[1],
                      rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(cm.shape[0]), [SHORT.get(l, l) for l in ylabels] if show_y else [""] * cm.shape[0], fontsize=7)
    else:
        ax.set_xticks(range(cm.shape[1]), xlabels, rotation=30, ha="right")
        ax.set_yticks(range(cm.shape[0]), ylabels)
        ax.set_xlabel("Predicted command")
        ax.set_ylabel("True command")
    ax.set_xticks(np.arange(-0.5, cm.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8 if compact else 10, loc="left")


def confusion_and_per_class(data, train, test, logp_imp, logp_sub, chain_imp):
    y = data["y"][test]
    probs_imp = np.exp(logp_imp[test])
    pred_imp_thr = I.predict_with_thresholds(probs_imp, chain_imp.thresholds)
    pred_imp_arg = logp_imp[test].argmax(1)
    pred_sub = logp_sub[test].argmax(1)
    acc = lambda p: float((p == y).mean())  # noqa: E731
    say(f"improved global chain on the notebook's test windows: threshold rule {acc(pred_imp_thr):.4f}, "
        f"argmax {acc(pred_imp_arg):.4f}; subject-specific {acc(pred_sub):.4f} (n = {len(y)})")

    chains = [("Improved global chain (threshold rule, as deployed)", pred_imp_thr),
              ("Improved global chain (argmax)", pred_imp_arg),
              ("Subject-specific chain", pred_sub)]
    rows = []
    for name, pred in chains:
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(COMMANDS)), zero_division=0)
        for k, c in enumerate(COMMANDS):
            rows.append(dict(Chain=name, Class=c, Precision=p[k], Recall=r[k], F1=f[k], Support=int(s[k])))
        rows.append(dict(Chain=name, Class="all (accuracy / macro-F1 / balanced acc)", Precision=acc(pred),
                         Recall=f1_score(y, pred, average="macro"), F1=balanced_accuracy_score(y, pred), Support=int(len(y))))
    per_class = pd.DataFrame(rows)
    write_csv(per_class, "defense_per_class_metrics.csv")

    cm_imp = confusion_matrix(y, pred_imp_thr, labels=range(len(COMMANDS)))
    cm_sub = confusion_matrix(y, pred_sub, labels=range(len(COMMANDS)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    plot_cm(axes[0], cm_imp, f"Improved global chain, accuracy {acc(pred_imp_thr):.3f}\n(one GaussianNB for everyone)")
    plot_cm(axes[1], cm_sub, f"Subject-specific chain, accuracy {acc(pred_sub):.3f}\n(the participant's own GaussianNB)")
    fig.suptitle("PyTorch GaussianNB chains on the notebook's held-out recordings (count and share of each true command)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_confusion_matrices.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    plot_cm(ax, cm_imp, f"Improved global PyTorch GaussianNB, accuracy {acc(pred_imp_thr):.3f}\n"
                        "notebook's held-out recordings, count and share of each true command")
    fig.tight_layout()
    fig.savefig(HERE / "defense_improved_global_confusion_matrix.png")
    plt.close(fig)
    say("wrote defense_confusion_matrices.png, defense_improved_global_confusion_matrix.png")
    return dict(improved_threshold=acc(pred_imp_thr), improved_argmax=acc(pred_imp_arg), subject=acc(pred_sub),
                pred_sub=pred_sub, pred_imp=pred_imp_thr)


# ------------------------------------------------------------------------------------------------------ 2. per participant
def per_participant_chart():
    df = pd.read_csv(PYTORCH_DIR / "gaussiannb_subject_specific_per_participant.csv")
    df = df.sort_values("Accuracy").reset_index(drop=True)
    weighted = float((df["Accuracy"] * df["Test windows"]).sum() / df["Test windows"].sum())
    colors = [ORANGE if m == "own" else AQUA for m in df["Model"]]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(df["Participant"], df["Accuracy"], color=colors, height=0.62)
    for i, r in df.iterrows():
        ax.text(r["Accuracy"] + 0.01, i, f"{r['Accuracy']:.2f}", va="center", fontsize=8, color=INK)
        note = f"{int(r['Train windows'])} train windows, {int(r['Commands in train'])} commands"
        if r["Accuracy"] > 0.35:
            ax.text(0.01, i, note, va="center", fontsize=7, color=SURFACE)
        else:
            ax.text(r["Accuracy"] + 0.07, i, note, va="center", fontsize=7, color=INK2)
    ax.axvline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax.text(CHANCE + 0.005, -0.9, "chance", fontsize=8, color=INK2)
    ax.axvline(weighted, color=INK, linewidth=1, linestyle=":")
    ax.text(weighted + 0.005, len(df) - 0.3, f"all test windows {weighted:.3f}", fontsize=8, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Window accuracy on the participant's held-out recordings")
    ax.set_title("Subject-specific GaussianNB: accuracy per participant (PyTorch notebook split)", loc="left", fontsize=11)
    ax.legend(handles=[Patch(color=ORANGE, label="own model"), Patch(color=AQUA, label="global fallback (one command in training)")],
              loc="lower right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "defense_per_participant_accuracy.png")
    plt.close(fig)
    say(f"wrote defense_per_participant_accuracy.png (window-weighted accuracy {weighted:.4f})")
    return weighted


# ------------------------------------------------------------------------------------------------------ 3. vote gate
def vote_gate(data, test, logp_by_chain):
    rows, points = [], {}
    y, groups = data["y"], data["groups"]
    for name, logp in logp_by_chain.items():
        agree, correct = [], []
        for grp in np.unique(groups[test]):
            idx = np.flatnonzero(test & (groups == grp))          # contiguous, time-ordered windows of one recording
            lp = logp[idx]
            ctx = I.context_log_posteriors(lp, I.CONTEXT_WINDOWS)
            dec, votes = ctx.argmax(1), lp.argmax(1)
            for i in range(S.VOTE_WINDOWS - 1, len(idx)):           # decisions with a full 5-window context, as live
                agree.append(int((votes[i - S.VOTE_WINDOWS + 1:i + 1] == dec[i]).sum()))
                correct.append(bool(dec[i] == y[idx[i]]))
        agree, correct = np.array(agree), np.array(correct)
        pts = []
        for k in range(1, S.VOTE_WINDOWS + 1):
            m = agree >= k
            cov, accu = float(m.mean()), float(correct[m].mean()) if m.any() else float("nan")
            rows.append({"Chain": name, "Agreement needed": f"{k} of {S.VOTE_WINDOWS}", "Coverage": cov,
                         "Accuracy of accepted decisions": accu, "Decisions": int(len(agree)), "Accepted": int(m.sum()),
                         "Deployed": "yes" if k == S.MIN_AGREEMENT else ""})
            pts.append((k, cov, accu))
        points[name] = pts
    df = pd.DataFrame(rows)
    write_csv(df, "defense_vote_gate.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for (name, pts), color in zip(points.items(), (BLUE, ORANGE)):
        ks, cov, accu = zip(*pts)
        ax.plot(cov, accu, color=color, linewidth=2, marker="o", markersize=7, label=name)
        for k, c, a in pts:
            offset = (6, 5) if k != 1 else (6, -13)
            ax.annotate(f"{k}/5", (c, a), textcoords="offset points", xytext=offset, fontsize=8, color=INK2)
            if k == S.MIN_AGREEMENT:
                ax.scatter([c], [a], s=180, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=5)
    ax.axhline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.02, CHANCE + 0.01, "chance", fontsize=8, color=INK2)
    ax.set_xlabel("Coverage (share of 5-window decisions the gate lets through)")
    ax.set_ylabel("Accuracy of the accepted decisions")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_title("Vote-agreement gate: k of the last 5 window votes must agree with the context decision\n"
                 "(ring = deployed 4 of 5; notebook's held-out recordings, argmax votes)", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_vote_gate.png")
    plt.close(fig)
    say("wrote defense_vote_gate.png")
    return df


# ------------------------------------------------------------------------------------------------------ 4. confounds
def fit_eval_subject(Xtr, ytr, Xte, q, backend="torch"):
    """The per-participant chain (fit_user_model) on given training windows; argmax decisions on Xte, or None."""
    try:
        mean, std, model, _ = S.fit_user_model(Xtr, ytr, q, backend=backend)
    except ValueError:
        return None
    if model is None:
        return None
    return np.asarray(model.predict_log_proba((np.asarray(Xte, np.float64) - mean) / std)).argmax(1)


def confound_controls(data, train, test, pred_sub, subject_acc):
    X, y, groups, parts = data["X_sub"], data["y"], data["groups"], data["parts"]
    t, file_time = window_times(groups)
    participants = sorted(np.unique(parts[test]))
    rows = []

    # a. held-out participants (protocol C of the notebook's improved-vs-baseline table)
    ivb = pd.read_csv(PYTORCH_DIR / "gaussiannb_improved_vs_baseline_summary.csv")
    for chain_key, chain_name in (("Improved", "improved global chain"), ("Subject-specific", "subject-specific chain")):
        r = ivb[ivb["Protocol"].str.startswith("C.") & ivb["Chain"].str.startswith(chain_key)]
        if len(r):
            rows.append(dict(Control="Held-out participants (5-fold by participant)", Classifier=chain_name,
                             Value=float(r["Accuracy mean"].iloc[0]), Reference=CHANCE,
                             Reference_meaning="chance", Notebook="0.18", Reading="nothing transfers across people"))

    # b. time-only classifier within participant, same split
    hits = tot = 0
    for q in participants:
        tr, te = train & (parts == q), test & (parts == q)
        if tr.sum() < 10 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = KNeighborsClassifier(n_neighbors=1).fit(t[tr, None], y[tr])
        hits += int((clf.predict(t[te, None]) == y[te]).sum())
        tot += int(te.sum())
    rows.append(dict(Control="Acquisition time only, no EEG (label of the nearest training window in time)", Classifier="time of the window",
                     Value=hits / tot, Reference=CHANCE, Reference_meaning="chance", Notebook="0.97",
                     Reading="the split lets time alone predict the command"))

    # c. recording-level label permutation within participant
    accs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        hits = tot = 0
        for q in participants:
            tr, te = train & (parts == q), test & (parts == q)
            recs = np.unique(groups[tr])
            if len(recs) < 6 or te.sum() == 0:
                continue
            rec_label = {r: int(y[groups == r][0]) for r in recs}
            permuted = dict(zip(recs, rng.permutation([rec_label[r] for r in recs])))
            y_perm = np.array([permuted[g] for g in groups[tr]])
            pred = fit_eval_subject(X[tr], y_perm, X[te], q)
            if pred is None:
                continue
            hits += int((pred == y[te]).sum())
            tot += int(te.sum())
        accs.append(hits / tot)
    rows.append(dict(Control="Labels permuted across recordings within participant (3 seeds)", Classifier="subject-specific chain",
                     Value=float(np.mean(accs)), Reference=CHANCE, Reference_meaning="chance", Notebook="0.15",
                     Reading="the chain learns nothing from shuffled labels: no leak through features"))

    # d. temporal split: first 60 % of every command block trains, the last 40 % tests
    hits = tot = 0
    for q in participants:
        tr_mask = np.zeros(len(y), bool)
        te_mask = np.zeros(len(y), bool)
        for c in range(len(COMMANDS)):
            recs = sorted(np.unique(groups[(parts == q) & (y == c)]), key=lambda r: file_time[r])
            if len(recs) < 3:
                continue
            n_tr = int(np.ceil(0.6 * len(recs)))
            tr_mask |= np.isin(groups, recs[:n_tr])
            te_mask |= np.isin(groups, recs[n_tr:])
        if te_mask.sum() == 0:
            continue
        pred = fit_eval_subject(X[tr_mask], y[tr_mask], X[te_mask], q)
        if pred is None:
            continue
        hits += int((pred == y[te_mask]).sum())
        tot += int(te_mask.sum())
    rows.append(dict(Control="Train on the first 60 % of every command block, test on the last 40 %", Classifier="subject-specific chain",
                     Value=hits / tot, Reference=subject_acc, Reference_meaning="same chain, notebook split", Notebook="0.66",
                     Reading="accuracy falls with time distance to the training recordings"))

    # e. the two halves of one command block are separable (leave-one-recording-out inside the block)
    hits = tot = 0
    for q in participants:
        for c in range(len(COMMANDS)):
            recs = sorted(np.unique(groups[(parts == q) & (y == c)]), key=lambda r: file_time[r])
            if len(recs) < 4:
                continue
            half = {r: int(i >= len(recs) / 2) for i, r in enumerate(recs)}
            m = np.isin(groups, recs)
            Xb, hb, gb = X[m], np.array([half[g] for g in groups[m]]), groups[m]
            for r in recs:
                te = gb == r
                if len(np.unique(hb[~te])) < 2:
                    continue
                clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=1), SkGaussianNB())
                clf.fit(Xb[~te], hb[~te])
                hits += int((clf.predict(Xb[te]) == hb[te]).sum())
                tot += int(te.sum())
    rows.append(dict(Control="First half vs second half of the SAME command block", Classifier="LDA + GaussianNB, leave-one-recording-out",
                     Value=hits / tot, Reference=0.5, Reference_meaning="chance (2 classes)", Notebook="0.73",
                     Reading="windows carry the session's drift, not only the command"))

    # f. share of the subject chain's errors that go to the temporally adjacent command block
    adjacent = expected = n_err = 0
    yt = y[test]
    pt = parts[test]
    idx_test = np.flatnonzero(test)
    for q in participants:
        med = {}
        for c in range(len(COMMANDS)):
            m = (parts == q) & (y == c)
            if m.any():
                med[c] = float(np.median(t[m]))
        rank = {c: r for r, c in enumerate(sorted(med, key=med.get))}
        n_blocks = len(rank)
        m = (pt == q) & (pred_sub != yt)
        for true_c, pred_c in zip(yt[m], pred_sub[m]):
            if true_c not in rank or pred_c not in rank:
                continue
            n_err += 1
            adjacent += int(abs(rank[pred_c] - rank[true_c]) == 1)
            n_adj = int(rank[true_c] > 0) + int(rank[true_c] < n_blocks - 1)
            expected += n_adj / max(n_blocks - 1, 1)
    rows.append(dict(Control="Errors of the subject-specific chain landing on the adjacent block in time", Classifier="subject-specific chain",
                     Value=adjacent / max(n_err, 1), Reference=expected / max(n_err, 1), Reference_meaning="if errors ignored time",
                     Notebook="0.79", Reading="confusions follow the recording order"))

    df = pd.DataFrame(rows)
    df["Value"] = df["Value"].astype(float).round(4)
    df["Reference"] = df["Reference"].astype(float).round(4)
    write_csv(df, "defense_confound_controls.csv")
    return df


# ------------------------------------------------------------------------------------------------------ 5. frameworks
def framework_equivalence(data, train, test, chain_sub):
    X, y, parts = data["X_sub"], data["y"], data["parts"]
    same_agree = {b: [] for b in BACKENDS[1:]}
    same_maxdiff = {b: 0.0 for b in BACKENDS[1:]}
    sep_pred = {b: [] for b in BACKENDS}
    truth = []
    n_participants = 0
    for q in sorted(np.unique(parts[test])):
        mt, mtr = test & (parts == q), train & (parts == q)
        mean, std, model, status = chain_sub.resolve(q)
        if str(status).startswith("global fallback"):          # no per-participant model in the notebook's artifacts
            continue
        Zt = (np.asarray(X[mt], np.float64) - mean) / std
        lp_torch = I._log_softmax(np.asarray(model.predict_log_proba(Zt)))
        arrays = S.model_to_arrays(model)
        for b in BACKENDS[1:]:
            lp = I._log_softmax(np.asarray(S.model_from_arrays(arrays, b).predict_log_proba(Zt)))
            same_agree[b].append(lp.argmax(1) == lp_torch.argmax(1))
            same_maxdiff[b] = max(same_maxdiff[b], float(np.abs(np.exp(lp) - np.exp(lp_torch)).max()))
        preds = {b: fit_eval_subject(X[mtr], y[mtr], X[mt], q, backend=b) for b in BACKENDS}
        if any(p is None for p in preds.values()):
            continue
        for b in BACKENDS:
            sep_pred[b].append(preds[b])
        truth.append(y[mt])
        n_participants += 1
    truth = np.concatenate(truth)
    sep = {b: np.concatenate(v) for b, v in sep_pred.items()}
    rows = []
    for b in BACKENDS[1:]:
        a = np.concatenate(same_agree[b])
        rows.append({"Comparison": f"PyTorch vs {BACKEND_LABEL[b]}, same stored model arrays", "Windows": int(len(a)),
                     "Decision agreement": float(a.mean()), "Max |difference in class probability|": same_maxdiff[b],
                     "Accuracy PyTorch": float("nan"), "Accuracy other": float("nan")})
    for b in BACKENDS[1:]:
        rows.append({"Comparison": f"PyTorch vs {BACKEND_LABEL[b]}, separately fitted on the same training windows",
                     "Windows": int(len(truth)), "Decision agreement": float((sep[b] == sep["torch"]).mean()),
                     "Max |difference in class probability|": float("nan"),
                     "Accuracy PyTorch": float((sep["torch"] == truth).mean()), "Accuracy other": float((sep[b] == truth).mean())})
    df = pd.DataFrame(rows)
    write_csv(df, "defense_framework_equivalence.csv")
    say(f"framework equivalence over {n_participants} participants with own models")

    # the three notebooks' own protocol numbers and window geometry
    geo = []
    for b in BACKENDS:
        ci = I.ImprovedGaussianNBChain(backend=b)
        cs = S.SubjectSpecificGaussianNBChain(backend=b, users_dir=TMP_USERS.name)
        geo.append(dict(Framework=BACKEND_LABEL[b], window_size=ci.window_size, window_stride=ci.window_stride,
                        capture_samples=int(value_of(ci, "capture_samples")), capture_seconds=round(float(value_of(ci, "capture_seconds")), 2),
                        global_decision="per-class thresholds" if getattr(ci, "thresholds", None) is not None else "argmax",
                        subject_window_size=cs.window_size))
    geo = pd.DataFrame(geo)
    ivb = pd.read_csv(PYTORCH_DIR / "gaussiannb_improved_vs_baseline_summary.csv")
    nb_rows = []
    for key, name in (("Improved", "improved global chain"), ("Subject-specific", "subject-specific chain")):
        for proto, label in (("A.", "5-fold by recording (seed 42)"), ("C.", "5-fold by participant")):
            r = ivb[ivb["Protocol"].str.startswith(proto) & ivb["Chain"].str.startswith(key)]
            if len(r):
                nb_rows.append(dict(Framework="PyTorch", Chain=name, Protocol=label, Window_accuracy=float(r["Accuracy mean"].iloc[0]),
                                    File_accuracy=float(r["Per-recording Acc mean"].iloc[0]) if "Per-recording Acc mean" in r else float("nan")))
    for b, folder, prefix in (("tensorflow", "tensorflow", "gaussiannb_tf"), ("jax", "jax", "gaussiannb_jax")):
        f = GNB_DIR / folder / f"{prefix}_protocol_comparison.csv"
        if f.exists():
            pc = pd.read_csv(f)
            for _, r in pc.iterrows():
                chain = "improved global chain" if "improved" in r["Chain"] else "subject-specific chain"
                nb_rows.append(dict(Framework=BACKEND_LABEL[b], Chain=chain, Protocol=r["Protocol"],
                                    Window_accuracy=float(r["Window acc (mean)"]), File_accuracy=float(r["File acc, majority vote (mean)"])))
    nb = pd.DataFrame(nb_rows).merge(geo[["Framework", "window_size", "window_stride", "capture_samples"]], on="Framework", how="left")
    write_csv(nb, "defense_framework_notebooks.csv")
    return df, nb, geo


def framework_performance(data, train, test, chain_sub, nb=None):
    """Performance of the best chain (subject-specific) fitted and evaluated in each framework on the same test
    windows: accuracy, macro-F1, balanced accuracy, Cohen's kappa and per-command F1 (defense_framework_performance.csv),
    one figure with the scores and the per-command F1 per framework, and one figure with the three confusion
    matrices. The three backends are fitted separately on the same training windows (PyTorch notebook split, the
    participants with an own model), so the only difference between them is the implementation; the notebooks' own
    protocol-A numbers (each with its own window geometry) are appended from defense_framework_notebooks.csv."""
    X, y, parts = data["X_sub"], data["y"], data["parts"]
    preds = {b: [] for b in BACKENDS}
    truth = []
    n_participants = 0
    for q in sorted(np.unique(parts[test])):
        mt, mtr = test & (parts == q), train & (parts == q)
        _, _, _, status = chain_sub.resolve(q)
        if str(status).startswith("global fallback"):
            continue
        p = {b: fit_eval_subject(X[mtr], y[mtr], X[mt], q, backend=b) for b in BACKENDS}
        if any(v is None for v in p.values()):
            continue
        for b in BACKENDS:
            preds[b].append(p[b])
        truth.append(y[mt])
        n_participants += 1
    truth = np.concatenate(truth)
    preds = {b: np.concatenate(v) for b, v in preds.items()}
    labels = list(range(len(COMMANDS)))
    rows = []
    for b in BACKENDS:
        pr = preds[b]
        f1s = f1_score(truth, pr, labels=labels, average=None, zero_division=0)
        row = {"Framework": BACKEND_LABEL[b], "Chain": "subject-specific chain", "Windows": int(len(truth)),
               "Participants": n_participants, "Accuracy": float((pr == truth).mean()),
               "Macro-F1": float(f1_score(truth, pr, labels=labels, average="macro", zero_division=0)),
               "Balanced accuracy": float(balanced_accuracy_score(truth, pr)),
               "Cohen's kappa": float(cohen_kappa_score(truth, pr)),
               "Agreement with PyTorch": float((pr == preds["torch"]).mean())}
        for c, f in zip(COMMANDS, f1s):
            row[f"F1 {c}"] = float(f)
        rows.append(row)
    df = pd.DataFrame(rows)
    if nb is not None and len(nb):
        own = nb[(nb["Chain"] == "subject-specific chain") & nb["Protocol"].str.startswith("5-fold by recording")]
        own = own[["Framework", "Window_accuracy", "File_accuracy", "window_size", "window_stride"]].rename(columns={
            "Window_accuracy": "Notebook protocol A window accuracy", "File_accuracy": "Notebook protocol A per-recording accuracy",
            "window_size": "Notebook window size", "window_stride": "Notebook window stride"})
        df = df.merge(own, on="Framework", how="left")
    write_csv(df, "defense_framework_performance.csv")
    say("framework performance: " + ", ".join(f"{r['Framework']} acc {r['Accuracy']:.4f} F1 {r['Macro-F1']:.4f}" for r in rows))

    colors = {"PyTorch": "#ee4c2c", "TensorFlow": "#ff8c00", "JAX": "#3987e5"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.05, 1.5]})
    metrics = ["Accuracy", "Macro-F1", "Balanced accuracy", "Cohen's kappa"]
    w = 0.26
    xs = np.arange(len(metrics))
    for i, r in enumerate(rows):
        vals = [r[m] for m in metrics]
        bars = ax1.bar(xs + (i - 1) * w, vals, w, color=colors[r["Framework"]], label=r["Framework"])
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax1.axhline(CHANCE, color="#555555", ls="--", lw=1)
    ax1.text(xs[0] - 1.5 * w, CHANCE + 0.01, "chance", fontsize=8, color="#555555")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(metrics, fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("score on the same held-out windows")
    ax1.set_title(f"Subject-specific chain per framework\n({len(truth)} test windows, {n_participants} participants, fitted separately)", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    xs2 = np.arange(len(COMMANDS))
    for i, r in enumerate(rows):
        vals = [r[f"F1 {c}"] for c in COMMANDS]
        bars = ax2.bar(xs2 + (i - 1) * w, vals, w, color=colors[r["Framework"]], label=r["Framework"])
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(COMMANDS, fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("F1 score per command")
    ax2.set_title("Per-command F1 of the subject-specific chain per framework", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right", frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Gaussian Naive Bayes performance across PyTorch, TensorFlow and JAX: the same chain, the same windows, three implementations",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_framework_performance.png")
    plt.close(fig)
    say("wrote defense_framework_performance.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, b in zip(axes, BACKENDS):
        pr = preds[b]
        plot_cm(ax, confusion_matrix(truth, pr, labels=labels),
                f"{BACKEND_LABEL[b]}, accuracy {(pr == truth).mean():.3f}, macro-F1 "
                f"{f1_score(truth, pr, labels=labels, average='macro', zero_division=0):.3f}")
    fig.suptitle(f"Subject-specific chain fitted and evaluated in each framework on the same {len(truth)} held-out windows "
                 "(count and share of each true command)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_framework_confusion_matrices.png")
    plt.close(fig)
    say("wrote defense_framework_confusion_matrices.png")
    return df


# ------------------------------------------------------------------------------------------------------ 5b. the project's own RF and MLP
RF_JAX_DIR = REPO_ROOT / "prediction-random-forest" / "JAX"
DL_JAX_DIR = REPO_ROOT / "prediction_deep_learning" / "JAX"


def _import_project_module(path, name):
    """Import one of the project's original scripts (they print a banner and define module constants on import)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_random_split(n, seed):
    """The 70/10/20 random split of rows used by both original scripts (main())."""
    n_test, n_val = int(n * 0.2), int(n * 0.1)
    np.random.seed(seed)
    idx = np.random.permutation(n)
    return idx[:n - n_test - n_val], idx[n - n_test - n_val:n - n_test], idx[n - n_test:]


def _scores(yt, yp):
    labels = list(range(len(COMMANDS)))
    return dict(Accuracy=float((yp == yt).mean()),
                **{"Macro-F1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))},
                **{"Balanced accuracy": float(balanced_accuracy_score(yt, yp))})


def _old_loader_with_files(RF, base_path, desired_rows):
    """load_and_process_data of the original scripts, reproduced step by step (same walk, same parser, same padding
    and truncation, same label normalisation, same zero padding to the widest row), with the index of the source file
    recorded for every row so that the rows can be split by file."""
    all_samples, all_labels, file_ids = [], [], []
    n_files = 0
    for root, dirs, files in os.walk(str(base_path)):
        if "output" in root:
            continue
        csv_files = [f for f in files if f.endswith(".csv")]
        if not csv_files:
            continue
        class_label_raw = os.path.basename(root)
        if class_label_raw.startswith(("group", "individual", "Test")):
            continue
        class_label = RF.normalize_class_label(class_label_raw)
        for csv_file in csv_files:
            try:
                df = RF.read_csv_flexible(os.path.join(root, csv_file))
                if df.empty:
                    continue
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    continue
                df = df[numeric_cols]
                if len(df) < desired_rows:
                    df = pd.concat([df, pd.DataFrame(0, index=range(desired_rows - len(df)), columns=df.columns)], ignore_index=True)
                elif len(df) > desired_rows:
                    df = df.iloc[:desired_rows]
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
                vals = df.values.astype(np.float32)
                all_samples.extend(list(vals))
                all_labels.extend([class_label] * len(vals))
                file_ids.extend([n_files] * len(vals))
                n_files += 1
            except Exception:  # noqa: BLE001  (the original swallows every parse error)
                pass
    max_features = max(len(s) for s in all_samples)
    X = np.zeros((len(all_samples), max_features), np.float32)
    for i, sample in enumerate(all_samples):
        X[i, :min(len(sample), max_features)] = sample[:max_features]
    y = np.array(all_labels)
    class_names = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int32)
    return X, y_encoded, class_names, np.array(file_ids)


def _project_model_job(job):
    """One fit of one original model (runs in a worker process on the CPU): job = dict(model, Xtr, ytr, Xte, Xval, yval)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import io, contextlib
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):      # the scripts print a banner on import
        if job["model"] == "JAX random forest":
            RF = _import_project_module(RF_JAX_DIR / "jax_random_forest.py", "jax_random_forest_worker")
            m = RF.JAXRandomForest(n_trees=RF.N_TREES, max_depth=RF.MAX_DEPTH, bootstrap_ratio=RF.BOOTSTRAP_RATIO, seed=RF.SEED)
            m.fit(np.asarray(job["Xtr"], np.float32), np.asarray(job["ytr"], np.int32), verbose=False)
            pred = np.asarray(m.predict(np.asarray(job["Xte"], np.float32)), np.int64)
        else:
            DL = _import_project_module(DL_JAX_DIR / "deep_learning_jax.py", "deep_learning_jax_worker")
            import jax.numpy as jnp
            Xtr, ytr, Xval, yval = job["Xtr"], job["ytr"], job.get("Xval"), job.get("yval")
            if Xval is None:   # hold out 1/8 of the training rows for the script's early stopping
                rng = np.random.default_rng(DL.SEED); perm = rng.permutation(len(ytr)); k = len(ytr) // 8
                Xval, yval, Xtr, ytr = Xtr[perm[:k]], ytr[perm[:k]], Xtr[perm[k:]], ytr[perm[k:]]
            state, _, _ = DL.train_model(jnp.asarray(Xtr, jnp.float32), jnp.asarray(ytr, jnp.int32),
                                         jnp.asarray(Xval, jnp.float32), jnp.asarray(yval, jnp.int32), DL.N_EPOCHS, DL.BATCH_SIZE)
            _, pred = DL.eval_step(state, jnp.asarray(job["Xte"], jnp.float32), jnp.zeros(len(job["Xte"]), jnp.int32))
            pred = np.asarray(pred, np.int64)
    return dict(Model=job["model"], Condition=job["condition"], Seed=job["seed"], n_train=int(len(job["ytr"])), n_test=int(len(job["yte"])),
                **_scores(job["yte"], pred), seconds=time.perf_counter() - t1)


def project_models_benchmark(data, data_dir=None, n_repeats=6, max_rows_per_file=160, row_subsample=30000, window_subsample=None, n_jobs=4):
    """The project's ORIGINAL random forest (JAXRandomForest, prediction-random-forest/JAX/jax_random_forest.py) and
    deep-learning model (the Flax MLP of prediction_deep_learning/JAX/deep_learning_jax.py), run as written, under
    three conditions, so that the old renditions can be compared with the GNB chains on the same dataset:

      1. "as reported": the two scripts' own loader on Data_clean (the first 160 rows of every BrainFlow CSV export,
         every numeric column, label from the folder name) and their own 70/10/20 random split of ROWS (seed 42),
         which is how the 94.36 % and 93.60 % in the scripts' headers were obtained;
      2. "same code, split by recording": the same loader and rows, but the 20 % test rows are whole files
         (GroupShuffleSplit by file, seed 0), so no row of a test file is in training;
      3. "same code on the verified windows": the two models trained and tested on the loader's 12,089 windows with
         the GNB chains' 570 features, under the six recording-grouped hold-outs of the main notebook's cell 36
         (GroupShuffleSplit, 20 % of recordings, seeds 0 to 5), which is the protocol of the GNB benchmark.

    The models and their hyperparameters are the scripts' own (40 trees of depth 8; hidden layers 256-128-64-32,
    100 epochs with early stopping); only the data split changes. Because the original forest trains at roughly one
    tree per 20 s on 20,000 rows, conditions 1 and 2 use a random subsample of row_subsample rows of the loader's
    291,200 (every file keeps some rows, so the random-row leak and the split by file are both preserved). The fits
    run in n_jobs worker processes on the CPU. Writes defense_project_models_benchmark.csv (means),
    defense_project_models_benchmark_runs.csv and defense_project_models_benchmark.png. Needs flax and optax."""
    from sklearn.model_selection import GroupShuffleSplit
    from joblib import Parallel, delayed
    import io, contextlib
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / "Data_clean"
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), plt.rc_context():   # the script sets a seaborn style on import
        RF = _import_project_module(RF_JAX_DIR / "jax_random_forest.py", "jax_random_forest")
    for _n in dir(RF):                                   # the stored pickle was written from __main__
        if isinstance(getattr(RF, _n), type):
            setattr(sys.modules["__main__"], _n, getattr(RF, _n))
    reported = {}
    try:
        import pickle
        with open(RF_JAX_DIR / "trained_classifier_rf_jax.pkl", "rb") as f:
            reported["JAX random forest"] = float(pickle.load(f)["test_accuracy"])
        with open(DL_JAX_DIR / "trained_classifier_dl_jax.pkl", "rb") as f:
            reported["JAX deep learning (MLP)"] = float(pickle.load(f)["test_accuracy"])
    except Exception as e:  # noqa: BLE001
        say(f"stored pickles not read: {e}")
    say(f"stored test accuracies in the project's pickles: {reported}")
    models = ("JAX random forest", "JAX deep learning (MLP)")

    # ---- conditions 1 and 2: the scripts' own loader (rows of the BrainFlow CSV exports)
    X_rows, y_raw, class_names, file_of_row = _old_loader_with_files(RF, data_dir, max_rows_per_file)
    # The scripts' normalize_class_label leaves Data_clean's misspelled folders (B ackward, Foreward, Fowward, TakOff,
    # backard) as classes of their own; map every folder name to the six commands with the loader's label table so
    # that the models are trained on the same six classes they were reported on.
    try:
        from signal_quality.salvage import folder_label
        norm = lambda c: folder_label(str(c)) or RF.normalize_class_label(str(c))  # noqa: E731
    except Exception:  # noqa: BLE001
        fix = {"b ackward": "backward", "backard": "backward", "foreward": "forward", "fowward": "forward", "takoff": "takeoff"}
        norm = lambda c: fix.get(str(c).lower(), RF.normalize_class_label(str(c)))  # noqa: E731
    mapped = {i: norm(c) for i, c in enumerate(class_names)}
    bad = sorted(set(mapped.values()) - set(COMMANDS))
    assert not bad, f"unmapped folder labels: {bad}"
    y_rows = np.array([COMMANDS.index(mapped[int(v)]) for v in y_raw], np.int32)
    say("folder labels mapped to the six commands: " + ", ".join(f"{c} -> {mapped[i]}" for i, c in enumerate(class_names) if str(c) != mapped[i]))
    n_rows_all, n_files_all = len(y_rows), len(np.unique(file_of_row))
    if row_subsample is not None and row_subsample < n_rows_all:
        sub = np.sort(np.random.default_rng(0).choice(n_rows_all, row_subsample, replace=False))
    else:
        sub = np.arange(n_rows_all)
    X_rows, y_rows, file_of_row = X_rows[sub], y_rows[sub], file_of_row[sub]
    X_rows_std = (X_rows - X_rows.mean(0)) / (X_rows.std(0) + 1e-8)   # the MLP script standardises its rows
    say(f"old loader: {n_rows_all:,} rows of {X_rows.shape[1]} columns from {n_files_all} BrainFlow exports "
        f"({time.perf_counter() - t0:.0f} s); using {len(sub):,} rows of {len(np.unique(file_of_row))} files")
    jobs = []
    tr, va, te = _old_random_split(len(y_rows), RF.SEED)
    c1 = "1. as reported: own loader, random split of rows"
    jobs.append(dict(model=models[0], condition=c1, seed=RF.SEED, Xtr=X_rows[tr], ytr=y_rows[tr], Xte=X_rows[te], yte=y_rows[te]))
    jobs.append(dict(model=models[1], condition=c1, seed=RF.SEED, Xtr=X_rows_std[tr], ytr=y_rows[tr], Xte=X_rows_std[te], yte=y_rows[te],
                     Xval=X_rows_std[va], yval=y_rows[va]))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X_rows, y_rows, file_of_row))
    c2 = "2. same code and rows, split by recording"
    jobs.append(dict(model=models[0], condition=c2, seed=0, Xtr=X_rows[tr], ytr=y_rows[tr], Xte=X_rows[te], yte=y_rows[te]))
    jobs.append(dict(model=models[1], condition=c2, seed=0, Xtr=X_rows_std[tr], ytr=y_rows[tr], Xte=X_rows_std[te], yte=y_rows[te]))

    # ---- condition 3: the verified windows and the GNB chains' 570 features, six recording-grouped hold-outs
    X, y, groups = data["X_imp"], data["y"], data["groups"]
    if window_subsample is not None and window_subsample < len(y):
        keep = np.sort(np.random.default_rng(0).choice(len(y), window_subsample, replace=False))
        X, y, groups = X[keep], y[keep], groups[keep]
    c3 = "3. same code on the verified windows (570 features), split by recording"
    for seed in range(n_repeats):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = imp.transform(X[tr]).astype(np.float32), imp.transform(X[te]).astype(np.float32)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        jobs.append(dict(model=models[0], condition=c3, seed=seed, Xtr=Xtr, ytr=y[tr], Xte=Xte, yte=y[te]))
        jobs.append(dict(model=models[1], condition=c3, seed=seed, Xtr=(Xtr - mu) / sd, ytr=y[tr], Xte=(Xte - mu) / sd, yte=y[te]))
    say(f"{len(jobs)} fits of the original models in {n_jobs} worker processes (the forest is the slow part)")
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_project_model_job)(j) for j in jobs)
    for r in rows:
        r["Reported"] = reported.get(r["Model"], float("nan")) if r["Condition"].startswith("1.") else float("nan")
        say(f"{r['Model']}, {r['Condition'][:2]} seed {r['Seed']}: accuracy {r['Accuracy']:.4f}, macro-F1 {r['Macro-F1']:.4f} in {r['seconds']:.0f} s")
    df = pd.DataFrame(rows)
    write_csv(df, "defense_project_models_benchmark_runs.csv")
    summ = (df.groupby(["Model", "Condition"], sort=False)
              .agg(runs=("Seed", "size"), Accuracy=("Accuracy", "mean"), Accuracy_sd=("Accuracy", "std"),
                   **{"Macro-F1": ("Macro-F1", "mean"), "Macro-F1_sd": ("Macro-F1", "std"),
                      "Balanced accuracy": ("Balanced accuracy", "mean"), "Reported": ("Reported", "first"),
                      "n_train": ("n_train", "mean"), "n_test": ("n_test", "mean"), "seconds": ("seconds", "mean")}).reset_index())
    summ["rows_in_loader"] = n_rows_all
    summ["files_in_loader"] = n_files_all
    summ["rows_used"] = len(sub)
    write_csv(summ, "defense_project_models_benchmark.csv")

    # ---- figure
    conds = [c1, c2, c3]
    cond_label = {c1: f"as reported: own loader, random split of rows ({len(sub):,} rows)",
                  c2: "same code and rows, split by recording",
                  c3: f"same code on the verified windows (570 features), {n_repeats} recording-grouped hold-outs"}
    colors = ["#9e9e9e", "#ff8c00", "#3987e5"]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    names = list(models)
    w = 0.8 / (len(conds) + 1)
    for j, c in enumerate(conds):
        for i, name in enumerate(names):
            r = summ[(summ["Model"] == name) & (summ["Condition"] == c)]
            if not len(r):
                continue
            v = float(r["Accuracy"].iloc[0]); sd = float(r["Accuracy_sd"].iloc[0]) if int(r["runs"].iloc[0]) > 1 else 0.0
            bar = ax.bar(i + (j - (len(conds) - 1) / 2) * w, v, w, color=colors[j], yerr=sd if sd else None, capsize=3,
                         label=cond_label[c] if i == 0 else None)
            ax.text(bar[0].get_x() + w / 2, v + 0.02 + sd, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, name in enumerate(names):
        if name in reported:
            ax.plot([i - 0.45, i + 0.45], [reported[name]] * 2, color="#333333", lw=1.2, ls=":",
                    label="test accuracy stored in the project's pickle" if i == 0 else None)
    gnb = PYTORCH_DIR / "final_model_benchmark_project_models_summary.csv"
    if gnb.exists():
        b = pd.read_csv(gnb)
        for key, lab, col in (("subject-specific", "GNB subject-specific chain, same six hold-outs", "#ee4c2c"), ("improved", "GNB improved global chain, same six hold-outs", "#d62728")):
            r = b[b["Model"].str.contains(key)]
            if len(r):
                ax.axhline(float(r["Accuracy Mean"].iloc[0]), color=col, lw=1.0, ls="-." if key == "improved" else "-", label=lab)
    ax.axhline(CHANCE, color="#555555", ls="--", lw=1)
    ax.text(len(names) - 0.55, CHANCE + 0.012, "chance", fontsize=8, color="#555555")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title("The project's original random forest and MLP, run as written on Data_clean, under three data splits", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "defense_project_models_benchmark.png")
    plt.close(fig)
    say(f"wrote defense_project_models_benchmark.png in {time.perf_counter() - t0:.0f} s")
    return summ


# ------------------------------------------------------------------------------------------------------ 6. latency
def _median_ms(fn, repeat, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 90))


JAX_GPU_SCRIPT = r'''
import json, sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
import jax
devices = sorted({d.platform for d in jax.devices()})
if "gpu" not in devices and "cuda" not in devices:
    print(json.dumps({"skipped": f"no GPU backend ({devices})"})); raise SystemExit
from algorithm_gnb_jax import GaussianNBJAX
rng = np.random.default_rng(0); rows = []
for n in (600, 9670):
    X = rng.normal(size=(n, 5)).astype(np.float32); y = rng.integers(0, 6, n); Xq = rng.normal(size=(5, 5)).astype(np.float32)
    m = GaussianNBJAX(); t0 = time.perf_counter(); m.fit(X, y); fit_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); GaussianNBJAX().fit(X, y); fit2_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); np.asarray(m.predict_log_proba(Xq)); first = (time.perf_counter() - t0) * 1e3
    ts = []
    for _ in range(200):
        t0 = time.perf_counter(); np.asarray(m.predict_log_proba(Xq)); ts.append((time.perf_counter() - t0) * 1e3)
    rows.append(dict(Model="GaussianNBJAX", Device="GPU", n_train=n, first_fit_ms=fit_ms, second_fit_ms=fit2_ms, first_predict_ms=first,
                     steady_predict_ms=float(np.median(ts)), steady_predict_p90_ms=float(np.percentile(ts, 90))))
print(json.dumps(rows))
'''


def bench_decision_models():
    import torch
    from gaussiannb_model import GaussianNB as TorchGaussianNB
    rows = []
    rng = np.random.default_rng(0)
    for n in (600, 9670):
        X = rng.normal(size=(n, 5)).astype(np.float32)
        yv = rng.integers(0, len(COMMANDS), n)
        Xq = rng.normal(size=(5, 5)).astype(np.float32)
        # PyTorch, CPU and CUDA
        for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
            Xt, yt, Xqt = (torch.as_tensor(a).to(dev) for a in (X, yv, Xq))
            m = TorchGaussianNB(5, len(COMMANDS)).to(dev)
            sync()
            t0 = time.perf_counter()
            m.fit(Xt, yt)
            sync()
            fit_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            TorchGaussianNB(5, len(COMMANDS)).to(dev).fit(Xt, yt)
            sync()
            fit2_ms = (time.perf_counter() - t0) * 1e3
            with torch.no_grad():
                t0 = time.perf_counter()
                m.predict_proba(Xqt)
                sync()
                first = (time.perf_counter() - t0) * 1e3

                def call(m=m, Xqt=Xqt, sync=sync):
                    m.predict_proba(Xqt)
                    sync()
                med, p90 = _median_ms(call, 200)
            rows.append(dict(Model="PyTorch GaussianNB", Device=dev.upper(), n_train=n, first_fit_ms=fit_ms, second_fit_ms=fit2_ms,
                             first_predict_ms=first, steady_predict_ms=med, steady_predict_p90_ms=p90))
        # TensorFlow, CPU (no GPU visible to this TensorFlow build)
        try:
            from subject_gnb import tensorflow_backend
            TF = tensorflow_backend().GaussianNBTF
            m = TF()
            t0 = time.perf_counter()
            m.fit(X, yv)
            fit_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            TF().fit(X, yv)
            fit2_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            np.asarray(m.predict_log_proba(Xq))
            first = (time.perf_counter() - t0) * 1e3
            med, p90 = _median_ms(lambda: np.asarray(m.predict_log_proba(Xq)), 200)
            rows.append(dict(Model="TensorFlow GaussianNBTF", Device="CPU", n_train=n, first_fit_ms=fit_ms, second_fit_ms=fit2_ms,
                             first_predict_ms=first, steady_predict_ms=med, steady_predict_p90_ms=p90))
        except Exception as e:  # noqa: BLE001
            say(f"TensorFlow benchmark skipped: {e!r}")
        # JAX, CPU (pinned by the backend module)
        try:
            from subject_gnb import jax_backend
            JX = jax_backend().GaussianNBJAX
            m = JX()
            t0 = time.perf_counter()
            m.fit(X, yv)
            fit_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            JX().fit(X, yv)
            fit2_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            np.asarray(m.predict_log_proba(Xq))
            first = (time.perf_counter() - t0) * 1e3
            med, p90 = _median_ms(lambda: np.asarray(m.predict_log_proba(Xq)), 200)
            rows.append(dict(Model="JAX GaussianNBJAX", Device="CPU", n_train=n, first_fit_ms=fit_ms, second_fit_ms=fit2_ms,
                             first_predict_ms=first, steady_predict_ms=med, steady_predict_p90_ms=p90))
        except Exception as e:  # noqa: BLE001
            say(f"JAX CPU benchmark skipped: {e!r}")
    # JAX on the GPU, in a fresh process so the backend module's CPU pin does not apply
    try:
        env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
        out = subprocess.run([sys.executable, "-c", JAX_GPU_SCRIPT, str(GNB_DIR / "jax")], capture_output=True, text=True,
                             timeout=600, env=env)
        last = [l for l in out.stdout.strip().splitlines() if l.startswith("{") or l.startswith("[")]
        payload = json.loads(last[-1]) if last else {"skipped": out.stderr[-300:]}
        if isinstance(payload, dict):
            say(f"JAX GPU benchmark skipped: {payload.get('skipped')}")
        else:
            rows.extend(dict(r, Model="JAX GaussianNBJAX") for r in payload)
    except Exception as e:  # noqa: BLE001
        say(f"JAX GPU benchmark skipped: {e!r}")
    df = pd.DataFrame(rows)
    for c in ("first_fit_ms", "second_fit_ms", "first_predict_ms", "steady_predict_ms", "steady_predict_p90_ms"):
        df[c] = df[c].astype(float).round(3)
    write_csv(df, "defense_decision_model_latency.csv")
    return df


def bench_chains(data, participant="G03/person03"):
    paths = [g for g, p in zip(data["groups"], data["parts"]) if p == participant and Path(g).exists()]
    if not paths:
        raise FileNotFoundError(f"no Data_clean recording of {participant} on this machine")
    rec = read_openbci(sorted(paths)[0])
    raw, channels, fs = rec["raw"], rec["channels"], rec["sfreq"]
    rows = []
    users_tmp = TMP_USERS.name
    for b in BACKENDS:
        for name, chain in (("Improved global chain", I.ImprovedGaussianNBChain(backend=b)),
                            ("Subject-specific chain", S.SubjectSpecificGaussianNBChain(backend=b, users_dir=users_tmp))):
            n = int(value_of(chain, "capture_samples"))
            start = min(int(10 * fs), raw.shape[1] - n)
            cap = np.ascontiguousarray(raw[:, start:start + n])
            X = chain.features_from_raw(cap, channels, fs)
            for _ in range(5):                                        # warm-up (JAX compiles the capture shape once)
                chain.predict_windows(chain.features_from_raw(cap, channels, fs), participant)
            tf_ms, td_ms = [], []
            for _ in range(50):
                t0 = time.perf_counter()
                X = chain.features_from_raw(cap, channels, fs)
                t1 = time.perf_counter()
                chain.predict_windows(X, participant)
                t2 = time.perf_counter()
                tf_ms.append((t1 - t0) * 1e3)
                td_ms.append((t2 - t1) * 1e3)
            tf_ms, td_ms = np.array(tf_ms), np.array(td_ms)
            rows.append({"Chain": name, "Framework": BACKEND_LABEL[b], "Capture samples": n, "Capture seconds": round(n / fs, 2),
                         "Windows": int(len(X)), "Features": X.shape[1], "Signal to features ms": float(np.median(tf_ms)),
                         "Features to decision ms": float(np.median(td_ms)), "Total ms (median)": float(np.median(tf_ms + td_ms)),
                         "Total ms (p90)": float(np.percentile(tf_ms + td_ms, 90))})
            say(f"{name} / {BACKEND_LABEL[b]}: {np.median(tf_ms + td_ms):.1f} ms per capture")
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.endswith("ms") or "ms" in c:
            df[c] = df[c].astype(float).round(2)
    write_csv(df, "defense_latency_per_capture.csv")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"{r['Chain']}\n{r['Framework']}" for _, r in df.iterrows()]
    yp = np.arange(len(df))
    ax.barh(yp, df["Signal to features ms"], color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=1.5,
            label="signal to features (filter, windows, features, scaling)")
    ax.barh(yp, df["Features to decision ms"], left=df["Signal to features ms"], color=ORANGE, height=0.6,
            edgecolor=SURFACE, linewidth=1.5, label="features to decision (projection, GaussianNB, context)")
    ax.set_xlim(0, float(df["Total ms (median)"].max()) * 1.22)
    for i, r in df.iterrows():
        ax.text(r["Total ms (median)"] + 0.8, i, f"{r['Total ms (median)']:.1f} ms", va="center", fontsize=8, color=INK)
    ax.set_yticks(yp, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Milliseconds per live capture (median of 50, CPU, quality gate excluded)")
    ax.set_title("End-to-end latency of one 5-window capture, by chain and framework", loc="left", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_latency_per_capture.png")
    plt.close(fig)
    say("wrote defense_latency_per_capture.png")
    return df


# ------------------------------------------------------------------------------------------------------ 7. shared helpers
# Categorical slots 4-6 of the reference palette, so the six commands take slots 1-6 in COMMANDS order; the markers
# are the secondary encoding where six colours share one panel (scatter, curves).
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"
COMMAND_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
COMMAND_MARKERS = ["o", "s", "^", "D", "v", "P"]
SHORT = {"backward": "back", "forward": "fwd", "landing": "land", "left": "left", "right": "right", "takeoff": "take"}
TEST_PARTICIPANT = "G03/person03"          # the participant the tests use
WINDOW_SECONDS = I.WINDOW_SIZE / FS
CONTEXT_SECONDS = (I.WINDOW_SIZE + (I.CONTEXT_WINDOWS - 1) * I.WINDOW_STRIDE) / FS


def recording_decisions(logp, groups, y, test):
    """One decision per held-out recording: argmax of the sum of its windows' log-posteriors (the notebook's
    _recording_decisions). Returns (recordings, true label, decision)."""
    recs, y_rec, p_rec = [], [], []
    for grp in np.unique(groups[test]):
        idx = np.flatnonzero(test & (groups == grp))
        recs.append(grp)
        y_rec.append(int(y[idx[0]]))
        p_rec.append(int(logp[idx].sum(0).argmax()))
    return np.array(recs), np.array(y_rec), np.array(p_rec)


def context_decisions(logp, groups, test, k=I.CONTEXT_WINDOWS, vote_windows=S.VOTE_WINDOWS):
    """Per test window, in cache order: the k-window context decision, its max posterior, how many of the last
    vote_windows window votes agree with it, and whether a full context was available (the GUI decides per capture,
    so only full-context positions count as decisions)."""
    idx_test = np.flatnonzero(test)
    pos = {int(i): n for n, i in enumerate(idx_test)}
    n = len(idx_test)
    dec, agree = np.zeros(n, int), np.zeros(n, int)
    conf, full = np.zeros(n), np.zeros(n, bool)
    ctx_all = np.zeros((n, logp.shape[1]))
    for grp in np.unique(groups[test]):
        idx = np.flatnonzero(test & (groups == grp))
        lp = logp[idx]
        ctx = I.context_log_posteriors(lp, k)
        d, votes = ctx.argmax(1), lp.argmax(1)
        probs = np.exp(I._log_softmax(ctx))
        for i, j in enumerate(idx):
            p = pos[int(j)]
            dec[p], conf[p], ctx_all[p] = d[i], probs[i, d[i]], ctx[i]
            agree[p] = int((votes[max(0, i - vote_windows + 1):i + 1] == d[i]).sum())
            full[p] = i >= vote_windows - 1
    return dict(index=idx_test, decision=dec, confidence=conf, agreement=agree, full=full, context=ctx_all)


def wilson(k, n, z=1.959964):
    """Wilson score interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def wolpaw_bits(p, n=len(COMMANDS)):
    """Bits per decision (Wolpaw et al. 1998) for accuracy p over n equiprobable commands."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1)))


def subject_chain_logp(Xtr, ytr, ptr, Xte, pte, n_components=5, min_windows=S.MIN_WINDOWS):
    """The notebook's _subject_chain: ParticipantScaler on the training windows, then SubjectSpecificGaussianNB
    (own LDA + own GaussianNB for every participant with >= min_windows training windows and >= 2 commands, the
    global projection + GaussianNB as the fallback). Returns (log-posteriors of Xte, the fitted models)."""
    sc = ParticipantScaler().fit(Xtr, ptr)
    model = SubjectSpecificGaussianNB(n_components=n_components, shrinkage="auto", min_windows=min_windows).fit(
        sc.transform(Xtr, ptr), ytr, ptr)
    return model.predict_log_proba(sc.transform(Xte, pte), pte), model


def context_accuracy(logp, groups, y, test):
    """The notebook's Context-5 accuracy: the 5-window context decision at every test window (partial contexts at
    the start of a recording included)."""
    cd = context_decisions(logp, groups, test)
    return float((cd["decision"] == y[cd["index"]]).mean())


def own_model_participants():
    pp = pd.read_csv(PYTORCH_DIR / "gaussiannb_subject_specific_per_participant.csv")
    own = sorted(pp[pp["Model"] == "own"]["Participant"])
    fallback = pp[pp["Model"] != "own"]
    return own, fallback


def command_legend(ax, loc="upper right", markers=False, **kw):
    handles = [Line2D([], [], marker=COMMAND_MARKERS[k] if markers else "s", linestyle="", color=COMMAND_COLORS[k],
                      markersize=7, label=c) for k, c in enumerate(COMMANDS)]
    return ax.legend(handles=handles, loc=loc, fontsize=8, **kw)


# ------------------------------------------------------------------------------------------------------ 8. confusion matrices
def per_recording_confusions(data, test, logp_by_chain):
    """One decision per held-out recording for both chains, 6 x 6 row-normalised with counts."""
    y, groups = data["y"], data["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    out = {}
    for ax, (name, logp) in zip(axes, logp_by_chain.items()):
        recs, y_rec, p_rec = recording_decisions(logp, groups, y, test)
        acc = float((y_rec == p_rec).mean())
        cm = confusion_matrix(y_rec, p_rec, labels=range(len(COMMANDS)))
        plot_cm(ax, cm, f"{name}, per-recording accuracy {acc:.3f}\n(n = {len(y_rec)} held-out recordings)")
        out[name] = dict(accuracy=acc, recordings=int(len(y_rec)), correct=int((y_rec == p_rec).sum()))
        say(f"{name}: per-recording accuracy {acc:.4f} over {len(y_rec)} recordings")
    fig.suptitle("One decision per held-out recording: argmax of the sum of its windows' log-posteriors "
                 "(count and share of each true command)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_per_recording_confusion_matrices.png")
    plt.close(fig)
    say("wrote defense_per_recording_confusion_matrices.png")
    return out


def gated_confusions(data, test, logp_by_chain):
    """The 5-window context decision at the deployed 4-of-5 vote gate: 6 true x 7 predicted, the last column being
    the withheld ('uncertain') decisions; per-class coverage in a CSV. Both chains with argmax votes, as in the
    vote-gate figure."""
    y = data["y"]
    labels = COMMANDS + ["uncertain"]
    rows, out = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (name, logp) in zip(axes, logp_by_chain.items()):
        cd = context_decisions(logp, data["groups"], test)
        m = cd["full"]
        yt, dec = y[cd["index"]][m], cd["decision"][m]
        accepted = cd["agreement"][m] >= S.MIN_AGREEMENT
        pred = np.where(accepted, dec, len(COMMANDS))
        cm = confusion_matrix(yt, pred, labels=range(len(COMMANDS) + 1))[:len(COMMANDS)]
        cov = float(accepted.mean())
        acc_acc = float((dec[accepted] == yt[accepted]).mean())
        plot_cm(ax, cm, f"{name}: coverage {cov:.2f}, accuracy of accepted {acc_acc:.3f}\n"
                        f"({S.MIN_AGREEMENT} of {S.VOTE_WINDOWS} window votes must agree; n = {int(m.sum())} decisions)",
                xlabels=labels)
        for k, c in enumerate(COMMANDS + ["all"]):
            mc = np.ones(len(yt), bool) if c == "all" else (yt == k)
            acc_m = accepted & mc
            correct = (dec == yt) if c == "all" else (dec == k)
            rows.append({"Chain": name, "Class": c, "Decisions": int(mc.sum()), "Accepted": int(acc_m.sum()),
                         "Withheld": int((~accepted & mc).sum()), "Coverage": float(accepted[mc].mean()),
                         "Accuracy of accepted": float(correct[acc_m].mean()) if acc_m.any() else float("nan"),
                         "Accuracy counting withheld as wrong": float((acc_m & correct).sum() / mc.sum())})
        out[name] = dict(coverage=cov, accuracy_accepted=acc_acc, decisions=int(m.sum()), accepted=int(accepted.sum()))
    fig.suptitle("Deployed vote gate on the 5-window context decision: the last column is what the drone never receives",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_gated_confusion_matrix.png")
    plt.close(fig)
    df = pd.DataFrame(rows)
    write_csv(df, "defense_gated_confusion_matrix.csv")
    say("wrote defense_gated_confusion_matrix.png")
    return out


def load_hybrid_cache(data):
    """The notebook's hybrid (time statistics + relative band powers) windows, the original chain's input."""
    zh = np.load(latest("hybrid_windows_*.npz"), allow_pickle=True)
    if not (np.array_equal(zh["groups"].astype(str), data["groups"]) and np.array_equal(zh["y"].astype(np.int64), data["y"])):
        raise RuntimeError("the hybrid feature cache does not hold the same windows as the improved / subject caches")
    return zh["X"].astype(np.float32)


def baseline_predictions(Xh, y, train, test):
    """The notebook's original chain rebuilt with sklearn on the training windows only (cell 8 / cell 15): drop
    zero-variance features, median imputation, Yeo-Johnson power transform, PCA keeping 95 % of the variance,
    GaussianNB. Returns the test log-posteriors and the PCA dimensionality."""
    keep = Xh[train].var(0) > 1e-15
    pipe = make_pipeline(SimpleImputer(strategy="median"), ParallelPowerTransformer(method="yeo-johnson", standardize=True),
                         PCA(n_components=0.95, random_state=42), SkGaussianNB())
    pipe.fit(Xh[train][:, keep], y[train])
    return pipe.predict_log_proba(Xh[test][:, keep]), int(pipe[-2].n_components_)


def progression_confusions(data, train, test, pred_imp, pred_sub):
    """Three panels on the same test windows: the original chain (rebuilt from the hybrid cache), the improved global
    chain and the subject-specific chain."""
    y = data["y"][test]
    t0 = time.perf_counter()
    logp_base, n_pca = baseline_predictions(load_hybrid_cache(data), data["y"], train, test)
    pred_base = logp_base.argmax(1)
    acc = lambda p: float((p == y).mean())  # noqa: E731
    say(f"original chain rebuilt in {time.perf_counter() - t0:.0f} s: PCA {n_pca} dims, accuracy {acc(pred_base):.4f} "
        f"(notebook fold 1: 0.260)")
    panels = [(f"Original chain, accuracy {acc(pred_base):.3f}\n(150 hybrid features, power transform, PCA {n_pca}, one GaussianNB)", pred_base),
              (f"Improved global chain, accuracy {acc(pred_imp):.3f}\n(570 features, participant z-score, LDA 5, one GaussianNB)", pred_imp),
              (f"Subject-specific chain, accuracy {acc(pred_sub):.3f}\n(435 features, own z-score, own LDA 5, own GaussianNB)", pred_sub)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (title, pred) in zip(axes, panels):
        plot_cm(ax, confusion_matrix(y, pred, labels=range(len(COMMANDS))), title)
    fig.suptitle(f"From the original GaussianNB to the subject-specific chain on the same {len(y)} held-out windows "
                 "(count and share of each true command)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_progression_confusion_matrices.png")
    plt.close(fig)
    say("wrote defense_progression_confusion_matrices.png")
    return dict(original=acc(pred_base), original_pca_dims=n_pca, improved=acc(pred_imp), subject=acc(pred_sub))


def heldout_participant_confusion(data):
    """The subject-specific chain under the notebook's protocol C (5-fold GroupKFold by participant): held-out
    people only have the global fallback, pooled confusion matrix over the folds."""
    X, y, parts = data["X_sub"], data["y"], data["parts"]
    yt, pt, fold_acc = [], [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=parts)):
        logp, _ = subject_chain_logp(X[tr], y[tr], parts[tr], X[te], parts[te])
        yt.append(y[te])
        pt.append(logp.argmax(1))
        fold_acc.append(float((pt[-1] == y[te]).mean()))
    yt, pt = np.concatenate(yt), np.concatenate(pt)
    acc = float((yt == pt).mean())
    say(f"held-out participants, subject-specific chain: pooled accuracy {acc:.4f}, folds {np.round(fold_acc, 3).tolist()} "
        f"(notebook mean 0.189)")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    plot_cm(ax, confusion_matrix(yt, pt, labels=range(len(COMMANDS))),
            f"Subject-specific chain on held-out participants, accuracy {acc:.3f}\n"
            f"(5-fold by participant, pooled over folds, n = {len(yt)} windows; chance {CHANCE:.3f})")
    fig.tight_layout()
    fig.savefig(HERE / "defense_heldout_participant_confusion_matrix.png")
    plt.close(fig)
    say("wrote defense_heldout_participant_confusion_matrix.png")
    return dict(accuracy=acc, folds=fold_acc, windows=int(len(yt)))


def participant_confusions(data, test, pred_sub):
    """Small multiples: one row-normalised 6 x 6 per participant with an own model, ordered by accuracy."""
    own, fallback = own_model_participants()
    yt, pt = data["y"][test], data["parts"][test]
    accs = {q: float((pred_sub[pt == q] == yt[pt == q]).mean()) for q in own}
    order = sorted(own, key=lambda q: -accs[q])
    ncol = 6
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.0 * nrow + 0.8))
    for n, q in enumerate(order):
        ax = axes.flat[n]
        m = pt == q
        cm = confusion_matrix(yt[m], pred_sub[m], labels=range(len(COMMANDS)))
        last_row = n + ncol >= len(order)
        plot_cm(ax, cm, f"{q}   {accs[q]:.2f}   ({int(m.sum())} windows)", compact=True, show_x=last_row, show_y=(n % ncol == 0))
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    fb_note = "; ".join(f"{r['Participant']} ({r['Model']}, {int(r['Commands in train'])} command in training, "
                        f"{int(r['Test windows'])} test windows, accuracy {r['Accuracy']:.2f}) omitted" for _, r in fallback.iterrows())
    fig.suptitle("Subject-specific chain per participant: share of each true command on the participant's held-out windows, "
                 "ordered by accuracy", fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"Rows are true commands, columns predicted commands, cells the row share. {fb_note}.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(HERE / "defense_participant_confusion_matrices.png")
    plt.close(fig)
    say(f"wrote defense_participant_confusion_matrices.png ({len(order)} participants, {fb_note})")
    return accs


# ------------------------------------------------------------------------------------------------------ 9. figures
def lda_scatter(data, test, chain_imp, chain_sub, participant=TEST_PARTICIPANT):
    """The first two discriminant axes of one participant's own LDA on their test windows, next to the same windows
    through the global chain's LDA; both from the stored projections."""
    m = test & (data["parts"] == participant)
    y = data["y"][m]
    mean, std, model, status = chain_sub.resolve(participant)
    P_sub = model.transform((np.asarray(data["X_sub"][m], np.float64) - mean) / std)
    P_imp = chain_imp._projected(data["X_imp"][m], participant)
    lp_sub = model.predict_log_proba((np.asarray(data["X_sub"][m], np.float64) - mean) / std)
    lp_imp = chain_imp.log_posteriors(data["X_imp"][m], participant)
    acc_sub, acc_imp = float((lp_sub.argmax(1) == y).mean()), float((lp_imp.argmax(1) == y).mean())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, P, title in zip(axes, (P_sub, P_imp),
                            (f"{participant}'s own LDA (subject-specific chain, {status}), window accuracy {acc_sub:.2f}",
                             f"The global chain's LDA on the same windows (improved global chain), window accuracy {acc_imp:.2f}")):
        for k, c in enumerate(COMMANDS):
            mk = y == k
            ax.scatter(P[mk, 0], P[mk, 1], s=26, color=COMMAND_COLORS[k], marker=COMMAND_MARKERS[k], edgecolors=SURFACE,
                       linewidths=0.6, label=c, alpha=0.9)
            cx, cy = P[mk, 0].mean(), P[mk, 1].mean()
            ax.text(cx, cy, c, fontsize=8, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor=COMMAND_COLORS[k], linewidth=1))
        ax.set_xlabel("discriminant axis 1")
        ax.set_ylabel("discriminant axis 2")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(color=GRID, linewidth=0.6)
    command_legend(axes[0], loc="best", markers=True)
    fig.suptitle(f"Why the per-participant projection separates: {int(m.sum())} held-out windows of {participant}, "
                 "coloured by the true command (text at each command's centroid)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_lda_scatter.png")
    plt.close(fig)
    say(f"wrote defense_lda_scatter.png ({participant}: own LDA {acc_sub:.3f}, global LDA {acc_imp:.3f} on {int(m.sum())} windows)")
    return dict(participant=participant, windows=int(m.sum()), own_lda_accuracy=acc_sub, global_lda_accuracy=acc_imp)


def decision_timeline(data, test, logp_sub, participant=TEST_PARTICIPANT):
    """One held-out recording per command of one participant, concatenated in time: stacked per-window posteriors of
    the subject-specific chain, the 5-window context decision, the vote gate's verdict, the true label as a band."""
    y, groups, parts = data["y"], data["groups"], data["parts"]
    t, file_time = window_times(groups)
    chosen = []
    for c in range(len(COMMANDS)):
        recs = sorted(np.unique(groups[test & (parts == participant) & (y == c)]), key=lambda r: file_time[r])
        if recs:
            chosen.append(recs[0])
    stride_s = I.WINDOW_STRIDE / FS
    fig, (ax_band, ax_post, ax_dec) = plt.subplots(3, 1, figsize=(14, 7.2), sharex=True,
                                                   gridspec_kw=dict(height_ratios=[0.45, 3.0, 1.6], hspace=0.08))
    x0 = 0.0
    n_dec = n_acc = n_acc_right = n_win_right = n_win = 0
    bounds = []
    for r in chosen:
        idx = np.flatnonzero(groups == r)
        lp = logp_sub[idx]
        probs = np.exp(lp)
        ctx = I.context_log_posteriors(lp)
        dec, votes = ctx.argmax(1), lp.argmax(1)
        agree = np.array([int((votes[max(0, i - S.VOTE_WINDOWS + 1):i + 1] == dec[i]).sum()) for i in range(len(idx))])
        full = np.arange(len(idx)) >= S.VOTE_WINDOWS - 1
        accepted = full & (agree >= S.MIN_AGREEMENT)
        truth = int(y[idx[0]])
        x = x0 + np.arange(len(idx)) * stride_s
        x1 = x0 + len(idx) * stride_s
        ax_band.axvspan(x0, x1, color=COMMAND_COLORS[truth], alpha=0.9, linewidth=0)
        ax_band.text((x0 + x1) / 2, 0.5, f"true: {COMMANDS[truth]}", ha="center", va="center", fontsize=8,
                     color=SURFACE if truth in (0, 5) else INK)
        lo = np.zeros(len(idx))
        for k in range(len(COMMANDS)):
            hi = lo + probs[:, k]
            ax_post.fill_between(x, lo, hi, color=COMMAND_COLORS[k], linewidth=0, step=None)
            lo = hi
        for i in range(len(idx)):
            if not full[i]:
                ax_dec.scatter(x[i], dec[i], s=18, marker="x", color=INK2, linewidths=0.8)
            elif accepted[i]:
                ax_dec.scatter(x[i], dec[i], s=34, marker="o", color=COMMAND_COLORS[dec[i]], edgecolors=SURFACE, linewidths=0.6)
            else:
                ax_dec.scatter(x[i], dec[i], s=34, marker="o", facecolors="none", edgecolors=COMMAND_COLORS[dec[i]], linewidths=1.2)
        ax_dec.hlines(truth, x0, x1, color=INK2, linewidth=0.8, linestyle=":")
        for ax in (ax_post, ax_dec):
            ax.axvline(x1, color=INK2, linewidth=0.8)
        bounds.append((x0, x1))
        n_dec += int(full.sum())
        n_acc += int(accepted.sum())
        n_acc_right += int((accepted & (dec == truth)).sum())
        n_win += len(idx)
        n_win_right += int((votes == truth).sum())
        x0 = x1
    ax_band.set_ylim(0, 1)
    ax_band.set_yticks([])
    ax_band.set_xlim(0, x0)
    for s in ax_band.spines.values():
        s.set_visible(False)
    ax_band.set_title(f"Subject-specific chain on one held-out recording per command of {participant}, concatenated in time "
                      f"(window stride {stride_s:.3f} s)", loc="left", fontsize=11)
    ax_post.set_ylim(0, 1)
    ax_post.set_ylabel("per-window posterior\n(stacked)")
    ax_post.grid(axis="y", color=GRID, linewidth=0.6)
    command_legend(ax_post, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6)
    ax_dec.set_yticks(range(len(COMMANDS)), COMMANDS, fontsize=8)
    ax_dec.set_ylim(-0.6, len(COMMANDS) - 0.4)
    ax_dec.set_ylabel("5-window context\ndecision")
    ax_dec.set_xlabel("seconds since the start of the concatenation (recording boundaries marked)")
    ax_dec.grid(axis="y", color=GRID, linewidth=0.6)
    ax_dec.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=6, label="gate accepted (4 of 5 votes agree)"),
                           Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK2, markersize=6, label="withheld as uncertain"),
                           Line2D([], [], marker="x", linestyle="", color=INK2, markersize=5, label="no full context yet"),
                           Line2D([], [], linestyle=":", color=INK2, label="true command")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=8)
    fig.savefig(HERE / "defense_decision_timeline.png")
    plt.close(fig)
    out = dict(participant=participant, recordings=[Path(r).parent.name for r in chosen], windows=n_win,
               window_accuracy=n_win_right / max(n_win, 1), decisions=n_dec, accepted=n_acc,
               accuracy_accepted=n_acc_right / max(n_acc, 1))
    say(f"wrote defense_decision_timeline.png ({len(chosen)} recordings, {n_win} windows, window accuracy "
        f"{out['window_accuracy']:.3f}, {n_acc}/{n_dec} decisions accepted at {out['accuracy_accepted']:.3f})")
    return out


def permutation_test(data, train, test, n_perm=100, seed=0, n_jobs=4):
    """Null distribution of the subject-specific chain's accuracy: the labels of the training recordings permuted
    within participant (the confound control's scheme), n_perm times; empirical p-value of the observed accuracy
    computed on the same code path (per-participant refit, participants whose fit is possible). The permutations
    are drawn sequentially (reproducible); the per-participant fits of one permutation run in threads."""
    from joblib import Parallel, delayed
    X, y, groups, parts = data["X_sub"], data["y"], data["groups"], data["parts"]
    participants = sorted(np.unique(parts[test]))

    def run(rng=None):
        jobs = []
        for q in participants:
            tr, te = train & (parts == q), test & (parts == q)
            recs = np.unique(groups[tr])
            if len(recs) < 6 or te.sum() == 0:
                continue
            if rng is None:
                y_fit = y[tr]
            else:
                rec_label = {r: int(y[groups == r][0]) for r in recs}
                permuted = dict(zip(recs, rng.permutation([rec_label[r] for r in recs])))
                y_fit = np.array([permuted[g] for g in groups[tr]])
            jobs.append((q, tr, te, y_fit))
        preds = Parallel(n_jobs=n_jobs, backend="threading")(delayed(fit_eval_subject)(X[tr], y_fit, X[te], q) for q, tr, te, y_fit in jobs)
        hits = tot = used = 0
        for (q, tr, te, _), pred in zip(jobs, preds):
            if pred is None:
                continue
            hits += int((pred == y[te]).sum())
            tot += int(te.sum())
            used += 1
        return hits / tot, tot, used

    t0 = time.perf_counter()
    observed, n_windows, n_used = run()
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        null.append(run(rng)[0])
        if (i + 1) % 20 == 0:
            say(f"permutation {i + 1}/{n_perm} ({time.perf_counter() - t0:.0f} s)")
    null = np.array(null)
    p_value = float((1 + (null >= observed).sum()) / (1 + n_perm))
    out = dict(n_permutations=int(n_perm), observed_accuracy=float(observed), null_mean=float(null.mean()),
               null_std=float(null.std()), null_min=float(null.min()), null_max=float(null.max()), p_value=p_value,
               windows=int(n_windows), participants=int(n_used))
    say(f"permutation test: observed {observed:.4f}, null {null.mean():.4f} +- {null.std():.4f} (max {null.max():.4f}), "
        f"p = {p_value:.4f} over {n_perm} permutations")
    pd.DataFrame({"Permutation": np.arange(1, n_perm + 1), "Accuracy": null}).to_csv(HERE / "defense_permutation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(null, bins=np.linspace(0.08, 0.26, 37), color=INK2, alpha=0.8, edgecolor=SURFACE, linewidth=0.8,
            label=f"{n_perm} label permutations across recordings within participant")
    ax.axvline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax.text(CHANCE + 0.003, ax.get_ylim()[1] * 0.92, "chance", fontsize=8, color=INK2)
    ax.axvline(observed, color=ORANGE, linewidth=2.2)
    ax.annotate(f"observed {observed:.3f}\np = {p_value:.3f}", (observed, ax.get_ylim()[1] * 0.75), textcoords="offset points",
                xytext=(-8, 0), ha="right", fontsize=9, color=INK)
    ax.set_xlim(0.08, max(0.27, observed + 0.04))
    ax.set_xlabel(f"Window accuracy of the subject-specific chain ({n_windows} held-out windows, {n_used} participants refitted)")
    ax.set_ylabel("permutations")
    ax.set_title("Permutation test: the chain learns nothing from shuffled recording labels", loc="left", fontsize=11)
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_permutation.png")
    plt.close(fig)
    say("wrote defense_permutation.png")
    return out


def confidence_figure(data, test, logp_by_chain):
    """Max posterior of the 5-window context decision and the number of agreeing window votes, correct against
    wrong decisions, per chain (full-context positions)."""
    y = data["y"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    out = {}
    for row, ((name, logp), color) in enumerate(zip(logp_by_chain.items(), (BLUE, ORANGE))):
        cd = context_decisions(logp, data["groups"], test)
        m = cd["full"]
        correct = cd["decision"][m] == y[cd["index"]][m]
        conf, agree = cd["confidence"][m], cd["agreement"][m]
        ax = axes[row, 0]
        # the summed log-posteriors saturate (max posterior 1.000 for most decisions, right or wrong), so the margin
        # 1 - max posterior is shown on a log axis; floor 1e-9
        margin = np.clip(1 - conf, 1e-9, 1)
        bins = np.logspace(-9, 0, 37)
        ax.hist(margin[correct], bins=bins, color=color, alpha=0.75, edgecolor=SURFACE, linewidth=0.6,
                weights=np.full(correct.sum(), 1 / correct.sum()), label=f"correct (n = {int(correct.sum())})")
        ax.hist(margin[~correct], bins=bins, color=INK2, alpha=0.55, edgecolor=SURFACE, linewidth=0.6,
                weights=np.full((~correct).sum(), 1 / (~correct).sum()), label=f"wrong (n = {int((~correct).sum())})")
        ax.set_xscale("log")
        ax.axvline(np.median(margin[correct]), color=color, linewidth=1.2, linestyle="--")
        ax.axvline(np.median(margin[~correct]), color=INK2, linewidth=1.2, linestyle="--")
        ax.set_xlabel("1 - max posterior of the context decision (log scale; left = more confident; floor 1e-9)")
        ax.set_ylabel("share of decisions")
        sat_c, sat_w = float((conf[correct] > 0.99).mean()), float((conf[~correct] > 0.99).mean())
        ax.set_title(f"{name}: posterior confidence\n({sat_c:.0%} of correct and {sat_w:.0%} of wrong decisions have a max posterior above 0.99)",
                     loc="left", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax = axes[row, 1]
        ks = np.arange(1, S.VOTE_WINDOWS + 1)
        share_c = np.array([(agree[correct] == k).mean() for k in ks])
        share_w = np.array([(agree[~correct] == k).mean() for k in ks])
        ax.bar(ks - 0.2, share_c, width=0.38, color=color, alpha=0.9, edgecolor=SURFACE, label="correct")
        ax.bar(ks + 0.2, share_w, width=0.38, color=INK2, alpha=0.65, edgecolor=SURFACE, label="wrong")
        for k, sc, sw in zip(ks, share_c, share_w):
            ax.text(k - 0.2, sc + 0.01, f"{sc:.0%}", ha="center", fontsize=7, color=INK)
            ax.text(k + 0.2, sw + 0.01, f"{sw:.0%}", ha="center", fontsize=7, color=INK2)
        ax.axvspan(S.MIN_AGREEMENT - 0.5, S.VOTE_WINDOWS + 0.5, color=GRID, alpha=0.35, linewidth=0)
        ax.text(S.MIN_AGREEMENT - 0.45, ax.get_ylim()[1] * 0.97, "gate accepts", fontsize=8, color=INK2, va="top")
        ax.set_xticks(ks, [f"{k} of {S.VOTE_WINDOWS}" for k in ks])
        ax.set_xlabel("window votes agreeing with the context decision")
        ax.set_ylabel("share of decisions")
        ax.set_title(f"{name}: vote agreement\n(how many of the last {S.VOTE_WINDOWS} window votes agree with the context decision)", loc="left", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        out[name] = dict(decisions=int(m.sum()), accuracy=float(correct.mean()), median_confidence_correct=float(np.median(conf[correct])),
                         median_confidence_wrong=float(np.median(conf[~correct])), share_above_0_99_correct=sat_c, share_above_0_99_wrong=sat_w,
                         share_unanimous_correct=float((agree[correct] == S.VOTE_WINDOWS).mean()),
                         share_unanimous_wrong=float((agree[~correct] == S.VOTE_WINDOWS).mean()))
    fig.suptitle("What separates right from wrong decisions: posterior confidence and vote agreement "
                 "(notebook's held-out recordings, full 5-window contexts)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_confidence.png")
    plt.close(fig)
    say("wrote defense_confidence.png")
    return out


def time_distance_figure(data, train, test, pred_sub):
    """Subject-chain window accuracy against the time gap between a test window and the nearest training window of
    the same participant (file timestamp plus window offset)."""
    y, groups, parts = data["y"], data["groups"], data["parts"]
    t, _ = window_times(groups)
    idx_test = np.flatnonzero(test)
    gap = np.full(len(idx_test), np.nan)
    for q in np.unique(parts[test]):
        tr = train & (parts == q)
        if not tr.any():
            continue
        tt = np.sort(t[tr])
        sel = parts[idx_test] == q
        tq = t[idx_test][sel]
        pos = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
        gap[sel] = np.minimum(np.abs(tq - tt[pos - 1]), np.abs(tq - tt[pos]))
    edges = [0, 2, 10, 15, 20, 30, 60, np.inf]
    labels = ["< 2 s", "2-10 s", "10-15 s", "15-20 s", "20-30 s", "30-60 s", "> 1 min"]
    correct = pred_sub == y[idx_test]
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (gap >= lo) & (gap < hi)
        if m.sum() == 0:
            continue
        low, high = wilson(int(correct[m].sum()), int(m.sum()))
        rows.append({"Gap to nearest training window": lab, "Windows": int(m.sum()), "Accuracy": float(correct[m].mean()),
                     "Wilson low": low, "Wilson high": high})
    df = pd.DataFrame(rows)
    write_csv(df, "defense_time_distance.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xp = np.arange(len(df))
    ax.bar(xp, df["Accuracy"], color=ORANGE, width=0.62, edgecolor=SURFACE)
    ax.errorbar(xp, df["Accuracy"], yerr=[df["Accuracy"] - df["Wilson low"], df["Wilson high"] - df["Accuracy"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for i, r in df.iterrows():
        ax.text(i, 0.03, f"n = {int(r['Windows'])}", ha="center", fontsize=8, color=SURFACE)
        ax.text(i, r["Wilson high"] + 0.02, f"{r['Accuracy']:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(correct.mean(), color=INK, linewidth=1, linestyle=":")
    ax.text(len(df) - 0.5, correct.mean() + 0.015, f"all test windows {correct.mean():.3f}", ha="right", fontsize=8, color=INK)
    ax.axhline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax.text(-0.4, CHANCE + 0.015, "chance", fontsize=8, color=INK2)
    ax.set_xticks(xp, df["Gap to nearest training window"])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("time between the test window and the nearest training window of the same participant")
    ax.set_ylabel("window accuracy (Wilson 95 % interval)")
    ax.set_title("Subject-specific chain: window accuracy against the time distance to the nearest training window\n"
                 "(a split by recording leaves almost every test window within a minute of training data)", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_time_distance.png")
    plt.close(fig)
    say("wrote defense_time_distance.png")
    return df.to_dict("records")


def learning_curve_spread(data, train, test):
    """The notebook's learning curve (cell 20: each participant's training windows restricted to the k earliest
    recordings per command in the notebook's file ordering, test windows unchanged) with one line per participant
    with an own model and the window-weighted mean, checked against gaussiannb_subject_learning_curve.csv."""
    X, y, groups, parts = data["X_sub"], data["y"], data["groups"], data["parts"]
    own, _ = own_model_participants()
    ref = pd.read_csv(PYTORCH_DIR / "gaussiannb_subject_learning_curve.csv")
    ref_acc = dict(zip(ref["Recordings per command (k)"].astype(str), ref["Accuracy"]))
    ks = [1, 2, 3, 4, 6, 8, "all"]
    rows, means = [], []
    for k in ks:
        keep = train.copy()
        if k != "all":
            keep = np.zeros(len(y), bool)
            for q in np.unique(parts[train]):
                for c in range(len(COMMANDS)):
                    m = train & (parts == q) & (y == c)
                    keep |= m & np.isin(groups, sorted(np.unique(groups[m]))[:k])
        logp, model = subject_chain_logp(X[keep], y[keep], parts[keep], X[test], parts[test])
        pred = logp.argmax(1)
        yt, pt = y[test], parts[test]
        mean_acc = float((pred == yt).mean())
        means.append(mean_acc)
        for q in own:
            m = pt == q
            rows.append({"Recordings per command (k)": str(k), "Participant": q, "Train windows": int((keep & (parts == q)).sum()),
                         "Own model": "yes" if model.has_model(q) else "no (fallback)", "Test windows": int(m.sum()),
                         "Accuracy": float((pred[m] == yt[m]).mean())})
        say(f"learning curve k = {k}: {int(keep.sum())} train windows, accuracy {mean_acc:.4f} (notebook {ref_acc.get(str(k), float('nan')):.4f})")
    df = pd.DataFrame(rows)
    write_csv(df, "defense_learning_curve_spread.csv")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xp = np.arange(len(ks))
    final = {q: float(df[(df["Participant"] == q) & (df["Recordings per command (k)"] == "all")]["Accuracy"].iloc[0]) for q in own}
    for q in own:
        d = df[df["Participant"] == q].set_index("Recordings per command (k)").loc[[str(k) for k in ks]]
        ax.plot(xp, d["Accuracy"], color=GRID, linewidth=1.2, marker="o", markersize=3, zorder=1)
    for q in (max(final, key=final.get), min(final, key=final.get)):      # label the extremes only; the rest is in the CSV
        ax.text(xp[-1] + 0.08, final[q], f"{q} {final[q]:.2f}", fontsize=7, color=INK2, va="center")
    ax.plot(xp, means, color=ORANGE, linewidth=2.4, marker="o", markersize=7, label="all test windows (notebook's learning curve)", zorder=3)
    for xi, mval in zip(xp, means):
        ax.text(xi, mval + 0.035, f"{mval:.2f}", ha="center", fontsize=8, color=INK, zorder=4)
    ax.axhline(CHANCE, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.02, CHANCE + 0.012, "chance", fontsize=8, color=INK2)
    ax.set_xticks(xp, [str(k) if k != "all" else f"all ({ref['Mean train recordings per command'].iloc[-1]:.1f} mean)" for k in ks])
    ax.set_xlim(-0.3, len(ks) - 0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("labelled recordings per command used for training (about 20 s each)")
    ax.set_ylabel("window accuracy on the participant's held-out recordings")
    ax.set_title("Learning curve of the subject-specific chain: every participant with an own model (grey) and the mean (orange)",
                 loc="left", fontsize=10)
    ax.legend(handles=[Line2D([], [], color=GRID, linewidth=1.2, marker="o", markersize=3, label=f"one participant ({len(own)} with an own model)"),
                       Line2D([], [], color=ORANGE, linewidth=2.4, marker="o", markersize=7, label="window-weighted mean over all test windows")],
              loc="lower right", fontsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_learning_curve_spread.png")
    plt.close(fig)
    say("wrote defense_learning_curve_spread.png")
    return [dict(k=str(k), accuracy=a, notebook=float(ref_acc.get(str(k), float("nan")))) for k, a in zip(ks, means)]


def channel_contribution(chain_sub):
    """Absolute LDA loadings of the subject chain aggregated per EEG channel: for every participant with an own model,
    |projection| summed over the five discriminant axes (features are z-scored, so loadings are comparable), each
    cross-channel covariance feature split equally between its two channels, normalised to sum 1 per participant,
    then averaged over participants; stacked by feature group."""
    names, channels = chain_sub.feature_names, chain_sub.channels
    groups_of = [("logcov_", "log-covariance 1-40 Hz"), ("dcov_", "log-covariance, first difference"),
                 ("gammacov_", "log-covariance 30-45 Hz"), ("_mt_log_", "multitaper band power")]
    group_names = [g for _, g in groups_of]
    chan_index = {c: i for i, c in enumerate(channels)}
    feat_chans = [re.findall(r"EXG Channel \d+", n) for n in names]
    feat_group = [next(k for k, (tag, _) in enumerate(groups_of) if tag in n) for n in names]
    models = chain_sub.notebook_models.models_
    contrib = np.zeros((len(models), len(channels), len(groups_of)))
    for p, (q, model) in enumerate(sorted(models.items())):
        w = np.abs(np.asarray(model.projection_, np.float64)).sum(1)
        w = w / w.sum()
        for f, (chs, g) in enumerate(zip(feat_chans, feat_group)):
            for c in chs:
                contrib[p, chan_index[c], g] += w[f] / len(chs)
    mean, total_sd = contrib.mean(0), contrib.sum(2).std(0)
    df = pd.DataFrame({"Channel": channels, **{g: mean[:, k] for k, g in enumerate(group_names)},
                       "Total": mean.sum(1), "Std over participants": total_sd})
    write_csv(df, "defense_channel_contribution.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    xp = np.arange(len(channels))
    bottom = np.zeros(len(channels))
    for k, (g, color) in enumerate(zip(group_names, (BLUE, ORANGE, AQUA, YELLOW))):
        ax.bar(xp, mean[:, k], bottom=bottom, color=color, width=0.66, edgecolor=SURFACE, linewidth=1, label=g)
        bottom += mean[:, k]
    ax.errorbar(xp, bottom, yerr=total_sd, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.axhline(1 / len(channels), color=INK2, linewidth=1, linestyle="--")
    ax.text(len(channels) - 0.5, 1 / len(channels) + 0.002, "equal share", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(xp, [c.replace("EXG Channel ", "ch ") for c in channels])
    ax.set_xlabel("EEG channel (board index of the 16-channel Cyton + Daisy; channel 15 excluded)")
    ax.set_ylabel("share of the absolute LDA loadings")
    ax.set_title(f"Which channels the per-participant projections lean on: mean over {len(models)} own models "
                 "(error bars: std over participants)", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_channel_contribution.png")
    plt.close(fig)
    top = df.sort_values("Total", ascending=False)["Channel"].head(3).tolist()
    say(f"wrote defense_channel_contribution.png (largest shares: {', '.join(top)})")
    return df.to_dict("records")


def dataset_figure_and_table(data, train, test, chain_imp, skip_load=False):
    """Windows per command per participant (train solid, test hatched) and the dataset table: what the loader found,
    loaded, skipped and kept (reloaded with eeg_data unless skip_load), plus the cache's windows and the split."""
    y, groups, parts = data["y"], data["groups"], data["parts"]
    rows = []
    root = REPO_ROOT / "Data_clean"
    if not skip_load:
        from eeg_data import load_recordings
        t0 = time.perf_counter()
        recs, skipped = load_recordings(root, chain_imp.config, verbose=False)
        n_files = sum(1 for p in root.rglob("*") if p.is_file())
        per_person = {}
        for r in recs:
            k, n = per_person.get(r["participant"], (0, 0))
            per_person[r["participant"]] = (k + r["n_blocks_kept"], n + r["n_blocks"])
        usable_people = sorted(p for p, (k, _) in per_person.items() if k > 0)
        other_board = sum(1 for s in skipped if "not " in str(s.get("error", "")))
        rows += [("Files under Data_clean", n_files, "eeg_data.load_recordings"),
                 ("Named OpenBCI recordings found", len(recs) + len(skipped), "eeg_data.load_recordings"),
                 (f"Recordings loaded ({chain_imp.config.board}, {len(chain_imp.config.channels)} channels)", len(recs), "eeg_data.load_recordings"),
                 ("Recordings skipped (other board, unreadable or unlabelled)", len(skipped), f"{other_board} of another board"),
                 ("2 s blocks assessed", sum(r["n_blocks"] for r in recs), "signal_quality policy on the kept channels"),
                 ("2 s blocks kept", sum(r["n_blocks_kept"] for r in recs),
                  f"{100 * sum(r['n_blocks_kept'] for r in recs) / max(1, sum(r['n_blocks'] for r in recs)):.1f} %"),
                 ("Recordings with usable data", sum(1 for r in recs if r["segments"]), "at least one kept block"),
                 ("Participants with usable data", f"{len(usable_people)} of {len(per_person)}",
                  "none usable: " + ", ".join(sorted(set(per_person) - set(usable_people)))),
                 ]
        say(f"Data_clean reloaded in {time.perf_counter() - t0:.0f} s: {len(recs)} recordings, {len(skipped)} skipped")
    own, fallback = own_model_participants()
    rows += [("Windows in the feature caches", len(y), f"{I.WINDOW_SIZE} samples ({WINDOW_SECONDS:.3f} s), stride {I.WINDOW_STRIDE}"),
             ("Recordings in the caches", len(np.unique(groups)), "one contiguous block of windows each"),
             ("Participants in the caches", len(np.unique(parts)), ", ".join(sorted(np.unique(parts)))),
             ("Windows per recording (median, min, max)",
              f"{int(np.median(np.unique(groups, return_counts=True)[1]))} ({np.unique(groups, return_counts=True)[1].min()}-{np.unique(groups, return_counts=True)[1].max()})", "")]
    for k, c in enumerate(COMMANDS):
        rows.append((f"Windows: {c}", int((y == k).sum()), f"train {int((train & (y == k)).sum())}, test {int((test & (y == k)).sum())}"))
    rows += [("Training windows (notebook split)", int(train.sum()), f"{len(np.unique(groups[train]))} recordings"),
             ("Test windows (notebook split)", int(test.sum()), f"{len(np.unique(groups[test]))} recordings"),
             ("Participants with an own model", len(own), "fallback: " + ", ".join(fallback["Participant"]))]
    df = pd.DataFrame(rows, columns=["Quantity", "Value", "Detail"])
    df["Detail"] = df["Detail"].fillna("")
    write_csv(df, "defense_dataset.csv")

    people = sorted(np.unique(parts), key=lambda q: -(parts == q).sum())
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xp = np.arange(len(people))
    bottom = np.zeros(len(people))
    for k, c in enumerate(COMMANDS):
        for split, hatch in ((train, None), (test, "////")):
            h = np.array([int((split & (parts == q) & (y == k)).sum()) for q in people])
            ax.bar(xp, h, bottom=bottom, color=COMMAND_COLORS[k], width=0.7, edgecolor=SURFACE, linewidth=0.8, hatch=hatch)
            bottom += h
    for i, q in enumerate(people):
        ax.text(i, bottom[i] + 8, str(int(bottom[i])), ha="center", fontsize=7, color=INK)
    ax.set_xticks(xp, people, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("windows")
    ax.set_title(f"Windows per participant and command in the feature caches ({len(y)} windows, {len(np.unique(groups))} recordings): "
                 "solid = training, hatched = held-out test", loc="left", fontsize=10)
    handles = [Patch(color=COMMAND_COLORS[k], label=c) for k, c in enumerate(COMMANDS)]
    handles += [Patch(facecolor=INK2, edgecolor=SURFACE, label="train"), Patch(facecolor=INK2, edgecolor=SURFACE, hatch="////", label="test")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(HERE / "defense_dataset.png")
    plt.close(fig)
    say("wrote defense_dataset.png")
    return df.to_dict("records")


# ------------------------------------------------------------------------------------------------------ 10. tables
def itr_table(data, test, logp_by_chain, window_pred):
    """Information transfer rate (Wolpaw, N = 6): single windows, the 5-window context and the gated decision."""
    y = data["y"]
    rows = []
    for name, logp in logp_by_chain.items():
        pw = window_pred[name]
        acc_w = float((pw == y[test]).mean())
        cd = context_decisions(logp, data["groups"], test)
        m = cd["full"]
        yt, dec = y[cd["index"]][m], cd["decision"][m]
        accepted = cd["agreement"][m] >= S.MIN_AGREEMENT
        acc_c, cov, acc_g = float((dec == yt).mean()), float(accepted.mean()), float((dec[accepted] == yt[accepted]).mean())
        for decision, secs, acc, cov_, n in ((f"single window ({WINDOW_SECONDS:.2f} s)", WINDOW_SECONDS, acc_w, 1.0, int(test.sum())),
                                            (f"5-window context ({CONTEXT_SECONDS:.2f} s)", CONTEXT_SECONDS, acc_c, 1.0, int(m.sum())),
                                            (f"gated context, {S.MIN_AGREEMENT} of {S.VOTE_WINDOWS} votes ({CONTEXT_SECONDS:.2f} s)", CONTEXT_SECONDS, acc_g, cov, int(accepted.sum()))):
            bits = wolpaw_bits(acc)
            rows.append({"Chain": name, "Decision": decision, "Seconds per decision": round(secs, 3), "Accuracy": acc,
                         "Coverage": cov_, "Bits per decision": bits, "Bits per minute": bits * 60 / secs * cov_, "n": n})
    df = pd.DataFrame(rows)
    write_csv(df, "defense_itr.csv")
    return df.to_dict("records")


def statistics_table(data, test, logp_by_chain, window_pred):
    """Wilson intervals, Cohen's kappa, McNemar between the chains on the test windows, and the paired per-fold
    differences over the notebook's protocol A with a Wilcoxon signed-rank test."""
    y, groups = data["y"], data["groups"]
    yt = y[test]
    rows = []
    correct = {}
    for name, logp in logp_by_chain.items():
        pw = window_pred[name]
        correct[name] = pw == yt
        k, n = int((pw == yt).sum()), int(len(yt))
        lo, hi = wilson(k, n)
        rows.append(dict(Statistic="Window accuracy", Chain=name, Estimate=k / n, **{"95 % low": lo, "95 % high": hi}, n=n, Detail="Wilson interval"))
        _, y_rec, p_rec = recording_decisions(logp, groups, y, test)
        k, n = int((y_rec == p_rec).sum()), int(len(y_rec))
        lo, hi = wilson(k, n)
        rows.append(dict(Statistic="Per-recording accuracy", Chain=name, Estimate=k / n, **{"95 % low": lo, "95 % high": hi}, n=n, Detail="Wilson interval"))
        cd = context_decisions(logp, groups, test)
        m = cd["full"]
        dec, yc = cd["decision"][m], y[cd["index"]][m]
        acc_m = cd["agreement"][m] >= S.MIN_AGREEMENT
        k, n = int((dec[acc_m] == yc[acc_m]).sum()), int(acc_m.sum())
        lo, hi = wilson(k, n)
        rows.append(dict(Statistic=f"Gated accuracy ({S.MIN_AGREEMENT} of {S.VOTE_WINDOWS}, accepted decisions)", Chain=name, Estimate=k / n,
                         **{"95 % low": lo, "95 % high": hi}, n=n, Detail=f"coverage {acc_m.mean():.3f} of {int(m.sum())} decisions"))
        rows.append(dict(Statistic="Cohen's kappa (windows)", Chain=name, Estimate=float(cohen_kappa_score(yt, pw)),
                         **{"95 % low": float("nan"), "95 % high": float("nan")}, n=int(len(yt)), Detail="0 = chance agreement, 1 = perfect"))
    names = list(logp_by_chain)
    a, b = correct[names[0]], correct[names[1]]
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    exact_p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    chi2 = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    rows.append(dict(Statistic=f"McNemar exact test, {names[1]} vs {names[0]} (windows)", Chain="both", Estimate=exact_p,
                     **{"95 % low": float("nan"), "95 % high": float("nan")}, n=int(len(yt)),
                     Detail=f"exact p = {exact_p:.1e}; {names[0]} right and {names[1]} wrong: {n01}, the reverse: {n10}; "
                            f"chi-square with continuity correction {chi2:.1f} (p = {stats.chi2.sf(chi2, 1):.1e})"))
    folds = pd.read_csv(PYTORCH_DIR / "gaussiannb_improved_vs_baseline_folds.csv")
    fa = folds[folds["Protocol"].str.startswith("A.")]
    chains = {"baseline": fa[fa["Chain"].str.startswith("Baseline")].set_index("Fold")["Accuracy"],
              "improved": fa[fa["Chain"].str.startswith("Improved")].set_index("Fold")["Accuracy"],
              "subject": fa[fa["Chain"].str.startswith("Subject")].set_index("Fold")["Accuracy"]}
    for hi_name, lo_name in (("improved", "baseline"), ("subject", "improved")):
        d = (chains[hi_name] - chains[lo_name]).sort_index()
        w = stats.wilcoxon(d.values)
        rows.append(dict(Statistic=f"Protocol A, window accuracy per fold: {hi_name} minus {lo_name}", Chain="both",
                         Estimate=float(d.mean()), **{"95 % low": float(d.min()), "95 % high": float(d.max())}, n=int(len(d)),
                         Detail=f"range over the 5 folds; Wilcoxon signed-rank p = {w.pvalue:.4f} "
                                f"(the smallest two-sided p with 5 folds is 0.0625); folds {np.round(d.values, 3).tolist()}"))
    df = pd.DataFrame(rows)
    write_csv(df, "defense_statistics.csv")
    return df.to_dict("records")


def subject_ablation(data, train, test, chain_sub):
    """The subject-specific chain with each feature group removed in turn and with 1-5 LDA dimensions, on the
    notebook split (the notebook's chain incl. fallback), window accuracy and the notebook's other metrics."""
    X, y, groups, parts = data["X_sub"], data["y"], data["groups"], data["parts"]
    n_ch = len(chain_sub.channels)
    n_cov = n_ch * (n_ch + 1) // 2
    blocks = {"log-covariance 1-40 Hz": (0, n_cov), "log-covariance, first difference": (n_cov, 2 * n_cov),
              "log-covariance 30-45 Hz": (2 * n_cov, 3 * n_cov), "multitaper band powers": (3 * n_cov, X.shape[1])}
    variants = [("all four groups, LDA 5 (deployed)", np.ones(X.shape[1], bool), 5)]
    for name, (a, b) in blocks.items():
        keep = np.ones(X.shape[1], bool)
        keep[a:b] = False
        variants.append((f"without {name}", keep, 5))
    for d in (1, 2, 3, 4):
        variants.append((f"all four groups, LDA {d}", np.ones(X.shape[1], bool), d))
    rows = []
    yt = y[test]
    for name, keep, d in variants:
        t0 = time.perf_counter()
        logp, _ = subject_chain_logp(X[train][:, keep], y[train], parts[train], X[test][:, keep], parts[test], n_components=d)
        pred = logp.argmax(1)
        _, y_rec, p_rec = recording_decisions(logp, groups[test], yt, np.ones(len(yt), bool))
        rows.append({"Variant": name, "Features": int(keep.sum()), "LDA dims": d, "Accuracy": float((pred == yt).mean()),
                     "Macro-F1": float(f1_score(yt, pred, average="macro")),
                     "Context-5 Acc": context_accuracy(logp, groups[test], yt, np.ones(len(yt), bool)),
                     "Per-recording Acc": float((y_rec == p_rec).mean())})
        say(f"ablation {name}: {int(keep.sum())} features, LDA {d}: accuracy {rows[-1]['Accuracy']:.4f} ({time.perf_counter() - t0:.0f} s)")
    df = pd.DataFrame(rows)
    df["Delta vs deployed"] = df["Accuracy"] - df["Accuracy"].iloc[0]
    write_csv(df, "defense_subject_ablation.csv")
    return df.to_dict("records")


def model_complexity(data, train, chain_imp, chain_sub, repeats=20):
    """Parameters per stage and artifact sizes of both chains, the fit time of one user's model, and the artifact
    sizes of the project's RandomForest and JAX MLP where they are in the repository."""
    kb = lambda p: round(Path(p).stat().st_size / 1024, 1) if Path(p).exists() else float("nan")  # noqa: E731
    n_imp, n_sub, n_proj, n_cls = len(chain_imp.feature_names), chain_sub.n_features, 5, len(COMMANDS)
    n_people_imp = len(chain_imp.training_scaler.stats_)
    models = chain_sub.notebook_models.models_
    n_people_sub = len(chain_sub.notebook_scaler.stats_)
    gnb_params = n_cls + 2 * n_cls * n_proj
    rows = [
        dict(Model="Improved global chain", Stage=f"participant z-score ({n_people_imp} participants + global)",
             Parameters=(n_people_imp + 1) * 2 * n_imp, Artifact=I.SCALER_FILE, **{"Size on disk (KB)": kb(PYTORCH_DIR / I.SCALER_FILE)},
             Note=f"mean and std of {n_imp} features; a new user adds {2 * n_imp} from a calibration recording"),
        dict(Model="Improved global chain", Stage=f"LDA projection ({n_imp} -> {n_proj})", Parameters=n_imp * n_proj,
             Artifact=I.PREP_FILE, **{"Size on disk (KB)": kb(PYTORCH_DIR / I.PREP_FILE)},
             Note="the fitted sklearn LDA also stores two 570 x 570 matrices the live chain never uses"),
        dict(Model="Improved global chain", Stage=f"GaussianNB ({n_cls} classes x {n_proj} dims) + thresholds",
             Parameters=gnb_params + n_cls, Artifact=I.MODEL_FILE, **{"Size on disk (KB)": kb(PYTORCH_DIR / I.MODEL_FILE)},
             Note="priors, means, variances, per-class thresholds"),
        dict(Model="Improved global chain", Stage="total", Parameters=(n_people_imp + 1) * 2 * n_imp + n_imp * n_proj + gnb_params + n_cls,
             Artifact="three files", **{"Size on disk (KB)": kb(PYTORCH_DIR / I.SCALER_FILE) + kb(PYTORCH_DIR / I.PREP_FILE) + kb(PYTORCH_DIR / I.MODEL_FILE)},
             Note="the 5.1 MB is the sklearn LDA object; the projection itself is 23 KB"),
        dict(Model="Subject-specific chain", Stage="one user's z-score", Parameters=2 * n_sub, Artifact=S.SUBJECT_SCALER_FILE,
             **{"Size on disk (KB)": kb(PYTORCH_DIR / S.SUBJECT_SCALER_FILE)}, Note=f"the artifact holds {n_people_sub} participants + global"),
        dict(Model="Subject-specific chain", Stage=f"one user's LDA projection ({n_sub} -> {n_proj})", Parameters=n_sub * n_proj,
             Artifact=S.SUBJECT_MODELS_FILE, **{"Size on disk (KB)": kb(PYTORCH_DIR / S.SUBJECT_MODELS_FILE)},
             Note=f"the artifact holds {len(models)} own models + the global fallback"),
        dict(Model="Subject-specific chain", Stage=f"one user's GaussianNB ({n_cls} classes x {n_proj} dims)", Parameters=gnb_params,
             Artifact=S.SUBJECT_MODELS_FILE, **{"Size on disk (KB)": float("nan")}, Note="priors, means, variances"),
        dict(Model="Subject-specific chain", Stage="total per user", Parameters=2 * n_sub + n_sub * n_proj + gnb_params,
             Artifact="model.npz per user in the GUI's store", **{"Size on disk (KB)": float("nan")},
             Note=f"all {len(models) + 1} models + scaler: {kb(PYTORCH_DIR / S.SUBJECT_MODELS_FILE) + kb(PYTORCH_DIR / S.SUBJECT_SCALER_FILE):.1f} KB"),
    ]
    # fit time of one user's model
    q = TEST_PARTICIPANT
    m = train & (data["parts"] == q)
    Xq, yq = data["X_sub"][m], data["y"][m]
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        S.fit_user_model(Xq, yq, q)
        ts.append((time.perf_counter() - t0) * 1e3)
    rows.append(dict(Model="Subject-specific chain", Stage=f"fit of one user's model ({len(yq)} windows of {q})", Parameters=float("nan"),
                     Artifact="", **{"Size on disk (KB)": float("nan")},
                     Note=f"median {np.median(ts):.0f} ms over {repeats} fits (p90 {np.percentile(ts, 90):.0f} ms): z-score, LDA, GaussianNB"))
    others = [("RandomForest (PyTorch notebook)", REPO_ROOT / "prediction-random-forest" / "pytorch" / "custom_rf.pt.gz", "gzip-compressed state"),
              ("RandomForest (JAX notebook)", REPO_ROOT / "prediction-random-forest" / "JAX" / "trained_classifier_rf_jax.pkl", "pickle"),
              ("JAX MLP (deep learning notebook)", REPO_ROOT / "prediction_deep_learning" / "JAX" / "trained_classifier_dl_jax.pkl", "pickle")]
    for name, path, note in others:
        if path.exists():
            rows.append(dict(Model=name, Stage="saved artifact", Parameters=float("nan"), Artifact=str(path.relative_to(REPO_ROOT)),
                             **{"Size on disk (KB)": kb(path)}, Note=f"{note}; parameters not counted (not loaded here)"))
    df = pd.DataFrame(rows)
    df["Parameters"] = df["Parameters"].astype("Int64")
    df["Note"] = df["Note"].fillna("")
    write_csv(df, "defense_model_complexity.csv")
    say(f"one user's model fits in {np.median(ts):.0f} ms (median of {repeats})")
    return df.to_dict("records")


def deployed_pr_roc(data, test, logp_by_chain):
    """One-vs-rest average precision and ROC AUC per command for both deployed chains on the test windows, with the
    curves (replaces the notebook's curves, which were computed on three old experiments)."""
    yt = data["y"][test]
    Y = label_binarize(yt, classes=range(len(COMMANDS)))
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    for col, (name, logp) in enumerate(logp_by_chain.items()):
        P = np.exp(logp[test])
        aps, aucs = [], []
        for k, c in enumerate(COMMANDS):
            ap, auc = float(average_precision_score(Y[:, k], P[:, k])), float(roc_auc_score(Y[:, k], P[:, k]))
            aps.append(ap)
            aucs.append(auc)
            rows.append({"Chain": name, "Class": c, "Average precision": ap, "ROC AUC": auc, "Support": int(Y[:, k].sum()),
                         "Prevalence (AP of a random scorer)": float(Y[:, k].mean())})
            pr, rc, _ = precision_recall_curve(Y[:, k], P[:, k])
            axes[0, col].plot(rc, pr, color=COMMAND_COLORS[k], linewidth=1.8, label=f"{c} (AP {ap:.2f})")
            fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
            axes[1, col].plot(fpr, tpr, color=COMMAND_COLORS[k], linewidth=1.8, label=f"{c} (AUC {auc:.2f})")
        rows.append({"Chain": name, "Class": "macro", "Average precision": float(np.mean(aps)), "ROC AUC": float(np.mean(aucs)),
                     "Support": int(len(yt)), "Prevalence (AP of a random scorer)": float(np.mean(Y.mean(0)))})
        ax = axes[0, col]
        ax.axhline(CHANCE, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{name}: one-vs-rest precision-recall, macro AP {np.mean(aps):.2f}", loc="left", fontsize=9)
        ax = axes[1, col]
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"{name}: one-vs-rest ROC, macro AUC {np.mean(aucs):.2f}", loc="left", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower left" if ax is axes[0, col] else "lower right", fontsize=7)
            ax.grid(color=GRID, linewidth=0.6)
    fig.suptitle(f"Per-command ranking quality of the deployed chains on the {len(yt)} held-out windows (posterior of each command "
                 "against the rest)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "defense_deployed_pr_roc.png")
    plt.close(fig)
    df = pd.DataFrame(rows)
    write_csv(df, "defense_deployed_pr_roc.csv")
    say("wrote defense_deployed_pr_roc.png")
    return df[df["Class"] == "macro"].to_dict("records")


def write_tables_md():
    """One markdown file with every table, ready to paste into slides."""
    parts = ["# Defense tables (generated by make_defense_figures.py)\n"]
    for name in ("defense_per_class_metrics.csv", "defense_vote_gate.csv", "defense_gated_confusion_matrix.csv",
                 "defense_confound_controls.csv", "defense_statistics.csv", "defense_itr.csv", "defense_subject_ablation.csv",
                 "defense_deployed_pr_roc.csv", "defense_time_distance.csv", "defense_channel_contribution.csv",
                 "defense_dataset.csv", "defense_model_complexity.csv",
                 "defense_framework_equivalence.csv", "defense_framework_notebooks.csv", "defense_framework_performance.csv", "defense_project_models_benchmark.csv",
                 "defense_decision_model_latency.csv", "defense_latency_per_capture.csv"):
        f = HERE / name
        if f.exists():
            parts.append(f"## {name}\n\n{md_table(pd.read_csv(f))}\n")
    (HERE / "defense_tables.md").write_text("\n".join(parts))


# ------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--only-latency", action="store_true", help="redo only the two latency tables and their figure")
    ap.add_argument("--skip-permutation", action="store_true", help="skip the 100-permutation null (about 5 min)")
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--skip-ablation", action="store_true", help="skip the subject chain's feature-group / LDA-dims ablation")
    ap.add_argument("--skip-learning-curve", action="store_true", help="skip the per-participant learning curve")
    ap.add_argument("--skip-heldout", action="store_true", help="skip the held-out-participant confusion matrix")
    ap.add_argument("--skip-dataset-load", action="store_true", help="do not reload Data_clean for the dataset table")
    ap.add_argument("--skip-project-models", action="store_true", help="skip the project's original RF and MLP run as written (slow)")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    summary = {}

    data = load_caches()
    train, test = notebook_split(data["X_sub"], data["y"], data["groups"])
    say(f"cache: {len(data['y'])} windows, {len(np.unique(data['groups']))} recordings, "
        f"{len(np.unique(data['parts']))} participants; notebook split: {train.sum()} train / {test.sum()} test windows")
    if args.only_latency:
        bench_decision_models()
        bench_chains(data)
        write_tables_md()
        say(f"latency redone in {time.perf_counter() - t_start:.0f} s")
        return 0

    chain_imp, chain_sub, logp_imp, logp_sub = chain_predictions(data, train, test)
    acc = confusion_and_per_class(data, train, test, logp_imp, logp_sub, chain_imp)
    summary["notebook_split_accuracy"] = {k: v for k, v in acc.items() if not k.startswith("pred_")}
    summary["per_participant_weighted_accuracy"] = per_participant_chart()
    gate = vote_gate(data, test, {"Improved global chain": logp_imp, "Subject-specific chain": logp_sub})
    summary["vote_gate_deployed"] = gate[gate["Deployed"] == "yes"].drop(columns=["Deployed"]).to_dict("records")

    if not args.skip_controls:
        controls = confound_controls(data, train, test, acc["pred_sub"], acc["subject"])
        summary["confound_controls"] = controls.to_dict("records")

    equiv, nb, geo = framework_equivalence(data, train, test, chain_sub)
    summary["framework_equivalence"] = equiv.to_dict("records")
    summary["framework_performance"] = framework_performance(data, train, test, chain_sub, nb).to_dict("records")

    # ---- the PyTorch chains in more detail (sections 7-10)
    logp_by_chain = {"Improved global chain": logp_imp, "Subject-specific chain": logp_sub}
    window_pred = {"Improved global chain": acc["pred_imp"], "Subject-specific chain": acc["pred_sub"]}
    summary["per_recording"] = per_recording_confusions(data, test, logp_by_chain)
    summary["gated"] = gated_confusions(data, test, logp_by_chain)
    summary["progression"] = progression_confusions(data, train, test, acc["pred_imp"], acc["pred_sub"])
    if not args.skip_heldout:
        summary["heldout_participants_subject_chain"] = heldout_participant_confusion(data)
    summary["per_participant_accuracy"] = participant_confusions(data, test, acc["pred_sub"])
    summary["lda_scatter"] = lda_scatter(data, test, chain_imp, chain_sub)
    summary["decision_timeline"] = decision_timeline(data, test, logp_sub)
    if not args.skip_permutation:
        summary["permutation_test"] = permutation_test(data, train, test, n_perm=args.permutations)
    summary["confidence"] = confidence_figure(data, test, logp_by_chain)
    summary["time_distance"] = time_distance_figure(data, train, test, acc["pred_sub"])
    if not args.skip_learning_curve:
        summary["learning_curve"] = learning_curve_spread(data, train, test)
    channel_contribution(chain_sub)
    summary["dataset"] = dataset_figure_and_table(data, train, test, chain_imp, skip_load=args.skip_dataset_load)
    summary["itr"] = itr_table(data, test, logp_by_chain, window_pred)
    summary["statistics"] = statistics_table(data, test, logp_by_chain, window_pred)
    if not args.skip_ablation:
        summary["subject_ablation"] = subject_ablation(data, train, test, chain_sub)
    summary["model_complexity"] = model_complexity(data, train, chain_imp, chain_sub)
    if not args.skip_project_models:
        summary["project_models_benchmark"] = project_models_benchmark(data).to_dict("records")
    summary["deployed_pr_roc"] = deployed_pr_roc(data, test, logp_by_chain)

    if not args.skip_latency:
        dm = bench_decision_models()
        ch = bench_chains(data)
        summary["latency_per_capture"] = ch.to_dict("records")
        summary["decision_model_latency"] = dm.to_dict("records")

    write_tables_md()
    (HERE / "defense_summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    say(f"done in {time.perf_counter() - t_start:.0f} s; tables in defense_tables.md, numbers in defense_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
