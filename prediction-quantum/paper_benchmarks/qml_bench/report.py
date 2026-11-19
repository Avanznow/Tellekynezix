"""REPORT.md, summary.csv, mcnemar.csv and figures from the files written by qml_bench.run.

    ~/qml_env/bin/python -m qml_bench.report ../results/paper/qml
Every number in REPORT.md is computed here from runs.csv, per_class.csv, confusions.json and predictions.npz.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import paths

# setup() goes before the matplotlib import because MPLCONFIGDIR is read at import time (hence the E402s below)
paths.setup()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
FAMILY = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "Liberation Sans"
plt.rcParams.update({"font.family": FAMILY, "font.size": 8.5})

NAMES = {"qsvm": "Quantum kernel SVM", "vqc": "VQC", "logreg": "Logistic regression", "gnb": "GaussianNB",
         "svm_rbf": "RBF SVM", "mlp": "MLP-64", "gnb_full": "GaussianNB (570)", "lda_gnb_full": "LDA + GaussianNB (570)",
         "mlp_full": "MLP-64 (570)"}
# row order of the results tables and bar order of the accuracy figure: quantum models, angle-input baselines, then the 570-feature ones
ORDER = ["qsvm", "vqc", "logreg", "gnb", "svm_rbf", "mlp", "gnb_full", "lda_gnb_full", "mlp_full"]
METRICS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "cohen_kappa"]
PROT = {"participant": "Participant-independent (GroupKFold by participant)",
        "recording": "Recording-grouped (seeded splits of the original benchmark)"}
COL = {"quantum (simulated)": "#eb6834", "classical": "#2a78d6"}


def md_table(df, index=True):
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rows = ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(d.columns)] + rows)


def pm(m, s, d=3):
    """Mean ± SD with d decimals; the ± part is dropped when the SD is NaN (a protocol with a single fold)."""
    return f"{m:.{d}f} ± {s:.{d}f}" if np.isfinite(s) else f"{m:.{d}f}"


def summarize(runs):
    """Mean and SD of every metric per (protocol, model), one row each; this is what summary.csv holds."""
    g = runs.groupby(["protocol", "model"])
    agg = {k: ["mean", "std"] for k in METRICS + ["fit_seconds", "latency_ms", "majority_share"]}
    s = g.agg(agg)
    # flatten the (metric, stat) column MultiIndex to accuracy_mean, accuracy_std, ...
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_folds"] = g.size()
    s["family"] = g["family"].first()
    s["input"] = g["input"].first()
    # median rather than mean: the SVM count includes the support vectors, so it changes from fold to fold
    s["n_parameters"] = g["n_parameters"].median()
    return s.reset_index()


def mcnemar(preds, prot, folds, a, b):
    """Exact McNemar test of model a against model b on the identical test windows of prot, pooled over folds.
    Returns the two discordant counts and the two-sided exact p-value as one row of mcnemar.csv."""
    bb = cc = 0
    for f in folds:
        y = preds[f"{prot}|{f}|y_true"]
        ra, rb = preds[f"{prot}|{f}|{a}"] == y, preds[f"{prot}|{f}|{b}"] == y
        # only the discordant windows count: a right and b wrong, and the reverse
        bb += int((ra & ~rb).sum())
        cc += int((~ra & rb).sum())
    # under H0 the discordant windows split 50/50, hence the exact binomial; p = 1 when the models never disagree
    p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if bb + cc else 1.0
    return dict(protocol=prot, model_a=a, model_b=b, a_right_only=bb, b_right_only=cc, p_exact=p)


def figure(summary, out):
    """fig_qml_accuracy.png: mean ± SD accuracy and macro-F1 bars per model, one panel per protocol."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, prot in zip(axes, ["participant", "recording"]):
        d = summary[summary.protocol == prot].set_index("model").reindex([m for m in ORDER if m in set(summary.model)])
        x = np.arange(len(d))
        # two 0.38-wide bars per model, centered at ±0.2: solid accuracy on the left, hatched macro-F1 on the right
        ax.bar(x - 0.2, d["accuracy_mean"], 0.38, yerr=d["accuracy_std"], color=[COL[f] for f in d["family"]],
               ecolor="#0b0b0b", capsize=1.5, label="accuracy")
        ax.bar(x + 0.2, d["macro_f1_mean"], 0.38, yerr=d["macro_f1_std"], color=[COL[f] for f in d["family"]],
               ecolor="#0b0b0b", capsize=1.5, alpha=0.55, hatch="///", edgecolor="white", label="macro-F1")
        ax.axhline(1 / 6, ls=(0, (4, 3)), color="#52514e", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[m] for m in d.index], rotation=55, ha="right", fontsize=7)
        ax.set_title({"participant": "A  Participant-independent", "recording": "B  Recording-grouped"}[prot],
                     loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color="#d9d8d4", lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Score on held-out test windows")
    # color encodes the family and hatching the metric, so the legend needs proxy handles instead of ax.legend()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=COL["quantum (simulated)"], label="quantum (simulated)"), Patch(color=COL["classical"], label="classical"),
         Patch(facecolor="#888888", hatch="///", edgecolor="white", label="macro-F1 (hatched)"),
         Line2D([0], [0], ls=(0, (4, 3)), color="#52514e", lw=0.8, label="chance (1/6)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7)
    # the bottom 6 % of the figure is kept free for the shared legend
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_qml_accuracy.png", dpi=300)
    plt.close(fig)


def confusion_figure(conf, labels, out, prot="participant"):
    """fig_qml_confusion_<prot>.png: row-normalized confusion matrices of QSVM and VQC, summed over folds."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, m in zip(axes, ["qsvm", "vqc"]):
        mats = [np.array(v) for k, v in conf.items() if k.startswith(f"{prot}|") and k.endswith(f"|{m}")]
        if not mats:
            ax.axis("off")
            continue
        # sum the 6x6 counts over folds, then normalize rows: each row is the recall profile of that true command
        C = np.sum(mats, axis=0)
        R = C / C.sum(1, keepdims=True)
        # a vmax floor of 0.5 stops a near-chance matrix (cells around 0.17) from being drawn in the darkest blues
        ax.imshow(R, cmap="Blues", vmin=0, vmax=max(0.5, R.max()))
        # 0.35 is about where Blues gets too dark for black text
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if R[i, j] > 0.35 else "#0b0b0b")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{NAMES[m]} ({'participant-independent' if prot == 'participant' else 'recording-grouped'})",
                     fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / f"fig_qml_confusion_{prot}.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    """Build summary.csv, mcnemar.csv, the figures and REPORT.md from one run's output folder (argv[0], default
    paths.RESULTS) and write them back into it."""
    # run.py passes [out]; the CLI leaves argv None, so the folder comes from sys.argv, else the default
    out = Path((argv or sys.argv[1:] or [str(paths.RESULTS)])[0])
    runs = pd.read_csv(out / "runs.csv")
    pcls = pd.read_csv(out / "per_class.csv")
    conf = json.loads((out / "confusions.json").read_text())
    preds = dict(np.load(out / "predictions.npz"))
    env = json.loads((out / "environment.json").read_text())
    folds_df = pd.read_csv(out / "folds.csv")
    summary = summarize(runs)
    summary.to_csv(out / "summary.csv", index=False)
    figure(summary, out)
    for prot in ("participant", "recording"):
        confusion_figure(conf["confusions"], conf["labels"], out, prot)
    tests = []
    for prot in ("participant", "recording"):
        fl = sorted(runs.loc[runs.protocol == prot, "fold"].unique())
        avail = set(runs.loc[runs.protocol == prot, "model"])
        # each quantum model against its classical twin on the same angles, against the project's LDA + GaussianNB
        # decoder on the full 570 features, and the two quantum models against each other
        for a, b in [("qsvm", "svm_rbf"), ("vqc", "mlp"), ("qsvm", "lda_gnb_full"), ("vqc", "lda_gnb_full"),
                     ("qsvm", "vqc")]:
            if a in avail and b in avail:
                tests.append(mcnemar(preds, prot, fl, a, b))
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "mcnemar.csv", index=False)
    cfg = env["config"]
    L = ["# QML Benchmark Report", "",
         f"Generated by `benchmarks/qml/qml_bench/report.py` from the files of one executed run "
         f"({env['generated']}; Python {env['python']}, PennyLane {env['pennylane']}, torch {env['torch']}, "
         f"scikit-learn {env['scikit_learn']}; {env['platform']}). Every number below was computed from "
         "`runs.csv`, `per_class.csv`, `confusions.json` and `predictions.npz` in this folder. All quantum circuits were "
         "simulated exactly on a classical CPU; nothing ran on quantum hardware.", "",
         "## Data, inputs and protocols", "",
         f"- **Windows.** {env['n_windows']:,} windows (≤ {cfg['data']['max_windows_per_class']} per command, seed "
         f"{cfg['data']['subsample_seed']}; with 2000 per command this is the subsample of the original `prediction-quantum` benchmark) of the "
         "project's 570-feature cache (verified loader, 128-sample windows, stride 64, 15 channels at 125 Hz; "
         f"`{Path(env['feature_cache']).name}`).",
         f"- **Inputs.** Per fold, StandardScaler + PCA to {cfg['encoding']['n_qubits']} components + percentile angle "
         "map to [0, π] (the project's `FeatureEncoder`), fitted on the training part only; the quantum models and the "
         "angle baselines see exactly these angles. The `(570)` baselines see the 570 standardized features.",
         "- **Protocols.** " + "; ".join(f"{k}: {v}" for k, v in PROT.items()) + ". Validation recordings are taken "
         "from the training part of each fold and used only to choose the checkpoint of the VQC and the MLPs. Leakage "
         "checks (no recording, and in the participant protocol no participant, on two sides) passed for every fold.",
         f"- **Quantum kernel SVM.** ZZ feature map on a ring of {cfg['encoding']['n_qubits']} qubits, {cfg['qsvm']['reps']} "
         f"repetitions, fidelity kernel from exact statevectors, SVC (C = {cfg['qsvm']['C']}) on a stratified "
         f"training subsample of at most {cfg['qsvm']['max_train']} windows; the RBF SVM uses the same subsample.",
         f"- **VQC.** The project's classifier (`prediction_quantum/vqc.py`): RY angle embedding, "
         f"{cfg['vqc']['n_layers']} StronglyEntanglingLayers, Pauli-Z readout, linear head; Adam (lr {cfg['vqc']['lr']}), "
         f"batch {cfg['vqc']['batch_size']}, {cfg['vqc']['epochs']} epochs, best-validation checkpoint.",
         "- **No hyperparameter tuning** for any model; every model uses fixed, documented settings.", "",
         "## Fold composition", "", md_table(folds_df[folds_df.part == "test"][["protocol", "fold", "windows", "recordings",
                                                                       "participants", "test_participants"]], index=False), ""]
    for prot in ("participant", "recording"):
        d = summary[summary.protocol == prot].set_index("model")
        d = d.reindex([m for m in ORDER if m in d.index])
        L += [f"## Results: {PROT[prot]}", "",
              f"Mean ± SD over {int(d['n_folds'].max())} folds. Chance for six balanced commands is 0.167; the "
              f"majority-class share of the test folds averages {d['majority_share_mean'].mean():.3f}.", "",
              "| Model | Family | Input | Accuracy | Balanced acc. | Macro precision | Macro recall | Macro-F1 | Kappa | "
              "Fit (s) | Latency (ms/window) | Parameters |", "|" + "---|" * 12]
        for m, r in d.iterrows():
            L.append(f"| {NAMES[m]} | {r['family']} | {r['input']} | {pm(r['accuracy_mean'], r['accuracy_std'])} | "
                     f"{pm(r['balanced_accuracy_mean'], r['balanced_accuracy_std'])} | "
                     f"{pm(r['macro_precision_mean'], r['macro_precision_std'])} | "
                     f"{pm(r['macro_recall_mean'], r['macro_recall_std'])} | {pm(r['macro_f1_mean'], r['macro_f1_std'])} | "
                     f"{pm(r['cohen_kappa_mean'], r['cohen_kappa_std'])} | {r['fit_seconds_mean']:.2f} | "
                     f"{r['latency_ms_mean']:.3f} | {int(r['n_parameters']) if np.isfinite(r['n_parameters']) else '—'} |")
        q = runs[(runs.protocol == prot)]
        L += [""]
        # x_* columns are the extras run_model returns, prefixed by run.py (support vectors, kernel circuits, best epoch, ...)
        if "qsvm" in set(q.model):
            qq = q[q.model == "qsvm"]
            L.append(f"- Quantum kernel SVM: {int(qq['x_n_train_used'].mean())} training windows per fold, "
                     f"{int(qq['x_n_support'].mean())} support vectors on average; a device would need about "
                     f"{qq['x_hardware_kernel_evaluations'].mean():,.0f} kernel-circuit estimates per fold before shots.")
        if "vqc" in set(q.model):
            qv = q[q.model == "vqc"]
            L.append(f"- VQC: {int(qv['x_n_circuit_parameters'].iloc[0])} circuit parameters; best validation epoch "
                     f"{qv['x_best_epoch'].mean():.1f} on average; final validation loss {qv['x_final_val_loss'].mean():.3f} "
                     f"(ln 6 = 1.792 for an uninformed classifier); parameter-shift equivalent of training "
                     f"{qv['x_parameter_shift_equivalent'].mean():,.0f} circuit runs per fold.")
        # one row per model, one column per command after the unstack
        pc = pcls[(pcls.protocol == prot) & (pcls.model.isin(["qsvm", "vqc"]))].groupby(["model", "cls"])["f1"].mean().unstack()
        if len(pc):
            L += ["", "Per-command F1 of the quantum models (mean over folds):", "", md_table(pc.round(3).rename(index=NAMES)), ""]
        # an empty frame has no protocol column, so the filter is skipped when no model pair was available
        t = tests[tests.protocol == prot] if len(tests) else tests
        if len(t):
            L += ["Exact McNemar tests on the identical test windows, pooled over folds (a_right_only / b_right_only):", "",
                  "| Model A | Model B | A right only | B right only | p |", "|---|---|---|---|---|"]
            for _, r in t.iterrows():
                L.append(f"| {NAMES[r.model_a]} | {NAMES[r.model_b]} | {r.a_right_only} | {r.b_right_only} | {r.p_exact:.3g} |")
            L.append("")
    L += ["## Figures", "", "- `fig_qml_accuracy.png`: accuracy and macro-F1 of every model under both protocols.",
          "- `fig_qml_confusion_participant.png`, `fig_qml_confusion_recording.png`: row-normalized confusion matrices "
          "of the two quantum models, summed over folds.", "",
          "## Limitations", "",
          "- Simulation only: exact statevectors, no shot noise, no device noise, no queueing; latencies are CPU wall "
          "time of a Python call on this machine and say nothing about hardware execution.",
          "- Dimensionality: angle encoding uses one PCA component per qubit, so the circuits see a few components of the "
          "570 features; exact simulation cost grows as 2^n with the number of qubits and the kernel cost as n_train^2, "
          "which is why the kernel models use a training subsample.",
          "- No hyperparameter search for any model, a single feature map and a single ansatz.",
          "- Few folds (5 and 3); standard deviations mix split and initialization variance.",
          "- The recording-grouped protocol measures within-session separability on a block-design dataset (paper, "
          "Section 3.4); only the participant-independent protocol speaks to a new user.",
          "- The benchmark is offline and independent of the real-time drone-control pipeline; no model here was "
          "integrated into the GUI.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    print(f"report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
