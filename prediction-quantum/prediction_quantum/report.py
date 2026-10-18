"""BENCHMARK.md, generated from the CSV tables in results/ so that every number in the report is backed by a file."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def pm(mean, sd, nd=3):
    return fmt(mean, nd) if (sd is None or np.isnan(sd)) else f"{mean:.{nd}f} +/- {sd:.{nd}f}"


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def rel(path, base):
    return Path(path).relative_to(base).as_posix()


def write_report(out_dir, cfg, info, env, timing, report_path):
    out_dir, report_path = Path(out_dir), Path(report_path)
    base = report_path.parent
    summary = pd.read_csv(out_dir / "benchmark_summary.csv")
    results = pd.read_csv(out_dir / "benchmark_results.csv")
    splits = pd.read_csv(out_dir / "dataset_splits.csv")
    encoders = pd.read_csv(out_dir / "encoders.csv")
    noise_path = out_dir / "noise_sweep.csv"
    noise = pd.read_csv(noise_path) if noise_path.exists() else None
    conv = pd.read_csv(out_dir / "convergence.csv")
    seeds = cfg["split"]["seeds"]
    n_ref, L_ref = cfg["report"]["reference_qubits"], cfg["report"]["reference_layers"]
    chance = info["chance_level"]
    sub = info["subsample"]

    vqc = summary[summary.family == "VQC"].copy()
    best = vqc.sort_values("accuracy_mean", ascending=False).iloc[0]
    ref = vqc[(vqc.n_qubits == n_ref) & (vqc.n_layers == L_ref)]
    ref = ref.iloc[0] if len(ref) else best

    def row_of(model, n=None):
        d = summary[summary.model == model]
        if n is not None:
            d = d[d.n_qubits == n]
        return d.iloc[0] if len(d) else None

    def cmp_word(a, b):
        return "above" if a > b else ("below" if a < b else "level with")

    lr_ref, mlp_ref, gnb_ref = row_of("logreg_pca", int(best.n_qubits)), row_of("mlp_pca", int(best.n_qubits)), row_of("gnb_pca", int(best.n_qubits))
    gnb_full, mlp_full = row_of("gnb_full"), row_of("mlp_full")
    classical_best = summary[summary.family != "VQC"].sort_values("accuracy_mean", ascending=False).iloc[0]

    # ----------------------------------------------------------------------------------------------- headline
    lines = []
    lines += [f"# Benchmark: variational quantum classifier against classical baselines on Avatar EEG windows", "",
              f"Deliverable of upstream issue Avanznow/Tellekynezix#52 (the benchmark step of the QML chain #49-#52). "
              f"Every number below comes from a CSV in `{rel(out_dir, base)}/` written by one command "
              f"(`python -m prediction_quantum.run_benchmark --config configs/default.yaml`); see README.md for the environment. "
              f"Generated {env['generated']}.", ""]
    lines += ["## Headline", ""]
    lines += [f"- **Same data for every model.** {sub['n_windows_used']:,} windows (1.024 s, 15 channels, 125 Hz) of "
              f"{sub['n_recordings_used']} OpenBCI recordings from {sub['n_participants_used']} participants, through the GaussianNB pipeline's "
              f"verified loader and 570-feature extractor, split by recording ({len(seeds)} split{'s' if len(seeds) != 1 else ''}, "
              f"seeds {seeds}). Chance (majority-class share) is {chance:.3f}.",
              f"- **Accuracy.** The best circuit by mean test accuracy is **{best.label}**: {pm(best.accuracy_mean, best.accuracy_sd)} "
              f"(macro-F1 {pm(best.macro_f1_mean, best.macro_f1_sd)}). On the identical {int(best.n_qubits)}-dimensional input, logistic regression "
              f"reaches {pm(lr_ref.accuracy_mean, lr_ref.accuracy_sd)}, GaussianNB {pm(gnb_ref.accuracy_mean, gnb_ref.accuracy_sd)} and the "
              f"64-unit MLP {pm(mlp_ref.accuracy_mean, mlp_ref.accuracy_sd)}; the circuit is {cmp_word(best.accuracy_mean, lr_ref.accuracy_mean)} "
              f"logistic regression and {cmp_word(best.accuracy_mean, mlp_ref.accuracy_mean)} the MLP. The best classical model overall is "
              f"{classical_best.label} at {pm(classical_best.accuracy_mean, classical_best.accuracy_sd)}.",
              f"- **Cost.** Training the best circuit took {best.train_seconds_mean:.1f} s of optimiser time per seed against "
              f"{mlp_ref.train_seconds_mean:.1f} s for the MLP and {lr_ref.train_seconds_mean:.2f} s for logistic regression on the same input. "
              f"One window takes {best.latency_ms_mean:.2f} ms through the simulated circuit against {lr_ref.latency_ms_mean:.3f} ms for logistic "
              f"regression ({best.latency_ms_mean / lr_ref.latency_ms_mean:.0f}x) and {mlp_ref.latency_ms_mean:.3f} ms for the MLP. "
              f"Training evaluated the circuit {best.circuit_evaluations_training:,.0f} times per seed (one run per window per forward pass, "
              f"gradients by backprop on the simulator); the parameter-shift rule needed on hardware would have cost "
              f"{best.parameter_shift_equivalent_evaluations:,.0f} runs before shots.",
              ]
    if noise is not None:
        p_max = noise.noise_p.max()
        acc_p = noise.groupby(["label", "noise_p"]).accuracy.mean().unstack("noise_p")
        delta = (acc_p[p_max] - acc_p[0.0]).sort_values()
        worst = delta.index[0]
        changed_txt = ""
        if "changed_predictions" in noise:
            ch = noise[noise.noise_p == p_max].groupby("n_layers").changed_predictions.agg(["min", "max"])
            lo_L, hi_L = ch.index.min(), ch.index.max()
            changed_txt = (f" At p = {p_max} the {int(lo_L)}-layer circuits change {ch.loc[lo_L, 'min']:.0%}-{ch.loc[lo_L, 'max']:.0%} of their "
                           f"predictions, the {int(hi_L)}-layer circuits {ch.loc[hi_L, 'min']:.0%}-{ch.loc[hi_L, 'max']:.0%}.")
        lines += [f"- **Noise.** With a depolarising channel on every qubit after the embedding and after each layer, the accuracy of the "
                  f"{len(delta)} swept circuits changes by between {delta.min():+.3f} and {delta.max():+.3f} at p = {p_max} (chance {chance:.3f}); "
                  f"the deepest circuits lose most ({worst}: {acc_p.loc[worst, 0.0]:.3f} -> {acc_p.loc[worst, p_max]:.3f}), shallow ones barely move "
                  f"because the channel contracts every expectation value towards zero while the linear head's decision depends mostly on their "
                  f"direction.{changed_txt} Details in section 4.6."]
    d_lr, d_mlp = best.accuracy_mean - lr_ref.accuracy_mean, best.accuracy_mean - mlp_ref.accuracy_mean
    sd_ref = max(float(np.nan_to_num(best.accuracy_sd)), float(np.nan_to_num(lr_ref.accuracy_sd)))
    reading = (f"**Reading.** The best circuit {'beats' if d_lr > 0 else 'does not beat'} the linear model on its own {int(best.n_qubits)}-dimensional "
               f"input by {d_lr:+.3f} accuracy, {'within' if abs(d_lr) <= sd_ref else 'more than'} one standard deviation over seeds ({sd_ref:.3f}), "
               f"and is {cmp_word(best.accuracy_mean, mlp_ref.accuracy_mean)} the 64-unit MLP on the same input ({d_mlp:+.3f}). Against the best "
               f"classical model on the full 570 features it is {classical_best.accuracy_mean - best.accuracy_mean:.3f} lower, and it is "
               f"{best.latency_ms_mean / lr_ref.latency_ms_mean:.0f}x slower per window than logistic regression and "
               f"{best.latency_ms_mean / mlp_ref.latency_ms_mean:.0f}x slower than the MLP. ")
    reading += ("So on this problem the circuit adds some capacity over a linear read-out of the same components, far less than a small classical "
                "network, at orders of magnitude more compute. " if d_lr > 0 and d_mlp < 0 else
                "So on this problem the circuit shows no advantage over classical models on the same input, at orders of magnitude more compute. ")
    reading += "The value of this benchmark is the fair, reproducible comparison and its documented limits (section 5)."
    lines += ["", reading, ""]

    # ---------------------------------------------------------------------------------------------------- data
    lines += ["## 1. Data and split", "",
              "- **Source.** `Data_clean/` at the repository root (the OpenBCI recordings of the Avatar project), read with "
              "`prediction-gaussiannb/eeg_data.py` (`load_recordings`, board `16ch_125hz`, channel 15 excluded, the start-up and every failing "
              "2 s block dropped by the deployed signal-quality policy, 1-40 Hz zero-phase band-pass and 60 Hz notch per kept segment).",
              f"- **Windows.** 128 samples (1.024 s) with stride 64 cut inside each kept segment, the PyTorch GaussianNB notebook's rule including "
              f"the tail window; {info['n_windows']:,} windows from {info['n_recordings']} recordings and {info['n_participants']} participants "
              f"({info['n_recordings_skipped']} files of the other board or unreadable were skipped by the loader).",
              "- **Features.** `prediction-gaussiannb/improved_gnb_features.py::extract_features`: 570 per window = log-Euclidean covariance (120) + "
              "the same on the differenced signal (120) + Teager-Kaiser energy (30) + 10 log-scale statistics and relative band powers per channel "
              "(150) + the same on common-average-referenced channels (150).",
              f"- **Subsample.** At most {sub['max_windows_per_class']} windows per command, drawn once with seed {sub['seed']} before splitting: "
              f"{sub['n_windows_used']:,} windows used ({', '.join(f'{k} {v}' for k, v in sub['windows_per_class_used'].items())}).",
              f"- **Split.** By recording with `GroupShuffleSplit` (the protocol of the defense benchmark in `prediction-gaussiannb/defense/`): "
              f"{int(cfg['split']['test_fraction'] * 100)} % of the recordings are the test set and {int(cfg['split']['val_fraction'] * 100)} % of the "
              f"recordings are the validation set used only for per-epoch monitoring and checkpoint choice; one split per seed. The window indices "
              f"of every part are in `{rel(out_dir, base)}/split_seed<seed>.csv`, and every model (classical and quantum) reads them from there. "
              "Scaler, PCA and models are fitted on the training part only.", ""]
    rows = [(int(r.seed), r.part, f"{int(r.windows):,}", int(r.recordings), int(r.participants), *(int(r[c]) for c in D.CLASSES)) for _, r in splits.iterrows()]
    lines += [md_table(rows, ["seed", "part", "windows", "recordings", "participants", *D.CLASSES]), "",
              f"Source: `{rel(out_dir, base)}/dataset_splits.csv`, `{rel(out_dir, base)}/dataset.json`.", ""]

    # -------------------------------------------------------------------------------------------------- models
    lines += ["## 2. Models", "", "### 2.1 Encoding (classical pre-processing in front of the circuit)", "",
              "Angle embedding needs one feature per qubit, so the 570 standardised features are projected to `n_qubits` principal components "
              "(StandardScaler + PCA, training windows only). Each component is mapped to an RY angle in [0, pi]: the training 1st and 99th "
              "percentiles of the component go to 0 and pi and values beyond them are clipped, so the embedding never wraps around. The classical "
              "PCA baselines receive exactly these angle vectors, so \"same input\" holds literally. Variance kept by the PCA:", ""]
    rows = [(int(r.seed), int(r.n_qubits), f"{r.explained_variance_total:.3f}", r.explained_variance_ratio) for _, r in encoders.iterrows()]
    lines += [md_table(rows, ["seed", "qubits (components)", "variance explained", "per component"]), "",
              f"Source: `{rel(out_dir, base)}/encoders.csv`. Implementation: `prediction_quantum/encoding.py`.", "",
              "### 2.2 Variational quantum classifier", "",
              "`AngleEmbedding(RY)` on `n_qubits` wires -> `StronglyEntanglingLayers` with `n_layers` layers (3 rotation angles per qubit per layer, "
              "CNOT range (l mod (n-1)) + 1 in layer l) -> `expval(PauliZ)` on every qubit -> `nn.Linear(n_qubits, 6)` -> softmax cross-entropy. "
              f"Grid: {cfg['encoding']['n_qubits']} qubits x {cfg['vqc']['n_layers']} layers. Simulator `default.qubit` (state vector, PennyLane "
              f"{env['pennylane']}) with the torch interface, parameter broadcasting over the mini-batch and backprop differentiation. "
              "Weights are initialised uniformly in [0, 2 pi) (PennyLane's TorchLayer default) with the run's seed. Implementation: "
              "`prediction_quantum/vqc.py`.", "",
              "### 2.3 Classical baselines", "",
              "| model | input | implementation |", "|---|---|---|",
              "| Logistic regression (PCA-n) | the circuit's n angle vectors | sklearn `LogisticRegression` (multinomial, lbfgs, C = 1) - a linear model on the circuit's input, the fairest match |",
              "| GaussianNB (PCA-n) | the same angle vectors | sklearn `GaussianNB` |",
              "| MLP-64 (PCA-n) | the same angle vectors | torch `Linear(n, 64) -> ReLU -> Linear(64, 6)`, same loop, optimiser, epochs and batch size as the circuit |",
              "| GaussianNB (570 features) | all 570 standardised features | sklearn `GaussianNB` - the project's GaussianNB decoder family on the full feature set |",
              "| MLP-64 (570 features) | all 570 standardised features | the same torch MLP, as an unconstrained classical reference |", "",
              "### 2.4 Training protocol (circuit and MLPs)", "",
              f"Adam (lr {cfg['vqc']['lr']} for the circuit, {cfg['mlp']['lr']} for the MLPs), cross-entropy, batch size {cfg['vqc']['batch_size']}, "
              f"{cfg['vqc']['epochs']} epochs, data shuffled with the run's seed. Train and validation loss and accuracy are logged after every epoch "
              f"(`{rel(out_dir, base)}/runs/<run>/history.csv`); the parameters at the epoch with the best validation accuracy are kept "
              f"(`params.pt`) and evaluated once on the test part. Each run directory holds a `config.yaml` from which "
              "`python -m prediction_quantum.runs <config.yaml>` re-trains that run alone. Implementation: `prediction_quantum/train.py`, `runs.py`.", ""]

    # ------------------------------------------------------------------------------------------------- metrics
    lines += ["## 3. Metric definitions", "",
              "| metric | definition |", "|---|---|",
              "| accuracy | share of test windows whose predicted command is the true one (`sklearn.metrics.accuracy_score`) |",
              "| macro-F1 | unweighted mean over the six commands of the per-command F1 (`f1_score(average=\"macro\")`) |",
              "| balanced accuracy | mean over the six commands of the per-command recall (`balanced_accuracy_score`) |",
              "| train (s) | wall time of fitting: for torch models the forward, backward and optimiser steps over all epochs (validation passes excluded); for sklearn models `fit()` |",
              f"| latency (ms/window) | time to classify ONE window (batch of one) from its encoded input, median of {cfg['latency']['repeats']} calls after "
              f"{cfg['latency']['warmup']} warm-up calls, {cfg['compute']['torch_threads']} torch threads; the shared PCA encoding is timed separately (`encoder_latency_ms`) |",
              "| batched (us/window) | time per window when the whole test part is classified in batches of 256 (parameter broadcasting on the simulator) |",
              "| circuit evaluations | circuit runs counted by `qml.Tracker` on the device: one per window per forward pass (the hardware-equivalent count); `simulations` is the number of batched state-vector simulations actually performed |",
              "| parameter-shift equivalent | runs a parameter-shift gradient would need on hardware: windows forwarded in training x (1 + 2 x trainable circuit parameters), before shots |",
              "| parameters | trainable parameters incl. the classical head (GaussianNB: means, variances and priors) |", "",
              "Mean +/- standard deviation over the seeds (one recording-level split and one initialisation per seed).", ""]

    # ------------------------------------------------------------------------------------------------- results
    lines += ["## 4. Results", "", "### 4.1 All models", ""]
    rows = []
    for _, r in summary.iterrows():
        rows.append((r.label, r.input, f"{int(r.n_parameters):,}", pm(r.accuracy_mean, r.accuracy_sd), pm(r.macro_f1_mean, r.macro_f1_sd),
                     pm(r.balanced_accuracy_mean, r.balanced_accuracy_sd), pm(r.train_seconds_mean, r.train_seconds_sd, 2),
                     pm(r.latency_ms_mean, r.latency_ms_sd, 3), fmt(r.batched_us_per_window_mean, 1),
                     f"{r.circuit_evaluations_training:,.0f}" if not np.isnan(r.get("circuit_evaluations_training", np.nan)) else "-"))
    lines += [md_table(rows, ["model", "input", "parameters", "accuracy", "macro-F1", "balanced acc.", "train (s)", "latency (ms/window)",
                              "batched (us/window)", "circuit evals (training)"]), "",
              f"Source: `{rel(out_dir, base)}/benchmark_summary.csv` (per-run values in `benchmark_results.csv`). Chance = {chance:.3f}. "
              f"PCA encoding of one window adds {results.encoder_latency_ms.median():.3f} ms (median over runs) in front of every PCA model.", "",
              "Two context notes. The GaussianNB rows are plain sklearn `GaussianNB` on globally standardised features; the project's deployed "
              "GaussianNB decoder (`prediction-gaussiannb/`) adds per-participant normalisation and a fitted projection and reports about 0.49 window "
              "accuracy on recording-grouped folds of the same windows, so these rows are a baseline, not the deployed decoder. The MLP-64 on 570 "
              "features is in line with the defense benchmark's run of the project's original JAX MLP on the same windows under six recording-grouped "
              "hold-outs (0.702 +/- 0.030, `prediction-gaussiannb/defense/defense_tables.md`), which used 100 epochs and early stopping.", ""]

    lines += ["### 4.2 Circuit grid: qubits x depth", ""]
    qs, Ls = sorted(vqc.n_qubits.unique()), sorted(vqc.n_layers.unique())
    for metric, name in (("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("train_seconds", "train (s)"), ("latency_ms", "latency (ms/window)")):
        rows = []
        for q in qs:
            cells = []
            for L in Ls:
                r = vqc[(vqc.n_qubits == q) & (vqc.n_layers == L)]
                cells.append(pm(r[f"{metric}_mean"].iloc[0], r[f"{metric}_sd"].iloc[0], 2 if metric in ("train_seconds", "latency_ms") else 3) if len(r) else "-")
            rows.append((f"{int(q)} qubits", *cells))
        lines += [f"**{name}**", "", md_table(rows, ["", *(f"{int(L)} layer{'s' if L != 1 else ''}" for L in Ls)]), ""]
    lines += [f"Source: `{rel(out_dir, base)}/benchmark_summary.csv`.", ""]

    # convergence
    lines += ["### 4.3 Convergence", "", f"![convergence]({rel(out_dir, base)}/fig_convergence.png)", ""]
    cv = conv[(conv.model == "vqc") & (conv.n_qubits == n_ref)]
    cm = conv[(conv.model == "mlp_pca") & (conv.n_qubits == n_ref)]
    rows = []
    for (lab, L), d in sorted(cv.groupby(["label", "n_layers"]), key=lambda t: t[0][1]):
        g = d.groupby("epoch")[["train_loss", "val_loss", "val_acc"]].mean()
        be = results[results.label == lab].best_epoch.mean()
        rows.append((lab, fmt(g.val_loss.iloc[0]), fmt(g.val_loss.iloc[-1]), fmt(g.val_acc.iloc[0]), fmt(g.val_acc.max()), fmt(be, 1), fmt(d.epoch_seconds.mean(), 1)))
    if len(cm):
        g = cm.groupby("epoch")[["train_loss", "val_loss", "val_acc"]].mean()
        rows.append((cm.label.iloc[0], fmt(g.val_loss.iloc[0]), fmt(g.val_loss.iloc[-1]), fmt(g.val_acc.iloc[0]), fmt(g.val_acc.max()),
                     fmt(results[(results.model == "mlp_pca") & (results.n_qubits == n_ref)].best_epoch.mean(), 1), fmt(cm.epoch_seconds.mean(), 2)))
    lines += [md_table(rows, ["run (mean over seeds)", "val loss ep. 1", "val loss last", "val acc ep. 1", "best val acc", "best epoch", "s / epoch"]), "",
              f"The figure shows the {n_ref}-qubit circuits at every depth and the MLP on the same input (mean over seeds; dotted = training, solid = "
              f"validation). Source: `{rel(out_dir, base)}/convergence.csv`.", ""]

    # accuracy vs latency
    lines += ["### 4.4 Accuracy against inference latency", "", f"![accuracy against latency]({rel(out_dir, base)}/fig_accuracy_latency.png)", "",
              "Every model on one figure: x = median single-window inference latency on the CPU (log scale), y = test accuracy, bars = standard "
              "deviation over seeds. Circuits (blue) sit far to the right of the classical models at the same or lower accuracy; the batched "
              "throughput (table 4.1) narrows the gap on the simulator because the state-vector simulation is broadcast over the batch, which "
              "has no hardware analogue.", ""]

    # circuit evaluations
    lines += ["### 4.5 Circuit evaluations", ""]
    rows = []
    for _, r in vqc.iterrows():
        rows.append((r.label, f"{int(r.n_quantum_parameters)}", f"{r.circuit_evaluations_training:,.0f}", f"{r.simulations_training:,.0f}",
                     f"{r.parameter_shift_equivalent_evaluations:,.0f}", f"{r.circuit_evaluations_inference:,.0f}", f"{r.simulations_inference:,.0f}"))
    lines += [md_table(rows, ["circuit", "circuit params", "evaluations in training (incl. validation passes)", "batched simulations in training",
                              "parameter-shift equivalent (training part)", "evaluations on the test part", "batched simulations on the test part"]), "",
              f"Means over seeds. One evaluation = one circuit run for one window at one parameter setting (what `qml.Tracker` counts as an "
              f"execution). Training forwards every training window once per epoch ({cfg['vqc']['epochs']} epochs) plus the validation part once "
              "per epoch. On the simulator, gradients come from backprop at no extra circuit runs; on hardware each gradient step would need the "
              "parameter-shift column, and every expectation value would additionally need of the order of 1000 shots.", ""]

    # noise
    if noise is not None:
        lines += ["### 4.6 Sensitivity to simulated depolarising noise", "", f"![noise]({rel(out_dir, base)}/fig_noise.png)", ""]
        ps = sorted(noise.noise_p.unique())
        rows = []
        for (lab, q, L), d in sorted(noise.groupby(["label", "n_qubits", "n_layers"]), key=lambda t: (t[0][1], t[0][2])):
            g = d.groupby("noise_p").accuracy.mean()
            rows.append((lab, *(fmt(g.loc[p]) for p in ps)))
        lines += [md_table(rows, ["circuit", *(f"p = {p}" for p in ps)]), "",
                  f"Accuracy on the test part (seed{'s' if len(cfg['noise']['seeds']) != 1 else ''} {cfg['noise']['seeds']}) when the trained parameters are run on "
                  f"`default.mixed` with `qml.DepolarizingChannel(p)` on every qubit after the embedding and after each entangling layer (n_layers + 1 "
                  f"noise layers), inference only. The p = 0 column reproduces the noiseless accuracy of the same run. Circuits with more than "
                  f"{cfg['noise']['max_qubits']} qubits were not swept because density-matrix simulation scales as 4^n. Macro-F1 and balanced accuracy per p are "
                  f"in `{rel(out_dir, base)}/noise_sweep.csv`.", ""]
        if "mean_abs_expectation" in noise:
            rows = []
            for (lab, q, L), d in sorted(noise.groupby(["label", "n_qubits", "n_layers"]), key=lambda t: (t[0][1], t[0][2])):
                g = d.groupby("noise_p")[["mean_abs_expectation", "changed_predictions"]].mean()
                rows.append((lab, *(f"{g.loc[p, 'mean_abs_expectation']:.3f} / {g.loc[p, 'changed_predictions']:.0%}" for p in ps)))
            gref = noise[(noise.n_qubits == ref.n_qubits) & (noise.n_layers == ref.n_layers)].groupby("noise_p")[["mean_abs_expectation", "changed_predictions"]].mean()
            if not len(gref):
                gref = noise[noise.label == worst].groupby("noise_p")[["mean_abs_expectation", "changed_predictions"]].mean()
            lines += ["**What the noise does to the read-out** (mean |<Z>| over the test windows / share of windows whose predicted command changed from p = 0):", "",
                      md_table(rows, ["circuit", *(f"p = {p}" for p in ps)]), "",
                      f"A depolarising channel shrinks a qubit's Bloch vector by 1 - 4p/3 per application, so every expectation value is pulled towards "
                      f"zero: for the {ref.label} circuit the mean |<Z>| goes from {gref.loc[0.0, 'mean_abs_expectation']:.3f} at p = 0 to "
                      f"{gref.loc[p_max, 'mean_abs_expectation']:.3f} at p = {p_max} (a factor {gref.loc[p_max, 'mean_abs_expectation'] / gref.loc[0.0, 'mean_abs_expectation']:.2f}), "
                      f"while {gref.loc[p_max, 'changed_predictions']:.0%} of its test windows change command. The linear head's argmax depends mostly on the "
                      f"direction of the expectation vector, so a shallow circuit keeps most of its (low) accuracy until the contraction lets the head's bias "
                      f"terms dominate; circuits with more layers pass through more noise layers and lose more ({worst}: {acc_p.loc[worst, 0.0]:.3f} -> "
                      f"{acc_p.loc[worst, p_max]:.3f}). Small increases at low p are within the run-to-run spread of the noiseless accuracy and not a noise "
                      f"benefit. Source: `{rel(out_dir, base)}/noise_sweep.csv`.", ""]

    # --------------------------------------------------------------------------------------------- limitations
    lines += ["## 5. Limitations", "",
              "- **Simulator only.** Every circuit ran on PennyLane's CPU state-vector (`default.qubit`) or density-matrix (`default.mixed`) simulator. "
              "No hardware run; the latency, circuit-evaluation and noise numbers describe the simulation, not a device. On hardware every "
              "expectation value needs shots (not modelled), gradients need parameter-shift evaluations (only counted, section 4.5) and noise is not "
              "a uniform depolarising channel.",
              f"- **Subsampled, within-participant data.** At most {sub['max_windows_per_class']} windows per command were used, and the split is by "
              "recording, not by participant: as the GaussianNB notebooks document, held-out participants are at chance on this dataset, so these "
              "accuracies measure within-participant separability. Overlapping windows (stride 64 of 128) of one recording are always in the same part.",
              f"- **PCA reduction.** The circuit sees {cfg['encoding']['n_qubits']} principal components of 570 features "
              f"({', '.join(f'{int(r.n_qubits)}: {r.explained_variance_total:.0%}' for _, r in encoders[encoders.seed == seeds[0]].iterrows())} of the "
              "variance, seed 0). The 570-feature baselines show what that reduction costs; a different encoding (amplitude encoding, data "
              "re-uploading, more qubits) could move the circuit's numbers.",
              f"- **Few seeds.** {len(seeds)} seed{'s' if len(seeds) != 1 else ''} ({seeds}), each with its own recording-level split and initialisation; "
              "the noise sweep used " + f"seed{'s' if len(cfg['noise']['seeds']) != 1 else ''} {cfg['noise']['seeds']} only. "
              "The standard deviations are over seeds and conflate split and initialisation variance.",
              f"- **One optimiser setting.** Adam at lr {cfg['vqc']['lr']} (circuit) / {cfg['mlp']['lr']} (MLP), batch {cfg['vqc']['batch_size']}, "
              f"{cfg['vqc']['epochs']} epochs, best-validation checkpoint; no hyper-parameter search for any model, so no model is tuned.",
              "- **Depth and width grid.** Only StronglyEntanglingLayers with 1, 2 and 4 layers on 4, 6 and 8 qubits; no other ansatz, no "
              "re-uploading, no trainable encoding.",
              "- **Latency is CPU wall time** of a Python call on this machine (table in section 6) and depends on thread count and load; the "
              "ratios between models are more robust than the absolute values.", ""]

    # ----------------------------------------------------------------------------------------- reproduction
    lines += ["## 6. Reproduction and environment", "",
              "```bash", "cd prediction-quantum", "python -m prediction_quantum.run_benchmark --config configs/default.yaml --data-dir ../Data_clean", "```", "",
              f"The command rebuilds the feature cache from `Data_clean` when missing, writes every CSV, figure and this report, and skips finished "
              f"runs (`--force` redoes them; `--quick` is a few-minute smoke version). Compute time of the run that produced this report: "
              f"{timing.get('total_seconds_phases', float('nan')) / 60:.0f} min in total "
              f"(features and splits {timing.get('data_seconds', 0):.0f} s, training {timing.get('training_seconds', 0) / 60:.1f} min with "
              f"{cfg['compute']['n_jobs']} parallel processes, latency {timing.get('latency_seconds', 0):.0f} s, noise sweep {timing.get('noise_seconds', 0) / 60:.1f} min). "
              f"Source: `{rel(out_dir, base)}/timing.json`.", ""]
    rows = [(k, v) for k, v in env.items()]
    lines += [md_table(rows, ["environment", ""]), "", f"Source: `{rel(out_dir, base)}/environment.json`. Tests: `pytest tests/`.", ""]
    lines += ["## Files", "",
              f"| file | content |", "|---|---|",
              f"| `{rel(out_dir, base)}/benchmark_summary.csv` | one row per model: mean and sd of every metric over seeds |",
              f"| `{rel(out_dir, base)}/benchmark_results.csv` | one row per run (model x seed) |",
              f"| `{rel(out_dir, base)}/convergence.csv` | per-epoch train / validation loss and accuracy of every torch run |",
              f"| `{rel(out_dir, base)}/noise_sweep.csv` | accuracy, macro-F1, balanced accuracy of every swept circuit at every p |",
              f"| `{rel(out_dir, base)}/split_seed<k>.csv`, `dataset_splits.csv`, `dataset.json` | the exact windows of every part, counts, loader statistics |",
              f"| `{rel(out_dir, base)}/encoders.csv`, `encoders/` | PCA variance per encoder; the fitted encoders (joblib) |",
              f"| `{rel(out_dir, base)}/runs/<run>/` | `config.yaml`, `metrics.json`, `history.csv`, `params.pt` or `model.joblib`, `test_predictions.npy` |",
              f"| `{rel(out_dir, base)}/fig_*.png` | the three figures |",
              f"| `{rel(out_dir, base)}/environment.json`, `timing.json`, `config_used.yaml` | versions, machine, phase timings, the resolved config |", ""]
    report_path.write_text("\n".join(lines))
    return report_path
