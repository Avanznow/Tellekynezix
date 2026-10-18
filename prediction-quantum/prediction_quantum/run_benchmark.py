"""Run the whole benchmark from one config: features -> splits -> encoders -> training runs (parallel) -> latency ->
noise sweep -> CSV tables -> figures -> BENCHMARK.md.

    cd prediction-quantum
    python -m prediction_quantum.run_benchmark --config configs/default.yaml [--data-dir ../Data_clean] [--quick]
        [--force] [--features-only] [--report-only] [--n-jobs N] [--output-dir results]

Finished runs (results/runs/<id>/metrics.json) are skipped unless --force, so an interrupted benchmark resumes.
"""
import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from . import data as D
from . import figures, report
from .encoding import FeatureEncoder
from .metrics import classification_scores, median_latency_ms
from .noise import chance_level, noise_sweep
from .runs import load_trained, run_config, run_single, run_specs, versions
from .train import predict

PROJECT_DIR = D.PROJECT_DIR


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_paths(cfg, data_dir=None, output_dir=None):
    if data_dir:
        cfg["data"]["data_dir"] = data_dir
    if output_dir:
        cfg["output_dir"] = output_dir
    for key in ("data_dir", "cache_dir"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = str(p if p.is_absolute() else PROJECT_DIR / p)
    p = Path(cfg["output_dir"])
    cfg["output_dir"] = str(p if p.is_absolute() else PROJECT_DIR / p)
    return cfg


def apply_quick(cfg):
    """A few-minute smoke version of the benchmark (results_quick/): same code path, tiny budget."""
    cfg["split"]["seeds"] = [0]
    cfg["encoding"]["n_qubits"] = [4]
    cfg["vqc"]["n_layers"] = [1]
    cfg["vqc"]["epochs"] = cfg["mlp"]["epochs"] = 2
    cfg["data"]["max_windows_per_class"] = 150
    cfg["noise"]["probabilities"] = [0.0, 0.1]
    cfg["latency"]["repeats"] = 5
    cfg["report"]["reference_qubits"], cfg["report"]["reference_layers"] = 4, 1
    cfg["output_dir"] = "results_quick"
    return cfg


def cpu_name():
    try:
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        return m.group(1).strip() if m else platform.processor()
    except OSError:
        return platform.processor()


# ------------------------------------------------------------------------------------------------- 1. data
def phase_data(cfg, out_dir, force=False):
    d = cfg["data"]
    feats = D.build_features(d["data_dir"], d["cache_dir"], n_jobs=d.get("n_jobs_load", -1), verbose=True)
    y, groups, parts = feats["y"], feats["groups"], feats["participants"]
    keep = D.subsample_per_class(y, d["max_windows_per_class"], d["subsample_seed"])
    enc_dir = out_dir / "encoders"
    enc_dir.mkdir(parents=True, exist_ok=True)
    summaries, enc_rows = [], []
    for seed in cfg["split"]["seeds"]:
        path = out_dir / f"split_seed{seed}.csv"
        if force or not path.exists():
            part = D.make_split(y[keep], groups[keep], seed, cfg["split"]["test_fraction"], cfg["split"]["val_fraction"])
            D.write_split(path, keep, y, groups, parts, part)
        split = D.load_split(path)
        s = D.split_summary(split["frame"])
        s.insert(0, "seed", seed)
        summaries.append(s)
        Xtr = feats["X"][split["train"]]
        scaler_path = enc_dir / f"scaler_full_s{seed}.joblib"
        if force or not scaler_path.exists():
            joblib.dump(StandardScaler().fit(Xtr), scaler_path)
        for n in cfg["encoding"]["n_qubits"]:
            p = enc_dir / f"encoder_q{n}_s{seed}.joblib"
            if force or not p.exists():
                enc = FeatureEncoder(n, tuple(cfg["encoding"]["angle_range"]), tuple(cfg["encoding"]["clip_percentiles"])).fit(Xtr)
                joblib.dump(enc, p)
            else:
                enc = joblib.load(p)
            desc = enc.describe()
            enc_rows.append(dict(seed=seed, n_qubits=n, explained_variance_total=desc["explained_variance_total"],
                                 explained_variance_ratio=" ".join(f"{v:.3f}" for v in desc["explained_variance_ratio"])))
    pd.concat(summaries, ignore_index=True).to_csv(out_dir / "dataset_splits.csv", index=False)
    pd.DataFrame(enc_rows).to_csv(out_dir / "encoders.csv", index=False)
    info = dict(feats["meta"], subsample=dict(max_windows_per_class=d["max_windows_per_class"], seed=d["subsample_seed"],
                                               n_windows_used=int(len(keep)), n_recordings_used=int(len(np.unique(groups[keep]))),
                                               n_participants_used=int(len(np.unique(parts[keep]))),
                                               windows_per_class_used={c: int((y[keep] == i).sum()) for i, c in enumerate(D.CLASSES)}),
                chance_level=chance_level(y[keep]), feature_cache=feats["cache"])
    (out_dir / "dataset.json").write_text(json.dumps(info, indent=2, default=str))
    say(f"data: {len(keep)} windows of {info['subsample']['n_recordings_used']} recordings used; chance level {info['chance_level']:.3f}")
    return feats, info


# --------------------------------------------------------------------------------------------- 2. training
def phase_train(cfg, out_dir, force=False):
    specs = run_specs(cfg)
    rcs = [run_config(s, cfg, out_dir) for s in specs]
    cfg_data = dict(data_dir=cfg["data"]["data_dir"], cache_dir=cfg["data"]["cache_dir"])
    pending = [rc for rc in rcs if force or not (out_dir / "runs" / rc["run_id"] / "metrics.json").exists()]
    cost = {"vqc": 3, "mlp_full": 2, "mlp_pca": 1, "gnb_pca": 0, "gnb_full": 0, "logreg_pca": 0}
    pending.sort(key=lambda rc: -(cost[rc["model"]] * 100 + (rc["n_qubits"] or 0) * (rc["n_layers"] or 1)))
    say(f"training: {len(pending)} of {len(rcs)} runs to do in {cfg['compute']['n_jobs']} processes x {cfg['compute']['torch_threads']} threads")
    if pending:
        Parallel(n_jobs=cfg["compute"]["n_jobs"], backend="loky")(
            delayed(run_single)(rc, out_dir, cfg_data, force=force, log=say) for rc in pending)
    return rcs


# ------------------------------------------------------------------------------------- 3. latency and noise
def test_inputs(rc, out_dir, feats):
    split = D.load_split(out_dir / rc["split_csv"])
    enc = joblib.load(out_dir / rc["encoder"])
    return enc, enc.transform(feats["X"][split["test"]]).astype(np.float32), feats["y"][split["test"]], feats["X"][split["test"]]


def phase_latency(cfg, out_dir, rcs, feats, force=False):
    torch.set_num_threads(int(cfg["compute"]["torch_threads"]))
    rep, warm = cfg["latency"]["repeats"], cfg["latency"]["warmup"]
    for rc in rcs:
        path = out_dir / "runs" / rc["run_id"] / "metrics.json"
        m = json.loads(path.read_text())
        if "latency_ms" in m and not force:
            continue
        enc, Ate, yte, Xte = test_inputs(rc, out_dir, feats)
        model = load_trained(rc, out_dir)
        if rc["model"] in ("vqc", "mlp_pca", "mlp_full"):
            x1 = torch.as_tensor(Ate[:1])

            def one():
                with torch.no_grad():
                    model(x1)
            lat, p90 = median_latency_ms(one, rep, warm)
            predict(model, Ate[:256])                                   # warm-up for the batched measurement
            if rc["model"] == "vqc":
                with model.tracker() as tr:
                    t0 = time.perf_counter()
                    pred = predict(model, Ate, batch_size=256)
                    dt = time.perf_counter() - t0
                m["circuit_evaluations_inference"] = int(tr.totals.get("executions", 0))
                m["simulations_inference"] = int(tr.totals.get("simulations", 0))
            else:
                t0 = time.perf_counter()
                pred = predict(model, Ate, batch_size=256)
                dt = time.perf_counter() - t0
        else:
            x1 = Ate[:1].astype(np.float64)
            lat, p90 = median_latency_ms(lambda: model.predict(x1), rep, warm)
            t0 = time.perf_counter()
            pred = model.predict(Ate.astype(np.float64))
            dt = time.perf_counter() - t0
        enc_lat, _ = median_latency_ms(lambda: enc.transform(Xte[:1]), rep, warm)
        m.update(latency_ms=lat, latency_p90_ms=p90, batched_us_per_window=dt / len(yte) * 1e6, encoder_latency_ms=enc_lat,
                 latency_threads=int(cfg["compute"]["torch_threads"]), accuracy_reloaded=classification_scores(yte, pred)["accuracy"])
        path.write_text(json.dumps(m, indent=2))
        say(f"latency {rc['run_id']}: {lat:.3f} ms per window (p90 {p90:.3f}), batched {m['batched_us_per_window']:.1f} us/window")


def phase_noise(cfg, out_dir, rcs, feats, force=False):
    path = out_dir / "noise_sweep.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    torch.set_num_threads(int(cfg["compute"]["torch_threads"]))
    nz = cfg["noise"]
    frames = []
    for rc in rcs:
        if rc["model"] != "vqc" or rc["n_qubits"] > nz["max_qubits"] or rc["seed"] not in nz["seeds"]:
            continue
        say(f"noise sweep {rc['run_id']}")
        _, Ate, yte, _ = test_inputs(rc, out_dir, feats)
        model = load_trained(rc, out_dir)
        df = noise_sweep(model.state_dict(), rc["n_qubits"], rc["n_layers"], Ate, yte, nz["probabilities"], nz["batch_size"], log=say)
        for k, v in (("run_id", rc["run_id"]), ("label", rc["label"]), ("n_qubits", rc["n_qubits"]), ("n_layers", rc["n_layers"]), ("seed", rc["seed"])):
            df.insert(0, k, v)
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ------------------------------------------------------------------------------------------ 4. aggregation
NUMERIC = ["accuracy", "macro_f1", "balanced_accuracy", "train_seconds", "latency_ms", "batched_us_per_window", "encoder_latency_ms"]


def aggregate(out_dir, rcs):
    rows = [json.loads((out_dir / "runs" / rc["run_id"] / "metrics.json").read_text()) for rc in rcs]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    keys = ["label", "family", "model", "input", "n_qubits", "n_layers"]
    agg = dict(n_seeds=("seed", "size"), n_parameters=("n_parameters", "first"))
    for c in NUMERIC:
        agg[f"{c}_mean"], agg[f"{c}_sd"] = (c, "mean"), (c, "std")
    for c in ("n_quantum_parameters", "circuit_evaluations_training", "simulations_training", "parameter_shift_equivalent_evaluations",
              "circuit_evaluations_inference", "simulations_inference", "best_epoch", "n_test", "n_train"):
        if c in df:
            agg[c] = (c, "mean")
    summary = df.groupby(keys, dropna=False).agg(**agg).reset_index()
    order = {"VQC": 0, "Logistic regression": 1, "GaussianNB": 2, "MLP": 3}
    summary["_o"] = summary.family.map(order)
    summary = summary.sort_values(["_o", "n_qubits", "n_layers"], na_position="last").drop(columns="_o")
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    hist = []
    for rc in rcs:
        p = out_dir / "runs" / rc["run_id"] / "history.csv"
        if p.exists():
            h = pd.read_csv(p)
            for k in ("seed", "n_layers", "n_qubits", "family", "model", "label", "run_id"):
                h.insert(0, k, rc[k])
            hist.append(h)
    conv = pd.concat(hist, ignore_index=True)
    conv.to_csv(out_dir / "convergence.csv", index=False)
    return df, summary, conv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_DIR / "configs" / "default.yaml"))
    ap.add_argument("--data-dir", default=None, help="Data_clean directory (default: data.data_dir of the config)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny smoke benchmark into results_quick/")
    ap.add_argument("--force", action="store_true", help="redo finished runs")
    ap.add_argument("--features-only", action="store_true", help="build the feature cache and the splits, then stop")
    ap.add_argument("--report-only", action="store_true", help="only aggregate, draw and write the report from finished runs")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.quick:
        cfg = apply_quick(cfg)
    if a.n_jobs:
        cfg["compute"]["n_jobs"] = a.n_jobs
    cfg = resolve_paths(cfg, a.data_dir, a.output_dir)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    timing_path = out_dir / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    t_all = time.perf_counter()

    def timed(name, fn):
        t0 = time.perf_counter()
        r = fn()
        timing[name] = round(time.perf_counter() - t0, 1)
        timing_path.write_text(json.dumps(timing, indent=2))
        return r

    say(f"config {a.config} -> {out_dir}")
    feats, info = timed("data_seconds", lambda: phase_data(cfg, out_dir, a.force))
    if a.features_only:
        say("features and splits written; stopping (--features-only)")
        return
    if not a.report_only:
        rcs = timed("training_seconds", lambda: phase_train(cfg, out_dir, a.force))
        timed("latency_seconds", lambda: phase_latency(cfg, out_dir, rcs, feats, a.force))
        timed("noise_seconds", lambda: phase_noise(cfg, out_dir, rcs, feats, a.force))
    else:
        rcs = [run_config(s, cfg, out_dir) for s in run_specs(cfg)]
    df, summary, conv = aggregate(out_dir, rcs)
    env = dict(versions(), cpu=cpu_name(), logical_cpus=os.cpu_count(), torch_threads=cfg["compute"]["torch_threads"],
               n_jobs=cfg["compute"]["n_jobs"], platform=platform.platform(), generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    timing["total_seconds_this_invocation"] = round(time.perf_counter() - t_all, 1)
    timing["total_seconds_phases"] = round(sum(v for k, v in timing.items() if k.endswith("_seconds") and not k.startswith("total")), 1)
    timing_path.write_text(json.dumps(timing, indent=2))
    figures.make_all(out_dir, cfg, info["chance_level"])
    report_path = PROJECT_DIR / ("BENCHMARK.md" if not a.quick else "results_quick/BENCHMARK_quick.md")
    report.write_report(out_dir, cfg, info, env, timing, report_path)
    say(f"done: {len(df)} runs, report {report_path}, {timing['total_seconds_this_invocation']} s this invocation")
    best = summary[summary.family == "VQC"].sort_values("accuracy_mean", ascending=False).iloc[0]
    say(f"best circuit: {best.label} accuracy {best.accuracy_mean:.4f} +/- {0 if np.isnan(best.accuracy_sd) else best.accuracy_sd:.4f}")


if __name__ == "__main__":
    main()
