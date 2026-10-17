"""The improved GaussianNB live path reproduces the notebook's windows, features and class probabilities.

Replays a Data_clean recording through prediction-gaussiannb/pytorch/improved_gnb_inference.py and compares with
  * the notebook's improved-feature cache (predictions_local/feature_cache/improved_windows_*.npz, written by cell 5:
    the windows it trained and evaluated on, in time order per recording), and
  * the notebook's evaluation of those windows (cells 8, 11: ParticipantScaler -> saved projection -> TorchGaussianNB
    predict_proba, per-class thresholds), recomputed here from the artifacts the notebook saved (cell 13).
Skipped when Data_clean or the cache is not present on this machine.
"""
import collections
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

GNB_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = GNB_DIR / "pytorch"
REPO_ROOT = GNB_DIR.parent
for _p in (MODEL_DIR, GNB_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import improved_gnb_inference as I  # noqa: E402
from gaussiannb_model import GaussianNB as TorchGaussianNB  # noqa: E402
from improved_gnb_features import participant_of  # noqa: E402
from signal_quality import QualityAssessor, QualityGatedPredictor, may_execute  # noqa: E402
from signal_quality.acquisition import read_openbci  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402

CACHE_DIR = REPO_ROOT / "predictions_local" / "feature_cache"
BLOCK = 250  # one 2 s quality block at 125 Hz; the loader drops block 0 (start-up transient)


# ----------------------------------------------------------------------------------------------- the notebook's code
def notebook_window_starts(n_rows, window_size=128, stride=64):
    """Cell 5, verbatim."""
    starts = list(range(0, max(1, n_rows - window_size + 1), stride))
    if not starts or starts[-1] != max(0, n_rows - window_size):
        starts.append(max(0, n_rows - window_size))
    return np.asarray(starts)


def notebook_predict_with_thresholds(probabilities, thresholds):
    """Cell 11, verbatim."""
    safe_thresholds = np.clip(thresholds, 1e-6, None)
    adjusted = probabilities / safe_thresholds[np.newaxis, :]
    return np.argmax(adjusted, axis=1)


def notebook_context_logp(logp, files, k=5):
    """Cell 15, verbatim (causal 5-window context within each recording)."""
    out = np.empty_like(logp)
    for f in np.unique(files):
        m = np.where(files == f)[0]
        cs = np.cumsum(logp[m], axis=0); ctx = cs.copy(); ctx[k:] = cs[k:] - cs[:-k]; out[m] = ctx
    return out


def notebook_probabilities(X_cached, participant):
    """Cells 8 + 11 on cached features with the saved artifacts, loaded independently of the inference module."""
    ck = torch.load(MODEL_DIR / I.MODEL_FILE, map_location="cpu", weights_only=False)
    prep = joblib.load(MODEL_DIR / I.PREP_FILE)
    scaler = joblib.load(MODEL_DIR / I.SCALER_FILE)
    model = TorchGaussianNB(num_features=ck["num_features"], num_classes=ck["num_classes"])
    model.load_state_dict(ck["model_state_dict"])
    Z = scaler.transform(X_cached, np.array([participant] * len(X_cached)))
    with torch.no_grad():
        probs = model.predict_proba(torch.from_numpy(prep.transform(Z).astype(np.float32))).numpy()
    return probs, np.asarray(ck["class_thresholds"]), [str(c) for c in ck["label_encoder_classes"]]


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    for name in (I.MODEL_FILE, I.PREP_FILE, I.SCALER_FILE):
        if not (MODEL_DIR / name).exists():
            pytest.skip(f"{name} not present")
    return I.ImprovedGaussianNBChain(calibration_dir=tmp_path_factory.mktemp("gnb_calibration"))


@pytest.fixture(scope="module")
def recording():
    """One Data_clean recording with >= 10 cached windows: (path, cached features, cached labels)."""
    caches = sorted(CACHE_DIR.glob("improved_windows_*.npz"), key=lambda f: f.stat().st_mtime)
    if not caches:
        pytest.skip("notebook feature cache not present")
    z = np.load(caches[-1], allow_pickle=True)
    groups = z["groups"].astype(str)
    counts = collections.Counter(groups)
    for path in sorted(counts):
        if counts[path] >= 10 and Path(path).exists():
            rows = np.where(groups == path)[0]
            assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)), "cached windows of a recording are contiguous"
            return path, z["X"][rows].astype(np.float32), z["y"][rows].astype(int)
    pytest.skip("no cached Data_clean recording present on this machine")


def chain_assessor(chain):
    """The app's quality policy (signal_quality/default_config.json), as the loader uses it."""
    return QualityAssessor(load_default_config())


def capture_slice(cap, start, n):
    out = dict(cap)
    for key in ("raw", "counters", "timestamps"):
        out[key] = cap[key][:, start:start + n] if key == "raw" else cap[key][start:start + n]
    out.update(captured_monotonic=time.monotonic(), stream_id=f"replay:{start}")
    return out


# ----------------------------------------------------------------------------------------------- tests
def test_window_cutting_is_the_notebooks_rule():
    for n in (8, 100, 128, 129, 191, 192, 193, 384, 500, 1000):
        assert np.array_equal(I.window_starts(n), notebook_window_starts(n)), n
    seg = np.arange(1000 * 15, dtype=float).reshape(1000, 15)
    W = I.cut_windows(seg)
    assert W.shape == (len(notebook_window_starts(1000)), 15, 128)
    assert np.array_equal(W[3, 2], seg[192:320, 2])        # window 3 of channel 2 = samples 192..319
    assert np.array_equal(W[-1, 0], seg[872:1000, 0])      # last window ends at the end of the segment
    assert I.cut_windows(seg[:7]).shape[0] == 0            # the notebook skips segments shorter than 8 samples


def test_replay_reproduces_notebook_windows_features_and_probabilities(chain, recording):
    path, X_cached, y_cached = recording
    participant = participant_of(path)
    assert participant in chain.training_scaler.stats_   # the notebook used this participant's train statistics

    # Loader-equivalent path on the raw file: 2 s quality blocks -> segments -> filtering -> windows -> features
    segments, report = chain.segments_from_capture(read_openbci(path), chain_assessor(chain))
    X_live = np.concatenate([chain.features_from_segment(s) for s in segments])
    assert report["block_reasons"][0] == ["startup_block"]
    assert X_live.shape == X_cached.shape == (len(y_cached), 570)
    np.testing.assert_allclose(X_live, X_cached, rtol=1e-5, atol=1e-5)

    # Same class probabilities as the notebook's evaluation of these windows
    probs_nb, thresholds, classes = notebook_probabilities(X_cached, participant)
    probs_live = chain.predict_proba(X_live, participant)
    np.testing.assert_allclose(probs_live, probs_nb, atol=1e-6)
    assert chain.classes == classes and np.allclose(chain.thresholds, thresholds)

    # Same per-window decisions (thresholds) and the same causal 5-window context as the notebook
    decision = chain.predict_windows(X_live, participant)
    assert decision["window_labels"] == [classes[i] for i in notebook_predict_with_thresholds(probs_nb, thresholds)]
    logp = decision["log_posteriors"]
    np.testing.assert_allclose(np.exp(logp), probs_nb, atol=1e-6)
    files = np.array([path] * len(logp))
    ctx_nb = notebook_context_logp(logp, files)
    np.testing.assert_allclose(I.context_log_posteriors(logp), ctx_nb, atol=1e-9)
    assert decision["context_windows"] == 5 and decision["windows"] == len(y_cached)
    ctx_probs = np.exp(ctx_nb[-1] - ctx_nb[-1].max()); ctx_probs /= ctx_probs.sum()
    assert decision["index"] == int(notebook_predict_with_thresholds(ctx_probs[None, :], thresholds)[0])
    assert decision["label"] == classes[decision["index"]]
    assert abs(decision["confidence"] - ctx_probs[decision["index"]]) < 1e-9
    assert abs(sum(decision["probabilities"].values()) - 1) < 1e-9


def test_live_capture_path_equals_loader_path_on_the_same_samples(chain, recording):
    """features_from_raw (what the gate callback runs on an accepted capture) filters the capture as one segment:
    on the samples of the loader's kept segment it gives the loader's windows exactly."""
    path, X_cached, _ = recording
    cap = read_openbci(path)
    segments, report = chain.segments_from_capture(cap, chain_assessor(chain))
    assert len(segments) == 1, "test recording expected to have one kept run of 2 s blocks"
    kept = [b for b, r in enumerate(report["block_reasons"]) if not r]
    start, stop = kept[0] * BLOCK, (kept[-1] + 1) * BLOCK
    X_live = chain.features_from_raw(cap["raw"][:, start:stop], cap["channels"], cap["sfreq"])
    np.testing.assert_allclose(X_live, X_cached, rtol=1e-5, atol=1e-5)


def test_quality_gated_capture_gives_the_same_command(chain, recording):
    path, _, _ = recording
    participant = participant_of(path)
    cap = read_openbci(path)
    gate = QualityGatedPredictor(QualityAssessor(load_default_config()))
    n = chain.capture_samples
    assert n == 128 + 4 * 64 and abs(chain.capture_seconds - n / 125) < 1e-12
    live = capture_slice(cap, BLOCK, n)                 # after the start-up block, inside the kept blocks
    result = chain.predict_capture(live, participant, gate)
    assert result["status"] == "accepted", result.get("withholding_reason")
    assert may_execute(result)
    direct = chain.predict_raw(live["raw"], live["channels"], live["sfreq"], participant)
    expected = "land" if direct["label"] == "landing" else direct["label"]
    assert result["prediction_label"] == expected
    assert result["confidence"] == direct["confidence"]
    assert chain.last_decision["windows"] == 5 and chain.last_decision["context_windows"] == 5
    assert chain.last_decision["participant_status"] == "training participant"

    # Synthetic sources and captures shorter than the quality minimum never produce a command
    synthetic = dict(capture_slice(cap, BLOCK, n), source_kind="synthetic")
    assert chain.predict_capture(synthetic, participant, gate)["status"] == "withheld"
    assert chain.last_decision is None
    short = capture_slice(cap, BLOCK, 100)
    assert chain.predict_capture(short, participant, gate)["status"] == "withheld"


def test_channel_mapping_from_brainflow_and_openbci_names(chain, recording):
    path, _, _ = recording
    cap = read_openbci(path)
    raw, names = cap["raw"], cap["channels"]
    by_name = chain.select_channels(raw, names)
    assert by_name.shape == (15, raw.shape[1]) and np.array_equal(by_name, raw[:15])
    # BrainFlow capture of the 16-channel board: rows EEG_1..EEG_16 in board order = EXG Channel 0..15
    brainflow = chain.select_channels(raw, [f"EEG_{i}" for i in range(1, 17)])
    assert np.array_equal(brainflow, by_name)
    # OpenBCI names in another order are picked by name
    order = np.random.default_rng(0).permutation(16)
    assert np.array_equal(chain.select_channels(raw[order], [names[i] for i in order]), by_name)
    with pytest.raises(ValueError):
        chain.select_channels(raw[:8], [f"EEG_{i}" for i in range(1, 9)])
    with pytest.raises(ValueError):
        chain.select_channels(raw, [f"EEG_{i}" for i in range(16, 0, -1)])
    with pytest.raises(ValueError):
        chain.features_from_raw(raw, names, sfreq=250.0)


def test_calibration_registers_a_new_user_and_unknown_users_fall_back(chain, recording, tmp_path):
    path, X_cached, _ = recording
    X = X_cached.astype(np.float64)
    assert chain.participant_status("someone-new") == "global fallback"
    Z = chain.scaler.transform(X, np.array(["someone-new"] * len(X)))
    np.testing.assert_allclose(Z, (X - chain.training_scaler.global_mean_) / chain.training_scaler.global_std_)

    entry = chain.calibrate_from_recording("someone-new", path)
    assert entry["n_windows"] == len(X) and entry["n_blocks_kept"] >= 1
    assert chain.participant_status("someone-new") == "calibrated"
    Z = chain.scaler.transform(X, np.array(["someone-new"] * len(X)))
    np.testing.assert_allclose(Z, (X - X.mean(0)) / (X.std(0) + chain.scaler.eps))
    assert chain.predict_windows(X, "someone-new")["participant_status"] == "calibrated"

    # Stored next to the app data, reloaded by a fresh chain; the training artifacts are not touched
    stored = sorted(chain.calibration_dir.glob("*.npz"))
    assert [f.name for f in stored] == ["someone-new.npz"]
    again = I.ImprovedGaussianNBChain(calibration_dir=chain.calibration_dir)
    assert again.participant_status("someone-new") == "calibrated"
    np.testing.assert_allclose(again.scaler.stats_["someone-new"][0], X.mean(0))
    assert "someone-new" not in joblib.load(MODEL_DIR / I.SCALER_FILE).stats_

    chain.forget_calibration("someone-new")
    assert chain.participant_status("someone-new") == "global fallback" and not stored[0].exists()
    with pytest.raises(ValueError):
        chain.calibrate("too-short", X[:1])
    with pytest.raises(ValueError):
        chain.calibrate("", X)

    # Calibration from a live-style capture dict goes through the loader's block logic
    cap = read_openbci(path)
    entry = chain.calibrate_from_capture("from-capture", dict(cap, stream_id="brainflow:2:test"))
    assert entry["n_windows"] == len(X) and entry["source"].startswith("capture:brainflow")
    chain.forget_calibration("from-capture")


def test_cli_replay(chain, recording, tmp_path, capsys):
    path, _, _ = recording
    I.main(["--file", path, "--calibration-dir", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert out["participant"] == participant_of(path) and out["segments"][0]["windows"] >= 10
    assert out["segments"][0]["label"] in chain.classes
