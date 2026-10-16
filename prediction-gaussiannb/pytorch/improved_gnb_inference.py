"""Live inference for the improved GaussianNB chain (the deployment side of the training notebook's cells 8-13).

    raw capture (channels, samples) in uV from the 16-channel 125 Hz board
      -> the trained channels (EXG Channel 0-14; channel 15 was excluded at training time)
      -> the loader's filtering (eeg_data._filter: mean removal, zero-phase 1-40 Hz band-pass, 60 Hz notch)
      -> 128-sample windows with a 64-sample stride (the notebook's window cutting)
      -> improved_gnb_features.extract_features                                      570 features per window
      -> ParticipantScaler.transform with the user's id (own statistics after calibration, else the global ones)
      -> the saved train-only projection (LDA, 5 dims)
      -> TorchGaussianNB log-posteriors per window
      -> causal context: sum of the log-posteriors of the last CONTEXT_WINDOWS windows
      -> the notebook's per-class thresholds (argmax of probability / threshold)      -> one command

Artifacts (written by the notebook, never modified here): torch_gaussiannb_improved_features.pth,
torch_gaussiannb_preprocessing.joblib and torch_gaussiannb_participant_scaler.joblib in this folder.

Calibration of a new user: 30-60 s of unlabelled EEG -> the same filtering / windows / features ->
ParticipantScaler.add_participant. The statistics are stored under CALIBRATION_DIR (one .npz per participant,
AVATAR_GNB_CALIBRATION_DIR overrides the location) and registered again when the chain is loaded. The training
artifacts stay untouched. A participant without statistics is standardised with the global training statistics.

Nothing here imports BrainFlow or Qt: the GUI passes the capture dict that signal_quality.acquisition produces and
the quality gate it already uses, and this module only transforms the accepted samples.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

MODEL_DIR = Path(__file__).resolve().parent
GNB_DIR = MODEL_DIR.parent
REPO_ROOT = GNB_DIR.parent
for _p in (MODEL_DIR, GNB_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eeg_data import LoadConfig, _filter  # noqa: E402  (the loader's filtering, applied unchanged)
from gaussiannb_model import GaussianNB as TorchGaussianNB  # noqa: E402
from improved_gnb_features import FEATURE_VERSION, ParticipantScaler, extract_features, feature_names  # noqa: E402
from signal_quality import QualityAssessor, QualityGatedPredictor  # noqa: E402
from signal_quality.acquisition import read_openbci  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402

WINDOW_SIZE = 128          # samples per window (1.024 s at 125 Hz), as in the notebook
WINDOW_STRIDE = 64         # max(16, WINDOW_SIZE // 2) in the notebook
CONTEXT_WINDOWS = 5        # causal context: sum of the log-posteriors of the last 5 windows (~3.1 s of EEG)
MIN_SEGMENT_ROWS = 8       # the notebook skips segments shorter than this
CALIBRATION_SECONDS = 30   # default length of a calibration recording (30-60 s recommended)
MODEL_FILE = "torch_gaussiannb_improved_features.pth"
PREP_FILE = "torch_gaussiannb_preprocessing.joblib"
SCALER_FILE = "torch_gaussiannb_participant_scaler.joblib"
DEFAULT_CALIBRATION_DIR = Path(os.environ.get("AVATAR_GNB_CALIBRATION_DIR",
                                              REPO_ROOT / "predictions_local" / "gnb_calibration"))
# TensorFlow backend: the chains trained by prediction-gaussiannb/tensorflow/gaussiannb_train.ipynb (same features,
# GaussianNBTF as the decision model, 125-sample windows with a 62-sample stride and no tail window, no thresholds)
TF_MODEL_DIR = GNB_DIR / "tensorflow"
TF_GLOBAL_FILE = "gaussiannb_tf_improved_global.pkl"
DEFAULT_CALIBRATION_DIR_TF = Path(os.environ.get("AVATAR_GNB_CALIBRATION_DIR_TF",
                                                 REPO_ROOT / "predictions_local" / "gnb_calibration_tf"))
BACKENDS = ("torch", "tensorflow")
_BRAINFLOW_NAME = re.compile(r"^EEG_(\d+)$")


def window_starts(n_rows, size=WINDOW_SIZE, stride=WINDOW_STRIDE, tail=True):
    """Window start samples of a segment with n_rows samples. tail=True: the PyTorch notebook's rule (a last window
    ending at the segment end is added when the stride does not land there); tail=False: the TensorFlow notebook's
    rule (starts 0, stride, 2*stride, ... only; nothing when the segment is shorter than one window)."""
    if not tail:
        return np.arange(0, max(0, n_rows - size + 1), stride)
    starts = list(range(0, max(1, n_rows - size + 1), stride))
    if not starts or starts[-1] != max(0, n_rows - size):
        starts.append(max(0, n_rows - size))
    return np.asarray(starts)


def cut_windows(segment, size=WINDOW_SIZE, stride=WINDOW_STRIDE, tail=True):
    """Filtered segment (samples, channels) -> windows (n, channels, size); empty when the segment is too short."""
    segment = np.asarray(segment, np.float64)
    n_rows = segment.shape[0]
    if n_rows < MIN_SEGMENT_ROWS or (not tail and n_rows < size):
        return np.empty((0, segment.shape[1], min(size, max(n_rows, 1))))
    if not np.isfinite(segment).all():
        raise ValueError("non-finite samples in the filtered segment")
    win_len = min(size, n_rows)
    return sliding_window_view(segment, win_len, axis=0)[window_starts(n_rows, size, stride, tail)]


def context_log_posteriors(logp, k=CONTEXT_WINDOWS):
    """Causal context over one recording: row t becomes the sum of rows max(0, t-k+1) .. t (notebook's _context_logp)."""
    logp = np.asarray(logp, np.float64)
    cs = np.cumsum(logp, axis=0)
    ctx = cs.copy()
    ctx[k:] = cs[k:] - cs[:-k]
    return ctx


def predict_with_thresholds(probabilities, thresholds):
    """The notebook's decision rule: argmax of probability / per-class threshold (row-wise)."""
    safe = np.clip(np.asarray(thresholds, np.float64), 1e-6, None)
    return np.argmax(np.asarray(probabilities, np.float64) / safe[np.newaxis, :], axis=1)


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _safe_name(participant):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(participant)).strip("_") or "participant"


class _LinearProjection:
    """The TensorFlow notebook's projection (a (features, k) matrix) with the Pipeline's transform() interface."""

    def __init__(self, matrix):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float64)

    def transform(self, Z):
        return np.asarray(Z, np.float64) @ self.matrix


class ImprovedGaussianNBChain:
    """Loads the notebook's artifacts and runs them on raw captures, calibration recordings and replayed files.

    backend="torch": the PyTorch notebook's artifacts (pytorch/, 128-sample windows, per-class thresholds).
    backend="tensorflow": the TensorFlow notebook's gaussiannb_tf_improved_global.pkl (tensorflow/, 125-sample windows
    with a 62-sample stride and no tail window, GaussianNBTF, argmax decision). Same features, same participant
    statistics logic; calibrations of the two backends are stored apart because their windows differ.
    """

    def __init__(self, model_dir=None, calibration_dir=None, device="cpu", backend="torch"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        self.model_dir = Path(model_dir) if model_dir is not None else (MODEL_DIR if backend == "torch" else TF_MODEL_DIR)
        default_calibration = DEFAULT_CALIBRATION_DIR if backend == "torch" else DEFAULT_CALIBRATION_DIR_TF
        self.calibration_dir = Path(calibration_dir) if calibration_dir is not None else default_calibration
        self.device = torch.device(device)
        self.window_size, self.window_stride, self.window_tail = WINDOW_SIZE, WINDOW_STRIDE, True
        if backend == "tensorflow":
            self._init_tensorflow()
        else:
            self._init_torch()
        self.calibrations = {}
        self.load_calibrations()
        self.last_decision = None

    def _init_tensorflow(self):
        from subject_gnb import tensorflow_backend
        tfb = tensorflow_backend()
        payload = tfb.load_tf_pickle(self.model_dir / TF_GLOBAL_FILE)
        chain = payload["chain"]
        if chain.get("kind") != "global" or payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{TF_GLOBAL_FILE} is not the improved global chain of feature version {FEATURE_VERSION}")
        cfg = dict(payload["load_config"])
        self.config = LoadConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()
                                    if k in ("board", "exclude_channels", "quality_filter", "bandpass_hz", "notch_hz", "filter_order")})
        self.channels = list(self.config.channels)
        self.sfreq = float(self.config.sfreq)
        if float(payload.get("sampling_rate", self.sfreq)) != self.sfreq:
            raise ValueError("sampling rate of the TensorFlow chain does not match its LoadConfig")
        self.classes = [str(c) for c in payload["label_names"]]
        self.thresholds = None                      # the TensorFlow notebook decides by argmax
        self.feature_names = feature_names(self.channels)
        self.projection = f"LDA ({int(np.asarray(chain['projection']).shape[1])} dims, TensorFlow notebook)"
        self.checkpoint_info = {}
        self.window_size, self.window_stride, self.window_tail = int(payload["window_size"]), int(payload["window_stride"]), False
        self.training_scaler = chain["scaler"]
        if not isinstance(self.training_scaler, ParticipantScaler) or self.training_scaler.n_features_in_ != len(self.feature_names):
            raise ValueError("participant scaler of the TensorFlow chain does not match the feature set")
        self._copy_scaler()
        self.prep = _LinearProjection(chain["projection"])
        if self.prep.matrix.shape[0] != len(self.feature_names):
            raise ValueError("projection of the TensorFlow chain does not match the feature set")
        self.model = chain["model"]                 # GaussianNBTF (predict_log_proba = unnormalised log-posteriors)
        if int(self.model.n_classes_) != len(self.classes) or int(self.model.n_features_) != self.prep.matrix.shape[1]:
            raise ValueError("GaussianNBTF shape does not match the projection / classes")

    def _copy_scaler(self):
        self.scaler = ParticipantScaler(min_windows=self.training_scaler.min_windows, eps=self.training_scaler.eps)
        self.scaler.global_mean_ = self.training_scaler.global_mean_.copy()
        self.scaler.global_std_ = self.training_scaler.global_std_.copy()
        self.scaler.stats_ = {k: (m.copy(), s.copy()) for k, (m, s) in self.training_scaler.stats_.items()}
        self.scaler.n_features_in_ = self.training_scaler.n_features_in_

    def _init_torch(self):
        ck = torch.load(self.model_dir / MODEL_FILE, map_location="cpu", weights_only=False)
        cfg = dict(ck["load_config"])
        self.config = LoadConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()
                                    if k in ("board", "exclude_channels", "quality_filter", "bandpass_hz", "notch_hz", "filter_order")})
        self.channels = list(self.config.channels)
        self.sfreq = float(self.config.sfreq)
        self.classes = [str(c) for c in ck["label_encoder_classes"]]
        self.thresholds = np.asarray(ck["class_thresholds"], np.float64)
        self.feature_names = list(ck["feature_names"])
        self.projection = ck.get("projection")
        self.checkpoint_info = {k: ck[k] for k in ("train_accuracy", "test_accuracy", "test_f1_macro",
                                                   "test_balanced_accuracy", "feature_set") if k in ck}
        if FEATURE_VERSION not in str(ck.get("feature_set", "")):
            raise ValueError(f"checkpoint feature set {ck.get('feature_set')!r} is not {FEATURE_VERSION}")
        if self.feature_names != feature_names(self.channels):
            raise ValueError("checkpoint feature names do not match improved_gnb_features.feature_names(channels)")
        if list(cfg.get("channels", self.channels)) != self.channels:
            raise ValueError("checkpoint channel list does not match its LoadConfig")
        if self.thresholds.shape != (len(self.classes),) or ck["num_classes"] != len(self.classes):
            raise ValueError("class thresholds / classes mismatch in the checkpoint")
        self.prep = joblib.load(self.model_dir / PREP_FILE)
        self.training_scaler = joblib.load(self.model_dir / SCALER_FILE)
        if not isinstance(self.training_scaler, ParticipantScaler) or self.training_scaler.n_features_in_ != len(self.feature_names):
            raise ValueError("participant scaler does not match the feature set")
        self._copy_scaler()
        n_proj = int(self.prep.transform(np.zeros((1, len(self.feature_names)))).shape[1])
        if n_proj != int(ck["num_features"]):
            raise ValueError(f"projection gives {n_proj} dims, checkpoint expects {ck['num_features']}")
        self.model = TorchGaussianNB(num_features=int(ck["num_features"]), num_classes=len(self.classes))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()

    # ----------------------------------------------------------------------------------------------- capture geometry
    @property
    def capture_samples(self):
        """Samples needed for a full CONTEXT_WINDOWS-window context in one capture."""
        return self.window_size + (CONTEXT_WINDOWS - 1) * self.window_stride

    @property
    def capture_seconds(self):
        return self.capture_samples / self.sfreq

    # ----------------------------------------------------------------------------------------------- raw -> features
    def select_channels(self, raw, channels):
        """Rows of the trained channels, in training order, from a capture with named channels.

        Accepts the OpenBCI GUI names ("EXG Channel i", any order, picked by name) or BrainFlow names
        ("EEG_<row>") of a full 16-channel board capture in board order, where the i-th EEG row is EXG Channel i.
        """
        raw = np.asarray(raw, np.float64)
        channels = [str(c) for c in channels]
        if raw.ndim != 2 or raw.shape[0] != len(channels):
            raise ValueError("raw must be (channels, samples) with one name per row")
        if all(c in channels for c in self.channels):
            return raw[[channels.index(c) for c in self.channels]]
        n_board = len([f"EXG Channel {i}" for i in range(16)])
        if len(channels) == n_board and all(_BRAINFLOW_NAME.match(c) for c in channels):
            rows = [int(_BRAINFLOW_NAME.match(c)[1]) for c in channels]
            if rows != sorted(rows):
                raise ValueError("BrainFlow EEG rows must be in board order")
            return raw[[int(c.split()[-1]) for c in self.channels]]
        raise ValueError(f"cannot map capture channels {channels} to the trained channels {self.channels}")

    def filter_segment(self, raw_selected):
        """Selected raw rows (channels, samples) -> the loader's filtered segment (samples, channels) float32."""
        return _filter(np.asarray(raw_selected, np.float64).T, self.config)

    def features_from_segment(self, filtered):
        """One filtered segment (samples, channels) -> features (n_windows, 570) float32."""
        W = cut_windows(filtered, self.window_size, self.window_stride, self.window_tail)
        if len(W) == 0:
            return np.empty((0, len(self.feature_names)), np.float32)
        return extract_features(W, fs=self.sfreq)

    def features_from_raw(self, raw, channels, sfreq):
        """Raw capture rows (channels, samples) in uV -> features of its windows (filtered as one segment)."""
        if float(sfreq) != self.sfreq:
            raise ValueError(f"capture is {sfreq} Hz, model expects {self.sfreq} Hz")
        return self.features_from_segment(self.filter_segment(self.select_channels(raw, channels)))

    # ----------------------------------------------------------------------------------------------- features -> posteriors
    def _projected(self, X, participant):
        X = np.asarray(X, np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected (n, {len(self.feature_names)}) features")
        Z = self.scaler.transform(X, np.array([str(participant)] * len(X)))
        return np.ascontiguousarray(self.prep.transform(Z).astype(np.float32))

    def _joint_log_likelihood(self, P):
        """Unnormalised log-posteriors (n, classes) of projected windows, by the backend's GaussianNB."""
        if self.backend == "tensorflow":
            return np.asarray(self.model.predict_log_proba(P), np.float64)
        with torch.no_grad():
            return self.model._joint_log_likelihood(torch.from_numpy(P).to(self.device)).cpu().numpy().astype(np.float64)

    def predict_proba(self, X, participant):
        """Per-window class probabilities exactly as the notebook evaluates them (softmax of the NB log-posteriors)."""
        if len(X) == 0:
            return np.empty((0, len(self.classes)))
        P = self._projected(X, participant)
        if self.backend == "tensorflow":
            return np.exp(_log_softmax(self._joint_log_likelihood(P)))
        with torch.no_grad():
            return self.model.predict_proba(torch.from_numpy(P).to(self.device)).cpu().numpy()

    def log_posteriors(self, X, participant):
        """Per-window normalised log-posteriors (n, classes), the quantity summed by the context."""
        if len(X) == 0:
            return np.empty((0, len(self.classes)))
        return _log_softmax(self._joint_log_likelihood(self._projected(X, participant)))

    def decide(self, logp, use_thresholds=True):
        """Command from the per-window log-posteriors of one capture, time-ordered: context sum of the last
        CONTEXT_WINDOWS windows, normalised, then the notebook's threshold rule (k = 1 reduces to the notebook's
        window-level decision)."""
        logp = np.asarray(logp, np.float64)
        if logp.ndim != 2 or len(logp) == 0:
            raise ValueError("need at least one window")
        ctx = context_log_posteriors(logp)[-1]
        probs = np.exp(_log_softmax(ctx))
        thresholds = self.thresholds if use_thresholds else None      # the TensorFlow chain has none: argmax
        index = int(predict_with_thresholds(probs[None, :], thresholds)[0]) if thresholds is not None else int(np.argmax(ctx))
        window_votes = predict_with_thresholds(np.exp(logp), thresholds) if thresholds is not None else logp.argmax(axis=1)
        return dict(label=self.classes[index], index=index, confidence=float(probs[index]),
                    probabilities={c: float(p) for c, p in zip(self.classes, probs)},
                    context_windows=int(min(CONTEXT_WINDOWS, len(logp))), windows=int(len(logp)),
                    window_labels=[self.classes[i] for i in window_votes])

    def predict_windows(self, X, participant, use_thresholds=True):
        """Features of one capture's windows -> decision dict (also keeps the per-window log-posteriors)."""
        logp = self.log_posteriors(X, participant)
        decision = self.decide(logp, use_thresholds)
        decision.update(participant=str(participant), participant_status=self.participant_status(participant),
                        log_posteriors=logp)
        self.last_decision = decision
        return decision

    def predict_raw(self, raw, channels, sfreq, participant, use_thresholds=True):
        """Raw capture rows -> decision dict (the whole live path without the quality gate)."""
        X = self.features_from_raw(raw, channels, sfreq)
        if len(X) == 0:
            raise ValueError(f"capture too short for one {self.window_size}-sample window")
        return self.predict_windows(X, participant, use_thresholds)

    def predict_capture(self, capture, participant, gate=None):
        """Quality-gated live prediction on a signal_quality capture dict (what the GUI passes).

        The gate assesses the raw samples first; the chain only runs on an accepted capture, and the result is the
        gate's decision dict (prediction_label, confidence, quality, expiry). The gate maps 'landing' to 'land'.
        """
        gate = gate or QualityGatedPredictor(QualityAssessor(load_default_config()))
        self.last_decision = None

        def callback(samples):
            d = self.predict_raw(samples, capture["channels"], capture["sfreq"], participant)
            return dict(prediction_label=d["label"], confidence=d["confidence"])

        return gate.predict(predict=callback, **capture)

    # ----------------------------------------------------------------------------------------------- calibration
    def participant_status(self, participant):
        key = str(participant)
        if key in self.calibrations:
            return "calibrated"
        if key in self.training_scaler.stats_:
            return "training participant"
        return "global fallback"

    def segments_from_capture(self, capture, assessor=None):
        """The loader's segmentation of a longer capture: 2 s blocks (start-up block dropped, failing blocks dropped,
        runs split where packets were lost), each run filtered on its own -> list of (samples, channels) segments."""
        assessor = assessor or QualityAssessor(load_default_config())
        if float(capture["sfreq"]) != self.sfreq:
            raise ValueError(f"capture is {capture['sfreq']} Hz, model expects {self.sfreq} Hz")
        raw = self.select_channels(capture["raw"], capture["channels"])
        counters, timestamps = capture.get("counters"), capture.get("timestamps")
        fs = self.sfreq
        block = int(round(2.0 * fs))
        n_blocks = raw.shape[1] // block
        usable = np.zeros(n_blocks, dtype=bool)
        reasons = []
        for b in range(n_blocks):
            a, e = b * block, (b + 1) * block
            if b == 0:
                reasons.append(["startup_block"])
                continue
            if not self.config.quality_filter:
                usable[b] = True
                reasons.append([])
                continue
            report = assessor.assess(raw[:, a:e], sfreq=fs, channels=self.channels, units=capture.get("units", "uV"),
                                     counters=None if counters is None else counters[a:e],
                                     timestamps=None if timestamps is None else timestamps[a:e],
                                     counter_step=capture.get("counter_step"),
                                     timestamp_kind=capture.get("timestamp_kind", "arrival"))
            usable[b] = bool(report["usable"])
            reasons.append(report["reasons"])
        segments, run = [], []
        max_gap = assessor.config.max_arrival_gap_seconds
        for b in range(n_blocks):
            if usable[b] and run and counters is not None and timestamps is not None:
                a = b * block
                step = (counters[a] - counters[a - 1]) % 256
                gap = timestamps[a] - timestamps[a - 1]
                if step != capture.get("counter_step") or not 0 <= gap <= max_gap:
                    segments.append(run)
                    run = []
            if usable[b]:
                run.append(b)
            elif run:
                segments.append(run)
                run = []
        if run:
            segments.append(run)
        data = [self.filter_segment(raw[:, r[0] * block:(r[-1] + 1) * block]) for r in segments]
        return data, dict(n_blocks=int(n_blocks), n_blocks_kept=int(usable.sum()), block_reasons=reasons)

    def calibration_features(self, capture, assessor=None):
        """Longer unlabelled capture -> (features of all its windows, loader-style block report)."""
        segments, report = self.segments_from_capture(capture, assessor)
        feats = [self.features_from_segment(s) for s in segments]
        X = np.concatenate(feats) if feats else np.empty((0, len(self.feature_names)), np.float32)
        report["n_windows"] = int(len(X))
        return X, report

    def calibrate(self, participant, X_calibration, source="capture"):
        """Register a participant from the features of their unlabelled calibration windows and store them."""
        key = str(participant)
        if not key.strip():
            raise ValueError("participant id must not be empty")
        X = np.asarray(X_calibration, np.float64)
        self.scaler.add_participant(X, key)
        mean, std = self.scaler.stats_[key]
        entry = dict(participant=key, n_windows=int(len(X)), feature_version=FEATURE_VERSION, source=str(source),
                     created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.calibrations[key] = entry
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.calibration_dir / f"{_safe_name(key)}.npz", mean=mean, std=std, meta=json.dumps(entry))
        return entry

    def calibrate_from_capture(self, participant, capture, assessor=None):
        X, report = self.calibration_features(capture, assessor)
        if len(X) < self.scaler.min_windows:
            raise ValueError(f"only {len(X)} usable calibration windows ({report['n_blocks_kept']}/{report['n_blocks']} "
                             f"2 s blocks passed quality); record again with the headset settled")
        entry = self.calibrate(participant, X, source=f"capture:{capture.get('stream_id', 'live')}")
        entry.update(n_blocks=report["n_blocks"], n_blocks_kept=report["n_blocks_kept"])
        return entry

    def calibrate_from_recording(self, participant, path, assessor=None):
        """Calibration from a named OpenBCI GUI recording (.txt) of the user, e.g. recorded with the OpenBCI GUI."""
        capture = read_openbci(path)
        X, report = self.calibration_features(capture, assessor)
        if len(X) < self.scaler.min_windows:
            raise ValueError(f"only {len(X)} usable calibration windows in {Path(path).name}")
        entry = self.calibrate(participant, X, source=f"recording:{Path(path).name}")
        entry.update(n_blocks=report["n_blocks"], n_blocks_kept=report["n_blocks_kept"])
        return entry

    def load_calibrations(self):
        """Register every stored calibration whose feature version matches the model (training stats are not touched)."""
        self.calibrations = {}
        for f in sorted(self.calibration_dir.glob("*.npz")) if self.calibration_dir.is_dir() else []:
            try:
                z = np.load(f, allow_pickle=False)
                entry = json.loads(str(z["meta"]))
                if entry.get("feature_version") != FEATURE_VERSION or z["mean"].shape != (len(self.feature_names),):
                    continue
                self.scaler.stats_[entry["participant"]] = (z["mean"].astype(np.float64), z["std"].astype(np.float64))
                self.calibrations[entry["participant"]] = entry
            except (OSError, ValueError, KeyError) as error:
                print(f"[GaussianNB] ignoring calibration file {f.name}: {error}")
        return self.calibrations

    def forget_calibration(self, participant):
        key = str(participant)
        self.calibrations.pop(key, None)
        f = self.calibration_dir / f"{_safe_name(key)}.npz"
        if f.exists():
            f.unlink()
        if key in self.training_scaler.stats_:
            m, s = self.training_scaler.stats_[key]
            self.scaler.stats_[key] = (m.copy(), s.copy())
        else:
            self.scaler.stats_.pop(key, None)

    # ----------------------------------------------------------------------------------------------- offline replay
    def replay_recording(self, path, participant=None, assessor=None):
        """Run a recording through the loader-equivalent path: one decision per kept segment plus per-window output."""
        from improved_gnb_features import participant_of
        capture = read_openbci(path)
        participant = participant_of(path) if participant is None else participant
        segments, report = self.segments_from_capture(capture, assessor)
        out = []
        for seg in segments:
            X = self.features_from_segment(seg)
            if len(X):
                d = self.predict_windows(X, participant)
                out.append(dict(samples=int(seg.shape[0]), windows=int(len(X)), label=d["label"],
                                confidence=d["confidence"], probabilities=d["probabilities"],
                                window_labels=d["window_labels"]))
        return dict(path=str(path), participant=participant, participant_status=self.participant_status(participant),
                    segments=out, **report)


def main(argv=None):
    p = argparse.ArgumentParser(description="Replay an OpenBCI recording through the improved GaussianNB live path.")
    p.add_argument("--file", required=True, help="named OpenBCI GUI recording (.txt), 16 channels at 125 Hz")
    p.add_argument("--participant", default=None, help="participant id (default: <group>/<person> from the path)")
    p.add_argument("--calibrate-from", default=None, help="recording used to calibrate the participant first")
    p.add_argument("--calibration-dir", default=None, help="where calibrations are stored (default: %s)" % DEFAULT_CALIBRATION_DIR)
    p.add_argument("--backend", default="torch", choices=BACKENDS, help="GaussianNB implementation: PyTorch (default) or TensorFlow")
    a = p.parse_args(argv)
    chain = ImprovedGaussianNBChain(calibration_dir=a.calibration_dir, backend=a.backend)
    if a.calibrate_from:
        if not a.participant:
            p.error("--calibrate-from needs --participant")
        print(json.dumps(chain.calibrate_from_recording(a.participant, a.calibrate_from), indent=2))
    print(json.dumps(chain.replay_recording(a.file, a.participant), indent=2))


if __name__ == "__main__":
    main()
