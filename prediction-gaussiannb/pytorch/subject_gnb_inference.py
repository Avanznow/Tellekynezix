"""Subject-specific GaussianNB: the live path and the labelled training flow (deployment side of the notebook's
section "Subject-specific GaussianNB", cells 17-20).

Live path, on a raw capture (channels, samples) in uV from the 16-channel 125 Hz board:

    the trained channels (EXG Channel 0-14; BrainFlow rows EEG_1..16 map by position)
      -> three aligned views of the same samples, each filtered as one segment:
           filt  : the loader's filtering (eeg_data._filter: mean removal, zero-phase 1-40 Hz band-pass, 60 Hz notch)
           raw   : mean removal only (the loader with bandpass_hz=None, notch_hz=None)
           gamma : improved_gnb_features.bandpass_segment on the raw view (30-45 Hz band-pass + 60 Hz notch)
      -> the same 128-sample / 64-stride windows cut from every view (the notebook's rule)
      -> improved_gnb_features.extract_subject_features                      435 features per window
      -> z-score with the USER's own feature statistics (ParticipantScaler statistics of their labelled windows)
      -> the USER's SubjectSpecificGaussianNB model (LDA projection <= 5 dims + TorchGaussianNB fitted on their windows)
      -> causal context: sum of the log-posteriors of the last 5 windows -> argmax = the context decision
      -> vote-agreement gate: the per-window decisions of the last 5 windows must agree with the context decision in
         at least 4 of 5 cases; otherwise the decision is "uncertain" and the quality gate withholds it.

Training flow (the guided session in the GUI, or offline from labelled recordings): every cue of the interleaved
schedule yields one capture of the cue period; it is cut into 2 s quality blocks exactly like the loader (first block
and failing blocks dropped, runs split on packet loss), every kept run gives the three views, windows and features,
and the windows are labelled with the cue. All of a user's labelled windows (across sessions) fit their statistics and
their model, stored under USERS_DIR/<id>/ (AVATAR_GNB_USERS_DIR overrides):

    training_windows.npz   X (n, 435) float32, y, cue_id, session_id, recorded (unix time)      every labelled window
    scaler.npz             mean, std of the 435 features over the user's windows               ParticipantScaler statistics
    model.npz              projection, classes, GaussianNB priors / means / variances            the user's model
    meta.json              sessions, windows and cues per command, learning-curve expectation, cue-grouped CV accuracy
    sessions/<id>/cues.npz + cues.json   the raw EEG of every cue (re-training with another feature version, audits)

The notebook's artifacts torch_gaussiannb_subject_models.joblib / torch_gaussiannb_subject_scaler.joblib are loaded,
when present, for the 17 Data_clean training participants and as the global fallback; they are never modified.
Nothing here imports BrainFlow or Qt at module level: BrainFlowStream imports brainflow when it starts.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent
GNB_DIR = MODEL_DIR.parent
REPO_ROOT = GNB_DIR.parent
for _p in (MODEL_DIR, GNB_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eeg_data import LABELS, LoadConfig, _filter  # noqa: E402
from gaussiannb_model import GaussianNB as TorchGaussianNB  # noqa: E402
from improved_gnb_features import (SUBJECT_FEATURE_VERSION, ParticipantScaler, bandpass_segment,  # noqa: E402
                                   extract_subject_features, participant_of, subject_feature_names)
from improved_gnb_inference import (CONTEXT_WINDOWS, MODEL_FILE, WINDOW_SIZE, WINDOW_STRIDE,  # noqa: E402
                                    _log_softmax, _safe_name, context_log_posteriors, cut_windows)
from signal_quality import QualityAssessor, QualityGatedPredictor  # noqa: E402
from signal_quality.acquisition import brainflow_capture, read_openbci  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402
from signal_quality.salvage import folder_label  # noqa: E402
from subject_gnb import ProjectedTorchGaussianNB, SubjectSpecificGaussianNB, tensorflow_backend  # noqa: E402

COMMANDS = list(LABELS)                       # alphabetical = the notebook's LabelEncoder order (0..5)
assert COMMANDS == sorted(COMMANDS) and len(COMMANDS) == 6
SUBJECT_MODELS_FILE = "torch_gaussiannb_subject_models.joblib"
SUBJECT_SCALER_FILE = "torch_gaussiannb_subject_scaler.joblib"
DEFAULT_USERS_DIR = Path(os.environ.get("AVATAR_GNB_USERS_DIR", REPO_ROOT / "predictions_local" / "gnb_users"))
# TensorFlow backend: GaussianNBTF as the decision model, the TensorFlow notebook's window geometry (125 samples,
# stride 62, no tail window) and its artifact; users trained with it are stored apart (their features differ).
BACKENDS = ("torch", "tensorflow")
TF_MODEL_DIR = GNB_DIR / "tensorflow"
TF_SUBJECT_FILE = "gaussiannb_tf_subject_specific.pkl"
TF_WINDOW_SIZE, TF_WINDOW_STRIDE = 125, 62
DEFAULT_USERS_DIR_TF = Path(os.environ.get("AVATAR_GNB_USERS_DIR_TF", REPO_ROOT / "predictions_local" / "gnb_users_tf"))
MIN_WINDOWS = 60               # SUBJ_MIN_WINDOWS in the notebook: own model only with this many labelled windows
N_COMPONENTS, SHRINKAGE = 5, "auto"
VOTE_WINDOWS = 5               # the vote-agreement gate looks at the last 5 window decisions ...
MIN_AGREEMENT = 4              # ... and acts only when at least 4 agree with the context decision
BLOCK_SECONDS = 2.0            # the loader's quality block
# Guided session defaults: 6 rounds x 6 commands x 20 s cues with 5 s rest gaps = 15 min (+ a 3 s settle)
DEFAULT_ROUNDS, DEFAULT_CUE_SECONDS, DEFAULT_REST_SECONDS, SETTLE_SECONDS = 6, 20.0, 5.0, 3.0
# Learning curve of the notebook (cell 20, gaussiannb_subject_learning_curve.csv): window accuracy of the
# subject-specific chain when each user contributes the k earliest recordings per command (one Data_clean recording
# is ~10 s of EEG, about 15 kept windows). The GUI counts one cue as one recording.
LEARNING_CURVE = ((1, 0.537), (2, 0.656), (3, 0.693), (4, 0.745), (6, 0.813), (8, 0.854))
CHANCE = 1.0 / len(COMMANDS)
_BRAINFLOW_NAME = re.compile(r"^EEG_(\d+)$")


# ====================================================================================================== pure helpers
def raw_config(config):
    """The loader configuration of the raw view: same board / channels / quality rule, no band-pass, no notch."""
    return LoadConfig(board=config.board, exclude_channels=config.exclude_channels,
                      quality_filter=config.quality_filter, bandpass_hz=None, notch_hz=None,
                      filter_order=config.filter_order)


def select_channels(raw, channels, trained_channels):
    """Rows of the trained channels, in training order: OpenBCI names picked by name, or a full 16-row BrainFlow
    capture (EEG_<row> in board order) where the i-th EEG row is EXG Channel i."""
    raw = np.asarray(raw, np.float64)
    channels = [str(c) for c in channels]
    if raw.ndim != 2 or raw.shape[0] != len(channels):
        raise ValueError("raw must be (channels, samples) with one name per row")
    if all(c in channels for c in trained_channels):
        return raw[[channels.index(c) for c in trained_channels]]
    if len(channels) == 16 and all(_BRAINFLOW_NAME.match(c) for c in channels):
        rows = [int(_BRAINFLOW_NAME.match(c)[1]) for c in channels]
        if rows != sorted(rows):
            raise ValueError("BrainFlow EEG rows must be in board order")
        return raw[[int(c.split()[-1]) for c in trained_channels]]
    raise ValueError(f"cannot map capture channels {channels} to the trained channels {trained_channels}")


def segment_views(raw_selected, config):
    """Selected raw rows (channels, samples) -> (filt, raw, gamma), each (samples, channels) float32, filtered as one
    segment: the loader's filtering, mean removal only, and the 30-45 Hz band-pass of the raw view."""
    x = np.asarray(raw_selected, np.float64).T
    filt = _filter(x, config)
    rawd = _filter(x, raw_config(config))
    gamma = bandpass_segment(rawd.astype(np.float64), fs=config.sfreq)
    return filt, rawd, gamma


def features_from_views(filt, rawd, gamma, fs, n_features, size=WINDOW_SIZE, stride=WINDOW_STRIDE, tail=True):
    """Three views of one segment -> subject features (n_windows, 435); empty when the segment is too short."""
    W_f, W_r, W_g = (cut_windows(filt, size, stride, tail), cut_windows(rawd, size, stride, tail),
                     cut_windows(gamma, size, stride, tail))
    if len(W_f) == 0:
        return np.empty((0, n_features), np.float32)
    return extract_subject_features(W_f, W_r, W_g, fs=fs)


def kept_runs(raw_selected, capture, config, channels, assessor):
    """The loader's segmentation of a capture: 2 s blocks, the first block dropped (start-up transient of a recording,
    reaction time after a cue), failing blocks dropped, runs split where packets were lost. Returns the raw samples of
    every run as (channels, samples) plus the block report."""
    fs = config.sfreq
    counters, timestamps = capture.get("counters"), capture.get("timestamps")
    block = int(round(BLOCK_SECONDS * fs))
    n_blocks = raw_selected.shape[1] // block
    usable = np.zeros(n_blocks, dtype=bool)
    reasons = []
    for b in range(n_blocks):
        a, e = b * block, (b + 1) * block
        if b == 0:
            reasons.append(["startup_block"])
            continue
        if not config.quality_filter:
            usable[b] = True
            reasons.append([])
            continue
        report = assessor.assess(raw_selected[:, a:e], sfreq=fs, channels=channels, units=capture.get("units", "uV"),
                                 counters=None if counters is None else counters[a:e],
                                 timestamps=None if timestamps is None else timestamps[a:e],
                                 counter_step=capture.get("counter_step"),
                                 timestamp_kind=capture.get("timestamp_kind", "arrival"))
        usable[b] = bool(report["usable"])
        reasons.append(report["reasons"])
    runs, run = [], []
    max_gap = assessor.config.max_arrival_gap_seconds
    for b in range(n_blocks):
        if usable[b] and run and counters is not None and timestamps is not None:
            a = b * block
            step = (counters[a] - counters[a - 1]) % 256
            gap = timestamps[a] - timestamps[a - 1]
            if step != capture.get("counter_step") or not 0 <= gap <= max_gap:
                runs.append(run)
                run = []
        if usable[b]:
            run.append(b)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    segments = [raw_selected[:, r[0] * block:(r[-1] + 1) * block] for r in runs]
    return segments, dict(n_blocks=int(n_blocks), n_blocks_kept=int(usable.sum()), block_reasons=reasons)


def _pct(x):
    """Percentage text with conventional rounding (0.745 -> '75')."""
    return str(int(np.floor(100.0 * float(x) + 0.5)))


def expected_accuracy(recordings_per_command):
    """Window accuracy the notebook's learning curve predicts for this many labelled recordings (cues) per command."""
    k = float(recordings_per_command)
    if k <= 0:
        return CHANCE
    xs, ys = zip(*LEARNING_CURVE)
    if k <= xs[0]:
        return float(CHANCE + (ys[0] - CHANCE) * k / xs[0])
    return float(np.interp(k, xs, ys))


def build_schedule(commands=COMMANDS, rounds=DEFAULT_ROUNDS, seed=None):
    """Interleaved cue order: every round presents every command once in a fresh random order, and no command is cued
    twice in a row across the round boundary (what makes a user's model mean the command, not the session state)."""
    rng = random.Random(seed)
    schedule = []
    for r in range(int(rounds)):
        order = list(commands)
        rng.shuffle(order)
        if schedule and order[0] == schedule[-1]["command"] and len(order) > 1:
            j = rng.randrange(1, len(order))
            order[0], order[j] = order[j], order[0]
        for command in order:
            schedule.append(dict(index=len(schedule), round=r + 1, command=str(command)))
    return schedule


class TrainingPlan:
    """Timing of a guided session: settle, then for every cue a rest gap followed by the cue itself."""

    def __init__(self, rounds=DEFAULT_ROUNDS, cue_seconds=DEFAULT_CUE_SECONDS, rest_seconds=DEFAULT_REST_SECONDS,
                 commands=COMMANDS, seed=None, settle_seconds=SETTLE_SECONDS):
        if int(rounds) < 1 or float(cue_seconds) < 2 * BLOCK_SECONDS:
            raise ValueError(f"need >= 1 round and cues of >= {2 * BLOCK_SECONDS:.0f} s (the first 2 s block is dropped)")
        self.rounds, self.cue_seconds, self.rest_seconds = int(rounds), float(cue_seconds), float(rest_seconds)
        self.settle_seconds, self.commands = float(settle_seconds), [str(c) for c in commands]
        self.seed = int(time.time()) if seed is None else int(seed)
        self.schedule = build_schedule(self.commands, self.rounds, self.seed)

    @property
    def n_cues(self):
        return len(self.schedule)

    @property
    def total_seconds(self):
        return self.settle_seconds + self.n_cues * (self.rest_seconds + self.cue_seconds)

    def as_dict(self):
        k = self.rounds
        return dict(rounds=self.rounds, cue_seconds=self.cue_seconds, rest_seconds=self.rest_seconds,
                    settle_seconds=self.settle_seconds, commands=self.commands, seed=self.seed, n_cues=self.n_cues,
                    total_seconds=self.total_seconds, total_minutes=round(self.total_seconds / 60, 1),
                    expected_accuracy=expected_accuracy(k),
                    expectation=(f"{self.rounds} cues per command: the notebook's learning curve gives about "
                                 f"{_pct(expected_accuracy(k))} % window accuracy "
                                 f"(4 recordings per command ~{_pct(expected_accuracy(4))} %, "
                                 f"6 ~{_pct(expected_accuracy(6))} %)"))


# ====================================================================================================== model (de)serialisation
def model_to_arrays(model):
    """A fitted projected GaussianNB (either backend) -> the framework-free arrays stored in model.npz."""
    if hasattr(model, "to_arrays"):            # TensorFlow backend
        return model.to_arrays()
    nb = model.nb_
    return dict(projection=np.asarray(model.projection_, np.float64), classes=np.asarray(model.classes_, np.int64),
                n_classes=np.int64(model.n_classes_), n_train=np.int64(getattr(model, "n_train_", 0)),
                absent_floor=np.float64(model.absent_floor),
                priors=nb.class_priors.detach().cpu().numpy().astype(np.float32),
                means=nb.means.detach().cpu().numpy().astype(np.float32),
                variances=nb.variances.detach().cpu().numpy().astype(np.float32))


def model_from_arrays(z, backend="torch"):
    """model.npz arrays -> a projected GaussianNB of the requested backend (the arrays are framework-free)."""
    if backend == "tensorflow":
        return tensorflow_backend().ProjectedGaussianNBTF.from_arrays(z)
    model = ProjectedTorchGaussianNB(n_components=N_COMPONENTS, shrinkage=SHRINKAGE, absent_floor=float(z["absent_floor"]))
    model.projection_ = np.ascontiguousarray(z["projection"], dtype=np.float64)
    model.classes_ = np.asarray(z["classes"], np.int64)
    model.n_classes_ = int(z["n_classes"])
    model.n_train_ = int(z["n_train"])
    means = np.asarray(z["means"], np.float32)
    nb = TorchGaussianNB(num_features=means.shape[1], num_classes=means.shape[0])
    with torch.no_grad():
        nb.class_priors.copy_(torch.from_numpy(np.asarray(z["priors"], np.float32)))
        nb.means.copy_(torch.from_numpy(means))
        nb.variances.copy_(torch.from_numpy(np.asarray(z["variances"], np.float32)))
        nb.is_fitted = torch.tensor(True)
    model.nb_ = nb.eval()
    return model


def fit_user_model(X, y, participant, min_windows=MIN_WINDOWS, backend="torch"):
    """A user's labelled windows -> (mean, std, model or None): the notebook's per-participant chain. The statistics
    are the ParticipantScaler statistics of these windows; the model is SubjectSpecificGaussianNB.add_participant on
    the standardised windows (None with fewer than min_windows windows or a single command: global fallback)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.int64)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be (n, features) with one label per row")
    scaler = ParticipantScaler()
    if len(X) < scaler.min_windows:
        raise ValueError(f"need at least {scaler.min_windows} labelled windows")
    mean, std = X.mean(0), X.std(0) + scaler.eps        # ParticipantScaler.fit statistics of this participant
    Z = (X - mean) / std
    ssg = SubjectSpecificGaussianNB(n_components=N_COMPONENTS, shrinkage=SHRINKAGE, min_windows=min_windows,
                                    n_classes=len(COMMANDS), backend=backend)
    ssg.models_, ssg.skipped_ = {}, {}
    model = ssg.add_participant(Z, y, participant)
    return mean, std, model, ssg.skipped_.get(str(participant))


def cue_grouped_cv(X, y, cue_ids, participant, min_windows=MIN_WINDOWS, max_folds=12, backend="torch"):
    """Honest self-check on the user's own data: statistics and model fitted without the held-out cues (grouped
    k-fold over cue ids), window accuracy and 5-window-context accuracy on the held-out cues. None when some command
    has fewer than two cues (then no fold can hold a cue out and still train every command)."""
    from sklearn.model_selection import GroupKFold
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.int64)
    cue_ids = np.asarray(cue_ids).astype(str)
    per_command = {c: len(np.unique(cue_ids[y == c])) for c in np.unique(y)}
    if len(per_command) < 2 or min(per_command.values()) < 2:
        return None
    n_splits = int(min(max_folds, min(per_command.values())))
    correct = ctx_correct = total = 0
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=cue_ids):
        mean, std, model, _ = fit_user_model(X[tr], y[tr], participant, min_windows=min(min_windows, len(tr)), backend=backend)
        if model is None:
            return None
        logp = model.predict_log_proba((X[te] - mean) / std)
        correct += int((logp.argmax(1) == y[te]).sum())
        for cue in np.unique(cue_ids[te]):
            m = np.where(cue_ids[te] == cue)[0]
            ctx_correct += int((context_log_posteriors(logp[m]).argmax(1) == y[te][m]).sum())
        total += len(te)
    return dict(folds=n_splits, windows=int(total), accuracy=correct / total, context_accuracy=ctx_correct / total)


# ====================================================================================================== the chain
class UserModel:
    """What is stored for one user: statistics, model (or None -> fallback), the labelled windows and metadata."""

    def __init__(self, participant, mean, std, model, meta, X=None, y=None, cue_ids=None, session_ids=None):
        self.participant, self.mean, self.std, self.model, self.meta = str(participant), mean, std, model, dict(meta)
        self.X, self.y, self.cue_ids, self.session_ids = X, y, cue_ids, session_ids

    @property
    def n_windows(self):
        return int(self.meta.get("n_windows", 0))

    @property
    def cues_per_command(self):
        return {c: int(self.meta.get("cues_per_command", {}).get(c, 0)) for c in COMMANDS}

    @property
    def min_cues(self):
        return min(self.cues_per_command.values())

    def status(self):
        if self.model is None:
            return f"too few labelled windows ({self.n_windows} < {MIN_WINDOWS}, {len(self.meta.get('commands', []))} commands): global fallback"
        return f"own model ({self.n_windows} windows, {self.min_cues} cues/command, expect ~{_pct(expected_accuracy(self.min_cues))} %)"


class SubjectSpecificGaussianNBChain:
    """Per-user subject-specific chain: three views -> 435 features -> the user's statistics and model -> 5-window
    context -> vote-agreement gate. Loads the users' models from users_dir and, when present, the notebook's
    artifacts (training participants + global fallback)."""

    def __init__(self, users_dir=None, model_dir=None, device="cpu", load_notebook_artifacts=True, backend="torch"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        self.model_dir = Path(model_dir) if model_dir is not None else (MODEL_DIR if backend == "torch" else TF_MODEL_DIR)
        default_users = DEFAULT_USERS_DIR if backend == "torch" else DEFAULT_USERS_DIR_TF
        self.users_dir = Path(users_dir) if users_dir is not None else default_users
        self.device = torch.device(device)
        if backend == "tensorflow":
            self.window_size, self.window_stride, self.window_tail = TF_WINDOW_SIZE, TF_WINDOW_STRIDE, False
        else:
            self.window_size, self.window_stride, self.window_tail = WINDOW_SIZE, WINDOW_STRIDE, True
        self._tf_payload = None
        if backend == "tensorflow" and load_notebook_artifacts and (self.model_dir / TF_SUBJECT_FILE).exists():
            self._tf_payload = tensorflow_backend().load_tf_pickle(self.model_dir / TF_SUBJECT_FILE)
        self.config = self._load_config()
        self.channels = list(self.config.channels)
        self.sfreq = float(self.config.sfreq)
        self.classes = list(COMMANDS)
        self.feature_names = subject_feature_names(self.channels)
        self.n_features = len(self.feature_names)
        self.notebook_models = self.notebook_scaler = None
        if self._tf_payload is not None:
            payload = self._tf_payload
            if (payload.get("feature_version") != SUBJECT_FEATURE_VERSION
                    or (int(payload["window_size"]), int(payload["window_stride"])) != (self.window_size, self.window_stride)):
                print(f"[GaussianNB subject] {TF_SUBJECT_FILE} has another feature version or window geometry; ignored")
            else:
                models, scaler = tensorflow_backend().subject_models_from_pickle(payload, n_classes=len(self.classes))
                if isinstance(scaler, ParticipantScaler) and scaler.n_features_in_ == self.n_features and models.global_.projection_.shape[0] == self.n_features:
                    self.notebook_models, self.notebook_scaler = models, scaler
                else:
                    print(f"[GaussianNB subject] {TF_SUBJECT_FILE} does not match the 435-feature set; ignored")
        elif backend == "torch" and load_notebook_artifacts and (self.model_dir / SUBJECT_MODELS_FILE).exists() and (self.model_dir / SUBJECT_SCALER_FILE).exists():
            models = joblib.load(self.model_dir / SUBJECT_MODELS_FILE)
            scaler = joblib.load(self.model_dir / SUBJECT_SCALER_FILE)
            if (isinstance(models, SubjectSpecificGaussianNB) and isinstance(scaler, ParticipantScaler)
                    and scaler.n_features_in_ == self.n_features and models.global_.projection_.shape[0] == self.n_features):
                self.notebook_models, self.notebook_scaler = models, scaler
            else:
                print("[GaussianNB subject] notebook artifacts do not match the 435-feature set; ignored")
        self.users = {}
        self.load_users()
        self.last_decision = None

    def _load_config(self):
        """The loader configuration of the global checkpoint (same board, channels and filtering), else the default."""
        if self.backend == "tensorflow":
            if self._tf_payload is not None:
                cfg = dict(self._tf_payload.get("load_config", {}))
                return LoadConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()
                                     if k in ("board", "exclude_channels", "quality_filter", "bandpass_hz", "notch_hz", "filter_order")})
            return LoadConfig()
        ck_path = self.model_dir / MODEL_FILE
        if ck_path.exists():
            try:
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                cfg = dict(ck["load_config"])
                return LoadConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()
                                     if k in ("board", "exclude_channels", "quality_filter", "bandpass_hz", "notch_hz", "filter_order")})
            except Exception as error:  # noqa: BLE001  (a broken global checkpoint must not block the subject chain)
                print(f"[GaussianNB subject] could not read {ck_path.name} ({error}); using the default LoadConfig")
        return LoadConfig()

    # ----------------------------------------------------------------------------------------------- geometry
    @property
    def capture_samples(self):
        return self.window_size + (CONTEXT_WINDOWS - 1) * self.window_stride

    @property
    def capture_seconds(self):
        return self.capture_samples / self.sfreq

    # ----------------------------------------------------------------------------------------------- raw -> features
    def select_channels(self, raw, channels):
        return select_channels(raw, channels, self.channels)

    def features_from_segment(self, raw_selected):
        """One raw segment (channels, samples), already channel-selected -> features of its windows (n, 435)."""
        filt, rawd, gamma = segment_views(raw_selected, self.config)
        return features_from_views(filt, rawd, gamma, self.sfreq, self.n_features, self.window_size, self.window_stride, self.window_tail)

    def features_from_raw(self, raw, channels, sfreq):
        """Raw capture rows (channels, samples) in uV -> features of its windows (filtered as one segment)."""
        if float(sfreq) != self.sfreq:
            raise ValueError(f"capture is {sfreq} Hz, model expects {self.sfreq} Hz")
        return self.features_from_segment(self.select_channels(raw, channels))

    def segments_from_capture(self, capture, assessor=None):
        """Loader-style segmentation of a longer capture -> (raw segments (channels, samples), block report)."""
        assessor = assessor or QualityAssessor(load_default_config())
        if float(capture["sfreq"]) != self.sfreq:
            raise ValueError(f"capture is {capture['sfreq']} Hz, model expects {self.sfreq} Hz")
        raw = self.select_channels(capture["raw"], capture["channels"])
        return kept_runs(raw, capture, self.config, self.channels, assessor)

    def features_from_capture(self, capture, assessor=None):
        """Longer capture -> (features of all windows of all kept runs, block report)."""
        segments, report = self.segments_from_capture(capture, assessor)
        feats = [self.features_from_segment(s) for s in segments]
        X = np.concatenate(feats) if feats else np.empty((0, self.n_features), np.float32)
        report["n_windows"] = int(len(X))
        return X, report

    # ----------------------------------------------------------------------------------------------- participants
    def resolve(self, participant):
        """(mean, std, model, status) used for this participant: own model > notebook training participant > global
        fallback (the user's statistics when they have some, else the notebook's global statistics)."""
        key = str(participant)
        user = self.users.get(key)
        if user is not None and user.model is not None:
            return user.mean, user.std, user.model, "own model"
        if self.notebook_models is not None and self.notebook_models.has_model(key) and key in self.notebook_scaler.stats_:
            mean, std = self.notebook_scaler.stats_[key]
            return mean, std, self.notebook_models.models_[key], "training participant"
        if self.notebook_models is None:
            raise ValueError(f"no trained model for '{key}' and no fallback artifacts in {self.model_dir}")
        if user is not None:
            return user.mean, user.std, self.notebook_models.global_, "global fallback (own statistics)"
        return self.notebook_scaler.global_mean_, self.notebook_scaler.global_std_, self.notebook_models.global_, "global fallback"

    def has_own_model(self, participant):
        user = self.users.get(str(participant))
        return user is not None and user.model is not None

    def participant_status(self, participant):
        key = str(participant)
        if key in self.users:
            return self.users[key].status()
        if self.notebook_models is not None and self.notebook_models.has_model(key):
            return "training participant"
        return "no model yet: global fallback" if self.notebook_models is not None else "no model yet"

    def participant_info(self, participant):
        """Everything the GUI shows about a user: status, windows and cues per command, expectation, self-check."""
        key = str(participant)
        user = self.users.get(key)
        info = dict(participant=key, status=self.participant_status(key), has_model=self.has_own_model(key),
                    n_windows=0, cues_per_command={c: 0 for c in COMMANDS}, windows_per_command={c: 0 for c in COMMANDS},
                    min_cues=0, expected_accuracy=CHANCE, cue_cv=None, sessions=[], min_windows=MIN_WINDOWS,
                    learning_curve=[dict(recordings_per_command=k, accuracy=a) for k, a in LEARNING_CURVE])
        if user is not None:
            info.update(n_windows=user.n_windows, cues_per_command=user.cues_per_command, min_cues=user.min_cues,
                        windows_per_command={c: int(user.meta.get("windows_per_command", {}).get(c, 0)) for c in COMMANDS},
                        expected_accuracy=expected_accuracy(user.min_cues), cue_cv=user.meta.get("cue_cv"),
                        sessions=list(user.meta.get("sessions", [])))
        return info

    # ----------------------------------------------------------------------------------------------- features -> decision
    def log_posteriors(self, X, participant):
        X = np.asarray(X, np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected (n, {self.n_features}) features")
        if len(X) == 0:
            return np.empty((0, len(self.classes)))
        mean, std, model, _ = self.resolve(participant)
        return model.predict_log_proba((X - mean) / std)

    def decide(self, logp, min_agreement=MIN_AGREEMENT, vote_windows=VOTE_WINDOWS):
        """Context decision of one time-ordered capture plus the vote-agreement gate."""
        logp = np.asarray(logp, np.float64)
        if logp.ndim != 2 or len(logp) == 0:
            raise ValueError("need at least one window")
        ctx = context_log_posteriors(logp)[-1]
        probs = np.exp(_log_softmax(ctx))
        index = int(np.argmax(ctx))
        window_votes = logp.argmax(axis=1)
        votes = window_votes[-vote_windows:]
        agreement = int((votes == index).sum())
        certain = agreement >= min_agreement
        return dict(label=self.classes[index] if certain else "uncertain", command=self.classes[index], index=index,
                    certain=bool(certain), agreement=agreement, votes=int(len(votes)), votes_needed=int(min_agreement),
                    confidence=float(probs[index]), probabilities={c: float(p) for c, p in zip(self.classes, probs)},
                    context_windows=int(min(CONTEXT_WINDOWS, len(logp))), windows=int(len(logp)),
                    window_labels=[self.classes[i] for i in window_votes])

    def predict_windows(self, X, participant, **gate_kwargs):
        logp = self.log_posteriors(X, participant)
        decision = self.decide(logp, **gate_kwargs)
        decision.update(participant=str(participant), participant_status=self.resolve(participant)[3], log_posteriors=logp)
        self.last_decision = decision
        return decision

    def predict_raw(self, raw, channels, sfreq, participant, **gate_kwargs):
        X = self.features_from_raw(raw, channels, sfreq)
        if len(X) == 0:
            raise ValueError(f"capture too short for one {self.window_size}-sample window")
        return self.predict_windows(X, participant, **gate_kwargs)

    def predict_capture(self, capture, participant, gate=None, **gate_kwargs):
        """Quality-gated live prediction on a signal_quality capture dict. The quality gate assesses the raw samples
        first; an uncertain decision (vote gate) is returned as a withheld result with reason 'uncertain_decision: ...'."""
        gate = gate or QualityGatedPredictor(QualityAssessor(load_default_config()))
        self.last_decision = None

        def callback(samples):
            d = self.predict_raw(samples, capture["channels"], capture["sfreq"], participant, **gate_kwargs)
            return dict(prediction_label=d["command"] if d["certain"] else None, confidence=d["confidence"])

        result = gate.predict(predict=callback, **capture)
        d = self.last_decision
        if d is not None and not d["certain"] and result.get("status") == "withheld":
            result = gate.withhold(f"uncertain_decision: {d['agreement']}/{d['votes']} window votes agree with "
                                   f"'{d['command']}', need {d['votes_needed']}", result.get("quality"),
                                   stream_id=capture.get("stream_id"))
        return result

    # ----------------------------------------------------------------------------------------------- users on disk
    def user_dir(self, participant):
        return self.users_dir / _safe_name(participant)

    def load_users(self):
        self.users = {}
        if not self.users_dir.is_dir():
            return self.users
        for d in sorted(p for p in self.users_dir.iterdir() if p.is_dir()):
            try:
                user = self._read_user(d)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                print(f"[GaussianNB subject] ignoring user folder {d.name}: {error}")
                continue
            if user is not None:
                self.users[user.participant] = user
        return self.users

    def _read_user(self, d, with_windows=False):
        if not (d / "meta.json").exists() or not (d / "scaler.npz").exists():
            return None
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("feature_version") != SUBJECT_FEATURE_VERSION:
            raise ValueError(f"feature version {meta.get('feature_version')!r} is not {SUBJECT_FEATURE_VERSION}")
        geometry = (int(meta.get("window_size", WINDOW_SIZE)), int(meta.get("window_stride", WINDOW_STRIDE)))
        if geometry != (self.window_size, self.window_stride):
            raise ValueError(f"trained with {geometry[0]}-sample windows (stride {geometry[1]}), this {self.backend} chain "
                             f"uses {self.window_size} / {self.window_stride}")
        s = np.load(d / "scaler.npz", allow_pickle=False)
        mean, std = s["mean"].astype(np.float64), s["std"].astype(np.float64)
        if mean.shape != (self.n_features,):
            raise ValueError("scaler statistics do not match the feature set")
        model = model_from_arrays(np.load(d / "model.npz", allow_pickle=False), self.backend) if (d / "model.npz").exists() else None
        user = UserModel(meta["participant"], mean, std, model, meta)
        if with_windows and (d / "training_windows.npz").exists():
            w = np.load(d / "training_windows.npz", allow_pickle=False)
            user.X, user.y = w["X"].astype(np.float32), w["y"].astype(np.int64)
            user.cue_ids, user.session_ids = w["cue_id"].astype(str), w["session_id"].astype(str)
        return user

    def training_windows(self, participant):
        """All stored labelled windows of a user -> (X, y, cue_ids, session_ids) or empty arrays."""
        d = self.user_dir(participant)
        if (d / "training_windows.npz").exists():
            w = np.load(d / "training_windows.npz", allow_pickle=False)
            return (w["X"].astype(np.float32), w["y"].astype(np.int64), w["cue_id"].astype(str),
                    w["session_id"].astype(str), w["recorded"].astype(np.float64))
        return (np.empty((0, self.n_features), np.float32), np.empty(0, np.int64), np.empty(0, str),
                np.empty(0, str), np.empty(0, np.float64))

    def train_user(self, participant, X_new, y_new, cue_ids_new, session, replace=False, self_check=True):
        """Append a session's labelled windows to the user's store, refit statistics and model from ALL their windows,
        persist everything and register the user. `session` is the session's metadata dict (id, source, plan, cues)."""
        key = str(participant)
        if not key.strip():
            raise ValueError("participant id must not be empty")
        X_new = np.asarray(X_new, np.float32)
        y_new = np.asarray(y_new, np.int64)
        cue_ids_new = np.asarray(cue_ids_new).astype(str)
        if X_new.ndim != 2 or X_new.shape[1] != self.n_features or len(X_new) != len(y_new) != len(cue_ids_new):
            raise ValueError(f"windows must be (n, {self.n_features}) with one label and cue id per row")
        if not np.isfinite(X_new).all():
            raise ValueError("non-finite feature values")
        if (y_new < 0).any() or (y_new >= len(COMMANDS)).any():
            raise ValueError("labels must index COMMANDS")
        session = dict(session)
        session_id = str(session.get("id") or time.strftime("%Y%m%d-%H%M%S"))
        session["id"] = session_id
        if replace:
            X_old, y_old, cues_old, sessions_old, rec_old = (np.empty((0, self.n_features), np.float32), np.empty(0, np.int64),
                                                             np.empty(0, str), np.empty(0, str), np.empty(0, np.float64))
        else:
            X_old, y_old, cues_old, sessions_old, rec_old = self.training_windows(key)
        X = np.concatenate([X_old, X_new]) if len(X_old) else X_new
        y = np.concatenate([y_old, y_new]) if len(y_old) else y_new
        cue_ids = np.concatenate([cues_old, np.array([f"{session_id}/{c}" for c in cue_ids_new])]) if len(cues_old) else np.array([f"{session_id}/{c}" for c in cue_ids_new])
        session_ids = np.concatenate([sessions_old, np.array([session_id] * len(y_new))]) if len(sessions_old) else np.array([session_id] * len(y_new))
        recorded = np.concatenate([rec_old, np.full(len(y_new), float(session.get("started_unix", time.time())))]) if len(rec_old) else np.full(len(y_new), float(session.get("started_unix", time.time())))
        if len(X) < 2:
            raise ValueError("need at least two labelled windows")
        mean, std, model, skipped = fit_user_model(X, y, key, backend=self.backend)
        cue_cv = cue_grouped_cv(X, y, cue_ids, key, backend=self.backend) if (self_check and model is not None) else None
        old_meta = self.users[key].meta if key in self.users and not replace else {}
        sessions = [s for s in old_meta.get("sessions", []) if s.get("id") != session_id]
        sessions.append(dict(id=session_id, source=session.get("source", "live"), started=session.get("started"),
                             n_windows=int(len(y_new)), n_cues=int(len(np.unique(cue_ids_new))),
                             windows_per_command={c: int((y_new == i).sum()) for i, c in enumerate(COMMANDS)},
                             plan=session.get("plan"), cues=session.get("cues")))
        meta = dict(participant=key, feature_version=SUBJECT_FEATURE_VERSION, n_features=self.n_features,
                    framework=self.backend, window_size=self.window_size, window_stride=self.window_stride,
                    channels=self.channels, sfreq=self.sfreq, commands=[COMMANDS[i] for i in np.unique(y)],
                    n_windows=int(len(y)), windows_per_command={c: int((y == i).sum()) for i, c in enumerate(COMMANDS)},
                    cues_per_command={c: int(len(np.unique(cue_ids[y == i]))) for i, c in enumerate(COMMANDS)},
                    min_windows=MIN_WINDOWS, n_components=N_COMPONENTS, shrinkage=SHRINKAGE,
                    model="own" if model is not None else f"global fallback ({skipped})",
                    model_classes=[COMMANDS[i] for i in model.classes_] if model is not None else [],
                    cue_cv=cue_cv, learning_curve=[dict(recordings_per_command=k, accuracy=a) for k, a in LEARNING_CURVE],
                    created=old_meta.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                    updated=time.strftime("%Y-%m-%dT%H:%M:%S"), sessions=sessions)
        min_cues = min(meta["cues_per_command"].values())
        meta["expected_accuracy"] = expected_accuracy(min_cues)
        d = self.user_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "training_windows.npz", X=X, y=y, cue_id=cue_ids.astype(str),
                            session_id=session_ids.astype(str), recorded=recorded)
        np.savez(d / "scaler.npz", mean=mean, std=std, n_windows=np.int64(len(y)))
        if model is not None:
            np.savez(d / "model.npz", **model_to_arrays(model))
        elif (d / "model.npz").exists():
            (d / "model.npz").unlink()
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        self.users[key] = UserModel(key, mean, std, model, meta)
        return self.users[key]

    def save_session_raw(self, participant, session_id, cues):
        """Store the raw EEG of every cue of a session (cues: list of dicts with capture, command, index, report)."""
        d = self.user_dir(participant) / "sessions" / _safe_name(session_id)
        d.mkdir(parents=True, exist_ok=True)
        arrays, index = {}, []
        for k, cue in enumerate(cues):
            cap = cue["capture"]
            arrays[f"raw_{k:03d}"] = np.asarray(cap["raw"], np.float32)
            arrays[f"counters_{k:03d}"] = np.asarray(cap["counters"], np.float64)
            arrays[f"timestamps_{k:03d}"] = np.asarray(cap["timestamps"], np.float64)
            index.append(dict(k=k, index=cue["index"], round=cue.get("round"), command=cue["command"],
                              channels=list(cap["channels"]), sfreq=float(cap["sfreq"]), counter_step=cap.get("counter_step"),
                              samples=int(np.asarray(cap["raw"]).shape[1]), report=cue.get("report")))
        np.savez_compressed(d / "cues.npz", **arrays)
        (d / "cues.json").write_text(json.dumps(index, indent=1, default=str))
        return d

    def forget_user(self, participant):
        """Remove a user's model, statistics and stored windows (raw session data included)."""
        import shutil
        key = str(participant)
        self.users.pop(key, None)
        d = self.user_dir(key)
        if d.is_dir():
            shutil.rmtree(d)

    # ----------------------------------------------------------------------------------------------- offline training / replay
    def train_from_recordings(self, participant, labelled_paths, session_id=None, assessor=None, source="recordings", self_check=True):
        """Train (or extend) a user's model from labelled recordings: [(path, command), ...]. Every recording is one
        cue of that command; the loader's block rule applies to each recording."""
        assessor = assessor or QualityAssessor(load_default_config())
        feats, labels, cue_ids, cues = [], [], [], []
        for path, command in sorted((str(p), str(c)) for p, c in labelled_paths):
            if command not in COMMANDS:
                raise ValueError(f"unknown command {command!r} for {path}")
            X, report = self.features_from_capture(read_openbci(path), assessor)
            cues.append(dict(path=path, command=command, n_windows=int(len(X)), n_blocks=report["n_blocks"],
                             n_blocks_kept=report["n_blocks_kept"]))
            if len(X):
                feats.append(X)
                labels.append(np.full(len(X), COMMANDS.index(command), np.int64))
                cue_ids.append(np.array([Path(path).name] * len(X)))
        if not feats:
            raise ValueError("no usable windows in the given recordings")
        session = dict(id=session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + source, source=source,
                       started=time.strftime("%Y-%m-%dT%H:%M:%S"), started_unix=time.time(), cues=cues)
        return self.train_user(participant, np.concatenate(feats), np.concatenate(labels), np.concatenate(cue_ids), session,
                               self_check=self_check)

    def train_from_folder(self, participant, folder, **kwargs):
        """Labelled recordings of one person as in Data_clean: <folder>/<command folder>/OpenBCI-RAW-*.txt."""
        paths = []
        for p in sorted(Path(folder).rglob("*.txt")):
            label = folder_label(p.parent.name)
            if label in COMMANDS and p.open(errors="ignore").read(21) == "%OpenBCI Raw EXG Data":
                paths.append((p, label))
        if not paths:
            raise ValueError(f"no named OpenBCI recordings in labelled command folders under {folder}")
        return self.train_from_recordings(participant, paths, source=f"folder:{Path(folder).name}", **kwargs)

    def replay_recording(self, path, participant=None, assessor=None, **gate_kwargs):
        """One recording through the loader-equivalent path: per kept segment the context decision with the vote
        gate and the per-window labels (plus their accuracy when the folder gives the command)."""
        capture = read_openbci(path)
        participant = participant_of(path) if participant is None else participant
        truth = folder_label(Path(path).parent.name)
        segments, report = self.segments_from_capture(capture, assessor)
        out = []
        for seg in segments:
            X = self.features_from_segment(seg)
            if len(X):
                d = self.predict_windows(X, participant, **gate_kwargs)
                entry = dict(samples=int(seg.shape[1]), windows=int(len(X)), label=d["label"], command=d["command"],
                             certain=d["certain"], agreement=f"{d['agreement']}/{d['votes']}", confidence=d["confidence"],
                             probabilities=d["probabilities"], window_labels=d["window_labels"])
                if truth in COMMANDS:
                    entry["window_accuracy"] = float(np.mean([w == truth for w in d["window_labels"]]))
                out.append(entry)
        return dict(path=str(path), participant=str(participant), participant_status=self.participant_status(participant),
                    truth=truth if truth in COMMANDS else None, segments=out, **report)


# ====================================================================================================== acquisition + session
class BrainFlowStream:
    """Continuous acquisition for a training session: one board session for the whole session; read() returns the
    samples received since the previous read() / discard() as a signal_quality capture dict."""

    def __init__(self, board_id=None, serial_port=None, ring_buffer_samples=450_000):
        self.board_id, self.serial_port, self.ring_buffer_samples = board_id, serial_port, ring_buffer_samples
        self.board = None

    def start(self):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        board_id = BoardIds.CYTON_DAISY_BOARD.value if self.board_id is None else int(self.board_id)
        params = BrainFlowInputParams()
        if board_id in (BoardIds.CYTON_BOARD.value, BoardIds.CYTON_DAISY_BOARD.value):
            params.serial_port = self.serial_port or self._detect_serial_port()
        self._board_id = board_id
        self._real = board_id in (BoardIds.CYTON_BOARD.value, BoardIds.CYTON_DAISY_BOARD.value)
        self._counter_step = 2 if board_id == BoardIds.CYTON_DAISY_BOARD.value else 1
        self.descr = BoardShim.get_board_descr(board_id)
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream(self.ring_buffer_samples)

    @staticmethod
    def _detect_serial_port():
        from serial.tools import list_ports
        for p in list_ports.comports():
            text = (p.description + " " + p.hwid).lower()
            if "usb" in text or "ftdi" in text or "serial" in text:
                return p.device
        raise RuntimeError("Could not auto-detect the Cyton serial port")

    def discard(self):
        self.board.get_board_data()

    def read(self):
        data = self.board.get_board_data()
        capture = brainflow_capture(data, self.descr, real=self._real, counter_step=self._counter_step)
        capture.update(captured_monotonic=time.monotonic(), stream_id=f"brainflow:{self._board_id}:{uuid.uuid4()}")
        return capture

    def stop(self):
        if self.board is not None:
            try:
                self.board.stop_stream()
            finally:
                self.board.release_session()
                self.board = None


class TrainingSession:
    """One guided labelled session. For every cue of the plan: a rest gap, then the cue; the samples of the cue
    period are read from the stream, quality-checked per 2 s block, windowed, featurised and labelled with the cue.
    At the end (or when stopped early, with what was collected) the user's statistics and model are fitted from all
    their stored windows and persisted. `present(event)` receives progress dicts for the GUI."""

    def __init__(self, chain, participant, plan, stream, present=None, should_stop=None, sleep=time.sleep,
                 clock=time.monotonic, assessor=None, session_id=None, keep_raw=True):
        self.chain, self.participant, self.plan, self.stream = chain, str(participant), plan, stream
        self.present = present or (lambda event: None)
        self.should_stop = should_stop or (lambda: False)
        self.sleep, self.clock = sleep, clock
        self.assessor = assessor or QualityAssessor(load_default_config())
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.keep_raw = keep_raw
        self.cues = []
        self.events = []
        self.stopped = False

    def _emit(self, **event):
        event.setdefault("session_id", self.session_id)
        event.setdefault("participant", self.participant)
        event.setdefault("n_cues", self.plan.n_cues)
        event.setdefault("windows_total", int(sum(c["n_windows"] for c in self.cues)))
        self.events.append(event)
        self.present(event)

    def _countdown(self, seconds, **event):
        """One event per second with the remaining time; returns False when stopped."""
        end = self.clock() + float(seconds)
        remaining = float(seconds)
        while remaining > 0:
            if self.should_stop():
                self.stopped = True
                return False
            self._emit(remaining=int(np.ceil(remaining)), **event)
            self.sleep(min(1.0, remaining))
            remaining = end - self.clock()
        return True

    def run(self):
        plan = self.plan
        started = time.time()
        self.stream.start()
        try:
            self._emit(phase="settle", message="Streaming; settling the headset", command="", index=-1, round=0,
                       remaining=int(np.ceil(plan.settle_seconds)))
            if not self._countdown(plan.settle_seconds, phase="settle", command="", index=-1, round=0):
                return self._finish(started)
            for cue in plan.schedule:
                base = dict(index=cue["index"], round=cue["round"], rounds=plan.rounds, command=cue["command"])
                if plan.rest_seconds > 0 and not self._countdown(plan.rest_seconds, phase="rest", **base):
                    return self._finish(started)
                self.stream.discard()
                if not self._countdown(plan.cue_seconds, phase="cue", **base):
                    return self._finish(started)
                capture = self.stream.read()
                X, report = self.chain.features_from_capture(capture, self.assessor)
                self.cues.append(dict(index=cue["index"], round=cue["round"], command=cue["command"], capture=capture,
                                      X=X, n_windows=int(len(X)), report=dict(n_blocks=report["n_blocks"],
                                                                              n_blocks_kept=report["n_blocks_kept"]),
                                      samples=int(np.asarray(capture["raw"]).shape[1])))
                self._emit(phase="cue_done", n_windows=int(len(X)), n_blocks=report["n_blocks"],
                           n_blocks_kept=report["n_blocks_kept"], **base)
        finally:
            self.stream.stop()
        return self._finish(started)

    def _finish(self, started):
        feats = [c["X"] for c in self.cues if c["n_windows"]]
        summary = dict(session_id=self.session_id, participant=self.participant, stopped_early=self.stopped,
                       cues_done=len(self.cues), n_cues=self.plan.n_cues, plan=self.plan.as_dict(),
                       windows_total=int(sum(c["n_windows"] for c in self.cues)),
                       windows_per_cue=[dict(index=c["index"], command=c["command"], n_windows=c["n_windows"],
                                             n_blocks_kept=c["report"]["n_blocks_kept"], n_blocks=c["report"]["n_blocks"])
                                        for c in self.cues],
                       cues_without_windows=[c["index"] for c in self.cues if not c["n_windows"]])
        if self.keep_raw and self.cues:
            summary["raw_dir"] = str(self.chain.save_session_raw(self.participant, self.session_id, self.cues))
        if not feats:
            summary.update(trained=False, message=("stopped before the first cue finished; nothing was stored" if not self.cues
                                                   else "no usable windows were recorded (every 2 s block of every cue failed "
                                                        "the quality checks); nothing was stored"))
            self._emit(phase="finished", **{k: v for k, v in summary.items() if k != "plan"})
            return summary
        X = np.concatenate(feats)
        y = np.concatenate([np.full(c["n_windows"], COMMANDS.index(c["command"]), np.int64) for c in self.cues if c["n_windows"]])
        cue_ids = np.concatenate([np.array([f"cue{c['index']:03d}-{c['command']}"] * c["n_windows"]) for c in self.cues if c["n_windows"]])
        session = dict(id=self.session_id, source="live", started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
                       started_unix=started, plan=self.plan.as_dict(),
                       cues=[dict(index=c["index"], command=c["command"], n_windows=c["n_windows"], **c["report"]) for c in self.cues])
        user = self.chain.train_user(self.participant, X, y, cue_ids, session)
        summary.update(trained=user.model is not None, status=user.status(), n_windows_user=user.n_windows,
                       cues_per_command=user.cues_per_command, expected_accuracy=expected_accuracy(user.min_cues),
                       cue_cv=user.meta.get("cue_cv"), model=user.meta.get("model"),
                       message=(f"model fitted on {user.n_windows} labelled windows" if user.model is not None
                                else f"{user.n_windows} labelled windows so far; {MIN_WINDOWS} and two commands are needed"))
        self._emit(phase="finished", **{k: v for k, v in summary.items() if k != "plan"})
        return summary


def session_plan_from_env():
    """The GUI's plan: AVATAR_GNB_TRAIN_ROUNDS / _CUE_SECONDS / _REST_SECONDS (defaults 6 / 20 / 5)."""
    def env(name, default, cast):
        try:
            return cast(os.environ.get(name, default))
        except (TypeError, ValueError):
            return cast(default)
    return TrainingPlan(rounds=env("AVATAR_GNB_TRAIN_ROUNDS", DEFAULT_ROUNDS, int),
                        cue_seconds=env("AVATAR_GNB_TRAIN_CUE_SECONDS", DEFAULT_CUE_SECONDS, float),
                        rest_seconds=env("AVATAR_GNB_TRAIN_REST_SECONDS", DEFAULT_REST_SECONDS, float))


# ====================================================================================================== CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="Subject-specific GaussianNB: train a user from labelled recordings, replay, status.")
    p.add_argument("--users-dir", default=None, help="where users are stored (default: %s, TensorFlow: %s)" % (DEFAULT_USERS_DIR, DEFAULT_USERS_DIR_TF))
    p.add_argument("--backend", default="torch", choices=BACKENDS, help="GaussianNB implementation: PyTorch (default) or TensorFlow")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-folder", help="train <participant> from <folder>/<command>/OpenBCI-RAW-*.txt")
    t.add_argument("--participant", required=True)
    t.add_argument("--folder", required=True)
    t.add_argument("--no-self-check", action="store_true", help="skip the cue-grouped cross-validation")
    r = sub.add_parser("replay", help="replay a recording through the live path")
    r.add_argument("--file", required=True)
    r.add_argument("--participant", default=None, help="default: <group>/<person> from the path")
    s = sub.add_parser("status", help="what is stored for a participant")
    s.add_argument("--participant", required=True)
    f = sub.add_parser("forget", help="delete a participant's stored model, statistics and windows")
    f.add_argument("--participant", required=True)
    a = p.parse_args(argv)
    chain = SubjectSpecificGaussianNBChain(users_dir=a.users_dir, backend=a.backend)
    if a.cmd == "train-folder":
        user = chain.train_from_folder(a.participant, a.folder, self_check=not a.no_self_check)
        print(json.dumps(dict(participant=user.participant, status=user.status(), meta={k: v for k, v in user.meta.items()
                              if k not in ("sessions", "learning_curve")}), indent=2, default=str))
    elif a.cmd == "replay":
        print(json.dumps(chain.replay_recording(a.file, a.participant), indent=2, default=str))
    elif a.cmd == "status":
        print(json.dumps(chain.participant_info(a.participant), indent=2, default=str))
    elif a.cmd == "forget":
        chain.forget_user(a.participant)
        print(f"forgot {a.participant}")


if __name__ == "__main__":
    main()
