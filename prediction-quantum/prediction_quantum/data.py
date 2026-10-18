"""EEG windows and features for the QML benchmark, through the GaussianNB pipeline, with a recording-level split.

Every window seen by the quantum and the classical models comes from the same chain the GaussianNB pipelines use
(prediction-gaussiannb/eeg_data.py and prediction-gaussiannb/improved_gnb_features.py):

    Data_clean -> load_recordings (16ch_125hz board, channel 15 excluded, quality-filtered 2 s blocks,
    1-40 Hz band-pass + 60 Hz notch)  ->  128-sample windows, stride 64, cut inside each kept segment
    (the PyTorch notebook's rule, including the tail window)  ->  extract_features: 570 features per window

Labels are indices into eeg_data.LABELS = (backward, forward, landing, left, right, takeoff). The features are
cached as an .npz keyed by the loader configuration, the window rule and the feature version; the cache is not
committed (results/cache/ is ignored by Git) but is rebuilt from Data_clean by run_benchmark.

Splits are by recording (GroupShuffleSplit on the recording path, the protocol of the defense benchmark in
prediction-gaussiannb/defense/make_defense_figures.py): 20 % of the recordings are held out for testing and a
further slice of the training recordings is held out for per-epoch validation. No window of a test or validation
recording is ever used for fitting the scaler, the PCA or a model. The exact window indices of every split are
written to results/split_seed<seed>.csv and every model reads them from there.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import GroupShuffleSplit

PKG_DIR = Path(__file__).resolve().parent          # prediction-quantum/prediction_quantum
PROJECT_DIR = PKG_DIR.parent                        # prediction-quantum
REPO_ROOT = PROJECT_DIR.parent
GNB_DIR = REPO_ROOT / "prediction-gaussiannb"
for _p in (REPO_ROOT, GNB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _block_shadowing_namespace_packages(names=("jax", "tensorflow")):
    """prediction-gaussiannb/ holds folders named jax/ and tensorflow/. With that directory on sys.path and the real
    libraries not installed (this benchmark needs neither), `import jax` would return an empty namespace package and
    PennyLane's optional-dependency probe (`jax.__version__`) would crash. Registering None makes the import raise
    ImportError, which is what an absent library looks like. A really installed library wins over a namespace
    portion (PEP 420) and is left untouched."""
    import importlib.util
    for name in names:
        if name in sys.modules:
            continue
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin is None:        # namespace package: only the shadowing folders exist
            sys.modules[name] = None


_block_shadowing_namespace_packages()

from eeg_data import LABELS, LoadConfig, load_recordings  # noqa: E402
from improved_gnb_features import FEATURE_VERSION, extract_features, participant_of  # noqa: E402

CLASSES = list(LABELS)
N_CLASSES = len(CLASSES)
WINDOW_SIZE = 128        # samples per window (1.024 s at 125 Hz), as in the GaussianNB notebooks
WINDOW_STRIDE = 64       # max(16, WINDOW_SIZE // 2)
MIN_SEGMENT_ROWS = 8     # segments shorter than this are skipped
N_FEATURES = 570         # for 15 channels: 120 + 120 + 30 + 150 + 150


def window_starts(n_rows, size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Start samples of the windows of a segment with n_rows samples (the PyTorch notebook's rule: regular
    starts, plus a last window ending at the segment end when the stride does not land there)."""
    starts = list(range(0, max(1, n_rows - size + 1), stride))
    if not starts or starts[-1] != max(0, n_rows - size):
        starts.append(max(0, n_rows - size))
    return np.asarray(starts)


def cut_windows(segment, size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Filtered segment (samples, channels) -> windows (n, channels, size); empty when the segment is too short."""
    segment = np.asarray(segment, np.float64)
    n_rows = segment.shape[0]
    if n_rows < MIN_SEGMENT_ROWS:
        return np.empty((0, segment.shape[1], min(size, max(n_rows, 1))))
    win_len = min(size, n_rows)
    return sliding_window_view(segment, win_len, axis=0)[window_starts(n_rows, size, stride)]


def feature_spec(load_config=LoadConfig()):
    return dict(load_config.as_dict(), window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE,
                min_segment_rows=MIN_SEGMENT_ROWS, feature_version=FEATURE_VERSION, labels=CLASSES)


def feature_cache_key(load_config=LoadConfig()):
    return hashlib.sha256(json.dumps(feature_spec(load_config), sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_features(data_dir, cache_dir, load_config=LoadConfig(), n_jobs=-1, verbose=True):
    """Features of every usable window under data_dir, from the cache when it exists.

    Returns dict(X (n, 570) float32, y (n,) int64, groups (n,) recording path relative to data_dir,
    participants (n,) str, cache (path of the .npz), meta (dict)).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = feature_cache_key(load_config)
    npz, meta_path = cache_dir / f"features_{key}.npz", cache_dir / f"features_{key}.json"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return dict(X=z["X"].astype(np.float32), y=z["y"].astype(np.int64), groups=z["groups"].astype(str),
                    participants=z["participants"].astype(str), cache=str(npz), meta=meta)

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"EEG data directory not found: {data_dir} (pass --data-dir or set data.data_dir)")
    t0 = time.perf_counter()
    recordings, skipped = load_recordings(data_dir, load_config, n_jobs=n_jobs, verbose=verbose)
    feats, labels, groups, n_short, n_nonfinite = [], [], [], 0, 0
    for r in recordings:
        rel = str(Path(r["path"]).relative_to(data_dir)) if Path(r["path"]).is_absolute() else r["path"]
        for seg in r["segments"]:
            W = cut_windows(seg)
            if len(W) == 0:
                n_short += 1
                continue
            if not np.isfinite(W).all():
                n_nonfinite += int((~np.isfinite(W)).sum())
                W = np.nan_to_num(W)
            feats.append(extract_features(W, fs=load_config.sfreq))
            labels.extend([CLASSES.index(r["label"])] * len(W))
            groups.extend([rel] * len(W))
    if not feats:
        raise ValueError(f"no usable windows under {data_dir}")
    X = np.concatenate(feats).astype(np.float32)
    y = np.asarray(labels, np.int64)
    groups = np.asarray(groups, dtype=str)
    participants = np.array([participant_of(g) for g in groups])
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature values")
    meta = dict(spec=feature_spec(load_config), data_dir=str(data_dir), n_windows=int(len(y)), n_features=int(X.shape[1]),
                n_recordings=int(len(np.unique(groups))), n_recordings_loaded=len(recordings), n_recordings_skipped=len(skipped),
                n_participants=int(len(np.unique(participants))), windows_per_class={c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
                segments_too_short=n_short, nonfinite_samples_zeroed=n_nonfinite, seconds=round(time.perf_counter() - t0, 1))
    np.savez_compressed(npz, X=X, y=y, groups=groups, participants=participants)
    meta_path.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"features: {X.shape} from {meta['n_recordings']} recordings, {meta['n_participants']} participants "
              f"in {meta['seconds']} s -> {npz}")
    return dict(X=X, y=y, groups=groups, participants=participants, cache=str(npz), meta=meta)


def subsample_per_class(y, max_per_class, seed):
    """Sorted indices keeping at most max_per_class windows of every class (fixed seed; None keeps everything)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, int(max_per_class), replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def make_split(y, groups, seed, test_fraction=0.2, val_fraction=0.1):
    """Recording-level split -> array of 'train' / 'val' / 'test' per window.

    GroupShuffleSplit holds out test_fraction of the recordings (the defense benchmark's protocol, one hold-out per
    seed); a second GroupShuffleSplit on the remaining recordings holds out val_fraction (of all recordings) for
    per-epoch validation. Recordings never straddle two parts.
    """
    idx = np.arange(len(y))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(idx, y, groups))
    rel_val = val_fraction / (1.0 - test_fraction)
    tr2, va = next(GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed).split(tr, y[tr], groups[tr]))
    part = np.empty(len(y), dtype=object)
    part[tr[tr2]], part[tr[va]], part[te] = "train", "val", "test"
    return part.astype(str)


def write_split(path, keep, y, groups, participants, part):
    """results/split_seed<k>.csv: one row per subsampled window with its index into the feature cache."""
    df = pd.DataFrame(dict(window_index=keep, recording=groups[keep], participant=participants[keep],
                           label=[CLASSES[i] for i in y[keep]], part=part))
    df.to_csv(path, index=False)
    return df


def load_split(path):
    """Split CSV -> dict(train=idx, val=idx, test=idx) of indices into the feature cache, plus the frame."""
    df = pd.read_csv(path)
    out = {p: df.loc[df["part"] == p, "window_index"].to_numpy() for p in ("train", "val", "test")}
    assert not (set(df.loc[df.part == "train", "recording"]) & set(df.loc[df.part == "test", "recording"]))
    assert not (set(df.loc[df.part == "train", "recording"]) & set(df.loc[df.part == "val", "recording"]))
    out["frame"] = df
    return out


def split_summary(df):
    rows = []
    for p in ("train", "val", "test"):
        d = df[df.part == p]
        rows.append(dict(part=p, windows=len(d), recordings=d.recording.nunique(), participants=d.participant.nunique(),
                         **{c: int((d.label == c).sum()) for c in CLASSES}))
    return pd.DataFrame(rows)
