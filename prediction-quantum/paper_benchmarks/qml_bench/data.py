"""Windows, labels, recordings and participants from the project's feature cache.

The cache was built by prediction-quantum/prediction_quantum/data.py through the verified GaussianNB loader
(prediction-gaussiannb/eeg_data.py) and the 570-feature extractor (improved_gnb_features.extract_features):
128-sample windows (1.024 s at 125 Hz), stride 64, cut inside quality-screened segments of 883 recordings from
18 participants of the 16-channel board. It is read, never rewritten.
"""
from pathlib import Path

import numpy as np

from .paths import FEATURE_CACHE_DIR

CLASSES = ("backward", "forward", "landing", "left", "right", "takeoff")


def load_windows(cache_dir=FEATURE_CACHE_DIR):
    """Read the one features_<key>.npz in cache_dir. Returns dict(X (n, 570) float32, y (n,) int64 indices into
    CLASSES, groups (n,) recording path of each window, participants (n,) str, cache (path of the .npz))."""
    files = sorted(Path(cache_dir).glob("features_*.npz"))
    # <key> hashes the loader settings, window rule and feature version, so a second cache would be a different
    # configuration: refuse to guess
    if len(files) != 1:
        raise FileNotFoundError(f"expected one features_*.npz in {cache_dir}, found {len(files)}")
    z = np.load(files[0], allow_pickle=True)
    return dict(X=z["X"].astype(np.float32), y=z["y"].astype(np.int64), groups=z["groups"].astype(str),
                participants=z["participants"].astype(str), cache=str(files[0]))


def subsample_per_class(y, max_per_class, seed):
    """Identical to prediction_quantum.data.subsample_per_class (same windows as the original benchmark)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y == c)
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, int(max_per_class), replace=False)
        keep.append(idx)
    # back in cache order: the draws above are concatenated class by class; the original sorts too, and callers
    # compare this against other sorted index arrays (fold coverage, the matched split check)
    return np.sort(np.concatenate(keep))
