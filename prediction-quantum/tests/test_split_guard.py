"""The split-CSV guard and the report's path helper (no EEG data needed)."""
import numpy as np
import pandas as pd
import pytest

from prediction_quantum import data as D
from prediction_quantum import report


def _feats(n_per_rec=4, n_rec=6):
    """A toy feature cache: n_rec recordings of n_per_rec windows each, labels cycling through the classes."""
    groups = np.repeat([f"G01/person01/{D.CLASSES[i % 6]}/rec{i}.txt" for i in range(n_rec)], n_per_rec)
    y = np.repeat(np.arange(n_rec) % 6, n_per_rec)
    return dict(X=np.zeros((len(y), 3), np.float32), y=y, groups=groups, participants=np.repeat("G01/person01", len(y)), cache="toy")


def _split(feats, path):
    keep = np.arange(len(feats["y"]))
    part = np.where(keep < 8, "test", "train")
    return D.write_split(path, keep, feats["y"], feats["groups"], feats["participants"], part)


def test_split_written_for_the_cache_passes(tmp_path):
    feats = _feats()
    path = tmp_path / "split_seed0.csv"
    _split(feats, path)
    split = D.load_split(path, feats)
    assert len(split["test"]) == 8 and len(split["train"]) == 16


def test_split_names_are_posix_whatever_platform_wrote_them(tmp_path):
    feats = _feats()
    path = tmp_path / "split_seed0.csv"
    df = _split(feats, path)
    df["recording"] = df["recording"].str.replace("/", "\\", regex=False)     # as a Windows cache named them
    df.to_csv(path, index=False)
    split = D.load_split(path, feats)
    assert "\\" not in "".join(split["frame"]["recording"])


def test_split_of_a_cache_with_another_row_order_is_refused(tmp_path):
    feats = _feats()
    path = tmp_path / "split_seed0.csv"
    _split(feats, path)
    perm = np.random.default_rng(0).permutation(len(feats["y"]))
    moved = dict(feats, y=feats["y"][perm], groups=feats["groups"][perm])
    with pytest.raises(ValueError, match="do not match the feature cache"):
        D.load_split(path, moved)


def test_split_of_a_larger_cache_is_refused(tmp_path):
    feats = _feats()
    path = tmp_path / "split_seed0.csv"
    _split(feats, path)
    short = dict(feats, y=feats["y"][:10], groups=feats["groups"][:10])
    with pytest.raises(ValueError, match="window_index up to"):
        D.load_split(path, short)


def test_report_paths_are_relative_or_absolute_never_an_error(tmp_path):
    base = tmp_path / "project"
    (base / "results").mkdir(parents=True)
    assert report.rel(base / "results", base) == "results"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert report.rel(outside, base) == "../elsewhere"
    # a path on another drive (Windows) or any path that relpath refuses is written in full
    assert report.rel(base, base) == "."
