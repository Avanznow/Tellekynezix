"""Leakage-safe evaluation protocols on identical windows.

participant  GroupKFold over participants (5 folds): every test participant is absent from training, the
             protocol that answers whether a decoder works for a new user (protocol C of the paper).
recording    the three recording-grouped splits of the original quantum benchmark (results/split_seed<k>.csv),
             within-participant separability, kept for continuity with Sections 2.10 and 3.13 of the paper.

In both protocols a validation part is carved out of the training recordings only (GroupShuffleSplit by
recording) and used only to choose the checkpoint of the gradient-trained models. Scalers, PCA, kernels and
models are fitted on the training part of each fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .paths import RECORDING_SPLITS


def participant_folds(idx, y, groups, participants, n_folds=5, val_fraction=0.1, seed=0):
    """Protocol C: GroupKFold over participants on the windows in `idx`, with a validation part cut from the
    training recordings of each fold. `groups` is the recording id per window, `participants` the participant
    id; returns one dict per fold with sorted train/val/test index arrays into the full window set."""
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(idx, y[idx], participants[idx])):
        tr_idx, te_idx = idx[tr], idx[te]
        # val is cut from the training side only, by recording; test_size is rescaled because that side holds
        # only about 1 - 1/n_folds of the windows, so val is roughly val_fraction of all of them (GroupShuffleSplit
        # takes the fraction over recordings, not windows); seed + k gives each fold its own draw
        rel = val_fraction / (1.0 - 1.0 / n_folds)
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=rel, random_state=seed + k)
                    .split(tr_idx, y[tr_idx], groups[tr_idx]))
        folds.append(dict(protocol="participant", fold=k, train=np.sort(tr_idx[a]), val=np.sort(tr_idx[b]),
                          test=np.sort(te_idx)))
    return folds


def recording_folds(seeds=(0, 1, 2), split_dir=RECORDING_SPLITS):
    """Load the fixed recording-grouped splits of the original quantum benchmark, one per seed, from
    split_dir/split_seed<k>.csv (columns `part` in train/val/test and `window_index` into the full window set).
    Returns the same fold dicts as participant_folds so both protocols go through one evaluation loop."""
    folds = []
    for s in seeds:
        df = pd.read_csv(split_dir / f"split_seed{s}.csv")
        part = {p: np.sort(df.loc[df.part == p, "window_index"].to_numpy()) for p in ("train", "val", "test")}
        folds.append(dict(protocol="recording", fold=int(s), **part))
    return folds


def check_no_leakage(fold, groups, participants):
    """Raise if a recording (or, for the participant protocol, a participant) is on two sides."""
    tr = set(groups[fold["train"]]) | set(groups[fold["val"]])
    if tr & set(groups[fold["test"]]):
        raise AssertionError("recording on both sides")
    # val picks the checkpoint, so it must not share recordings with train either
    if set(groups[fold["train"]]) & set(groups[fold["val"]]):
        raise AssertionError("recording in train and val")
    if fold["protocol"] == "participant":
        trp = set(participants[fold["train"]]) | set(participants[fold["val"]])
        if trp & set(participants[fold["test"]]):
            raise AssertionError("participant on both sides")
    return True
