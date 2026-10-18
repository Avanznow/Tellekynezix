"""Explicit physical BrainFlow/OpenBCI adapters, independent of task labels."""

import csv
import hashlib
from pathlib import Path
import re

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def brainflow_capture(data, descriptor, *, real, counter_step):
    data = np.asarray(data)
    eeg = descriptor.get("eeg_channels")
    rows = descriptor.get("num_rows")
    if (data.ndim != 2 or data.shape[0] != rows or not isinstance(eeg, list) or not eeg
            or len(set(eeg)) != len(eeg) or any(not isinstance(i, int) or not 0 <= i < rows for i in eeg)):
        raise ValueError("Unverified BrainFlow physical channel mapping")
    counter = descriptor.get("package_num_channel")
    timestamp = descriptor.get("timestamp_channel")
    if (not isinstance(counter, int) or not isinstance(timestamp, int)
            or counter in eeg or timestamp in eeg or counter == timestamp
            or not 0 <= counter < rows or not 0 <= timestamp < rows):
        raise ValueError("Missing/overlapping BrainFlow timing/counter channels")
    return dict(raw=data[eeg].copy(), sfreq=descriptor["sampling_rate"],
                channels=[f"EEG_{i}" for i in eeg], units="uV",
                counters=data[counter].copy(), timestamps=data[timestamp].copy(),
                counter_step=counter_step, timestamp_kind="arrival",
                source_kind="real" if real else "synthetic")


def read_openbci(path):
    """Select named physical electrodes; headerless/ambiguous exports fail."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        preamble = [stream.readline().strip() for _ in range(4)]
        names = [name.strip() for name in next(csv.reader([stream.readline()]), [])]
    count = re.fullmatch(r"%Number of channels = (\d+)", preamble[1])
    rate = re.fullmatch(r"%Sample Rate = ([0-9.]+) Hz", preamble[2])
    if preamble[0] != "%OpenBCI Raw EXG Data" or count is None or rate is None:
        raise ValueError("Requires a named OpenBCI physical acquisition header")
    c, fs = int(count[1]), float(rate[1])
    expected = { (16, 125.0): "%Board = OpenBCI_GUI$BoardCytonSerialDaisy",
                 (8, 250.0): "%Board = OpenBCI_GUI$BoardCytonSerial" }
    if expected.get((c, fs)) != preamble[3]:
        raise ValueError("Unsupported/unreviewed board/channel/rate combination")
    channels = [f"EXG Channel {i}" for i in range(c)]
    required = ["Sample Index", *channels, "Timestamp"]
    if any(names.count(name) != 1 for name in required):
        raise ValueError("Ambiguous/missing EEG or timing columns")
    values = np.loadtxt(path, delimiter=",", skiprows=5, usecols=[names.index(name) for name in required], ndmin=2)
    if not len(values):
        raise ValueError("Empty recording")
    return dict(raw=values[:, 1:1+c].T, sfreq=fs, channels=channels, units="uV",
                counters=values[:, 0], timestamps=values[:, -1], counter_step=2 if c == 16 else 1,
                timestamp_kind="arrival", source_kind="real")
