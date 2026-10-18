"""Verified EEG loading and cleaning shared by the GaussianNB pipelines (PyTorch, TensorFlow, JAX).

What it fixes compared with the per-notebook loaders:

1. Channels and sampling rate. Only named OpenBCI GUI recordings are read, through
   signal_quality.acquisition.read_openbci, which selects the EEG channels by header name and
   reports the real sampling rate. The old loaders took 16 columns after the sample index, so
   8-channel recordings (G01/person01-07, 250 Hz) contributed accelerometer and digital-pin
   columns as "EEG", and were analysed as if sampled at 125 Hz. Headerless BrainFlow exports
   (whose channels 9-16 are all zero and whose rate is not recorded) are skipped. One board
   configuration is used per run (LoadConfig.board).
2. Labels. Action folders are normalised with signal_quality.salvage.folder_label, so folders
   such as "TakOff", "Take off", "OpenBCISession_take_off_2", "Foreward" and "backard" are
   labelled instead of dropped.
3. Signal quality. Each recording is assessed in non-overlapping 2 s blocks with the deployed
   signal_quality policy, on the channels kept. The start-up block and every failing block are
   dropped; consecutive passing blocks with continuous packet counters form a segment.
4. Filtering. Each segment is band-passed (zero-phase Butterworth) and notched at the mains
   frequency. Filtering per segment keeps artifacts in rejected blocks from ringing into kept data.

Recorded sessions (prediction_eegnet/record_session.py) are read too: every trial becomes one
labelled item, using the label of its cue marker, and only the task period from the cue to the
end marker is kept. It goes through the same quality checks and filtering.

Windows must be cut within segments (segment_id), and data split by file or participant.
"""

import csv
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signal_quality.acquisition import read_openbci  # noqa: E402
from signal_quality.assessment import QualityAssessor  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402
from signal_quality.salvage import folder_label  # noqa: E402

BOARDS = {"16ch_125hz": (16, 125.0), "8ch_250hz": (8, 250.0),
          "synthetic_16ch_250hz": (16, 250.0)}  # BrainFlow synthetic board: dry-run sessions only
LABELS = ("backward", "forward", "landing", "left", "right", "takeoff")
SESSION_BOARDS = {"cyton_daisy": "16ch_125hz", "cyton": "8ch_250hz", "synthetic": "synthetic_16ch_250hz"}


@dataclass(frozen=True)
class LoadConfig:
    # One board configuration per run. G01/person01-07 used the 8-channel board, everyone else the
    # 16-channel board; mixing them would mix montages and sampling rates.
    board: str = "16ch_125hz"
    # Channel 15 was dead (flat or railed) for 6 participants (signal_quality report). Excluding it
    # keeps 18 participants usable instead of 14 when every kept channel must pass.
    exclude_channels: tuple = ("EXG Channel 15",)
    quality_filter: bool = True        # drop failing 2 s blocks (always drops the start-up block)
    bandpass_hz: tuple = (1.0, 40.0)   # None to disable
    notch_hz: float = 60.0             # mains frequency verified from the recordings; None to disable
    filter_order: int = 4

    def __post_init__(self):
        if self.board not in BOARDS:
            raise ValueError(f"board must be one of {sorted(BOARDS)}")

    @property
    def sfreq(self):
        return BOARDS[self.board][1]

    @property
    def channels(self):
        n = BOARDS[self.board][0]
        return [f"EXG Channel {i}" for i in range(n) if f"EXG Channel {i}" not in self.exclude_channels]

    def as_dict(self):
        return asdict(self) | {"sfreq": self.sfreq, "channels": self.channels}


def _participant(path):
    parts = Path(path).parts
    group = next((p for p in parts if p.upper().startswith("G") and p[1:].isdigit()), "G??")
    person = next((p for p in parts if p.lower().startswith("person")), "person??")
    return f"{group.upper()}/{person.lower()}"


def _design_filters(config):
    fs = config.sfreq
    sos = None
    if config.bandpass_hz is not None:
        lo, hi = config.bandpass_hz
        sos = butter(config.filter_order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    notch = None
    if config.notch_hz is not None and config.notch_hz < fs / 2:
        notch = iirnotch(config.notch_hz, Q=30.0, fs=fs)
    return sos, notch


def _filter(x, config):
    """Zero-phase band-pass and notch along the time axis of x (samples, channels)."""
    sos, notch = _design_filters(config)
    y = x.astype(np.float64)
    y = y - y.mean(axis=0, keepdims=True)
    if sos is not None:
        y = sosfiltfilt(sos, y, axis=0)
    if notch is not None:
        y = filtfilt(notch[0], notch[1], y, axis=0)
    return y.astype(np.float32)


def load_recording(path, config=LoadConfig()):
    """One recording -> dict with kept, filtered segments (samples, channels); raises if unsupported."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cap = read_openbci(path)
    n_ch, fs = BOARDS[config.board]
    if (len(cap["channels"]), cap["sfreq"]) != (n_ch, fs):
        raise ValueError(f"recording is {len(cap['channels'])} ch at {cap['sfreq']} Hz, not {config.board}")
    keep = [cap["channels"].index(c) for c in config.channels]
    raw = cap["raw"][keep]                                   # (channels, samples), raw uV
    block = int(round(2.0 * fs))
    n_blocks = raw.shape[1] // block
    assessor = QualityAssessor(load_default_config())
    usable = np.zeros(n_blocks, dtype=bool)
    reasons = []
    for b in range(n_blocks):
        a, e = b * block, (b + 1) * block
        if b == 0:          # start-up transient: fails in 99.9% of recordings
            reasons.append(["startup_block"])
            continue
        if not config.quality_filter:
            usable[b] = True
            reasons.append([])
            continue
        report = assessor.assess(raw[:, a:e], sfreq=fs, channels=config.channels, units="uV",
                                 counters=cap["counters"][a:e], timestamps=cap["timestamps"][a:e],
                                 counter_step=cap["counter_step"], timestamp_kind=cap["timestamp_kind"])
        usable[b] = bool(report["usable"])
        reasons.append(report["reasons"])

    # Runs of consecutive usable blocks; split a run if packets were lost at a block boundary.
    segments, run = [], []
    max_gap = assessor.config.max_arrival_gap_seconds
    for b in range(n_blocks):
        if usable[b] and run:
            a = b * block
            step = (cap["counters"][a] - cap["counters"][a - 1]) % 256
            gap = cap["timestamps"][a] - cap["timestamps"][a - 1]
            if step != cap["counter_step"] or not 0 <= gap <= max_gap:
                segments.append(run)
                run = []
        if usable[b]:
            run.append(b)
        elif run:
            segments.append(run)
            run = []
    if run:
        segments.append(run)
    data = [_filter(raw[:, r[0] * block:(r[-1] + 1) * block].T, config) for r in segments]
    return dict(path=str(path), participant=_participant(path), label=folder_label(Path(path).parent.name),
                sfreq=fs, channels=list(config.channels), segments=data,
                n_blocks=int(n_blocks), n_blocks_kept=int(usable.sum()), block_reasons=reasons)


def _infer_counter_step(counters):
    """BrainFlow package-counter step (1 or 2) from the recording itself."""
    steps, counts = np.unique(np.mod(np.diff(counters), 256), return_counts=True)
    step = int(steps[np.argmax(counts)])
    if step not in (1, 2) or counts.max() < 0.9 * counts.sum():
        raise ValueError("Package counter does not advance regularly; continuity cannot be checked")
    return step


def load_session(directory, config=LoadConfig()):
    """One recorded session -> one item per trial with the cue-to-end task period as segments.

    Quality is assessed in 2 s blocks aligned to the cue: one block before the cue (filter context)
    and as many whole blocks as fit before the end marker. Passing blocks next to each other are
    filtered together; only the part from the cue onwards is returned.
    """
    directory = Path(directory)
    session = json.loads((directory / "session.json").read_text())
    if session.get("problems"):
        raise ValueError(f"marker check failed during recording: {session['problems']}")
    board = SESSION_BOARDS.get(session.get("board"))
    if board != config.board:
        raise ValueError(f"session board {session.get('board')!r} does not match LoadConfig.board {config.board!r}")
    data = np.load(directory / "brainflow_raw.npy")
    with (directory / "events.csv").open() as stream:
        events = list(csv.DictReader(stream))
    fs = float(session["sampling_rate"])
    channels = [f"EXG Channel {i}" for i in range(len(session["eeg_rows"]))]
    keep = [channels.index(c) for c in config.channels]
    raw = data[session["eeg_rows"]][keep]
    counters, timestamps = data[session["package_num_row"]], data[session["timestamp_row"]]
    step = _infer_counter_step(counters)
    assessor = QualityAssessor(load_default_config())
    block = int(round(2.0 * fs))

    def usable(a):
        if a < 0 or a + block > raw.shape[1]:
            return False, ["outside_recording"]
        if not config.quality_filter:
            return True, []
        report = assessor.assess(raw[:, a:a + block], sfreq=fs, channels=config.channels, units="uV",
                                 counters=counters[a:a + block], timestamps=timestamps[a:a + block],
                                 counter_step=step, timestamp_kind="arrival")
        return bool(report["usable"]), report["reasons"]

    trials = {}
    for e in events:
        trials.setdefault(e["trial_id"], {})[e["kind"]] = dict(e, sample=int(e["sample"]))
    items = []
    for trial_id, ev in trials.items():
        if set(ev) != {"baseline", "cue", "end"} or not ev["baseline"]["sample"] < ev["cue"]["sample"] < ev["end"]["sample"]:
            continue
        cue, end, label = ev["cue"]["sample"], ev["end"]["sample"], ev["cue"]["label"]
        n_task = (end - cue) // block
        starts = [cue - block] + [cue + k * block for k in range(n_task)]
        flags = [usable(a) for a in starts]
        # runs of passing blocks; keep the part of each run from the cue onwards
        segments, run = [], []
        for a, (ok, _) in zip(starts + [None], flags + [(False, [])]):
            if ok:
                run.append(a)
                continue
            if run and run[-1] >= cue:
                x = _filter(raw[:, run[0]:run[-1] + block].T, config)
                segments.append(x[max(0, cue - run[0]):])
            run = []
        items.append(dict(path=f"{directory}#{trial_id}", participant=session["participant_id"],
                          label=label if label in LABELS else None,
                          sfreq=fs, channels=list(config.channels), segments=segments,
                          n_blocks=int(n_task), n_blocks_kept=int(sum(ok for ok, _ in flags[1:])),
                          block_reasons=[r for _, r in flags[1:]], session=str(directory), trial_id=trial_id))
    return items


def load_item(path, config=LoadConfig()):
    """A recording file, or one session trial given as "<session folder>#<trial id>"."""
    path = str(path)
    if "#" in path and (Path(path.split("#", 1)[0]) / "session.json").exists():
        directory, trial_id = path.split("#", 1)
        match = [i for i in load_session(directory, config) if i["trial_id"] == trial_id]
        if not match:
            raise ValueError(f"trial {trial_id!r} not found in {directory}")
        return match[0]
    return load_recording(path, config)


def _try_load(path, config):
    try:
        return load_recording(path, config)
    except (ValueError, OSError, StopIteration) as error:
        return dict(path=str(path), error=str(error))


def _is_named_openbci(path):
    with open(path, errors="ignore") as stream:
        return stream.read(21) == "%OpenBCI Raw EXG Data"


def load_recordings(root, config=LoadConfig(), n_jobs=-1, verbose=True):
    """Load every supported recording under root. Returns (recordings, skipped)."""
    paths = sorted(p for p in Path(root).rglob("*.txt") if _is_named_openbci(p))
    results = Parallel(n_jobs=n_jobs)(delayed(_try_load)(p, config) for p in paths)
    # Cued sessions recorded with prediction_eegnet/record_session.py: one item per trial
    session_dirs = sorted(p.parent for p in Path(root).rglob("session.json")
                          if (p.parent / "brainflow_raw.npy").exists() and (p.parent / "events.csv").exists())
    for d in session_dirs:
        try:
            results.extend(load_session(d, config))
        except (ValueError, OSError, KeyError) as error:
            results.append(dict(path=str(d), error=str(error)))
    recordings = [r for r in results if "error" not in r and r["label"] is not None]
    skipped = [r for r in results if "error" in r or r.get("label") is None]
    if verbose:
        n_files = sum(1 for p in Path(root).rglob("*") if p.is_file())
        blocks = sum(r["n_blocks"] for r in recordings)
        kept = sum(r["n_blocks_kept"] for r in recordings)
        with_data = sum(1 for r in recordings if r["segments"])
        print(f"Files under root: {n_files}; named OpenBCI recordings: {len(paths)}; "
              f"recorded sessions: {len(session_dirs)} ({sum(1 for r in recordings if 'trial_id' in r)} trials)")
        print(f"Board {config.board}: {len(recordings)} recordings loaded, {len(skipped)} skipped "
              f"(other board or unreadable); headerless exports are not read")
        print(f"Channels used ({len(config.channels)}): excluded {list(config.exclude_channels) or 'none'}")
        print(f"2 s blocks kept: {kept}/{blocks} ({100 * kept / max(blocks, 1):.1f}%), "
              f"recordings with usable data: {with_data}/{len(recordings)}")
        per_person = {}
        for r in recordings:
            k, n = per_person.get(r["participant"], (0, 0))
            per_person[r["participant"]] = (k + r["n_blocks_kept"], n + r["n_blocks"])
        empty = sorted(p for p, (k, _) in per_person.items() if k == 0)
        print(f"Participants with usable data: {len(per_person) - len(empty)}/{len(per_person)}"
              + (f" (none usable: {', '.join(empty)})" if empty else ""))
    return recordings, skipped


def recordings_to_frame(recordings):
    """Long table: one row per kept sample, with src_filename, segment_id, participant_id, label_txt."""
    frames = []
    seg_id = 0
    for r in recordings:
        for seg in r["segments"]:
            df = pd.DataFrame(seg, columns=r["channels"])
            df["src_filename"] = r["path"]
            df["segment_id"] = seg_id
            df["participant_id"] = r["participant"]
            df["label_txt"] = r["label"]
            frames.append(df)
            seg_id += 1
    if not frames:
        raise ValueError("No usable EEG after loading and quality filtering.")
    out = pd.concat(frames, ignore_index=True)
    for col in ("src_filename", "participant_id", "label_txt"):
        out[col] = out[col].astype("category")
    out["segment_id"] = out["segment_id"].astype(np.int32)
    return out
