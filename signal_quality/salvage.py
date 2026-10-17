"""Export windows recovered by excluding persistently dead electrodes (and start-up windows).

Each participant's dead electrodes (flat or railed in >= 80% of their windows, the same
rule as the quality report) are removed, then every window is RE-ASSESSED by the real
rules on the remaining channels. Samples are exported unmodified (raw µV, float32);
excluded channels are NaN with a boolean mask, never zero-filled. 8- and 16-channel
boards are kept apart because their windows differ in shape.

Folder labels are exported as recorded. They are proxies confounded with recording
order (each class was recorded as one block); salvaging data does not fix that.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import tempfile
import warnings

import numpy as np

from .acquisition import file_sha256, read_openbci
from .assessment import QualityAssessor, QualityConfig
from .report import FindingPolicy, RecordingStats

ALIASES = {"backward": "backward", "backard": "backward", "forward": "forward", "foreward": "forward",
           "fowward": "forward", "landing": "landing", "land": "landing", "left": "left", "right": "right",
           "takeoff": "takeoff", "takoff": "takeoff"}
LABEL_WARNING = ("Folder labels are proxies without cue timing; in Data_clean every class was recorded as one "
                 "contiguous block, so labels are confounded with recording order. Evaluate on held-out "
                 "participants; recovered windows do not make the labels more trustworthy.")


@dataclass(frozen=True)
class SalvagePolicy:
    drop_startup_window: bool = True       # first window of a recording fails ~100% (start-up transient)
    max_excluded_fraction: float = 0.25    # skip participants needing more channels removed than this

    def __post_init__(self):
        if not 0 <= self.max_excluded_fraction < 1:
            raise ValueError("max_excluded_fraction must be in [0, 1)")


def folder_label(name):
    """Normalise an action-folder name ('B ackward', 'OpenBCISession_Left_3') to a class, or None."""
    compact = re.sub(r"[\s_-]+", "", name.lower())
    if compact.startswith("openbcisession"):
        compact = re.sub(r"\d+$", "", compact[len("openbcisession"):])
    return ALIASES.get(compact)


def _read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_openbci(path)


def _window(capture, a, b, keep=None):
    window = {k: v for k, v in capture.items() if k != "source_kind"}
    raw = capture["raw"][:, a:b]
    if keep is not None:
        raw = raw[keep]
        window["channels"] = [capture["channels"][i] for i in keep]
    window["raw"], window["counters"], window["timestamps"] = raw, capture["counters"][a:b], capture["timestamps"][a:b]
    return window


def salvage(replay_dir, out, policy=SalvagePolicy(), finding_policy=FindingPolicy()):
    replay_dir, out = Path(replay_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    summary = json.loads((replay_dir / "summary.json").read_text())
    config = QualityConfig(**summary["config"])
    if config.policy_id != summary["quality_policy_id"]:
        raise ValueError("Replay summary policy does not match its configuration")
    assessor = QualityAssessor(config)
    by_person = defaultdict(lambda: defaultdict(list))
    for line in (replay_dir / "windows.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_person[row["participant_id"]][row["source_path"]].append(row)

    boards = defaultdict(lambda: dict(X=[], mask=[], rows=[]))
    people, counts = {}, Counter()
    for person in sorted(by_person):
        sources = by_person[person]
        captures, stats = {}, RecordingStats()
        # Pass 1: full-montage assessment (must match the replay) -> this participant's dead electrodes.
        for source, rows in sources.items():
            if file_sha256(source) != rows[0]["source_sha256"]:
                raise ValueError(f"Source changed since the replay: {source}")
            capture = captures[source] = _read(source)
            for row in rows:
                a, b = row["source_start_sample"], row["source_end_sample_exclusive"]
                report = assessor.assess(**_window(capture, a, b))
                if report["raw_sha256"] != row["raw_sha256"] or report["usable"] != row["rule_usable"]:
                    raise ValueError(f"Re-assessment disagrees with the replay for {row['window_id']}")
                stats.add(report, finding_policy, first=a == 0)
        dead = stats.dead_electrodes(finding_policy)
        channels = next(iter(captures.values()))["channels"]
        board = f"{len(channels)}ch_{next(iter(captures.values()))['sfreq']:g}hz"
        entry = dict(board=board, windows=stats.windows, usable_full_montage=stats.usable,
                     dead_electrodes=dead, excluded=False, usable_after_exclusion=0, exported=0, recovered=0,
                     dropped_startup=0)
        people[person] = entry
        counts["windows"] += stats.windows
        counts["usable_full_montage"] += stats.usable
        if len(dead) > policy.max_excluded_fraction * len(channels):
            entry["excluded"] = True
            entry["reason"] = f"{len(dead)} of {len(channels)} channels dead (limit {policy.max_excluded_fraction:.0%})"
            counts["windows_in_excluded_participants"] += stats.windows
            continue
        keep = [i for i, name in enumerate(channels) if name not in dead]
        # Pass 2: re-assess on the remaining electrodes with the same rules.
        for source, rows in sources.items():
            capture = captures[source]
            if capture["channels"] != channels:
                raise ValueError(f"Participant {person} mixes channel layouts")
            for row in rows:
                a, b = row["source_start_sample"], row["source_end_sample_exclusive"]
                reduced = assessor.assess(**_window(capture, a, b, keep))
                if not reduced["usable"]:
                    continue
                counts["usable_after_exclusion"] += 1
                entry["usable_after_exclusion"] += 1
                if a == 0 and policy.drop_startup_window:
                    entry["dropped_startup"] += 1
                    counts["dropped_startup"] += 1
                    continue
                x = np.full((len(channels), b - a), np.nan, dtype=np.float32)
                x[keep] = capture["raw"][keep, a:b].astype(np.float32)
                mask = np.zeros(len(channels), dtype=bool)
                mask[keep] = True
                boards[board]["X"].append(x)
                boards[board]["mask"].append(mask)
                recovered = not row["rule_usable"]
                boards[board]["rows"].append(dict(
                    window_id=row["window_id"], participant_id=person, source_path=source,
                    source_sha256=row["source_sha256"], source_start_sample=a, source_end_sample_exclusive=b,
                    raw_sha256_full_montage=row["raw_sha256"], raw_sha256_exported_channels=reduced["raw_sha256"],
                    sfreq=capture["sfreq"], excluded_channels=dead, recovered=recovered,
                    action_folder=Path(source).parent.name, folder_label=folder_label(Path(source).parent.name)))
                entry["exported"] += 1
                entry["recovered"] += recovered
                counts["exported"] += 1
                counts["recovered"] += recovered
        # The quality report's salvage estimate assumes these same rules; the real re-assessment must agree.
        expected = round(stats.salvage_fraction(finding_policy) * stats.windows)
        if entry["usable_after_exclusion"] != expected:
            raise ValueError(f"{person}: re-assessment recovered {entry['usable_after_exclusion']} windows, "
                             f"report estimate {expected}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".salvage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "salvaged"
        stage.mkdir()
        board_info = {}
        for board, data in sorted(boards.items()):
            directory = stage / board
            directory.mkdir()
            X = np.stack(data["X"])
            masks = np.stack(data["mask"])
            np.save(directory / "X.npy", X)
            np.save(directory / "mask.npy", masks)
            with (directory / "windows.jsonl").open("w") as stream:
                for row in data["rows"]:
                    stream.write(json.dumps(row) + "\n")
            channels = [f"EXG Channel {i}" for i in range(X.shape[1])]
            common = [name for i, name in enumerate(channels) if masks[:, i].all()]
            board_info[board] = dict(windows=int(len(X)), shape=list(X.shape), channels=channels,
                                     common_channels=common,
                                     montage_options=montage_options(
                                         masks, np.array([r["participant_id"] for r in data["rows"]]), channels), participants=len({r["participant_id"] for r in data["rows"]}),
                                     recovered=int(sum(r["recovered"] for r in data["rows"])),
                                     label_counts=dict(Counter(r["folder_label"] for r in data["rows"])),
                                     X_sha256=file_sha256(directory / "X.npy"))
        result = dict(
            replay=str(replay_dir), replay_summary_sha256=file_sha256(replay_dir / "summary.json"),
            quality_policy_id=config.policy_id, salvage_policy=asdict(policy), finding_policy=asdict(finding_policy),
            counts=dict(counts), participants=people, boards=board_info,
            usable_pct_full_montage=round(100 * counts["usable_full_montage"] / counts["windows"], 1),
            usable_pct_after_exclusion=round(100 * counts["usable_after_exclusion"] / counts["windows"], 1),
            consistent_with_quality_report_estimate=True,
            samples="raw µV as recorded (float32); no filtering, re-referencing or normalisation",
            excluded_channels="NaN in X.npy and False in mask.npy; never zero-filled or interpolated",
            label_warning=LABEL_WARNING,
            loading="signal_quality.salvage.load_salvaged(<dir>, board, montage='common' | 'masked')")
        (stage / "summary.json").write_text(json.dumps(result, indent=1))
        (stage / "README.txt").write_text(
            "Salvaged EEG windows (signal_quality.salvage).\n\n"
            f"{result['counts'].get('exported', 0)} windows exported, {result['counts'].get('recovered', 0)} of them "
            "recovered by excluding dead electrodes.\nPer board folder: X.npy [windows, channels, samples] raw µV "
            "(excluded channels = NaN), mask.npy [windows, channels], windows.jsonl (provenance, folder labels).\n\n"
            "Load:  from signal_quality.salvage import load_salvaged\n"
            "       data = load_salvaged('<this folder>', '16ch_125hz', montage='common')\n\n"
            f"WARNING: {LABEL_WARNING}\n")
        stage.rename(out)
    return result


def montage_options(masks, participants, channels):
    """Best channel set for each montage size: windows/participants retained with every listed channel.

    Exact search over subsets of the channels that are missing anywhere (small in practice).
    """
    from itertools import combinations
    missing = [i for i in range(masks.shape[1]) if not masks[:, i].all()]
    if len(missing) > 16:
        raise ValueError("Too many partially missing channels for an exact montage search")
    best = {}
    for k in range(len(missing) + 1):
        for drop in combinations(missing, k):
            keep = [i for i in range(masks.shape[1]) if i not in drop]
            rows = masks[:, keep].all(axis=1)
            option = dict(channels_kept=len(keep), excluded_channels=[channels[i] for i in drop],
                          windows=int(rows.sum()), participants=int(len(set(participants[rows]))))
            if option["windows"] > best.get(len(keep), {}).get("windows", -1):
                best[len(keep)] = option
    return [best[n] for n in sorted(best, reverse=True)]


def load_salvaged(directory, board, montage="common"):
    """Load one board's salvaged windows.

    montage='common': only channels usable in EVERY exported window (no NaN) - a fixed model input.
    montage='masked': all channels; excluded ones are NaN, with the boolean mask returned.
    montage=[channel names]: exactly those channels, keeping only windows where all of them are usable
    (see ``montage_options`` in summary.json for the windows/participants each choice retains).
    """
    directory = Path(directory) / board
    X = np.load(directory / "X.npy", mmap_mode="r")
    mask = np.load(directory / "mask.npy")
    rows = [json.loads(line) for line in (directory / "windows.jsonl").read_text().splitlines() if line.strip()]
    channels = [f"EXG Channel {i}" for i in range(X.shape[1])]
    meta = dict(window_ids=[r["window_id"] for r in rows], participants=np.array([r["participant_id"] for r in rows]),
                labels=np.array([r["folder_label"] or "" for r in rows]), recovered=np.array([r["recovered"] for r in rows]),
                sfreq=rows[0]["sfreq"] if rows else None, label_warning=LABEL_WARNING)
    if isinstance(montage, str) and montage == "masked":
        return dict(X=X, mask=mask, channels=channels, **meta)
    if isinstance(montage, str):
        if montage != "common":
            raise ValueError("montage must be 'common', 'masked' or a list of channel names")
        keep = np.flatnonzero(mask.all(axis=0))
    else:
        unknown = set(montage) - set(channels)
        if unknown:
            raise ValueError(f"Unknown channels: {sorted(unknown)}")
        keep = np.array([channels.index(name) for name in montage])
    rows = np.flatnonzero(mask[:, keep].all(axis=1))
    selected = {k: (v[rows] if isinstance(v, np.ndarray) else [v[i] for i in rows] if isinstance(v, list) else v)
                for k, v in meta.items()}
    return dict(X=np.asarray(X[rows][:, keep, :]), mask=None, channels=[channels[i] for i in keep], **selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--keep-startup", action="store_true", help="Keep first windows that pass the rules")
    parser.add_argument("--max-excluded-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)
    result = salvage(args.replay, args.out, SalvagePolicy(drop_startup_window=not args.keep_startup,
                                                          max_excluded_fraction=args.max_excluded_fraction))
    c = result["counts"]
    print(f"Usable windows: {result['usable_pct_full_montage']}% with all electrodes -> "
          f"{result['usable_pct_after_exclusion']}% after excluding dead electrodes")
    print(f"Exported {c.get('exported', 0)} windows ({c.get('recovered', 0)} recovered); dropped "
          f"{c.get('dropped_startup', 0)} start-up windows; {c.get('windows_in_excluded_participants', 0)} windows "
          "from participants with too many dead electrodes")
    for board, info in result["boards"].items():
        print(f"  {board}: {info['windows']} windows, {info['participants']} participants, "
              f"common montage {len(info['common_channels'])}/{len(info['channels'])} channels")
        for option in info["montage_options"]:
            dropped = ", ".join(c.replace("EXG Channel ", "Ch ") for c in option["excluded_channels"]) or "none"
            print(f"     {option['channels_kept']:2d} channels: {option['windows']:5d} windows, "
                  f"{option['participants']:2d} participants (drop {dropped})")
    skipped = [p for p, e in result["participants"].items() if e["excluded"]]
    if skipped:
        print("  excluded participants: " + "; ".join(f"{p} ({result['participants'][p]['reason']})" for p in skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
