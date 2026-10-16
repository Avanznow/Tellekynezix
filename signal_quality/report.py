"""Actionable electrode-level quality reports for datasets and single recordings.

``dataset`` turns a published replay into per-participant, per-recording and
per-electrode tables plus a self-contained HTML dashboard. It re-assesses every
replayed window from its source file with the replay's own policy, verifies the
decisions match the replay exactly, and adds which electrodes failed and why.

``preflight`` checks one new recording in seconds, so faults are caught during
a collection session instead of after the dataset is assembled.

Findings name LIKELY causes and checks to perform. They are engineering
heuristics over rule flags, not verified hardware diagnoses or artifact labels.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import re
import sys
import tempfile
import warnings

from .acquisition import file_sha256, read_openbci
from .assessment import QualityAssessor, QualityConfig

ELECTRODE_FAULTS = ("flat_channel", "adc_saturation")
ARTIFACTS = ("excessive_amplitude", "excessive_sample_jump")


@dataclass(frozen=True)
class FindingPolicy:
    """Thresholds that turn rule flags into findings. Heuristics, fixed before reporting."""
    electrode_fault_fraction: float = 0.8   # one electrode flat/railed in >= 80% of windows
    global_rail_channel_fraction: float = 0.75  # >= 75% of electrodes railed together ...
    global_rail_window_fraction: float = 0.5    # ... in >= 50% of windows
    dropout_window_fraction: float = 0.2    # packet loss in >= 20% of windows
    artifact_window_fraction: float = 0.3   # amplitude/jump flags in >= 30% of windows
    artifact_channel_fraction: float = 0.5  # ... on >= 50% of electrodes at once
    startup_first_fraction: float = 0.8     # first window fails in >= 80% of recordings ...
    startup_excess: float = 0.3             # ... and >= 30 points more often than later windows
    systematic_participants: int = 3        # same dead electrode in >= 3 participants
    good_usable_fraction: float = 0.7
    unusable_below_fraction: float = 0.3


def verdict(usable_fraction, policy=FindingPolicy()):
    if usable_fraction >= policy.good_usable_fraction:
        return "good"
    if usable_fraction < policy.unusable_below_fraction:
        return "unusable"
    return "review"


class RecordingStats:
    """Window and per-electrode flag counts for one recording (or a pooled set)."""

    def __init__(self):
        self.windows = self.usable = 0
        self.reasons = Counter()
        self.channel_windows = Counter()                 # windows in which each electrode exists
        self.channel_flags = defaultdict(Counter)        # electrode -> reason -> windows flagged
        self.channel_flagged = Counter()                 # electrode -> windows flagged for any reason
        self.global_rail_windows = self.artifact_spread_windows = 0
        self.boards = Counter()
        self.position = Counter()                         # ("first"|"later", "windows"|"failed")
        # Start-up windows are reported separately, so dropout/movement findings use later windows only.
        self.later_reasons = Counter()
        self.later_artifact_spread = 0
        self.later_channel_windows = Counter()
        self.later_channel_flagged = Counter()
        self.window_flags = []                            # (flagged electrodes, capture-level fault) per window

    def add(self, report, policy, *, first=False):
        self.windows += 1
        where = "first" if first else "later"
        self.position[where, "windows"] += 1
        self.position[where, "failed"] += not report["usable"]
        flagged = frozenset(name for name, flags in report["channel_reasons"].items() if flags)
        channel_level = {flag for flags in report["channel_reasons"].values() for flag in flags}
        self.window_flags.append((flagged, any(r not in channel_level for r in report["reasons"])))
        self.usable += report["usable"]
        self.reasons.update(report["reasons"])
        channels = report["channels"]
        railed = 0
        artifact_channels = 0
        for name, flags in report["channel_reasons"].items():
            self.channel_windows[name] += 1
            if flags:
                self.channel_flagged[name] += 1
                self.channel_flags[name].update(flags)
            railed += "adc_saturation" in flags
            artifact_channels += any(flag in ARTIFACTS for flag in flags)
        self.global_rail_windows += railed >= policy.global_rail_channel_fraction * len(channels)
        spread = artifact_channels >= policy.artifact_channel_fraction * len(channels)
        self.artifact_spread_windows += spread
        if not first:
            self.later_reasons.update(report["reasons"])
            self.later_artifact_spread += spread
            for name, flags in report["channel_reasons"].items():
                self.later_channel_windows[name] += 1
                self.later_channel_flagged[name] += bool(flags)
        self.boards[f"{len(channels)}ch@{report['sfreq']:g}Hz"] += 1

    def merge(self, other):
        for name in ("windows", "usable", "global_rail_windows", "artifact_spread_windows"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.reasons.update(other.reasons)
        self.channel_windows.update(other.channel_windows)
        self.channel_flagged.update(other.channel_flagged)
        self.boards.update(other.boards)
        self.position.update(other.position)
        self.later_reasons.update(other.later_reasons)
        self.later_artifact_spread += other.later_artifact_spread
        self.later_channel_windows.update(other.later_channel_windows)
        self.later_channel_flagged.update(other.later_channel_flagged)
        self.window_flags.extend(other.window_flags)
        for name, flags in other.channel_flags.items():
            self.channel_flags[name].update(flags)

    @property
    def usable_fraction(self):
        return self.usable / self.windows if self.windows else 0.0

    def channel_fraction_after_startup(self, name):
        """Share of a channel's windows flagged, excluding each recording's start-up window."""
        total = self.later_channel_windows[name]
        return self.later_channel_flagged[name] / total if total else None

    def failure_rate(self, where):
        n = self.position[where, "windows"]
        return self.position[where, "failed"] / n if n else 0.0

    def dead_electrodes(self, policy):
        """Electrodes flat/railed in most windows: persistent contact or hardware faults."""
        return [name for name in sorted(self.channel_windows, key=_channel_order)
                if self.channel_fraction(name, ELECTRODE_FAULTS) >= policy.electrode_fault_fraction]

    def salvage_fraction(self, policy):
        """Share of windows that would pass if ONLY the persistently dead electrodes were excluded.

        An estimate of recoverable data, not a recommendation to drop channels silently:
        models must then be trained on the reduced montage (or the channel interpolated).
        """
        dead = set(self.dead_electrodes(policy))
        if not self.windows:
            return 0.0
        return sum(not capture_fault and flagged <= dead for flagged, capture_fault in self.window_flags) / self.windows

    def channel_fraction(self, name, reasons=None):
        total = self.channel_windows[name]
        if not total:
            return 0.0
        flagged = (self.channel_flagged[name] if reasons is None
                   else max(self.channel_flags[name][r] for r in reasons))
        return flagged / total

    def findings(self, policy):
        """Likely causes with the evidence that triggered them; empty when nothing stands out."""
        found = []
        if not self.windows:
            return found
        fraction = lambda count: count / self.windows  # noqa: E731
        global_rail = fraction(self.global_rail_windows) >= policy.global_rail_window_fraction
        if global_rail:
            found.append(dict(
                kind="reference_or_bias", severity="high",
                evidence=f"most electrodes at the ADC rail together in {100 * fraction(self.global_rail_windows):.0f}% of windows",
                action="Check the reference (SRB) and bias electrodes and their contact first, then the board gain setting; "
                       "when every channel rails together a shared connection is the usual cause."))
        dead = []
        for name in sorted(self.channel_windows, key=_channel_order):
            share = self.channel_fraction(name, ELECTRODE_FAULTS)
            if share >= policy.electrode_fault_fraction and not global_rail:
                top = max(ELECTRODE_FAULTS, key=lambda r: self.channel_flags[name][r])
                dead.append((name, share, top))
        for name, share, top in dead:
            flat, rail = (self.channel_fraction(name, (r,)) for r in ELECTRODE_FAULTS)
            what = ("stuck at the ADC rail (no signal)" if min(flat, rail) >= policy.electrode_fault_fraction
                    else "flat (no signal)" if top == "flat_channel" else "saturated at the ADC rail")
            found.append(dict(
                kind="electrode_contact", severity="high", channel=name,
                evidence=f"{name} {what} in {100 * share:.0f}% of its windows",
                action=f"Re-seat / re-gel {name}, check its lead and pin, and confirm the channel is enabled in the GUI."))
        first, later = self.failure_rate("first"), self.failure_rate("later")
        if (self.position["first", "windows"] >= 3 and first >= policy.startup_first_fraction
                and first - later >= policy.startup_excess):
            found.append(dict(
                kind="startup_transient", severity="medium",
                evidence=f"first 2 s window of a recording fails {100 * first:.0f}% of the time vs {100 * later:.0f}% for later windows",
                action="Discard the first seconds of every recording (or start saving only after the stream has settled)."))
        later_windows = self.position["later", "windows"]
        later_fraction = lambda count: count / later_windows if later_windows else 0.0  # noqa: E731
        drops = self.later_reasons["packet_discontinuity"] + self.later_reasons["acquisition_gap"]
        if later_fraction(drops) >= policy.dropout_window_fraction:
            found.append(dict(
                kind="wireless_dropouts", severity="medium",
                evidence=f"packet loss or acquisition gaps in {100 * later_fraction(drops):.0f}% of windows after start-up",
                action="Move the USB dongle closer (USB extension), check the battery, and reduce 2.4 GHz interference."))
        if (later_fraction(self.later_artifact_spread) >= policy.artifact_window_fraction
                and not global_rail and len(dead) < len(self.channel_windows) / 2):
            found.append(dict(
                kind="movement_or_muscle", severity="medium",
                evidence=f"large amplitude/jumps on most electrodes in {100 * later_fraction(self.later_artifact_spread):.0f}% of windows after start-up",
                action="Likely movement, jaw/muscle activity or cable motion: secure cables and ask the participant to stay relaxed and still."))
        if len(self.boards) > 1:
            found.append(dict(
                kind="mixed_hardware", severity="medium",
                evidence="recordings from different board configurations: " + ", ".join(sorted(self.boards)),
                action="Use one board configuration (e.g. 16-channel Cyton+Daisy at 125 Hz) for every session of a study."))
        return found


def _channel_order(name):
    match = re.search(r"(\d+)$", name)
    return (int(match[1]) if match else 10**6, name)


def assess_capture(capture, assessor, policy, on_window=None):
    """Assess consecutive non-overlapping windows exactly like ``replay`` does."""
    n = int(round(assessor.config.min_seconds * capture["sfreq"]))
    stats = RecordingStats()
    for start in range(0, capture["raw"].shape[1] - n + 1, n):
        window = {key: value for key, value in capture.items() if key != "source_kind"}
        window["raw"] = capture["raw"][:, start:start + n]
        window["counters"] = capture["counters"][start:start + n]
        window["timestamps"] = capture["timestamps"][start:start + n]
        report = assessor.assess(**window)
        stats.add(report, policy, first=start == 0)
        if on_window is not None:
            on_window(start, report)
    return stats


def _read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_openbci(path)


def preflight(path, config=None, policy=FindingPolicy()):
    """Check one recording; returns a JSON-safe dict (``passed`` is the overall decision)."""
    assessor = QualityAssessor(config)
    capture = _read(path)
    stats = assess_capture(capture, assessor, policy)
    findings = stats.findings(policy)
    channels = [dict(channel=name, flagged_fraction=stats.channel_fraction(name),
                     flagged_fraction_after_startup=stats.channel_fraction_after_startup(name),
                     reasons=dict(stats.channel_flags[name]))
                for name in capture["channels"]]
    return dict(path=str(Path(path).resolve()), source_sha256=file_sha256(path),
                board=f"{len(capture['channels'])}ch@{capture['sfreq']:g}Hz",
                seconds=capture["raw"].shape[1] / capture["sfreq"], windows=stats.windows,
                usable_windows=stats.usable, usable_fraction=stats.usable_fraction,
                verdict=verdict(stats.usable_fraction, policy) if stats.windows else "too_short",
                dead_electrodes=stats.dead_electrodes(policy),
                first_window_failed=bool(stats.position["first", "failed"]),
                usable_fraction_if_dead_excluded=stats.salvage_fraction(policy),
                passed=bool(stats.windows) and verdict(stats.usable_fraction, policy) == "good",
                reasons=dict(stats.reasons), channels=channels, findings=findings,
                quality_policy_id=assessor.config.policy_id, finding_policy=asdict(policy),
                interpretation="Rule flags and heuristic findings; not a verified hardware diagnosis")


def build_dataset_report(replay_dir, out, policy=FindingPolicy()):
    """Write participants/recordings/channels CSVs, findings.json, summary.json and report.html."""
    replay_dir, out = Path(replay_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    summary = json.loads((replay_dir / "summary.json").read_text())
    config = QualityConfig(**summary["config"])
    if config.policy_id != summary["quality_policy_id"]:
        raise ValueError("Replay summary policy does not match its configuration")
    assessor = QualityAssessor(config)
    by_source = defaultdict(list)
    with (replay_dir / "windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_source[row["source_path"]].append(row)

    recordings, participants = [], defaultdict(RecordingStats)
    dataset = RecordingStats()
    for source, rows in sorted(by_source.items()):
        digest = file_sha256(source)
        if any(row["source_sha256"] != digest for row in rows):
            raise ValueError(f"Source changed since the replay: {source}")
        expected = {row["source_start_sample"]: row for row in rows}

        def check(start, report, expected=expected, source=source):
            row = expected.pop(start, None)
            if (row is None or row["rule_usable"] != report["usable"]
                    or row["rule_reasons"] != report["reasons"] or row["raw_sha256"] != report["raw_sha256"]):
                raise ValueError(f"Re-assessment disagrees with the replay at {source} sample {start}")

        stats = assess_capture(_read(source), assessor, policy, on_window=check)
        if expected:
            raise ValueError(f"Replay windows missing from re-assessment: {source}")
        person = rows[0]["participant_id"]
        participants[person].merge(stats)
        dataset.merge(stats)
        worst = sorted(stats.channel_windows, key=lambda c: (-stats.channel_fraction(c), _channel_order(c)))
        recordings.append(dict(
            participant_id=person, action_folder=Path(source).parent.name, file=Path(source).name,
            board=next(iter(stats.boards)), windows=stats.windows, usable_windows=stats.usable,
            usable_pct=round(100 * stats.usable_fraction, 1), verdict=verdict(stats.usable_fraction, policy),
            top_reasons="; ".join(f"{r} ({c})" for r, c in stats.reasons.most_common(3)),
            worst_electrodes="; ".join(f"{c} {100 * stats.channel_fraction(c):.0f}%"
                                       for c in worst[:3] if stats.channel_fraction(c) > 0),
            findings="; ".join(f["kind"] + (f":{f['channel']}" if "channel" in f else "")
                               for f in stats.findings(policy)),
            source_path=source, source_sha256=digest))

    participant_rows, channel_rows, findings = [], [], {}
    for person in sorted(participants):
        stats = participants[person]
        # The start-up transient is reported once for the whole dataset, not per participant.
        findings[person] = [f for f in stats.findings(policy) if f["kind"] != "startup_transient"]
        participant_rows.append(dict(
            participant_id=person, recordings=sum(r["participant_id"] == person for r in recordings),
            windows=stats.windows, usable_windows=stats.usable, usable_pct=round(100 * stats.usable_fraction, 1),
            verdict=verdict(stats.usable_fraction, policy),
            dead_electrodes="; ".join(stats.dead_electrodes(policy)),
            usable_pct_if_dead_excluded=round(100 * stats.salvage_fraction(policy), 1),
            first_window_fail_pct=round(100 * stats.failure_rate("first"), 1),
            later_window_fail_pct=round(100 * stats.failure_rate("later"), 1),
            boards="; ".join(sorted(stats.boards)),
            top_reasons="; ".join(f"{r} ({c})" for r, c in stats.reasons.most_common(3)),
            findings="; ".join(f["kind"] + (f":{f['channel']}" if "channel" in f else "") for f in findings[person])))
        for name in sorted(stats.channel_windows, key=_channel_order):
            flags = stats.channel_flags[name]
            channel_rows.append(dict(
                participant_id=person, channel=name, windows=stats.channel_windows[name],
                flagged_windows=stats.channel_flagged[name],
                flagged_pct=round(100 * stats.channel_fraction(name), 1),
                **{f"{reason}_pct": round(100 * flags[reason] / stats.channel_windows[name], 1)
                   for reason in (*ELECTRODE_FAULTS, *ARTIFACTS)}))

    dead_in = defaultdict(list)
    for person in sorted(participants):
        for name in participants[person].dead_electrodes(policy):
            dead_in[name].append(person)
    dataset_findings = [f for f in dataset.findings(policy) if f["kind"] in ("startup_transient", "mixed_hardware")]
    for name, people in sorted(dead_in.items(), key=lambda kv: (-len(kv[1]), _channel_order(kv[0]))):
        if len(people) >= policy.systematic_participants:
            dataset_findings.append(dict(
                kind="systematic_electrode", severity="high", channel=name,
                evidence=f"{name} dead (flat/railed) for {len(people)} participants: {', '.join(people)}",
                action=f"Likely hardware rather than individual contact: inspect {name}'s cap lead, connector and "
                       "board pin, and confirm it is enabled in the GUI before the next session."))
    findings = {"(whole dataset)": dataset_findings, **findings}
    salvage = sum(participants[p].salvage_fraction(policy) * participants[p].windows for p in participants)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quality-report-", dir=out.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        for name, rows in (("participants.csv", participant_rows), ("recordings.csv", recordings),
                           ("channels.csv", channel_rows)):
            with (stage / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (stage / "findings.json").write_text(json.dumps(findings, indent=2))
        report_summary = dict(
            replay=str(replay_dir), replay_summary_sha256=file_sha256(replay_dir / "summary.json"),
            quality_policy_id=config.policy_id, finding_policy=asdict(policy),
            recordings=len(recordings), participants=len(participant_rows),
            windows=dataset.windows, usable_windows=dataset.usable,
            usable_pct=round(100 * dataset.usable_fraction, 1),
            usable_pct_if_dead_excluded=round(100 * salvage / dataset.windows, 1),
            first_window_fail_pct=round(100 * dataset.failure_rate("first"), 1),
            later_window_fail_pct=round(100 * dataset.failure_rate("later"), 1),
            verdicts=dict(Counter(r["verdict"] for r in participant_rows)),
            recording_verdicts=dict(Counter(r["verdict"] for r in recordings)),
            finding_counts=dict(Counter(f["kind"] for fs in findings.values() for f in fs)),
            replay_decisions_reproduced=True,
            interpretation="Rule flags plus heuristic findings; not verified hardware diagnoses or artifact ground truth",
            implementation_sha256={name: file_sha256(Path(__file__).parent / name)
                                   for name in ("assessment.py", "acquisition.py", "report.py")})
        (stage / "summary.json").write_text(json.dumps(report_summary, indent=2))
        (stage / "report.html").write_text(
            _html(report_summary, participant_rows, channel_rows, findings, recordings), encoding="utf-8")
        stage.rename(out)
    return report_summary


def _html(summary, participants, channels, findings, recordings):
    e = html.escape
    names = sorted({row["channel"] for row in channels}, key=_channel_order)
    grid = {(row["participant_id"], row["channel"]): row for row in channels}

    def cell(person, name):
        row = grid.get((person, name))
        if row is None:
            return '<td class="na" title="electrode not recorded">–</td>'
        pct = row["flagged_pct"]
        level = 0 if pct < 10 else 1 if pct < 30 else 2 if pct < 60 else 3
        tip = (f"{person} {name}: flagged {pct}% | flat {row['flat_channel_pct']}% | "
               f"rail {row['adc_saturation_pct']}% | amplitude {row['excessive_amplitude_pct']}% | "
               f"jump {row['excessive_sample_jump_pct']}%")
        return f'<td class="h{level}" title="{e(tip)}">{pct:.0f}</td>'

    heat = "".join(
        f'<tr><th scope="row">{e(p["participant_id"])}</th>'
        + "".join(cell(p["participant_id"], n) for n in names) + "</tr>" for p in participants)
    rows = "".join(
        f'<tr><td>{e(p["participant_id"])}</td><td><span class="v {p["verdict"]}">{p["verdict"]}</span></td>'
        f'<td class="num">{p["usable_pct"]:.1f}%</td><td class="num">{p["usable_pct_if_dead_excluded"]:.1f}%</td>'
        f'<td>{e(p["dead_electrodes"].replace("EXG Channel ", "Ch ")) or "–"}</td><td class="num">{p["windows"]}</td>'
        f'<td>{e(p["boards"])}</td><td>{e(p["top_reasons"])}</td></tr>'
        for p in participants)
    actions = []
    severity_rank = {"high": 0, "medium": 1}
    ordered = sorted(((person, f) for person, items in findings.items() for f in items),
                     key=lambda pf: severity_rank.get(pf[1]["severity"], 2))  # stable: dataset-level first
    for person, f in ordered:
        actions.append(
            f'<li class="{f["severity"]}"><b>{e(person)}</b> · {e(f["kind"].replace("_", " "))}'
            f'<div class="ev">{e(f["evidence"])}</div><div class="act">{e(f["action"])}</div></li>')
    worst = sorted(recordings, key=lambda r: (r["usable_pct"], r["participant_id"]))[:25]
    rec_rows = "".join(
        f'<tr><td>{e(r["participant_id"])}</td><td>{e(r["action_folder"])}</td><td title="{e(r["source_path"])}">{e(r["file"])}</td>'
        f'<td class="num">{r["usable_pct"]:.0f}%</td><td>{e(r["worst_electrodes"])}</td><td>{e(r["top_reasons"])}</td></tr>'
        for r in worst)
    v = summary["verdicts"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG Quality Report</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1f2328;--muted:#5d646d;--card:#fff;--line:#e3e1db;--h0:#e8f3ec;--h1:#fff1c9;--h2:#ffd3a8;--h3:#f4a3a3;
--good:#1f7a45;--review:#9a6400;--unusable:#b42318;--na:#f1f0ec}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181b;--fg:#e8e6e1;--muted:#a3a7ad;--card:#1f2226;
--line:#33373d;--h0:#1d3327;--h1:#4a3f17;--h2:#5c3a1a;--h3:#6e2626;--good:#6fd39b;--review:#f0c060;--unusable:#ff8a80;--na:#25282c}}}}
:root[data-theme="dark"]{{--bg:#16181b;--fg:#e8e6e1;--muted:#a3a7ad;--card:#1f2226;--line:#33373d;--h0:#1d3327;--h1:#4a3f17;
--h2:#5c3a1a;--h3:#6e2626;--good:#6fd39b;--review:#f0c060;--unusable:#ff8a80;--na:#25282c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 64px}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:36px 0 10px}}
p.sub{{color:var(--muted);margin:0 0 20px}}.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.tile b{{display:block;font-size:24px}}
.tile span{{color:var(--muted);font-size:13px}}.scroll{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}.heat td{{text-align:center;min-width:34px;font-variant-numeric:tabular-nums}}
.heat th[scope=row]{{white-space:nowrap}}.h0{{background:var(--h0)}}.h1{{background:var(--h1)}}.h2{{background:var(--h2)}}.h3{{background:var(--h3)}}
.na{{background:var(--na);color:var(--muted)}}.v{{font-weight:600}}.v.good{{color:var(--good)}}.v.review{{color:var(--review)}}.v.unusable{{color:var(--unusable)}}
ul.f{{list-style:none;padding:0;margin:0;display:grid;gap:8px}}ul.f li{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--review);
border-radius:8px;padding:10px 12px}}ul.f li.high{{border-left-color:var(--unusable)}}.ev{{color:var(--muted);font-size:13px}}.act{{margin-top:2px}}
.legend{{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:8px 0}}.legend i{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:4px}}
.note{{color:var(--muted);font-size:13px;margin-top:28px}}
</style></head><body><main>
<h1>EEG Quality Report</h1>
<p class="sub">Rule-based assessment of raw recordings · policy <code>{e(summary["quality_policy_id"])}</code> · replay decisions reproduced exactly</p>
<div class="tiles">
<div class="tile"><b>{summary["usable_pct"]:.1f}%</b><span>usable windows ({summary["usable_windows"]:,} of {summary["windows"]:,})</span></div>
<div class="tile"><b>{summary["recordings"]:,}</b><span>recordings · {summary["participants"]} participants</span></div>
<div class="tile"><b>{summary["usable_pct_if_dead_excluded"]:.1f}%</b><span>usable if only persistently dead electrodes were excluded (estimate)</span></div>
<div class="tile"><b>{summary["first_window_fail_pct"]:.0f}% vs {summary["later_window_fail_pct"]:.0f}%</b><span>first 2 s of a recording fail vs later windows</span></div>
<div class="tile"><b>{v.get("good", 0)} / {v.get("review", 0)} / {v.get("unusable", 0)}</b><span>participants good / review / unusable</span></div>
<div class="tile"><b>{sum(summary["finding_counts"].values())}</b><span>actionable findings</span></div>
</div>
<h2>What to fix</h2><ul class="f">{"".join(actions) or "<li>No findings.</li>"}</ul>
<h2>Electrodes flagged, % of windows</h2>
<div class="legend"><span><i class="h0"></i>&lt;10%</span><span><i class="h1"></i>10–30%</span><span><i class="h2"></i>30–60%</span><span><i class="h3"></i>≥60%</span><span><i class="na"></i>not recorded</span><span>Hover a cell for the reasons. A similar low percentage on every electrode of a participant usually comes from the failing start-up window of each recording, not from the electrodes.</span></div>
<div class="scroll"><table class="heat"><thead><tr><th>Participant</th>{"".join(f"<th>{e(n.replace('EXG Channel ', 'Ch '))}</th>" for n in names)}</tr></thead><tbody>{heat}</tbody></table></div>
<h2>Participants</h2>
<div class="scroll"><table><thead><tr><th>Participant</th><th>Verdict</th><th>Usable</th><th>Usable without dead electrodes</th><th>Dead electrodes</th><th>Windows</th><th>Board</th><th>Top reasons</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Worst 25 recordings</h2>
<div class="scroll"><table><thead><tr><th>Participant</th><th>Folder</th><th>File</th><th>Usable</th><th>Worst electrodes</th><th>Top reasons</th></tr></thead><tbody>{rec_rows}</tbody></table></div>
<p class="note">"Usable without dead electrodes" estimates windows that pass when only each participant's persistently flat/railed electrodes are excluded; using them requires a reduced-montage model or channel interpolation. Verdicts: good ≥ {100 * summary["finding_policy"]["good_usable_fraction"]:.0f}% usable windows, unusable &lt; {100 * summary["finding_policy"]["unusable_below_fraction"]:.0f}%.
Findings are heuristics over rule flags that point to likely causes and checks; they are not verified hardware diagnoses, artifact ground truth or evidence about task decoding.
Full tables: participants.csv, recordings.csv, channels.csv, findings.json.</p>
</main></body></html>
"""


def _print_preflight(result, stream=None):
    stream = sys.stdout if stream is None else stream
    print(f"{result['path']}\n  board {result['board']}, {result['seconds']:.1f} s, "
          f"{result['usable_windows']}/{result['windows']} usable windows "
          f"({100 * result['usable_fraction']:.0f}%) -> {result['verdict'].upper()}", file=stream)
    bad = [c for c in result["channels"] if c["flagged_fraction"] > 0]
    for c in sorted(bad, key=lambda c: -c["flagged_fraction"]):
        reasons = ", ".join(f"{r} {n}" for r, n in sorted(c["reasons"].items(), key=lambda kv: -kv[1]))
        print(f"  {c['channel']:15s} flagged {100 * c['flagged_fraction']:3.0f}%  ({reasons})", file=stream)
    for f in result["findings"]:
        print(f"  ! {f['kind']}: {f['evidence']}\n    -> {f['action']}", file=stream)
    if result["first_window_failed"]:
        print("  note: the first 2 s window failed; across Data_clean the start-up window fails in nearly every "
              "recording, so discard the first seconds rather than re-recording for this alone.", file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ds = sub.add_parser("dataset", help="Electrode-level report from a published replay")
    ds.add_argument("--replay", required=True, help="Replay directory (summary.json + windows.jsonl)")
    ds.add_argument("--out", required=True, help="New output directory")
    pf = sub.add_parser("preflight", help="Check one or more new OpenBCI TXT recordings")
    pf.add_argument("files", nargs="+")
    pf.add_argument("--config", help="QualityConfig JSON (default: signal_quality/default_config.json)")
    pf.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    if args.command == "dataset":
        summary = build_dataset_report(args.replay, args.out)
        print(json.dumps({k: summary[k] for k in ("usable_pct", "verdicts", "recording_verdicts", "finding_counts")}, indent=2))
        print(f"Report: {Path(args.out).resolve() / 'report.html'}")
        return 0
    config_path = Path(args.config) if args.config else Path(__file__).parent / "default_config.json"
    config = QualityConfig(**json.loads(config_path.read_text()))
    results, failed = [], 0
    for path in args.files:
        try:
            result = preflight(path, config)
        except (ValueError, OSError) as error:
            result = dict(path=str(path), passed=False, verdict="unreadable", error=str(error))
        failed += not result["passed"]
        results.append(result)
        if not args.json:
            if "error" in result:
                print(f"{path}\n  UNREADABLE: {result['error']}")
            else:
                _print_preflight(result)
    if args.json:
        print(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
