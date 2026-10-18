"""Threshold sensitivity of the quality rules, recomputed from a published replay.

Every amplitude/electrode rule is an "any channel" rule, so each window's channel
extremes (largest peak-to-peak and jump, highest rail/flat fraction, smallest
peak-to-peak) decide it for ANY threshold. Continuity faults have no threshold and
stay fixed. The baseline is verified to reproduce the replay's decisions exactly.

Outputs: one-at-a-time sweeps, a peak-to-peak × jump grid, decision margins, the
stability of participant verdicts over a plausible range, and (with reviews) false
acceptance / false rejection against independent human judgement for every setting.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from .acquisition import file_sha256
from .assessment import QualityConfig
from .report import FindingPolicy, verdict

TEMPLATE = Path(__file__).with_name("sensitivity_template.html")
THRESHOLD_RULES = {"flat_channel", "adc_saturation", "excessive_amplitude", "excessive_sample_jump", "excessive_line_noise"}

# Values swept for each threshold (the configured value is always added).
SWEEPS = {
    "max_ptp_uv": [250, 350, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000],
    "max_jump_uv": [100, 150, 250, 350, 500, 750, 1000, 1500, 2000, 5000],
    "max_rail_fraction": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    "max_flat_fraction": [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
}
# Swept only when the replay's policy has a verified mains frequency.
LINE_SWEEP = [0.2, 0.35, 0.5, 0.75, 0.9, 1.0]
# "Plausible" alternatives a reasonable lab might choose; verdict stability is judged over these.
PLAUSIBLE = {
    "max_ptp_uv": (500, 2000),
    "max_jump_uv": (250, 1000),
    "max_rail_fraction": (0.005, 0.05),
    "max_flat_fraction": (0.9, 1.0),
}
RULE_OF = {"max_ptp_uv": "excessive_amplitude", "max_jump_uv": "excessive_sample_jump",
           "max_rail_fraction": "adc_saturation", "max_flat_fraction": "flat_channel",
           "max_line_ratio": "excessive_line_noise"}
LABELS = {"max_ptp_uv": "Peak-to-peak limit (µV)", "max_jump_uv": "Sample-jump limit (µV)",
          "max_rail_fraction": "Rail-occupancy limit (fraction of samples)",
          "max_flat_fraction": "Flat-step limit (fraction of samples)",
          "max_line_ratio": "Line-noise limit (share of power at mains)"}


class WindowTable:
    """Per-window channel extremes and fixed (threshold-independent) faults from a replay."""

    def __init__(self, rows):
        self.rows = rows
        n = len(rows)
        self.participant = np.array([r["participant_id"] for r in rows])
        self.window_id = np.array([r["window_id"] for r in rows])
        self.first = np.array([r["source_start_sample"] == 0 for r in rows])
        self.measured = np.array([r["window_features"] is not None for r in rows])
        get = lambda key, default: np.array([r["window_features"][key] if r["window_features"] else default  # noqa: E731
                                             for r in rows], dtype=float)
        self.ptp_max = get("ptp_uv_max", np.nan)
        self.ptp_min = get("ptp_uv_min", np.nan)
        self.jump_max = get("max_jump_uv_max", np.nan)
        self.rail_max = get("rail_fraction_max", np.nan)
        self.flat_max = get("flat_fraction_max", np.nan)
        self.line_max = {f: get(f"line{f}_ratio_max", np.nan) for f in (50, 60)}
        # Faults no threshold here can change: continuity, timing, non-finite samples, etc.
        self.fixed_fault = np.array([bool(set(r["rule_reasons"]) - THRESHOLD_RULES) for r in rows]) | ~self.measured
        self.replay_usable = np.array([bool(r["rule_usable"]) for r in rows])
        self.n = n

    def flags(self, cfg):
        """Boolean arrays per rule (True = rule flags the window) under ``cfg``."""
        m = self.measured
        flags = {
            "flat_channel": m & ((self.ptp_min <= cfg.min_ptp_uv) | (self.flat_max >= cfg.max_flat_fraction)),
            "adc_saturation": m & (self.rail_max >= cfg.max_rail_fraction),
            "excessive_amplitude": m & (self.ptp_max > cfg.max_ptp_uv),
            "excessive_sample_jump": m & (self.jump_max > cfg.max_jump_uv),
        }
        if cfg.mains_hz is not None:
            flags["excessive_line_noise"] = m & (self.line_max[int(cfg.mains_hz)] > cfg.max_line_ratio)
        return flags

    def usable(self, cfg):
        flags = self.flags(cfg)
        bad = self.fixed_fault.copy()
        for f in flags.values():
            bad |= f
        return ~bad

    def verify(self, cfg):
        """The baseline must reproduce every replay decision and threshold-rule reason."""
        usable = self.usable(cfg)
        flags = self.flags(cfg)
        mismatched = int((usable != self.replay_usable).sum())
        for i, row in enumerate(self.rows):
            expected = set(row["rule_reasons"]) & THRESHOLD_RULES
            actual = {name for name, f in flags.items() if f[i]}
            if expected != actual:
                mismatched += 1
        if mismatched:
            raise ValueError(f"Sensitivity model disagrees with the replay on {mismatched} windows")
        return dict(windows=self.n, replay_decisions_reproduced=True)


def _metrics(table, usable, baseline_usable, policy):
    later = ~table.first
    per_person = defaultdict(lambda: [0, 0])
    for person, ok in zip(table.participant, usable):
        per_person[person][0] += 1
        per_person[person][1] += int(ok)
    verdicts = {p: verdict(ok / n, policy) for p, (n, ok) in per_person.items()}
    return dict(
        rejection_rate=float(1 - usable.mean()),
        rejection_rate_after_startup=float(1 - usable[later].mean()) if later.any() else None,
        newly_rejected=int((baseline_usable & ~usable).sum()),
        newly_accepted=int((~baseline_usable & usable).sum()),
        verdicts=dict(Counter(verdicts.values())), participant_verdicts=verdicts)


def _review_metrics(table, usable, reviews):
    """False acceptance/rejection of a setting against independent reviews (weighted when possible)."""
    index = {w: i for i, w in enumerate(table.window_id)}
    rows = [(index[w], label, weight) for w, (label, weight) in reviews.items() if w in index]
    if not rows:
        return None
    idx = np.array([r[0] for r in rows])
    human = np.array([r[1] for r in rows])
    w = np.array([r[2] for r in rows], dtype=float)
    rule = usable[idx]
    bad, good = human == 0, human == 1
    def rate(mask, numer):
        return float(w[mask & numer].sum() / w[mask].sum()) if mask.any() else None
    return dict(reviewed_windows=len(rows),
                false_acceptance_rate=rate(bad, rule), false_rejection_rate=rate(good, ~rule),
                agreement=float(w[(rule == (human == 1))].sum() / w.sum()))


def load_reviews(paths):
    """window_id -> (label, weight); several reviewers on one window use their majority (ties dropped)."""
    votes = defaultdict(list)
    weights = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                votes[row["window_id"]].append(int(row["reviewed_quality_label"]))
                weights[row["window_id"]] = float(row.get("sampling_weight") or 1.0)
    out = {}
    for window, labels in votes.items():
        share = sum(labels) / len(labels)
        if share != 0.5:
            out[window] = (int(share > 0.5), weights[window])
    return out


def _margins(table, cfg):
    """How far past its threshold each rule-flagged window is, and how many it rejects alone."""
    flags = table.flags(cfg)
    stats = {"excessive_line_noise": (table.line_max[int(cfg.mains_hz)] / cfg.max_line_ratio
                                      if cfg.mains_hz is not None else None),
             "excessive_amplitude": table.ptp_max / cfg.max_ptp_uv,
             "excessive_sample_jump": table.jump_max / cfg.max_jump_uv,
             "adc_saturation": table.rail_max / cfg.max_rail_fraction,
             "flat_channel": table.flat_max / cfg.max_flat_fraction}
    others = {name: table.fixed_fault | np.logical_or.reduce([f for k, f in flags.items() if k != name])
              for name in flags}
    out = {}
    for name, flagged in flags.items():
        ratio = stats[name][flagged]
        alone = flagged & ~others[name]
        out[name] = dict(
            windows_flagged=int(flagged.sum()), rejected_by_this_rule_alone=int(alone.sum()),
            borderline_within_1_5x=int(((ratio >= 1) & (ratio < 1.5)).sum()),
            far_beyond_2x=int((ratio >= 2).sum()),
            median_ratio_to_threshold=float(np.median(ratio)) if ratio.size else None)
    # A flat fraction cannot exceed 1, so ratios cannot show distance: a completely flat channel
    # (fraction 1, or zero peak-to-peak) is unambiguous, anything else flagged is borderline.
    flagged = flags["flat_channel"]
    fully_flat = flagged & ((table.flat_max >= 0.999) | (table.ptp_min <= cfg.min_ptp_uv))
    out["flat_channel"].update(borderline_within_1_5x=int((flagged & ~fully_flat).sum()),
                               far_beyond_2x=int(fully_flat.sum()), median_ratio_to_threshold=None,
                               note="far = channel completely flat (fraction 1 or zero peak-to-peak); "
                                    "ratios are not meaningful for a fraction capped at 1")
    return out


def analyse(replay_dir, reviews=None, policy=FindingPolicy()):
    replay_dir = Path(replay_dir).resolve()
    summary = json.loads((replay_dir / "summary.json").read_text())
    base = QualityConfig(**summary["config"])
    rows = [json.loads(line) for line in (replay_dir / "windows.jsonl").read_text().splitlines() if line.strip()]
    table = WindowTable(rows)
    check = table.verify(base)
    baseline_usable = table.usable(base)
    baseline = _metrics(table, baseline_usable, baseline_usable, policy)
    review_labels = load_reviews(reviews) if reviews else {}

    sweeps = {}
    sweep_plan = dict(SWEEPS, **({"max_line_ratio": LINE_SWEEP} if base.mains_hz is not None else {}))
    for param, values in sweep_plan.items():
        values = sorted(set(values) | {getattr(base, param)})
        points = []
        for value in values:
            cfg = replace(base, **{param: value})
            usable = table.usable(cfg)
            point = dict(value=value, is_baseline=value == getattr(base, param),
                         **_metrics(table, usable, baseline_usable, policy))
            point["rule_flags"] = int(table.flags(cfg)[RULE_OF[param]].sum())
            if review_labels:
                point["reviews"] = _review_metrics(table, usable, review_labels)
            points.append(point)
        sweeps[param] = points

    grid = []
    for ptp in SWEEPS["max_ptp_uv"]:
        for jump in SWEEPS["max_jump_uv"]:
            usable = table.usable(replace(base, max_ptp_uv=ptp, max_jump_uv=jump))
            grid.append(dict(max_ptp_uv=ptp, max_jump_uv=jump, rejection_rate=float(1 - usable.mean()),
                             rejection_rate_after_startup=float(1 - usable[~table.first].mean())))

    stability = []
    for person in sorted(baseline["participant_verdicts"]):
        base_verdict = baseline["participant_verdicts"][person]
        changes = []
        for param, (low, high) in PLAUSIBLE.items():
            for point in sweeps[param]:
                if low <= point["value"] <= high and point["participant_verdicts"][person] != base_verdict:
                    changes.append(f"{param}={point['value']:g}: {point['participant_verdicts'][person]}")
        stability.append(dict(participant_id=person, baseline_verdict=base_verdict, stable=not changes,
                              changes="; ".join(changes)))

    # Strip per-participant maps from sweep points for compact outputs (kept in the stability table).
    for points in sweeps.values():
        for point in points:
            point.pop("participant_verdicts")
    baseline.pop("participant_verdicts")
    result = dict(
        replay=str(replay_dir), replay_summary_sha256=file_sha256(replay_dir / "summary.json"),
        quality_policy_id=base.policy_id, baseline_config=asdict(base), verification=check,
        baseline=baseline, plausible_ranges={k: list(v) for k, v in PLAUSIBLE.items()},
        margins=_margins(table, base), sweeps=sweeps, grid=grid, stability=stability,
        stable_participants=sum(s["stable"] for s in stability), participants=len(stability),
        reviewed_windows=len(review_labels),
        interpretation=("Rejection rates describe the configured rules, not detection accuracy. "
                        "False acceptance/rejection appear only with independent reviews; thresholds chosen on "
                        "the same reviews are in-sample and need confirmation on held-out reviews."))
    return result


def write_outputs(result, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sensitivity-", dir=out.parent) as temporary:
        stage = Path(temporary) / "out"
        stage.mkdir()
        (stage / "sensitivity.json").write_text(json.dumps(result, indent=1))
        rows = []
        for param, points in result["sweeps"].items():
            for p in points:
                r = p.get("reviews") or {}
                rows.append(dict(threshold=param, value=p["value"], is_baseline=p["is_baseline"],
                                 rejection_rate=round(p["rejection_rate"], 5),
                                 rejection_rate_after_startup=round(p["rejection_rate_after_startup"], 5),
                                 newly_rejected=p["newly_rejected"], newly_accepted=p["newly_accepted"],
                                 good=p["verdicts"].get("good", 0), review=p["verdicts"].get("review", 0),
                                 unusable=p["verdicts"].get("unusable", 0),
                                 false_acceptance_rate=r.get("false_acceptance_rate"),
                                 false_rejection_rate=r.get("false_rejection_rate")))
        for name, table in (("sweeps.csv", rows), ("grid.csv", result["grid"]), ("participants.csv", result["stability"]),
                            ("margins.csv", [dict(rule=k, **{a: b for a, b in v.items() if a != "note"})
                                             for k, v in result["margins"].items()])):
            with (stage / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        payload = json.dumps(dict(result, labels=LABELS), allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
        (stage / "report.html").write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/null", payload),
                                           encoding="utf-8")
        stage.rename(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--reviews", nargs="*", default=[], help="Imported review annotations (.jsonl)")
    args = parser.parse_args(argv)
    result = analyse(args.replay, args.reviews)
    out = write_outputs(result, args.out)
    b = result["baseline"]
    print(f"Baseline reproduced for {result['verification']['windows']} windows: rejection "
          f"{100 * b['rejection_rate']:.1f}% ({100 * b['rejection_rate_after_startup']:.1f}% after start-up)")
    for param, points in result["sweeps"].items():
        if param not in PLAUSIBLE:  # e.g. the line-noise limit: swept, but no agreed plausible range
            print(f"  {param:18s} rejection {100 * min(p['rejection_rate'] for p in points):.1f}–"
                  f"{100 * max(p['rejection_rate'] for p in points):.1f}% over its sweep")
            continue
        low, high = PLAUSIBLE[param]
        inside = [p for p in points if low <= p["value"] <= high]
        print(f"  {param:18s} plausible {low:g}–{high:g}: rejection "
              f"{100 * min(p['rejection_rate'] for p in inside):.1f}–{100 * max(p['rejection_rate'] for p in inside):.1f}%")
    print(f"  participant verdicts stable over all plausible ranges: {result['stable_participants']}/{result['participants']}")
    print(f"Report: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
