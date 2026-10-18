"""Interactive EEG quality dashboard, blind review import and reviewer agreement.

``build``    writes ONE self-contained HTML file for a recording: channel quality grid,
             raw signal viewer with fault markers, window-by-window replay, the gate's
             accepted/withheld decision with its explanation, a session report with
             exports, and blind review controls. No server, hardware or BrainFlow needed.
``import``   validates a review record exported from the dashboard against the source
             recording and writes annotations accepted by ``signal_quality.train``.
``agreement`` compares independent reviews with rule decisions (and reviewers with
             each other): agreement, Cohen's kappa, false acceptance and false rejection.

Rule decisions are configured checks, not ground truth. Reviews are the only labels.
"""

import argparse
import base64
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np

from .acquisition import file_sha256, read_openbci
from .assessment import QualityAssessor, QualityConfig
from .explain import reason_text
from .gate import QualityGatedPredictor
from .report import FindingPolicy, RecordingStats, verdict

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
REVIEW_FORMAT = "avatar-quality-review-v1"
DEFAULT_CONFIG = Path(__file__).with_name("default_config.json")


def _participant(path):
    parts = Path(path).resolve().parts
    group = next((p for p in parts if re.fullmatch(r"G\d{2}", p, re.I)), None)
    person = next((p for p in parts if re.fullmatch(r"person\d{2}", p, re.I)), None)
    return f"{group.upper()}/{person.lower()}" if group and person else None


def _read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_openbci(path)


def _markers(window, report, config):
    """Sample offsets (within the window) that explain continuity and channel flags."""
    counters, times = window["counters"], window["timestamps"]
    discontinuities = np.flatnonzero(np.mod(np.diff(counters), 256) != window["counter_step"]) + 1
    gaps = np.flatnonzero(np.diff(times) > config.max_arrival_gap_seconds) + 1
    jumps = {}
    raw = np.asarray(window["raw"], dtype=np.float64)
    for i, name in enumerate(window["channels"]):
        if "excessive_sample_jump" in report["channel_reasons"][name]:
            jumps[name] = int(np.argmax(np.abs(np.diff(raw[i]))) + 1)
    return dict(discontinuities=discontinuities[:50].tolist(), gaps=gaps[:50].tolist(), jumps=jumps)


def analyse_recording(path, config=None, policy=FindingPolicy()):
    """Assess every 2 s window, run the real gate in replay order and summarise the session."""
    config = config if isinstance(config, QualityConfig) else QualityConfig(**(config or {}))
    assessor = QualityAssessor(config)
    capture = _read(path)
    digest = file_sha256(path)
    participant = _participant(path)
    sfreq, channels = capture["sfreq"], capture["channels"]
    n = int(round(config.min_seconds * sfreq))
    total = capture["raw"].shape[1]

    # Replay clock: each window is "captured" when its last sample arrives.
    replay_now = [0.0]
    gate = QualityGatedPredictor(assessor, clock=lambda: replay_now[0])
    model_calls = []

    def model(samples):  # stands in for a classifier; only records that the gate let input through
        model_calls.append(samples.shape)
        return "land"

    stats, windows, latencies = RecordingStats(), [], []
    for index, start in enumerate(range(0, total - n + 1, n)):
        window = {key: value for key, value in capture.items() if key != "source_kind"}
        window["raw"] = capture["raw"][:, start:start + n]
        window["counters"] = capture["counters"][start:start + n]
        window["timestamps"] = capture["timestamps"][start:start + n]
        before = time.perf_counter()
        report = assessor.assess(**window)
        latencies.append(1000 * (time.perf_counter() - before))
        stats.add(report, policy, first=start == 0)
        replay_now[0] = (start + n) / sfreq
        calls = len(model_calls)
        decision = gate.predict(window["raw"], model, sfreq=sfreq, channels=channels, units="uV",
                                counters=window["counters"], timestamps=window["timestamps"],
                                counter_step=window["counter_step"], source_kind="real",
                                stream_id=digest, captured_monotonic=replay_now[0])
        gate_reasons = ([] if decision["status"] == "accepted"
                        else [r.strip() for r in decision["withholding_reason"].split(";") if r.strip()])
        features = report["channel_features"]
        per_channel = {}
        for i, name in enumerate(channels):
            flags = report["channel_reasons"][name]
            per_channel[name] = dict(
                status="unassessed" if features is None else ("fail" if flags else "pass"), reasons=flags,
                **({} if features is None else {key: float(features[key][i]) for key in
                                                ("ptp_uv", "max_jump_uv", "rail_fraction", "flat_fraction")}))
        windows.append(dict(
            index=index, start=start, end=start + n, t0=start / sfreq, t1=(start + n) / sfreq,
            status=report["status"], reasons=report["reasons"], channels=per_channel,
            markers=_markers(window, report, config), raw_sha256=report["raw_sha256"],
            window_id=f"{participant or 'unknown'}:{digest}:rows{start}-{start + n}",
            gate=dict(status=decision["status"], reasons=gate_reasons,
                      model_called=len(model_calls) > calls)))
    if total % n:
        start = total - total % n
        windows.append(dict(index=len(windows), start=start, end=total, t0=start / sfreq, t1=total / sfreq,
                            status="unassessed", reasons=["insufficient_samples"],
                            channels={name: dict(status="unassessed", reasons=[]) for name in channels},
                            markers=dict(discontinuities=[], gaps=[], jumps={}), raw_sha256=None, window_id=None,
                            gate=dict(status="withheld", reasons=["insufficient_samples"], model_called=False)))

    session = dict(
        windows=stats.windows, usable_windows=stats.usable,
        rejected_windows=stats.windows - stats.usable, unassessed_samples=total % n,
        usable_pct=round(100 * stats.usable_fraction, 1),
        verdict=verdict(stats.usable_fraction, policy) if stats.windows else "too_short",
        reasons=dict(stats.reasons.most_common()),
        channel_flagged_pct={name: round(100 * stats.channel_fraction(name), 1) for name in channels},
        channel_reasons={name: dict(stats.channel_flags[name]) for name in channels},
        dead_electrodes=stats.dead_electrodes(policy),
        usable_pct_if_dead_excluded=round(100 * stats.salvage_fraction(policy), 1),
        first_window_failed=bool(stats.position["first", "failed"]),
        findings=stats.findings(policy),
        gate_accepted=sum(w["gate"]["status"] == "accepted" for w in windows),
        gate_withheld=sum(w["gate"]["status"] == "withheld" for w in windows if w["raw_sha256"]),
        assessment_latency_ms=dict(median=float(np.median(latencies)) if latencies else None,
                                   p95=float(np.percentile(latencies, 95)) if latencies else None))
    meta = dict(
        file=Path(path).name, path=str(Path(path).resolve()), source_sha256=digest, participant_id=participant,
        action_folder=Path(path).parent.name, board=f"{len(channels)}ch@{sfreq:g}Hz", sfreq=sfreq,
        channels=channels, samples=total, duration_seconds=total / sfreq, window_samples=n,
        counter_step=capture["counter_step"], quality_policy_id=config.policy_id, config=asdict(config),
        finding_policy=asdict(policy), generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        review_format=REVIEW_FORMAT,
        interpretation="Rule decisions are configured checks, not ground truth; reviews are independent labels.")
    return capture, dict(meta=meta, windows=windows, session=session, reason_text=reason_text(config))


def build(path, out=None, config=None, *, max_megabytes=60):
    """Write the dashboard HTML; returns its path and the analysis payload."""
    capture, payload = analyse_recording(path, config)
    signal = np.ascontiguousarray(capture["raw"], dtype="<f4")
    payload["signal"] = base64.b64encode(signal.tobytes()).decode("ascii")
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    if len(data) > max_megabytes * 1e6:
        raise ValueError(f"Recording too large for one dashboard file ({len(data) / 1e6:.0f} MB)")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/null", data)
    out = Path(out) if out else Path(path).with_suffix("").name + ".quality.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out, payload


# ---------------------------------------------------------------------------- reviews

def _load_record(record_path):
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if record.get("format") != REVIEW_FORMAT or not isinstance(record.get("reviews"), list):
        raise ValueError("Not an Avatar quality review record")
    if not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip():
        raise ValueError("Review record needs a reviewer identity")
    return record


def config_from_json(values):
    """Rebuild a QualityConfig from JSON written by a browser.

    JavaScript serialises 2.0 as 2, which would change the policy hash; restore the
    declared float fields so the same policy round-trips to the same policy ID.
    """
    defaults = asdict(QualityConfig())
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown quality config fields: {sorted(unknown)}")
    return QualityConfig(**{key: float(value) if isinstance(defaults[key], float) and value is not None
                                  and not isinstance(value, bool) else value
                            for key, value in values.items()})


def _portable(path, base):
    """Relative to the annotation file when nearby, otherwise absolute (both resolve in train.py)."""
    relative = os.path.relpath(path, base)
    return str(path) if relative.startswith("..") else relative


def import_reviews(record_path, out, config=None):
    """Validate a dashboard review record and write ``signal_quality.train`` annotations (JSONL)."""
    record_path, out = Path(record_path).resolve(), Path(out).resolve()
    record = _load_record(record_path)
    config = config if isinstance(config, QualityConfig) else config_from_json(config or record["config"])
    if config.policy_id != record.get("quality_policy_id"):
        raise ValueError("Review record was made under a different quality policy")
    source = Path(record["source_path"]).resolve()
    if file_sha256(source) != record.get("source_sha256"):
        raise ValueError("Source recording changed since review")
    _, payload = analyse_recording(source, config)
    by_start = {w["start"]: w for w in payload["windows"] if w["raw_sha256"]}
    participant = payload["meta"]["participant_id"]
    if participant is None:
        raise ValueError("Training annotations need a Gxx/personxx participant path")
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for i, review in enumerate(record["reviews"]):
        window = by_start.get(review.get("source_start_sample"))
        if (window is None or window["end"] != review.get("source_end_sample_exclusive")
                or window["raw_sha256"] != review.get("raw_sha256") or window["window_id"] != review.get("window_id")):
            raise ValueError(f"Review {i} does not match an assessed window of the source")
        if review.get("label") not in (0, 1) or isinstance(review.get("label"), bool):
            raise ValueError(f"Review {i} label must be 0 (unacceptable) or 1 (acceptable)")
        reviewed = datetime.fromisoformat(str(review.get("reviewed_at_utc", "")).replace("Z", "+00:00"))
        if reviewed.tzinfo is None:
            raise ValueError(f"Review {i} timestamp needs a timezone")
        rows.append(dict(
            window_id=window["window_id"], source_path=_portable(source, out.parent),
            source_sha256=record["source_sha256"], participant_id=participant, recording_id=record["source_sha256"],
            source_start_sample=window["start"], source_end_sample_exclusive=window["end"],
            sfreq=payload["meta"]["sfreq"], channels=payload["meta"]["channels"], units="uV",
            raw_sha256=window["raw_sha256"], rule_usable=window["status"] == "acceptable",
            rule_reasons=window["reasons"], quality_policy_id=config.policy_id,
            reviewed_quality_label=int(review["label"]), label_verified=True,
            reviewed_by=record["reviewer"].strip(), reviewed_at_utc=reviewed.isoformat(),
            review_blind_to_rules=bool(record.get("blind", False) and not review.get("rule_decision_seen", False)),
            review_note=review.get("note") or None,
            annotation_evidence=dict(path=_portable(record_path, out.parent),
                                     sha256=file_sha256(record_path), locator=f"reviews[{i}]")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return rows


def _kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = float((a == b).mean())
    expected = sum(float((a == k).mean() * (b == k).mean()) for k in (0, 1))
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _wilson(successes, n, z=1.96):
    """95% Wilson score interval for a proportion; None when there are no trials."""
    if not n:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def agreement(annotation_paths):
    """Reviewer-vs-rule and reviewer-vs-reviewer agreement over imported annotations."""
    rows = [json.loads(line) for path in annotation_paths
            for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No reviewed windows")
    reviewers = defaultdict(dict)
    for row in rows:
        reviewers[row["reviewed_by"]][row["window_id"]] = row
    result = dict(windows_reviewed=len({r["window_id"] for r in rows}), reviewers=sorted(reviewers),
                  interpretation="Rule decisions compared with independent human judgement; "
                                 "false acceptance = rule passed a window the reviewer rejected.")
    per = {}
    for name, reviewed in reviewers.items():
        values = list(reviewed.values())
        human = np.array([r["reviewed_quality_label"] for r in values])
        rule = np.array([int(r["rule_usable"]) for r in values])
        bad, good = human == 0, human == 1
        rejected = rule == 0
        entry = dict(
            windows=len(human), blind_windows=sum(bool(r.get("review_blind_to_rules")) for r in values),
            reviewer_acceptable=int(good.sum()), reviewer_unacceptable=int(bad.sum()),
            agreement=float((human == rule).mean()), cohens_kappa=_kappa(human, rule),
            false_acceptance_rate=float((rule[bad] == 1).mean()) if bad.any() else None,
            false_rejection_rate=float((rule[good] == 0).mean()) if good.any() else None,
            agreement_ci95=_wilson(int((human == rule).sum()), len(human)),
            false_acceptance_ci95=_wilson(int((rule[bad] == 1).sum()), int(bad.sum())),
            false_rejection_ci95=_wilson(int((rule[good] == 0).sum()), int(good.sum())),
            uncertain_reviews=sum(bool(r.get("review_uncertain")) for r in values))
        seconds = [r["review_seconds"] for r in values if isinstance(r.get("review_seconds"), (int, float))]
        if seconds:
            entry["median_review_seconds"] = float(np.median(seconds))
        # Which rules over-reject: among rule rejections citing a reason, how many the reviewer accepted.
        by_reason = defaultdict(lambda: [0, 0])
        for r, h in zip(values, human):
            if not r["rule_usable"]:
                for reason in set(r.get("rule_reasons") or []):
                    by_reason[reason][0] += 1
                    by_reason[reason][1] += int(h == 1)
        entry["rule_rejections_judged_acceptable_by_reason"] = {
            reason: dict(rule_rejections=n, reviewer_acceptable=k, share=k / n)
            for reason, (n, k) in sorted(by_reason.items(), key=lambda kv: -kv[1][0])}
        strata = defaultdict(list)
        for r, h in zip(values, human):
            if r.get("stratum"):
                strata[r["stratum"]].append(int(h))
        if strata:
            entry["reviewer_acceptable_share_by_stratum"] = {
                s: dict(windows=len(v), reviewer_acceptable=float(np.mean(v))) for s, v in sorted(strata.items())}
        weights = [r.get("sampling_weight") for r in values]
        if all(isinstance(w, (int, float)) and w > 0 for w in weights):
            w = np.asarray(weights, dtype=float)
            entry["weighted_to_replay"] = dict(
                interpretation="Sampling-weighted estimates for all replay windows (point estimates)",
                windows_represented=float(w.sum()),
                agreement=float(w[human == rule].sum() / w.sum()),
                reviewer_acceptable_share=float(w[good].sum() / w.sum()),
                false_acceptance_rate=float(w[bad & ~rejected].sum() / w[bad].sum()) if bad.any() else None,
                false_rejection_rate=float(w[good & rejected].sum() / w[good].sum()) if good.any() else None)
        per[name] = entry
    result["reviewer_vs_rules"] = per
    pairs = {}
    names = sorted(reviewers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(reviewers[a]) & set(reviewers[b]))
            if shared:
                x = [reviewers[a][w]["reviewed_quality_label"] for w in shared]
                y = [reviewers[b][w]["reviewed_quality_label"] for w in shared]
                pairs[f"{a} vs {b}"] = dict(shared_windows=len(shared),
                                            agreement=float(np.mean(np.equal(x, y))), cohens_kappa=_kappa(x, y))
    result["reviewer_vs_reviewer"] = pairs
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Write an interactive dashboard for one or more recordings")
    b.add_argument("files", nargs="+")
    b.add_argument("--out-dir", default=".", help="Directory for <recording>.quality.html files")
    b.add_argument("--config", default=str(DEFAULT_CONFIG))
    i = sub.add_parser("import", help="Validate a dashboard review record into training annotations")
    i.add_argument("record")
    i.add_argument("--out", required=True, help="New annotations .jsonl")
    a = sub.add_parser("agreement", help="Reviewer agreement with rules and with each other")
    a.add_argument("annotations", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "build":
        config = QualityConfig(**json.loads(Path(args.config).read_text()))
        failed = 0
        for path in args.files:
            try:
                out, payload = build(path, Path(args.out_dir) / (Path(path).stem + ".quality.html"), config)
            except (ValueError, OSError) as error:
                failed += 1
                print(f"{path}  NOT BUILT: {error}", file=sys.stderr)
                continue
            s = payload["session"]
            print(f"{out}  {s['usable_windows']}/{s['windows']} windows pass ({s['usable_pct']}%), "
                  f"gate accepted {s['gate_accepted']}")
        return 1 if failed else 0
    elif args.command == "import":
        rows = import_reviews(args.record, args.out)
        print(f"Wrote {len(rows)} reviewed windows to {args.out}")
    else:
        print(json.dumps(agreement(args.annotations), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
