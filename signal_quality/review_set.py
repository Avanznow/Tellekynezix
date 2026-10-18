"""Stratified blind review sets for validating the quality rules against human judgement.

``build``   samples windows from a published replay, stratified by rule decision and
            rejection type and spread evenly across participants, and writes ONE
            self-contained review page (no rule decisions, participants or file names
            inside it) plus a manifest holding everything needed for analysis.
``import``  validates a reviewer's exported record against the manifest and the
            original recordings and writes annotations for ``signal_quality.train``
            and ``python -m signal_quality.dashboard agreement`` (with sampling weights).

Sampling is stratified by (decision, category, participant). Each sampled window
carries weight = windows in its cell / windows sampled from that cell, so weighted
agreement estimates describe the whole replay, not just the sample.
"""

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np

from .acquisition import file_sha256, read_openbci
from .assessment import QualityAssessor, QualityConfig
from .dashboard import _portable, config_from_json

TEMPLATE = Path(__file__).with_name("review_set_template.html")
RECORD_FORMAT = "avatar-quality-review-set-v1"
ELECTRODE = {"flat_channel", "adc_saturation"}
CONTINUITY = {"packet_discontinuity", "acquisition_gap", "nonmonotonic_time", "nonadvancing_time"}
ARTIFACT = {"excessive_amplitude", "excessive_sample_jump", "excessive_line_noise"}

RUBRIC = "\n".join([
    "Judge whether this 2-second window is usable EEG for decoding, from the raw traces alone.",
    "ACCEPTABLE: every channel shows a plausible, varying EEG-like signal. Small slow drifts and moderate "
    "noise are fine.",
    "UNACCEPTABLE: one or more channels are flat or stuck, or large spikes, steps, bursts or movement "
    "artifacts dominate the window, or the signal is broken or discontinuous.",
    'If you are not sure, choose the closer label and tick "unsure".',
])


def category(row):
    """Rejection type of a replay window; accepted windows are 'pass'."""
    if row["rule_usable"]:
        return "pass"
    reasons = set(row["rule_reasons"])
    if row["source_start_sample"] == 0:
        return "startup"          # first window of a recording; fails almost always
    if reasons & ELECTRODE:
        return "electrode"
    if reasons & CONTINUITY:
        return "continuity"
    if reasons & ARTIFACT:
        return "artifact"
    return "other"


def _allocate(total, capacities):
    """Split ``total`` as evenly as possible over keys without exceeding their capacity."""
    alloc = {key: 0 for key in capacities}
    remaining = total
    while remaining > 0:
        open_keys = [k for k in capacities if alloc[k] < capacities[k]]
        if not open_keys:
            break
        share = max(1, remaining // len(open_keys))
        for key in open_keys:
            take = min(share, capacities[key] - alloc[key], remaining)
            alloc[key] += take
            remaining -= take
            if remaining == 0:
                break
    return alloc


def sample_windows(rows, size, seed=42, pass_fraction=0.5):
    """Stratified sample; returns (sampled rows with stratum/weight, plan)."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for row in rows:
        cells[(category(row), row["participant_id"])].append(row)
    by_category = defaultdict(dict)
    for (cat, person), members in cells.items():
        by_category[cat][person] = len(members)
    reject_cats = sorted(c for c in by_category if c != "pass")
    n_pass = min(round(size * pass_fraction), sum(by_category.get("pass", {}).values()))
    reject_capacity = {c: sum(by_category[c].values()) for c in reject_cats}
    per_category = {"pass": n_pass, **_allocate(size - n_pass, reject_capacity)}
    if sum(per_category.values()) < size and "pass" in by_category:  # few rejections: top up passes
        per_category["pass"] = min(per_category["pass"] + size - sum(per_category.values()),
                                   sum(by_category["pass"].values()))
    sampled, plan = [], {}
    for cat, n in per_category.items():
        if not n:
            continue
        people = sorted(by_category[cat])
        rng.shuffle(people)  # which participants get leftover windows is random, not alphabetical
        per_person = _allocate(n, {person: by_category[cat][person] for person in people})
        total = sum(by_category[cat].values())
        covered = sum(by_category[cat][person] for person, k in per_person.items() if k)
        # Windows of participants with no sampled window are represented by the rest of the stratum.
        scale = total / covered
        for person, k in sorted(per_person.items()):
            if not k:
                continue
            members = sorted(cells[(cat, person)], key=lambda r: r["window_id"])
            chosen = rng.sample(members, k)
            weight = len(members) / k * scale
            for row in chosen:
                sampled.append(dict(row, stratum=cat, sampling_weight=weight))
        plan[cat] = dict(windows_in_replay=total, sampled=n,
                         participants_sampled=sum(1 for k in per_person.values() if k),
                         participants_in_stratum=len(people),
                         windows_from_unsampled_participants=total - covered)
    rng.shuffle(sampled)
    return sampled, plan


def _read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_openbci(path)


def build_review_set(replay_dir, out, *, size=300, seed=42, pass_fraction=0.5, rubric=RUBRIC):
    """Write review_set.html (what reviewers open) and manifest.json (kept by the analyst)."""
    replay_dir, out = Path(replay_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 10 <= size <= 2000:
        raise ValueError("Review sets need between 10 and 2000 windows")
    summary = json.loads((replay_dir / "summary.json").read_text())
    config = QualityConfig(**summary["config"])
    if config.policy_id != summary["quality_policy_id"]:
        raise ValueError("Replay summary policy does not match its configuration")
    rows = [json.loads(line) for line in (replay_dir / "windows.jsonl").read_text().splitlines() if line.strip()]
    sampled, plan = sample_windows(rows, size, seed, pass_fraction)
    assessor = QualityAssessor(config)
    sources, items, page_items = {}, [], []
    for number, row in enumerate(sampled, 1):
        source = row["source_path"]
        if source not in sources:
            if file_sha256(source) != row["source_sha256"]:
                raise ValueError(f"Source changed since the replay: {source}")
            sources[source] = _read(source)
        capture = sources[source]
        a, b = row["source_start_sample"], row["source_end_sample_exclusive"]
        window = {k: v for k, v in capture.items() if k != "source_kind"}
        window["raw"], window["counters"], window["timestamps"] = capture["raw"][:, a:b], capture["counters"][a:b], capture["timestamps"][a:b]
        report = assessor.assess(**window)
        if report["raw_sha256"] != row["raw_sha256"] or report["usable"] != row["rule_usable"]:
            raise ValueError(f"Re-assessment disagrees with the replay for {row['window_id']}")
        item_id = f"item-{number:04d}"
        items.append(dict(item_id=item_id, window_id=row["window_id"], source_path=source,
                          source_sha256=row["source_sha256"], participant_id=row["participant_id"],
                          source_start_sample=a, source_end_sample_exclusive=b, raw_sha256=row["raw_sha256"],
                          channels=capture["channels"], sfreq=capture["sfreq"], rule_usable=row["rule_usable"],
                          rule_reasons=row["rule_reasons"], stratum=row["stratum"],
                          sampling_weight=row["sampling_weight"]))
        # The page gets ONLY what a reviewer needs to judge the raw signal.
        signal = np.ascontiguousarray(window["raw"], dtype="<f4")
        page_items.append(dict(item_id=item_id, channels=capture["channels"], sfreq=capture["sfreq"],
                               samples=signal.shape[1], signal=base64.b64encode(signal.tobytes()).decode("ascii")))
    identity = json.dumps(dict(items=[(i["item_id"], i["window_id"], i["raw_sha256"]) for i in items],
                               rubric=rubric, policy=config.policy_id), sort_keys=True)
    set_id = "rs-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    manifest = dict(review_set_id=set_id, record_format=RECORD_FORMAT, replay=str(replay_dir),
                    replay_summary_sha256=file_sha256(replay_dir / "summary.json"),
                    quality_policy_id=config.policy_id, config=summary["config"], seed=seed, size=len(items),
                    pass_fraction=pass_fraction, rubric=rubric,
                    rubric_sha256=hashlib.sha256(rubric.encode()).hexdigest(),
                    plan=plan, participants=len({i["participant_id"] for i in items}),
                    recordings=len({i["source_sha256"] for i in items}),
                    weighting=("weight = windows in (stratum, participant) cell / windows sampled from that cell, "
                               "scaled so each stratum's weights sum to its replay windows; participants with no "
                               "sampled window in a stratum are represented by the others (see plan)"),
                    items=items)
    payload = dict(review_set_id=set_id, record_format=RECORD_FORMAT, rubric=rubric, items=page_items)
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/null", data)
    out.mkdir(parents=True)
    (out / "review_set.html").write_text(html, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.txt").write_text(
        f"Review set {set_id}: {len(items)} windows from {manifest['participants']} participants.\n"
        "Send ONLY review_set.html to reviewers. Keep manifest.json: it contains the rule decisions.\n"
        "Reviewers open the page, enter their name, label every window and click 'Export review record'.\n"
        "Then: python -m signal_quality.review_set import <record.json> --set <this folder> --out reviews/<name>.jsonl\n")
    return manifest


def import_review_record(record_path, set_dir, out):
    """Validate an exported review record and write weighted training/agreement annotations."""
    record_path, set_dir, out = Path(record_path).resolve(), Path(set_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("format") != RECORD_FORMAT or record.get("review_set_id") != manifest["review_set_id"]:
        raise ValueError("Record does not belong to this review set")
    reviewer = record.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Review record needs a reviewer identity")
    config = config_from_json(manifest["config"])
    if config.policy_id != manifest["quality_policy_id"]:
        raise ValueError("Review set manifest policy mismatch")
    items = {item["item_id"]: item for item in manifest["items"]}
    reviews = record.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Record contains no reviews")
    seen = set()
    assessor, sources = QualityAssessor(config), {}
    evidence_sha = file_sha256(record_path)
    rows = []
    for i, review in enumerate(reviews):
        item = items.get(review.get("item_id"))
        if item is None or review["item_id"] in seen:
            raise ValueError(f"Review {i} has an unknown or repeated item")
        seen.add(review["item_id"])
        if review.get("label") not in (0, 1) or isinstance(review.get("label"), bool):
            raise ValueError(f"Review {i} label must be 0 (unacceptable) or 1 (acceptable)")
        reviewed = datetime.fromisoformat(str(review.get("reviewed_at_utc", "")).replace("Z", "+00:00"))
        if reviewed.tzinfo is None:
            raise ValueError(f"Review {i} timestamp needs a timezone")
        source = item["source_path"]
        if source not in sources:
            if file_sha256(source) != item["source_sha256"]:
                raise ValueError(f"Source recording changed since the review set was built: {source}")
            sources[source] = _read(source)
        capture = sources[source]
        a, b = item["source_start_sample"], item["source_end_sample_exclusive"]
        window = {k: v for k, v in capture.items() if k != "source_kind"}
        window["raw"], window["counters"], window["timestamps"] = capture["raw"][:, a:b], capture["counters"][a:b], capture["timestamps"][a:b]
        if assessor.assess(**window)["raw_sha256"] != item["raw_sha256"]:
            raise ValueError(f"Raw window changed for {item['item_id']}")
        seconds = review.get("seconds")
        rows.append(dict(
            window_id=item["window_id"], source_path=_portable(Path(source), out.parent),
            source_sha256=item["source_sha256"], participant_id=item["participant_id"],
            recording_id=item["source_sha256"], source_start_sample=a, source_end_sample_exclusive=b,
            sfreq=item["sfreq"], channels=item["channels"], units="uV", raw_sha256=item["raw_sha256"],
            rule_usable=item["rule_usable"], rule_reasons=item["rule_reasons"], quality_policy_id=config.policy_id,
            reviewed_quality_label=int(review["label"]), label_verified=True, reviewed_by=reviewer.strip(),
            reviewed_at_utc=reviewed.isoformat(), review_blind_to_rules=True,
            review_uncertain=bool(review.get("unsure", False)), review_note=review.get("note") or None,
            review_seconds=float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None,
            review_set_id=manifest["review_set_id"], stratum=item["stratum"],
            sampling_weight=item["sampling_weight"], rubric_sha256=manifest["rubric_sha256"],
            annotation_evidence=dict(path=_portable(record_path, out.parent), sha256=evidence_sha,
                                     locator=f"reviews[{i}]")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return dict(rows=len(rows), of_items=len(items), reviewer=reviewer.strip(),
                missing=len(items) - len(rows), strata=dict(Counter(r["stratum"] for r in rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Sample a stratified blind review set from a replay")
    b.add_argument("--replay", required=True)
    b.add_argument("--out", required=True, help="New directory for review_set.html + manifest.json")
    b.add_argument("--size", type=int, default=300)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--pass-fraction", type=float, default=0.5, help="Share of rule-accepted windows")
    i = sub.add_parser("import", help="Validate a reviewer's exported record")
    i.add_argument("record")
    i.add_argument("--set", required=True, help="Review set directory (with manifest.json)")
    i.add_argument("--out", required=True, help="New annotations .jsonl")
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_review_set(args.replay, args.out, size=args.size, seed=args.seed,
                                    pass_fraction=args.pass_fraction)
        print(f"Review set {manifest['review_set_id']}: {manifest['size']} windows, "
              f"{manifest['participants']} participants, {manifest['recordings']} recordings")
        for stratum, info in manifest["plan"].items():
            print(f"  {stratum:11s} {info['sampled']:4d} sampled of {info['windows_in_replay']:5d} "
                  f"({info['participants_sampled']} participants)")
        print(f"Send {Path(args.out) / 'review_set.html'} to reviewers; keep manifest.json private.")
    else:
        result = import_review_record(args.record, args.set, args.out)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
