"""Replay REAL acquisition files and export quality features for human review."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import tempfile
import time
import warnings

from .acquisition import file_sha256, read_openbci
from .assessment import QualityAssessor, QualityConfig


def replay(raw_root, out, config=None):
    root, out = Path(raw_root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise ValueError("raw_root must exist")
    if out.exists():
        raise FileExistsError(out)
    if root == out or root in out.parents:
        raise ValueError("Write derived quality data outside the source tree")
    assessor = QualityAssessor(config)
    counts, reasons, by_person, by_channels, latencies = Counter(), Counter(), {}, {}, []
    inventory = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quality-replay-", dir=out.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        with (stage / "windows.jsonl").open("w", encoding="utf-8") as stream:
            for source in sorted(p for p in root.rglob("*") if p.is_file()):
                counts["files_scanned"] += 1
                if source.suffix.lower() != ".txt":
                    counts["unsupported_export_files"] += 1
                    continue
                row = dict(source_path=str(source), status="unassessed")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        capture = read_openbci(source)
                    parts = source.relative_to(root).parts
                    group = next(p for p in parts if re.fullmatch(r"G\d{2}", p, re.I))
                    person = next(p for p in parts if re.fullmatch(r"person\d{2}", p, re.I))
                    subject = f"{group.upper()}/{person.lower()}"
                    digest = file_sha256(source)
                    n = int(round(assessor.config.min_seconds * capture["sfreq"]))
                    total = capture["raw"].shape[1]
                    row.update(status="assessed", source_sha256=digest, participant_id=subject,
                               channels=len(capture["channels"]), sfreq=capture["sfreq"],
                               samples=total, trailing_samples_unscored=total % n)
                    counts["files_assessed"] += 1
                    counts["trailing_samples_unscored"] += total % n
                    for start in range(0, total - n + 1, n):
                        window = {key: value for key, value in capture.items() if key != "source_kind"}
                        window["raw"] = capture["raw"][:, start:start+n]
                        window["counters"] = capture["counters"][start:start+n]
                        window["timestamps"] = capture["timestamps"][start:start+n]
                        before = time.perf_counter()
                        assessment = assessor.assess(**window)
                        latencies.append(1000 * (time.perf_counter() - before))
                        status = assessment["status"]
                        counts["windows_assessed"] += 1
                        counts[status] += 1
                        reasons.update(assessment["reasons"])
                        by_person.setdefault(subject, Counter())[status] += 1
                        by_channels.setdefault(str(len(capture["channels"])), Counter())[status] += 1
                        record = dict(window_id=f"{subject}:{digest}:rows{start}-{start+n}",
                            source_path=str(source), source_sha256=digest, participant_id=subject,
                            recording_id=digest, source_start_sample=start, source_end_sample_exclusive=start+n,
                            sfreq=capture["sfreq"], channels=capture["channels"], units="uV",
                            raw_sha256=assessment["raw_sha256"], rule_usable=assessment["usable"],
                            rule_reasons=assessment["reasons"], bad_channels=assessment["bad_channels"],
                            window_features=assessment["window_features"], quality_policy_id=assessor.config.policy_id,
                            reviewed_quality_label=None, label_verified=False, reviewed_by=None,
                            reviewed_at_utc=None, annotation_evidence=None)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                except (ValueError, OSError, StopIteration) as error:
                    row.update(status="unassessed", reason=str(error) or "participant_identity_unavailable")
                    counts["invalid_or_unsupported_named_files"] += 1
                inventory.append(row)
        import numpy as np
        summary = dict(source_root=str(root), source_data_unchanged=True, synthetic_samples=0,
                       quality_policy_id=assessor.config.policy_id, config=asdict(assessor.config),
                       counts=dict(counts), reason_counts=dict(reasons),
                       participants={k:dict(v) for k,v in by_person.items()},
                       channel_counts={k:dict(v) for k,v in by_channels.items()},
                       latency_ms=dict(median=float(np.median(latencies)) if latencies else None,
                                       p95=float(np.percentile(latencies, 95)) if latencies else None),
                       independently_reviewed_windows=0, artifact_detection_accuracy=None,
                       implementation_sha256={name:file_sha256(Path(__file__).parent/name)
                                              for name in ("assessment.py", "acquisition.py", "replay.py")},
                       interpretation="Rule decisions and feature extraction only; not artifact ground truth or measured classification accuracy",
                       trailing_samples="Reported but not scored as complete offline windows; live gate checks all supplied samples")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2))
        (stage / "inventory.json").write_text(json.dumps(inventory, indent=2))
        stage.rename(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="QualityConfig JSON; fixed before evaluation")
    args = parser.parse_args()
    config = QualityConfig(**json.loads(Path(args.config).read_text())) if args.config else None
    summary = replay(args.raw_root, args.out, config)
    print(json.dumps({key:summary[key] for key in ("counts", "reason_counts", "latency_ms", "artifact_detection_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
