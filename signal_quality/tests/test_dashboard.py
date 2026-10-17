"""Dashboard payload, real-gate replay, review import and agreement tests on generated fixtures."""

import base64
import json
import re

import numpy as np
import pytest

from signal_quality import QualityConfig
from signal_quality.dashboard import (agreement, analyse_recording, build, config_from_json,
                                      import_reviews, main)
from signal_quality.train import reviewed_features

from test_report import write_recording  # sibling test module (tests are not a package)


def _payload(html):
    data = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S).group(1)
    return json.loads(data)


def test_dashboard_shows_failing_channel_withheld_gate_and_unassessed_tail(tmp_path):
    source = write_recording(tmp_path / "G01" / "person01" / "Left" / "rec.txt", seconds=9, dead=(15,))
    out, payload = build(source, tmp_path / "rec.quality.html")
    windows = payload["windows"]
    assert len(windows) == 5 and windows[-1]["status"] == "unassessed" and windows[-1]["raw_sha256"] is None
    for w in windows[:4]:
        assert w["status"] == "unacceptable"
        assert w["channels"]["EXG Channel 15"]["status"] == "fail"
        assert set(w["channels"]["EXG Channel 15"]["reasons"]) == {"flat_channel", "adc_saturation"}
        assert all(w["channels"][f"EXG Channel {i}"]["status"] == "pass" for i in range(15))
        assert w["gate"]["status"] == "withheld" and not w["gate"]["model_called"]
        assert set(w["gate"]["reasons"]) == {"flat_channel", "adc_saturation"}
    assert payload["session"]["dead_electrodes"] == ["EXG Channel 15"]
    assert payload["session"]["gate_accepted"] == 0 and payload["session"]["gate_withheld"] == 4
    assert "ADC rail" in payload["reason_text"]["adc_saturation"]
    # The embedded payload round-trips and the signal is the exact raw recording (float32).
    html = out.read_text()
    assert "<title>EEG Quality Dashboard</title>" in html
    embedded = _payload(html)
    signal = np.frombuffer(base64.b64decode(embedded["signal"]), dtype="<f4").reshape(16, -1)
    assert signal.shape[1] == embedded["meta"]["samples"] and np.allclose(signal[15], -187500.02235174447)


def test_clean_windows_pass_to_the_model_and_markers_locate_faults(tmp_path):
    source = write_recording(tmp_path / "rec.txt", drop_packets=(300,), startup_spike=True)
    _, payload = analyse_recording(source)
    first, second, third = payload["windows"][:3]
    # The one-sample spike jumps up at sample 5 and back at 6; the marker points at the larger jump.
    assert first["status"] == "unacceptable" and first["markers"]["jumps"]["EXG Channel 0"] in (5, 6)
    assert second["reasons"] == ["packet_discontinuity"] and second["markers"]["discontinuities"] == [50]
    assert third["gate"]["status"] == "accepted" and third["gate"]["model_called"]
    assert payload["meta"]["participant_id"] is None and third["window_id"].startswith("unknown:")


def test_gate_recovery_rule_is_replayed_in_order(tmp_path):
    source = write_recording(tmp_path / "rec.txt", startup_spike=True)
    _, payload = analyse_recording(source, QualityConfig(recovery_good_windows=2))
    gates = [w["gate"] for w in payload["windows"]]
    assert gates[0]["status"] == "withheld"
    assert gates[1]["status"] == "withheld" and gates[1]["reasons"] == ["awaiting_quality_recovery"]
    assert gates[2]["status"] == "accepted"


def _browser_record(path, payload, labels, *, reviewer="r1", seen=()):
    """What the dashboard's 'Export review record' writes (note: JSON numbers lose '.0')."""
    config = {k: (int(v) if isinstance(v, float) and v.is_integer() else v)
              for k, v in payload["meta"]["config"].items()}
    windows = [w for w in payload["windows"] if w["raw_sha256"]]
    record = dict(format="avatar-quality-review-v1", tool="avatar-quality-dashboard",
                  source_path=payload["meta"]["path"], source_sha256=payload["meta"]["source_sha256"],
                  participant_id=payload["meta"]["participant_id"], quality_policy_id=payload["meta"]["quality_policy_id"],
                  config=config, reviewer=reviewer, blind=True, random_order=True,
                  exported_at_utc="2026-10-07T12:00:00.000Z",
                  reviews=[dict(window_id=w["window_id"], source_start_sample=w["start"],
                                source_end_sample_exclusive=w["end"], raw_sha256=w["raw_sha256"], label=label,
                                reviewed_at_utc="2026-10-07T12:00:00.000Z", note=None,
                                rule_decision_seen=i in seen)
                           for i, (w, label) in enumerate(zip(windows, labels))])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_config_from_browser_json_keeps_the_policy_id():
    browser = {k: (int(v) if isinstance(v, float) and v.is_integer() else v)
               for k, v in QualityConfig().__dict__.items()}
    assert browser["min_seconds"] == 2 and type(browser["min_seconds"]) is int
    assert config_from_json(browser).policy_id == QualityConfig().policy_id
    with pytest.raises(ValueError, match="Unknown"):
        config_from_json({**browser, "surprise": 1})


def test_imported_reviews_are_accepted_by_the_training_pipeline(tmp_path):
    annotations = []
    for p in range(5):
        source = write_recording(tmp_path / "raw" / "G02" / f"person0{p + 1}" / "Left" / "rec.txt",
                                 dead=(3,) if p % 2 else (), start=1000 + 50 * p)
        _, payload = analyse_recording(source)
        labels = [0 if p % 2 else 1] * 4
        record = _browser_record(tmp_path / "reviews" / f"r{p}.json", payload, labels, seen={1})
        out = tmp_path / "reviews" / f"ann{p}.jsonl"
        rows = import_reviews(record, out)
        assert [r["review_blind_to_rules"] for r in rows] == [True, False, True, True]
        assert rows[0]["annotation_evidence"]["path"] == f"r{p}.json"
        annotations.extend(out.read_text().splitlines())
    merged = tmp_path / "reviews" / "all.jsonl"
    merged.write_text("\n".join(annotations) + "\n")
    X, y, groups, ids, _ = reviewed_features(merged, QualityConfig())
    assert X.shape == (20, 39) and set(y) == {0, 1} and len(set(groups)) == 5
    with pytest.raises(FileExistsError):
        import_reviews(tmp_path / "reviews" / "r0.json", tmp_path / "reviews" / "ann0.jsonl")


@pytest.mark.parametrize("tamper,message", [
    ("raw", "does not match an assessed window"), ("label", "label must be"),
    ("source", "changed since review"), ("policy", "different quality policy")])
def test_import_rejects_mismatched_or_tampered_reviews(tmp_path, tamper, message):
    source = write_recording(tmp_path / "raw" / "G02" / "person01" / "Left" / "rec.txt")
    _, payload = analyse_recording(source)
    path = _browser_record(tmp_path / "r.json", payload, [1, 1, 1, 1])
    record = json.loads(path.read_text())
    if tamper == "raw":
        record["reviews"][2]["raw_sha256"] = "0" * 64
    elif tamper == "label":
        record["reviews"][0]["label"] = True
    elif tamper == "source":
        source.write_text(source.read_text() + "\n")
    else:
        record["config"]["max_ptp_uv"] = 2000
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        import_reviews(path, tmp_path / "ann.jsonl")


def test_agreement_reports_false_acceptance_rejection_and_reviewer_kappa(tmp_path):
    def rows(reviewer, labels, rules):
        return [dict(window_id=f"w{i}", reviewed_by=reviewer, reviewed_quality_label=label,
                     rule_usable=bool(rule), review_blind_to_rules=True)
                for i, (label, rule) in enumerate(zip(labels, rules))]
    rules = [1, 1, 1, 0, 0, 0]
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows("alice", [1, 1, 0, 0, 0, 1], rules)))
    b.write_text("\n".join(json.dumps(r) for r in rows("bob", [1, 1, 0, 0, 1, 1], rules)))
    result = agreement([a, b])
    alice = result["reviewer_vs_rules"]["alice"]
    assert alice["agreement"] == pytest.approx(4 / 6)
    assert alice["false_acceptance_rate"] == pytest.approx(1 / 3)  # rules passed 1 of alice's 3 bad windows
    assert alice["false_rejection_rate"] == pytest.approx(1 / 3)   # rules failed 1 of alice's 3 good windows
    pair = result["reviewer_vs_reviewer"]["alice vs bob"]
    assert pair["shared_windows"] == 6 and pair["agreement"] == pytest.approx(5 / 6)
    assert pair["cohens_kappa"] == pytest.approx((5 / 6 - 0.5) / 0.5)


def test_cli_builds_dashboards_and_reports_unreadable_files(tmp_path, capsys):
    good = write_recording(tmp_path / "good.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("not an OpenBCI file\n")
    assert main(["build", str(good), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "good.quality.html").exists()
    assert main(["build", str(good), str(empty), "--out-dir", str(tmp_path / "out2")]) == 1
    assert "NOT BUILT" in capsys.readouterr().err
