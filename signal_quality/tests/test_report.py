"""Electrode-level report and preflight tests on generated OpenBCI fixtures."""

import csv
import json

import numpy as np
import pytest

from signal_quality import QualityConfig
from signal_quality.replay import replay
from signal_quality.report import FindingPolicy, build_dataset_report, main, preflight

RAIL = -187500.02235174447


def write_recording(path, *, seconds=8, dead=(), drop_packets=(), startup_spike=False, spikes=(), start=1000):
    fs, c = 125, 16
    n = int(seconds * fs)
    t = np.arange(n) / fs
    raw = np.stack([-10000 - 100 * i + (20 + i) * np.sin(2 * np.pi * (8 + i / 5) * t + i / 9) for i in range(c)])
    for channel in dead:
        raw[channel] = RAIL
    for sample in ((5,) if startup_spike else ()) + tuple(spikes):  # one-sample spike on every channel
        raw[:, sample] += 2000
    counters = (2 * np.arange(n)) % 256
    for sample in drop_packets:  # one lost packet at each listed sample
        counters[sample:] = (counters[sample:] + 2) % 256
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        stream.write("%OpenBCI Raw EXG Data\n%Number of channels = 16\n%Sample Rate = 125 Hz\n"
                     "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n")
        writer = csv.writer(stream)
        writer.writerow(["Sample Index", *[f"EXG Channel {i}" for i in range(c)], "Timestamp", "Marker Channel"])
        for i in range(n):
            writer.writerow([counters[i], *raw[:, i], start + t[i], 0])
    return path


def test_preflight_names_a_stuck_electrode_and_estimates_salvage(tmp_path):
    result = preflight(write_recording(tmp_path / "rec.txt", dead=(15,)))
    assert result["windows"] == 4 and result["usable_windows"] == 0 and not result["passed"]
    assert result["dead_electrodes"] == ["EXG Channel 15"]
    # Every other electrode is clean, so excluding the one dead electrode recovers every window.
    assert result["usable_fraction_if_dead_excluded"] == 1.0
    [finding] = [f for f in result["findings"] if f["kind"] == "electrode_contact"]
    assert finding["channel"] == "EXG Channel 15" and "stuck at the ADC rail" in finding["evidence"]
    flagged = {c["channel"]: c["flagged_fraction"] for c in result["channels"]}
    assert flagged["EXG Channel 15"] == 1.0 and all(v == 0 for k, v in flagged.items() if k != "EXG Channel 15")
    json.dumps(result, allow_nan=False)


def test_preflight_passes_clean_recording_and_flags_dropouts(tmp_path):
    clean = preflight(write_recording(tmp_path / "clean.txt"))
    assert clean["passed"] and clean["verdict"] == "good" and clean["findings"] == []
    dropped = preflight(write_recording(tmp_path / "drops.txt", drop_packets=(100, 400)))
    assert not dropped["passed"]
    assert [f["kind"] for f in dropped["findings"]] == ["wireless_dropouts"]
    # Capture-level faults are not attributed to electrodes and cannot be salvaged by excluding any.
    assert dropped["dead_electrodes"] == [] and dropped["usable_fraction_if_dead_excluded"] == 0.5


def test_startup_only_packet_loss_is_not_reported_as_dropouts(tmp_path):
    result = preflight(write_recording(tmp_path / "startup.txt", drop_packets=(100,)))
    assert result["first_window_failed"] and result["usable_windows"] == 3
    assert all(f["kind"] != "wireless_dropouts" for f in result["findings"])


def test_preflight_cli_exit_code_reflects_quality(tmp_path, capsys):
    good = write_recording(tmp_path / "good.txt")
    bad = write_recording(tmp_path / "bad.txt", dead=(3,))
    assert main(["preflight", str(good)]) == 0
    assert main(["preflight", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "EXG Channel 3" in out and "Re-seat" in out
    unreadable = tmp_path / "not_openbci.txt"
    unreadable.write_text("hello\n")
    assert main(["preflight", "--json", str(unreadable)]) == 1
    assert json.loads(capsys.readouterr().out)[0]["verdict"] == "unreadable"


def test_dataset_report_reproduces_replay_and_finds_systematic_electrode(tmp_path):
    raw = tmp_path / "raw" / "Data_clean"
    for person in ("person01", "person02", "person03"):
        for k in range(2):
            write_recording(raw / "G03" / person / "Left" / f"rec{k}.txt", dead=(15,), startup_spike=True,
                            start=1000 + 100 * k)
    write_recording(raw / "G03" / "person04" / "Left" / "rec0.txt")
    write_recording(raw / "G03" / "person04" / "Left" / "rec1.txt", start=2000)
    replay(tmp_path / "raw", tmp_path / "replay")
    summary = build_dataset_report(tmp_path / "replay", tmp_path / "report")
    assert summary["replay_decisions_reproduced"] and summary["windows"] == 32
    assert summary["verdicts"] == {"unusable": 3, "good": 1}
    findings = json.loads((tmp_path / "report" / "findings.json").read_text())
    [systematic] = [f for f in findings["(whole dataset)"] if f["kind"] == "systematic_electrode"]
    assert systematic["channel"] == "EXG Channel 15" and "3 participants" in systematic["evidence"]
    rows = list(csv.DictReader((tmp_path / "report" / "participants.csv").open()))
    p1 = next(r for r in rows if r["participant_id"] == "G03/person01")
    # Only the start-up window (spike on every electrode) stays bad once electrode 15 is excluded.
    assert p1["dead_electrodes"] == "EXG Channel 15" and float(p1["usable_pct_if_dead_excluded"]) == 75.0
    assert float(p1["first_window_fail_pct"]) == 100.0 and float(p1["later_window_fail_pct"]) == 100.0
    html = (tmp_path / "report" / "report.html").read_text()
    assert "<title>EEG Quality Report</title>" in html and "systematic electrode" in html
    for name in ("channels.csv", "recordings.csv", "summary.json"):
        assert (tmp_path / "report" / name).exists()
    with pytest.raises(FileExistsError):
        build_dataset_report(tmp_path / "replay", tmp_path / "report")


def test_dataset_report_refuses_sources_changed_after_replay(tmp_path):
    source = write_recording(tmp_path / "raw" / "G01" / "person01" / "Left" / "rec.txt")
    replay(tmp_path / "raw", tmp_path / "replay")
    source.write_text(source.read_text().replace("Marker Channel", "Marker Channel "))
    with pytest.raises(ValueError, match="changed since the replay"):
        build_dataset_report(tmp_path / "replay", tmp_path / "report")


def test_startup_transient_is_reported_once_for_the_dataset(tmp_path):
    raw = tmp_path / "raw"
    for k in range(4):
        write_recording(raw / "G01" / "person01" / "Left" / f"rec{k}.txt", startup_spike=True, start=1000 + 100 * k)
    replay(raw, tmp_path / "replay")
    build_dataset_report(tmp_path / "replay", tmp_path / "report")
    findings = json.loads((tmp_path / "report" / "findings.json").read_text())
    assert [f["kind"] for f in findings["(whole dataset)"]] == ["startup_transient"]
    assert all(f["kind"] != "startup_transient" for f in findings["G01/person01"])


def test_finding_policy_and_config_are_recorded(tmp_path):
    result = preflight(write_recording(tmp_path / "rec.txt"), QualityConfig())
    assert result["finding_policy"] == FindingPolicy().__dict__
    assert result["quality_policy_id"] == QualityConfig().policy_id
