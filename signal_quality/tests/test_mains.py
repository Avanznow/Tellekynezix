"""Mains-frequency check: peak detection, recommendation rule and exact rejection impact."""

import csv
import json

import numpy as np
import pytest

from signal_quality import QualityAssessor, QualityConfig
from signal_quality.mains import analyse, classify, main, peak_db, write_outputs
from signal_quality.replay import replay


def write_line_noise_recording(path, *, mains, amplitude=60.0, seconds=12, start=1000, seed=0):
    """16-channel 125 Hz OpenBCI file: random 'EEG' plus a mains sinusoid (or none)."""
    fs, c = 125, 16
    n = int(seconds * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    raw = -10000 - 100 * np.arange(c)[:, None] + 15 * rng.standard_normal((c, n))
    if mains:
        raw = raw + amplitude * np.sin(2 * np.pi * mains * t + np.arange(c)[:, None] / 7)
    counters = (2 * np.arange(n)) % 256
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        stream.write("%OpenBCI Raw EXG Data\n%Number of channels = 16\n%Sample Rate = 125 Hz\n"
                     "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n")
        writer = csv.writer(stream)
        writer.writerow(["Sample Index", *[f"EXG Channel {i}" for i in range(c)], "Timestamp", "Marker Channel"])
        for i in range(n):
            writer.writerow([counters[i], *raw[:, i], start + t[i], 0])


def _dataset(tmp_path, mains_by_person):
    raw = tmp_path / "raw"
    for k, (person, mains) in enumerate(mains_by_person.items()):
        for j in range(2):
            write_line_noise_recording(raw / "G09" / person / "Left" / f"r{j}.txt", mains=mains,
                                       start=1000 + 50 * k + 10 * j, seed=10 * k + j)
    replay(raw, tmp_path / "replay")
    return tmp_path / "replay"


def test_peak_height_and_classification():
    freqs = np.arange(0, 62.5, 0.25)
    psd = np.ones_like(freqs)
    psd[(freqs >= 59.75) & (freqs <= 60.25)] = 20.0
    assert peak_db(freqs, psd, 60) == pytest.approx(10 * np.log10(20))
    assert peak_db(freqs, psd, 50) == pytest.approx(0.0)
    assert peak_db(freqs, psd, 100) is None  # above Nyquist
    assert classify(1.0, 13.0) == "60" and classify(12.0, 2.0) == "50"
    assert classify(7.0, 8.0) == "none" and classify(3.0, 4.0) == "none"


def test_60hz_dataset_is_recommended_with_record_only_config(tmp_path):
    replay_dir = _dataset(tmp_path, {"person01": 60, "person02": 60, "person03": 60, "person04": None})
    result = analyse(replay_dir)
    assert result["recommended_mains_hz"] == 60
    assert result["overall"]["clear_60"] == 6 and result["overall"]["clear_50"] == 0
    assert result["overall"]["no_clear_peak"] == 2
    suggested = {k: v for k, v in result["impact"]["suggested_config"].items() if not k.startswith("_")}
    config = QualityConfig(**suggested)
    assert config.mains_hz == 60 and config.max_line_ratio == 1.0
    assert any("notch" in r for r in result["recommendations"])
    record_only = next(r for r in result["impact"]["sweep"] if r["max_line_ratio"] == 1.0)
    assert record_only["currently_usable_newly_rejected"] == 0


def test_50hz_and_inconclusive_datasets(tmp_path):
    assert analyse(_dataset(tmp_path / "a", {"person01": 50, "person02": 50}))["recommended_mains_hz"] == 50
    mixed = analyse(_dataset(tmp_path / "b", {"person01": 50, "person02": 60}))
    assert mixed["recommended_mains_hz"] is None and mixed["impact"] is None
    assert "do not show a consistent mains peak" in mixed["recommendations"][0]


def test_impact_matches_rerunning_the_assessor_with_mains_enabled(tmp_path):
    replay_dir = _dataset(tmp_path, {"person01": 60, "person02": None})
    result = analyse(replay_dir)
    rows = [json.loads(line) for line in (replay_dir / "windows.jsonl").read_text().splitlines()]
    configured = next(r for r in result["impact"]["sweep"] if r["is_configured"])
    # Re-run the real rules with mains enabled on every window and count newly rejected usable windows.
    from signal_quality.acquisition import read_openbci
    assessor = QualityAssessor(QualityConfig(mains_hz=60))
    newly = 0
    for row in rows:
        capture = read_openbci(row["source_path"])
        a, b = row["source_start_sample"], row["source_end_sample_exclusive"]
        window = {k: v for k, v in capture.items() if k != "source_kind"}
        window["raw"], window["counters"], window["timestamps"] = capture["raw"][:, a:b], capture["counters"][a:b], capture["timestamps"][a:b]
        newly += row["rule_usable"] and not assessor.assess(**window)["usable"]
    assert configured["currently_usable_newly_rejected"] == newly > 0


def test_outputs_and_cli(tmp_path, capsys):
    replay_dir = _dataset(tmp_path, {"person01": 60, "person02": 60})
    out = write_outputs(analyse(replay_dir), tmp_path / "out")
    for name in ("report.html", "mains.json", "recordings.csv", "suggested_config.json"):
        assert (out / name).exists()
    assert "<title>Mains Frequency Check</title>" in (out / "report.html").read_text()
    with pytest.raises(FileExistsError):
        write_outputs(analyse(replay_dir), out)
    assert main(["--replay", str(replay_dir), "--out", str(tmp_path / "cli")]) == 0
    assert "Recommended mains_hz: 60" in capsys.readouterr().out
