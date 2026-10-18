"""Threshold sensitivity: exact reproduction, equivalence with re-running the assessor, outputs."""

import json

import numpy as np
import pytest

from signal_quality import QualityConfig
from signal_quality.replay import replay
from signal_quality.sensitivity import WindowTable, analyse, main, write_outputs

from test_mains import write_line_noise_recording
from test_report import write_recording


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("sens") / "raw"
    for p in range(1, 5):
        base = root / "G06" / f"person0{p}"
        write_recording(base / "Left" / "clean.txt", seconds=12, start=1000 + p)
        write_recording(base / "Right" / "spikes.txt", seconds=12, spikes=(300, 800, 1300), start=2000 + p)
        write_recording(base / "Takeoff" / "packets.txt", seconds=12, drop_packets=(600,), start=3000 + p)
        if p % 2:
            write_recording(base / "Forward" / "dead.txt", seconds=12, dead=(3,), start=4000 + p)
        # Line interference: strong and weak 60 Hz, and a 50 Hz recording, so line limits flag some windows.
        write_line_noise_recording(base / "Landing" / "line60_strong.txt", mains=60, amplitude=60, start=5000 + p, seed=p)
        write_line_noise_recording(base / "Backward" / "line60_weak.txt", mains=60, amplitude=4, start=6000 + p, seed=10 + p)
        write_line_noise_recording(base / "Right" / "line50.txt", mains=50, amplitude=60, start=7000 + p, seed=20 + p)
    return root


@pytest.fixture(scope="module")
def baseline_replay(raw_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("replay") / "baseline"
    replay(raw_root, out)
    return out


def _rows(replay_dir):
    return [json.loads(line) for line in (replay_dir / "windows.jsonl").read_text().splitlines()]


def test_baseline_reproduces_every_replay_decision(baseline_replay):
    result = analyse(baseline_replay)
    assert result["verification"]["replay_decisions_reproduced"]
    rows = _rows(baseline_replay)
    assert result["baseline"]["rejection_rate"] == pytest.approx(1 - np.mean([r["rule_usable"] for r in rows]))


@pytest.mark.parametrize("changes", [dict(max_ptp_uv=3000, max_jump_uv=2500), dict(max_ptp_uv=300),
                                     dict(max_jump_uv=100), dict(max_rail_fraction=0.5, max_flat_fraction=0.9),
                                     dict(mains_hz=60, max_line_ratio=0.5), dict(mains_hz=50, max_line_ratio=0.5)])
def test_model_matches_rerunning_the_assessor_with_other_thresholds(raw_root, baseline_replay, tmp_path, changes):
    config = QualityConfig(**changes)
    rerun = tmp_path / "rerun"
    replay(raw_root, rerun, config)
    table = WindowTable(_rows(baseline_replay))
    expected = np.array([r["rule_usable"] for r in _rows(rerun)])
    np.testing.assert_array_equal(table.usable(config), expected)
    if "mains_hz" in changes:  # the line rule must actually change decisions, not pass trivially
        baseline = np.array([r["rule_usable"] for r in _rows(baseline_replay)])
        assert (baseline & ~expected).sum() > 0 and (baseline & expected).sum() > 0


def test_tampered_replay_is_detected(baseline_replay, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(baseline_replay, copy)
    rows = _rows(copy)
    rows[3]["rule_usable"] = not rows[3]["rule_usable"]
    (copy / "windows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="disagrees with the replay"):
        analyse(copy)


def test_sweeps_are_monotonic_and_include_the_current_value(baseline_replay):
    result = analyse(baseline_replay)
    for param, points in result["sweeps"].items():
        assert sum(p["is_baseline"] for p in points) == 1
        rates = [p["rejection_rate"] for p in points]
        # A looser limit (larger value) can only accept more windows.
        assert all(a >= b - 1e-12 for a, b in zip(rates, rates[1:])), param
    base = next(p for p in result["sweeps"]["max_ptp_uv"] if p["is_baseline"])
    assert base["newly_rejected"] == base["newly_accepted"] == 0
    assert len(result["grid"]) == 100 and result["participants"] == 4


def test_margins_show_dead_electrodes_are_unambiguous(baseline_replay):
    margins = analyse(baseline_replay)["margins"]
    assert margins["adc_saturation"]["borderline_within_1_5x"] == 0
    assert margins["flat_channel"]["far_beyond_2x"] == margins["flat_channel"]["windows_flagged"] > 0
    assert margins["excessive_amplitude"]["windows_flagged"] > 0


def test_review_curves_use_majority_labels_and_weights(baseline_replay, tmp_path):
    rows = _rows(baseline_replay)
    reviews = tmp_path / "reviews.jsonl"
    # Reviewer agrees with the current rules everywhere: zero error at the current setting.
    reviews.write_text("\n".join(json.dumps(dict(window_id=r["window_id"], reviewed_quality_label=int(r["rule_usable"]),
                                                 sampling_weight=2.0)) for r in rows))
    result = analyse(baseline_replay, [reviews])
    assert result["reviewed_windows"] == len(rows)
    base = next(p for p in result["sweeps"]["max_jump_uv"] if p["is_baseline"])
    assert base["reviews"]["false_acceptance_rate"] == 0 and base["reviews"]["false_rejection_rate"] == 0
    strict = result["sweeps"]["max_jump_uv"][0]
    if strict["newly_rejected"]:
        assert strict["reviews"]["false_rejection_rate"] > 0


def test_outputs_and_cli(baseline_replay, tmp_path, capsys):
    out = write_outputs(analyse(baseline_replay), tmp_path / "out")
    for name in ("report.html", "sensitivity.json", "sweeps.csv", "grid.csv", "participants.csv", "margins.csv"):
        assert (out / name).exists()
    html = (out / "report.html").read_text()
    assert "<title>Threshold Sensitivity</title>" in html and "NaN" not in html
    with pytest.raises(FileExistsError):
        write_outputs(analyse(baseline_replay), out)
    assert main(["--replay", str(baseline_replay), "--out", str(tmp_path / "cli")]) == 0
    assert "Baseline reproduced" in capsys.readouterr().out


def test_line_rule_is_modelled_and_swept_when_mains_is_set(raw_root, tmp_path):
    from signal_quality.config import load_default_config
    config = load_default_config()
    assert config.mains_hz == 60 and config.max_line_ratio == 1.0
    replay(raw_root, tmp_path / "replay", config)
    result = analyse(tmp_path / "replay")
    assert result["verification"]["replay_decisions_reproduced"]
    line = result["sweeps"]["max_line_ratio"]
    assert [p["value"] for p in line][-1] == 1.0 and line[-1]["is_baseline"]
    assert line[-1]["newly_rejected"] == 0 and "excessive_line_noise" in result["margins"]
