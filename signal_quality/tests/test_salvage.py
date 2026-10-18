"""Salvage export: re-assessed recovery, exact samples, masks, montage options, safeguards."""

import json

import numpy as np
import pytest

from signal_quality.acquisition import read_openbci
from signal_quality.replay import replay
from signal_quality.salvage import SalvagePolicy, folder_label, load_salvaged, main, montage_options, salvage

from test_report import write_recording


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """Four participants (one with a dead Ch 15, one with Ch 2 and Ch 15 dead) and one participant
    with 5 of 16 channels dead (31%, above the 25% exclusion limit)."""
    root = tmp_path_factory.mktemp("salv")
    raw = root / "raw"
    layouts = {"person01": (), "person02": (15,), "person03": (15, 2), "person04": ()}
    for k, (person, dead) in enumerate(layouts.items()):
        for j, folder in enumerate(("Left", "B ackward")):
            write_recording(raw / "G07" / person / folder / f"rec{j}.txt", seconds=10, dead=dead,
                            startup_spike=True, start=1000 + 100 * k + 10 * j)
    write_recording(raw / "G08" / "person01" / "Right" / "rec.txt", seconds=10, dead=(0, 1, 2, 3, 4), start=9000)
    replay(raw, root / "replay")
    return root


def test_dead_electrodes_are_excluded_and_windows_reassessed(replayed, tmp_path):
    result = salvage(replayed / "replay", tmp_path / "out")
    people = result["participants"]
    assert people["G07/person02"]["dead_electrodes"] == ["EXG Channel 15"]
    assert people["G07/person03"]["dead_electrodes"] == ["EXG Channel 2", "EXG Channel 15"]
    assert people["G08/person01"]["excluded"] and "5 of 16" in people["G08/person01"]["reason"]
    # Each 16-channel participant: 2 recordings x 5 windows; start-up windows (spike) fail and are dropped.
    for person in ("G07/person01", "G07/person02", "G07/person03", "G07/person04"):
        assert people[person]["exported"] == 8
    assert people["G07/person02"]["recovered"] == 8 and people["G07/person01"]["recovered"] == 0
    assert result["counts"]["recovered"] == 16 and result["consistent_with_quality_report_estimate"]


def test_exported_samples_are_raw_and_excluded_channels_are_nan(replayed, tmp_path):
    salvage(replayed / "replay", tmp_path / "out")
    board = tmp_path / "out" / "16ch_125hz"
    X, masks = np.load(board / "X.npy"), np.load(board / "mask.npy")
    rows = [json.loads(line) for line in (board / "windows.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        source = read_openbci(row["source_path"])["raw"][:, row["source_start_sample"]:row["source_end_sample_exclusive"]]
        excluded = [int(c.split()[-1]) for c in row["excluded_channels"]]
        keep = [c for c in range(16) if c not in excluded]
        np.testing.assert_array_equal(X[i][keep], source[keep].astype(np.float32))
        assert np.isnan(X[i][excluded]).all() and masks[i].tolist() == [c in keep for c in range(16)]
        assert row["folder_label"] in ("left", "backward") and row["source_start_sample"] != 0


def test_loader_montages(replayed, tmp_path):
    salvage(replayed / "replay", tmp_path / "out")
    common = load_salvaged(tmp_path / "out", "16ch_125hz")
    assert common["X"].shape == (32, 14, 250) and not np.isnan(common["X"]).any()
    assert "EXG Channel 2" not in common["channels"] and "EXG Channel 15" not in common["channels"]
    fifteen = load_salvaged(tmp_path / "out", "16ch_125hz", [f"EXG Channel {i}" for i in range(15)])
    assert fifteen["X"].shape == (24, 15, 250) and set(fifteen["participants"]) == {"G07/person01", "G07/person02", "G07/person04"}
    masked = load_salvaged(tmp_path / "out", "16ch_125hz", "masked")
    assert masked["X"].shape == (32, 16, 250) and masked["mask"].shape == (32, 16)
    with pytest.raises(ValueError, match="Unknown channels"):
        load_salvaged(tmp_path / "out", "16ch_125hz", ["EXG Channel 99"])
    options = json.loads((tmp_path / "out" / "summary.json").read_text())["boards"]["16ch_125hz"]["montage_options"]
    assert [(o["channels_kept"], o["windows"], o["participants"]) for o in options] == [(16, 16, 2), (15, 24, 3), (14, 32, 4)]


def test_montage_options_pick_the_best_set_per_size():
    masks = np.ones((6, 4), dtype=bool)
    masks[0:3, 3] = False          # Ch 3 missing in 3 windows (participant a)
    masks[3, 1] = False            # Ch 1 missing in 1 window (participant b)
    options = montage_options(masks, np.array(list("aaabcc")), [f"Ch {i}" for i in range(4)])
    assert [(o["channels_kept"], o["windows"], o["excluded_channels"]) for o in options] == [
        (4, 2, []), (3, 5, ["Ch 3"]), (2, 6, ["Ch 1", "Ch 3"])]


def test_startup_option_and_safeguards(replayed, tmp_path):
    keep = salvage(replayed / "replay", tmp_path / "keep", SalvagePolicy(drop_startup_window=False))
    assert keep["counts"].get("dropped_startup", 0) == 0  # spiked start-up windows fail anyway
    with pytest.raises(FileExistsError):
        salvage(replayed / "replay", tmp_path / "keep")
    with pytest.raises(ValueError):
        SalvagePolicy(max_excluded_fraction=1.0)
    relaxed = salvage(replayed / "replay", tmp_path / "relaxed", SalvagePolicy(max_excluded_fraction=0.5))
    assert not relaxed["participants"]["G08/person01"]["excluded"]


def test_changed_source_is_refused(replayed, tmp_path):
    import shutil
    raw = tmp_path / "raw"
    shutil.copytree(replayed / "raw", raw)
    replay(raw, tmp_path / "replay")
    target = next(raw.rglob("rec0.txt"))
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="Source changed"):
        salvage(tmp_path / "replay", tmp_path / "out")


@pytest.mark.parametrize("name,label", [("B ackward", "backward"), ("backard", "backward"), ("Fowward", "forward"),
                                        ("OpenBCISession_Takeoff_3", "takeoff"), ("takoff", "takeoff"), ("misc", None)])
def test_folder_labels(name, label):
    assert folder_label(name) == label


def test_cli(replayed, tmp_path, capsys):
    assert main(["--replay", str(replayed / "replay"), "--out", str(tmp_path / "cli")]) == 0
    out = capsys.readouterr().out
    assert "after excluding dead electrodes" in out and "15 channels" in out
    assert "WARNING" in (tmp_path / "cli" / "README.txt").read_text()
