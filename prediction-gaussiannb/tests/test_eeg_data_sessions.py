"""eeg_data reads cued sessions from prediction_eegnet/record_session.py: one item per trial."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eeg_data as E  # noqa: E402

FS, BLOCK = 125, 250
LABELS = ["left", "right", "takeoff", "landing", "forward", "backward"]


def write_session(directory, *, board="cyton_daisy", rail_trial=None, problems=(), seed=0):
    """16-channel 125 Hz session: 6 trials (baseline 2 s, cue 4 s, rest 1 s) after a 3 s settle."""
    rng = np.random.default_rng(seed)
    trial_len = BLOCK + 2 * BLOCK + 125
    n = 3 * FS + len(LABELS) * trial_len + FS
    data = np.zeros((32, n))
    data[0] = (2 * np.arange(n)) % 256
    data[1:17] = rng.normal(0, 10, (16, n)) - 20_000 - 50 * np.arange(16)[:, None]
    data[30] = 1.7e9 + np.arange(n) / FS
    events, code = [], 1
    for k, label in enumerate(LABELS):
        base = 3 * FS + k * trial_len
        cue, end = base + BLOCK, base + 3 * BLOCK
        if rail_trial == k:                       # second task block: channel 3 stuck at the ADC rail
            data[4, cue + BLOCK:end] = 187_500.0
        for kind, sample in (("baseline", base), ("cue", cue), ("end", end)):
            data[31, sample] = code
            events.append(dict(trial_id=f"r01-{label}", label=label, kind=kind, sample=sample,
                               time_s=sample / FS, board_timestamp=data[30, sample], marker_value=code))
            code += 1
    directory.mkdir(parents=True)
    np.save(directory / "brainflow_raw.npy", data)
    with (directory / "events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    (directory / "session.json").write_text(json.dumps(dict(
        board=board, sampling_rate=FS, eeg_rows=list(range(1, 17)), marker_row=31, timestamp_row=30,
        package_num_row=0, participant_id="G09/person01", problems=list(problems))))
    return directory


def test_each_trial_is_a_labelled_cue_locked_item(tmp_path):
    items = E.load_session(write_session(tmp_path / "s1"), E.LoadConfig())
    assert [i["label"] for i in items] == LABELS
    assert all(i["participant"] == "G09/person01" and i["path"].endswith(i["trial_id"]) for i in items)
    for item in items:
        assert len(item["segments"]) == 1
        seg = item["segments"][0]
        assert seg.shape == (2 * BLOCK, 15)                # cue to end only, Channel 15 excluded
        assert np.isfinite(seg).all() and abs(float(seg.mean())) < 1.0   # band-passed: DC offset removed


def test_failing_block_is_dropped_and_rest_of_trial_kept(tmp_path):
    items = E.load_session(write_session(tmp_path / "s1", rail_trial=2), E.LoadConfig())
    bad = items[2]
    assert bad["n_blocks"] == 2 and bad["n_blocks_kept"] == 1
    assert [s.shape[0] for s in bad["segments"]] == [BLOCK]
    assert "adc_saturation" in bad["block_reasons"][1]


def test_load_recordings_finds_sessions_and_skips_unusable_ones(tmp_path):
    write_session(tmp_path / "G09" / "person01" / "a", seed=1)
    write_session(tmp_path / "G09" / "person01" / "b", problems=["marker 7 found 0 times"], seed=2)
    write_session(tmp_path / "G09" / "person01" / "c", board="cyton", seed=3)
    recs, skipped = E.load_recordings(tmp_path, E.LoadConfig(), n_jobs=1, verbose=False)
    assert len(recs) == 6 and {r["session"] for r in recs} == {str(tmp_path / "G09" / "person01" / "a")}
    reasons = " ".join(s["error"] for s in skipped)
    assert "marker check failed" in reasons and "does not match LoadConfig.board" in reasons
    frame = E.recordings_to_frame(recs)
    assert frame["src_filename"].nunique() == 6 and set(frame["label_txt"]) == set(LABELS)
    assert len(frame) == 6 * 2 * BLOCK


def test_irregular_counters_are_refused(tmp_path):
    d = write_session(tmp_path / "s1")
    data = np.load(d / "brainflow_raw.npy")
    data[0] = np.random.default_rng(0).integers(0, 256, data.shape[1])
    np.save(d / "brainflow_raw.npy", data)
    with pytest.raises(ValueError, match="continuity"):
        E.load_session(d, E.LoadConfig())


def test_load_item_reads_one_session_trial(tmp_path):
    d = write_session(tmp_path / "s1")
    item = E.load_item(f"{d}#r01-takeoff", E.LoadConfig())
    assert item["label"] == "takeoff" and item["segments"][0].shape == (2 * BLOCK, 15)
    with pytest.raises(ValueError, match="not found"):
        E.load_item(f"{d}#r09-left", E.LoadConfig())
