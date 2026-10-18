"""Stratified blind review sets: sampling, blinding, weights, import validation and weighted agreement."""

from collections import defaultdict
import json
import re

import pytest

from signal_quality import QualityConfig
from signal_quality.dashboard import agreement
from signal_quality.replay import replay
from signal_quality.review_set import build_review_set, category, import_review_record, main
from signal_quality.train import reviewed_features

from test_report import write_recording


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Six participants with clean, dead-electrode, packet-loss, artifact and start-up windows."""
    root = tmp_path_factory.mktemp("rs")
    raw = root / "raw"
    for p in range(1, 7):
        base = raw / "G05" / f"person0{p}"
        write_recording(base / "Left" / "clean.txt", seconds=12, start=1000 + p)
        write_recording(base / "Right" / "startup.txt", seconds=12, startup_spike=True, start=2000 + p)
        write_recording(base / "Takeoff" / "packets.txt", seconds=12, drop_packets=(600,), start=3000 + p)
        write_recording(base / "Landing" / "artifact.txt", seconds=12, spikes=(900,), start=4000 + p)
        if p % 2:
            write_recording(base / "Forward" / "dead.txt", seconds=12, dead=(7,), start=5000 + p)
    replay(raw, root / "replay")
    return root


def _page_payload(html):
    return json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S).group(1))


def test_sample_is_stratified_weighted_and_blind(dataset, tmp_path):
    manifest = build_review_set(dataset / "replay", tmp_path / "set", size=40, seed=1)
    rows = [json.loads(line) for line in (dataset / "replay" / "windows.jsonl").read_text().splitlines()]
    population = defaultdict(int)
    for row in rows:
        population[category(row)] += 1
    items = manifest["items"]
    assert len(items) == 40 and sum(i["rule_usable"] for i in items) == 20
    assert {i["stratum"] for i in items} == {"pass", "startup", "electrode", "continuity", "artifact"}
    # Weights reproduce the replay's window counts in every sampled stratum.
    weights = defaultdict(float)
    for item in items:
        weights[item["stratum"]] += item["sampling_weight"]
    assert {k: round(v) for k, v in weights.items()} == {k: population[k] for k in weights}
    assert manifest["participants"] == 6
    # 5 windows per rejection stratum cannot cover 6 participants: the gap is recorded, not hidden.
    assert any(p["windows_from_unsampled_participants"] > 0 for p in manifest["plan"].values())
    # The page carries only what a reviewer needs: no rule decisions, reasons, participants or files.
    html = (tmp_path / "set" / "review_set.html").read_text()
    payload = _page_payload(html)
    assert set(payload) == {"items", "record_format", "review_set_id", "rubric"}
    assert set(payload["items"][0]) == {"item_id", "channels", "sfreq", "samples", "signal"}
    visible = html.replace(re.search(r'"items":\[.*\]', html, re.S).group(0), "")
    for leak in ("rule_usable", "adc_saturation", "packet_discontinuity", "person0", "G05", "clean.txt", "stratum"):
        assert leak not in visible and all(leak not in json.dumps({k: v for k, v in it.items() if k != "signal"})
                                           for it in payload["items"])
    assert "Keep manifest.json" in (tmp_path / "set" / "README.txt").read_text()


def test_same_seed_same_set_and_size_limits(dataset, tmp_path):
    a = build_review_set(dataset / "replay", tmp_path / "a", size=20, seed=3)
    b = build_review_set(dataset / "replay", tmp_path / "b", size=20, seed=3)
    c = build_review_set(dataset / "replay", tmp_path / "c", size=20, seed=4)
    assert a["review_set_id"] == b["review_set_id"] != c["review_set_id"]
    with pytest.raises(ValueError, match="between 10 and 2000"):
        build_review_set(dataset / "replay", tmp_path / "d", size=5)
    with pytest.raises(FileExistsError):
        build_review_set(dataset / "replay", tmp_path / "a", size=20)


def _record(path, manifest, label_of, reviewer="alice", unsure=()):
    reviews = [dict(item_id=item["item_id"], label=label_of(item), unsure=item["item_id"] in unsure, note="",
                    reviewed_at_utc="2026-10-08T00:00:00.000Z", seconds=3.5)
               for item in manifest["items"]]
    path.write_text(json.dumps(dict(format="avatar-quality-review-set-v1", review_set_id=manifest["review_set_id"],
                                    reviewer=reviewer, exported_at_utc="2026-10-08T00:00:00.000Z", reviews=reviews)))
    return path


def test_imported_reviews_train_and_give_weighted_agreement(dataset, tmp_path):
    set_dir = tmp_path / "set"
    manifest = build_review_set(dataset / "replay", set_dir, size=40, seed=2)
    # Alice agrees with the rules except that she accepts every start-up window.
    alice = _record(tmp_path / "alice.json", manifest,
                    lambda i: 1 if i["rule_usable"] or i["stratum"] == "startup" else 0,
                    unsure={manifest["items"][0]["item_id"]})
    bob = _record(tmp_path / "bob.json", manifest, lambda i: int(i["rule_usable"]), reviewer="bob")
    result = import_review_record(alice, set_dir, tmp_path / "reviews" / "alice.jsonl")
    assert result == dict(rows=40, of_items=40, reviewer="alice", missing=0, strata=result["strata"])
    import_review_record(bob, set_dir, tmp_path / "reviews" / "bob.jsonl")
    X, y, groups, _, _ = reviewed_features(tmp_path / "reviews" / "bob.jsonl", QualityConfig())
    assert len(y) == 40 and set(y) == {0, 1} and len(set(groups)) == 6

    report = agreement([tmp_path / "reviews" / "alice.jsonl", tmp_path / "reviews" / "bob.jsonl"])
    bob_stats, alice_stats = report["reviewer_vs_rules"]["bob"], report["reviewer_vs_rules"]["alice"]
    assert bob_stats["agreement"] == 1 and bob_stats["false_acceptance_rate"] == 0
    assert bob_stats["weighted_to_replay"]["agreement"] == pytest.approx(1)
    startup = sum(i["stratum"] == "startup" for i in manifest["items"])
    assert alice_stats["false_rejection_rate"] == pytest.approx(startup / (20 + startup))
    assert alice_stats["reviewer_acceptable_share_by_stratum"]["startup"]["reviewer_acceptable"] == 1
    assert alice_stats["uncertain_reviews"] == 1 and alice_stats["median_review_seconds"] == 3.5
    # Weighted false rejection uses replay prevalence: start-up windows are rarer in the replay than in the sample.
    weights = {i["item_id"]: i["sampling_weight"] for i in manifest["items"]}
    w_start = sum(w for i, w in weights.items() if next(x for x in manifest["items"] if x["item_id"] == i)["stratum"] == "startup")
    w_pass = sum(w for i, w in weights.items() if next(x for x in manifest["items"] if x["item_id"] == i)["rule_usable"])
    assert alice_stats["weighted_to_replay"]["false_rejection_rate"] == pytest.approx(w_start / (w_start + w_pass))
    pair = report["reviewer_vs_reviewer"]["alice vs bob"]
    assert pair["shared_windows"] == 40 and pair["agreement"] == pytest.approx((40 - startup) / 40)


@pytest.mark.parametrize("tamper,message", [
    ("set", "does not belong"), ("repeat", "unknown or repeated"), ("unknown", "unknown or repeated"),
    ("label", "label must be"), ("source", "changed since the review set")])
def test_import_rejects_invalid_records(dataset, tmp_path, tamper, message):
    import shutil
    raw = tmp_path / "raw"
    shutil.copytree(dataset / "raw", raw)
    replay(raw, tmp_path / "replay")
    manifest = build_review_set(tmp_path / "replay", tmp_path / "set", size=20, seed=5)
    path = _record(tmp_path / "r.json", manifest, lambda i: 1)
    record = json.loads(path.read_text())
    if tamper == "set":
        record["review_set_id"] = "rs-0000000000000000"
    elif tamper == "repeat":
        record["reviews"][1]["item_id"] = record["reviews"][0]["item_id"]
    elif tamper == "unknown":
        record["reviews"][0]["item_id"] = "item-9999"
    elif tamper == "label":
        record["reviews"][0]["label"] = "good"
    else:
        target = manifest["items"][0]["source_path"]
        with open(target, "a") as stream:
            stream.write("\n")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        import_review_record(path, tmp_path / "set", tmp_path / "out.jsonl")


def test_cli_build_and_import(dataset, tmp_path, capsys):
    assert main(["build", "--replay", str(dataset / "replay"), "--out", str(tmp_path / "set"), "--size", "20"]) == 0
    assert "keep manifest.json private" in capsys.readouterr().out
    manifest = json.loads((tmp_path / "set" / "manifest.json").read_text())
    record = _record(tmp_path / "r.json", manifest, lambda i: int(i["rule_usable"]))
    assert main(["import", str(record), "--set", str(tmp_path / "set"), "--out", str(tmp_path / "a.jsonl")]) == 0
    assert '"missing": 0' in capsys.readouterr().out
