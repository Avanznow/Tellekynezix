"""The subject-specific GaussianNB training flow and live path reproduce the notebook's per-participant results.

Pushes one Data_clean participant's recordings through prediction-gaussiannb/pytorch/subject_gnb_inference.py:
  * the TRAIN recordings of the notebook's split (cell 7: StratifiedGroupKFold(5, shuffle, seed 42) by recording)
    through the training flow (every recording = one labelled cue) -> the user's statistics and model equal the
    notebook's torch_gaussiannb_subject_scaler.joblib / torch_gaussiannb_subject_models.joblib entries for that
    participant (cell 18);
  * the TEST recordings through the live path -> the per-window accuracy of gaussiannb_subject_specific_per_participant.csv;
  * the three views and 435 features equal the notebook's subject feature cache (cell 17) window for window.
Also: schedule interleaving, the vote-agreement gate on gated captures, persistence, the guided session runner on a
replayed stream, and the CLI. Skipped when Data_clean, the cache or the notebook's artifacts are not present.
"""
import collections
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

GNB_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = GNB_DIR / "pytorch"
REPO_ROOT = GNB_DIR.parent
for _p in (MODEL_DIR, GNB_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import subject_gnb_inference as S  # noqa: E402
import eeg_data as E  # noqa: E402
from improved_gnb_features import participant_of  # noqa: E402
from signal_quality import QualityAssessor, QualityGatedPredictor, may_execute  # noqa: E402
from signal_quality.acquisition import read_openbci  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402
from signal_quality.salvage import folder_label  # noqa: E402

CACHE_DIR = REPO_ROOT / "predictions_local" / "feature_cache"
PER_PARTICIPANT_CSV = MODEL_DIR / "gaussiannb_subject_specific_per_participant.csv"
LEARNING_CURVE_CSV = MODEL_DIR / "gaussiannb_subject_learning_curve.csv"
PREFERRED = "G03/person03"
BLOCK = 250


# ----------------------------------------------------------------------------------------------- the notebook's code
def notebook_split(X, y, groups):
    """Cell 7, verbatim: one fold of a stratified group split by recording, seed 42."""
    from sklearn.model_selection import StratifiedGroupKFold
    group_codes = np.unique(groups, return_inverse=True)[1]
    return next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=group_codes))


def notebook_context_logp(logp, k=5):
    cs = np.cumsum(logp, axis=0)
    ctx = cs.copy()
    ctx[k:] = cs[k:] - cs[:-k]
    return ctx


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def split():
    """Subject feature cache of the notebook + its split: dict with X, y, groups, parts, train mask."""
    caches = sorted(CACHE_DIR.glob("subject_windows_*.npz"), key=lambda f: f.stat().st_mtime)
    if not caches:
        pytest.skip("notebook subject feature cache not present")
    z = np.load(caches[-1], allow_pickle=True)
    X, y, groups = z["X"].astype(np.float32), z["y"].astype(np.int64), z["groups"].astype(str)
    assert X.shape[1] == 435
    tr, _ = notebook_split(X, y, groups)
    train = np.zeros(len(y), dtype=bool)
    train[tr] = True
    return dict(X=X, y=y, groups=groups, parts=np.array([participant_of(g) for g in groups]), train=train)


@pytest.fixture(scope="module")
def participant(split):
    """A Data_clean participant with an own notebook model whose recordings are all present: (id, train paths, test paths)."""
    if not PER_PARTICIPANT_CSV.exists():
        pytest.skip("per-participant csv not present")
    with PER_PARTICIPANT_CSV.open() as stream:
        rows = {r["Participant"]: r for r in csv.DictReader(stream)}
    candidates = [PREFERRED] + sorted(q for q in rows if q != PREFERRED)
    for q in candidates:
        if q not in rows or rows[q]["Model"] != "own":
            continue
        m = split["parts"] == q
        paths = sorted(np.unique(split["groups"][m]))
        if paths and all(Path(p).exists() for p in paths):
            train_paths = sorted(np.unique(split["groups"][m & split["train"]]))
            test_paths = sorted(np.unique(split["groups"][m & ~split["train"]]))
            return q, train_paths, test_paths, float(rows[q]["Accuracy"]), int(rows[q]["Test windows"])
    pytest.skip("no Data_clean participant with all recordings present on this machine")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    for name in (S.SUBJECT_MODELS_FILE, S.SUBJECT_SCALER_FILE):
        if not (MODEL_DIR / name).exists():
            pytest.skip(f"{name} not present")
    return S.SubjectSpecificGaussianNBChain(users_dir=tmp_path_factory.mktemp("gnb_users"))


@pytest.fixture(scope="module")
def trained(chain, participant):
    """The participant trained through the flow from the notebook's TRAIN recordings (one cue per recording)."""
    q, train_paths, _, _, _ = participant
    labelled = [(p, folder_label(Path(p).parent.name)) for p in train_paths]
    return chain.train_from_recordings(q, labelled, session_id="notebook-train-split")


def assessor():
    return QualityAssessor(load_default_config())


def capture_slice(cap, start, n, **extra):
    out = dict(cap)
    for key in ("raw", "counters", "timestamps"):
        out[key] = cap[key][:, start:start + n] if key == "raw" else cap[key][start:start + n]
    out.update(captured_monotonic=time.monotonic(), stream_id=f"replay:{start}:{n}", **extra)
    return out


# ----------------------------------------------------------------------------------------------- pure parts
def test_schedule_is_interleaved_and_reproducible():
    for seed in (0, 1, 7, 42):
        schedule = S.build_schedule(S.COMMANDS, rounds=6, seed=seed)
        assert len(schedule) == 36 and [c["index"] for c in schedule] == list(range(36))
        for r in range(1, 7):
            commands = [c["command"] for c in schedule if c["round"] == r]
            assert sorted(commands) == sorted(S.COMMANDS), "every round presents every command once"
        assert all(a["command"] != b["command"] for a, b in zip(schedule, schedule[1:])), "no command twice in a row"
        assert schedule == S.build_schedule(S.COMMANDS, rounds=6, seed=seed)
    assert S.build_schedule(S.COMMANDS, 6, 1) != S.build_schedule(S.COMMANDS, 6, 2)
    plan = S.TrainingPlan(rounds=6, cue_seconds=20, rest_seconds=5, seed=3)
    assert plan.n_cues == 36 and abs(plan.total_seconds - (3 + 36 * 25)) < 1e-9
    d = plan.as_dict()
    assert d["total_minutes"] == 15.1 and "75 %" in d["expectation"] and "81 %" in d["expectation"]
    with pytest.raises(ValueError):
        S.TrainingPlan(rounds=1, cue_seconds=3)


def test_learning_curve_expectation_matches_the_notebook():
    assert S.expected_accuracy(0) == pytest.approx(1 / 6)
    assert S.expected_accuracy(4) == pytest.approx(0.745, abs=0.01) and S.expected_accuracy(6) == pytest.approx(0.813, abs=0.01)
    assert S.expected_accuracy(5) == pytest.approx((0.745 + 0.813) / 2, abs=0.01)
    assert S.expected_accuracy(100) == pytest.approx(0.854, abs=0.01)
    if LEARNING_CURVE_CSV.exists():
        with LEARNING_CURVE_CSV.open() as stream:
            rows = {r["Recordings per command (k)"]: float(r["Accuracy"]) for r in csv.DictReader(stream)}
        for k, acc in S.LEARNING_CURVE:
            assert rows[str(k)] == pytest.approx(acc, abs=5e-4), k


def test_vote_gate_decision_rule(chain):
    classes = len(S.COMMANDS)
    one_hot = lambda i: np.log(np.full(classes, 1e-3)) + np.eye(classes)[i] * 10  # noqa: E731
    certain = np.stack([one_hot(3)] * 5)
    d = chain.decide(certain)
    assert d["label"] == "left" and d["certain"] and d["agreement"] == 5 and d["votes"] == 5 and d["votes_needed"] == 4
    mixed = np.stack([one_hot(3), one_hot(1), one_hot(3), one_hot(1), one_hot(3)])    # 3 of 5 agree -> uncertain
    d = chain.decide(mixed)
    assert d["label"] == "uncertain" and not d["certain"] and d["command"] == "left" and d["agreement"] == 3
    four = np.stack([one_hot(1), one_hot(3), one_hot(3), one_hot(3), one_hot(3)])     # 4 of 5 -> act
    assert chain.decide(four)["label"] == "left"
    assert chain.decide(four, min_agreement=5)["label"] == "uncertain"
    assert chain.decide(np.stack([one_hot(2)] * 3))["label"] == "uncertain"           # fewer than 4 votes never acts
    assert chain.decide(np.stack([one_hot(2)] * 12))["window_labels"] == ["landing"] * 12
    np.testing.assert_allclose(notebook_context_logp(mixed)[-1], S.context_log_posteriors(mixed)[-1])


# ----------------------------------------------------------------------------------------------- notebook equivalence
def test_three_views_and_features_equal_the_notebook_cache(chain, participant, split):
    q, train_paths, test_paths, _, _ = participant
    n_checked = 0
    for path in train_paths + test_paths:
        cap = read_openbci(path)
        X_live, report = chain.features_from_capture(cap, assessor())
        rows = np.where(split["groups"] == path)[0]
        assert report["block_reasons"][0] == ["startup_block"]
        assert X_live.shape == (len(rows), 435)
        np.testing.assert_allclose(X_live, split["X"][rows], rtol=1e-5, atol=1e-5)
        # the block segmentation is the loader's
        loaded = E.load_recording(path, chain.config)
        assert (report["n_blocks"], report["n_blocks_kept"]) == (loaded["n_blocks"], loaded["n_blocks_kept"])
        segments, _ = chain.segments_from_capture(cap, assessor())
        assert [s.shape[1] for s in segments] == [s.shape[0] for s in loaded["segments"]]
        n_checked += len(rows)
    assert n_checked >= 100


def test_training_flow_reproduces_the_notebook_participant_model(chain, participant, split, trained):
    q, train_paths, _, _, _ = participant
    m = (split["parts"] == q) & split["train"]
    X_train, y_train = split["X"][m].astype(np.float64), split["y"][m]
    # the user's labelled windows are the notebook's training windows of this participant
    X_stored, y_stored, cue_ids, session_ids, _ = chain.training_windows(q)
    assert X_stored.shape == X_train.shape and np.array_equal(y_stored, y_train)
    np.testing.assert_allclose(X_stored, X_train, rtol=1e-5, atol=1e-5)
    assert set(session_ids) == {"notebook-train-split"} and len(np.unique(cue_ids)) == len(train_paths)
    # statistics = the notebook scaler's statistics of this participant (fitted on the same training windows)
    mean_nb, std_nb = chain.notebook_scaler.stats_[q]
    np.testing.assert_allclose(trained.mean, mean_nb, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(trained.std, std_nb, rtol=1e-6, atol=1e-6)
    # model = the notebook's per-participant model: same posteriors on the training windows
    model_nb = chain.notebook_models.models_[q]
    assert trained.model is not None and list(trained.model.classes_) == list(model_nb.classes_)
    assert trained.model.projection_.shape == model_nb.projection_.shape == (435, 5)
    Z = (X_train - trained.mean) / trained.std
    np.testing.assert_allclose(trained.model.predict_log_proba(Z), model_nb.predict_log_proba(Z), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.abs(trained.model.projection_), np.abs(model_nb.projection_), rtol=1e-6, atol=1e-8)
    # what the GUI shows
    assert trained.status().startswith("own model (") and trained.n_windows == len(y_train)
    assert sum(trained.cues_per_command.values()) == len(train_paths)
    info = chain.participant_info(q)
    assert info["has_model"] and info["expected_accuracy"] == pytest.approx(S.expected_accuracy(trained.min_cues))
    assert info["cue_cv"]["folds"] >= 2 and 0.5 < info["cue_cv"]["accuracy"] <= 1.0
    assert info["sessions"][0]["id"] == "notebook-train-split" and info["sessions"][0]["n_cues"] == len(train_paths)
    # stored under users_dir/<id>/ and the notebook's artifacts are untouched
    d = chain.user_dir(q)
    assert sorted(p.name for p in d.iterdir()) == ["meta.json", "model.npz", "scaler.npz", "training_windows.npz"]
    meta = json.loads((d / "meta.json").read_text())
    assert meta["feature_version"] == S.SUBJECT_FEATURE_VERSION and meta["model"] == "own"
    assert set(meta["windows_per_command"]) == set(S.COMMANDS)


def test_live_path_gives_the_notebook_per_participant_accuracy(chain, participant, split, trained):
    q, _, test_paths, accuracy_csv, n_test_csv = participant
    correct = total = 0
    window_labels = {}
    for path in test_paths:
        truth = folder_label(Path(path).parent.name)
        out = chain.replay_recording(path, q)
        assert out["participant_status"].startswith("own model") and out["truth"] == truth
        for seg in out["segments"]:
            correct += sum(w == truth for w in seg["window_labels"])
            total += seg["windows"]
            assert seg["label"] in (seg["command"], "uncertain") and seg["agreement"].endswith("/5") or seg["windows"] < 5
        window_labels[path] = [w for seg in out["segments"] for w in seg["window_labels"]]
    assert total == n_test_csv
    assert correct / total == pytest.approx(accuracy_csv, abs=1e-12)
    # the same per-window decisions as the notebook's artifacts on the cached test windows
    m = (split["parts"] == q) & ~split["train"]
    Z = chain.notebook_scaler.transform(split["X"][m], split["parts"][m])
    pred_nb = chain.notebook_models.predict(Z, split["parts"][m])
    live = np.concatenate([[S.COMMANDS.index(w) for w in window_labels[p]] for p in sorted(np.unique(split["groups"][m]))])
    assert np.array_equal(live, pred_nb)
    # the vote gate only passes decisions that agree: on the accepted segments the accuracy is at least the window accuracy
    certain = [(seg["command"] == folder_label(Path(p).parent.name)) for p in test_paths
               for seg in chain.replay_recording(p, q)["segments"] if seg["certain"]]
    assert certain and np.mean(certain) >= accuracy_csv


def test_quality_gated_capture_applies_the_vote_gate(chain, participant, trained):
    q, _, test_paths, _, _ = participant
    gate = QualityGatedPredictor(QualityAssessor(load_default_config()))
    n = chain.capture_samples
    assert n == 384 and abs(chain.capture_seconds - 384 / 125) < 1e-12
    accepted = withheld = None
    for path in test_paths:
        cap = read_openbci(path)
        for start in range(BLOCK, cap["raw"].shape[1] - n, 64):
            live = capture_slice(cap, start, n)
            result = chain.predict_capture(live, q, gate)
            d = chain.last_decision
            if result["status"] == "accepted" and accepted is None:
                assert may_execute(result) and d["certain"] and d["agreement"] >= 4 and d["windows"] == 5
                assert result["prediction_label"] == ("land" if d["command"] == "landing" else d["command"])
                assert result["confidence"] == d["confidence"]
                accepted = result
            elif result["status"] == "withheld" and d is not None and withheld is None:
                assert not d["certain"] and d["label"] == "uncertain"
                assert result["withholding_reason"].startswith("uncertain_decision: ") and str(d["agreement"]) in result["withholding_reason"]
                assert result["quality"]["usable"] is True and not may_execute(result)
                withheld = result
            if accepted is not None and withheld is not None:
                break
        if accepted is not None and withheld is not None:
            break
    assert accepted is not None, "no accepted decision in the test recordings"
    assert withheld is not None, "no uncertain decision in the test recordings"
    # a stricter gate (all 5 must agree) withholds a 4/5 decision; synthetic and short captures never act
    cap = read_openbci(test_paths[0])
    live = capture_slice(cap, BLOCK, n)
    direct = chain.predict_raw(live["raw"], live["channels"], live["sfreq"], q)
    if direct["agreement"] == 4:
        assert chain.predict_capture(live, q, gate, min_agreement=5)["withholding_reason"].startswith("uncertain_decision")
    assert chain.predict_capture(capture_slice(cap, BLOCK, n, source_kind="synthetic"), q, gate)["status"] == "withheld"
    assert chain.last_decision is None
    assert chain.predict_capture(capture_slice(cap, BLOCK, 100), q, gate)["status"] == "withheld"
    # BrainFlow channel names of the 16-channel board map by position
    brainflow = chain.select_channels(cap["raw"], [f"EEG_{i}" for i in range(1, 17)])
    assert np.array_equal(brainflow, chain.select_channels(cap["raw"], cap["channels"]))
    with pytest.raises(ValueError):
        chain.features_from_raw(cap["raw"], cap["channels"], sfreq=250.0)


def test_users_persist_reload_fall_back_and_forget(chain, participant, trained, tmp_path):
    q, train_paths, test_paths, _, _ = participant
    again = S.SubjectSpecificGaussianNBChain(users_dir=chain.users_dir)
    assert again.participant_status(q) == trained.status() and again.has_own_model(q)
    np.testing.assert_allclose(again.users[q].mean, trained.mean)
    X = chain.features_from_capture(read_openbci(test_paths[0]), assessor())[0]
    np.testing.assert_allclose(again.log_posteriors(X, q), chain.log_posteriors(X, q), atol=1e-6)
    # an unknown user falls back to the notebook's global chain with the global statistics
    assert chain.participant_status("nobody") == "no model yet: global fallback"
    mean, std, model, status = chain.resolve("nobody")
    assert status == "global fallback" and model is chain.notebook_models.global_
    np.testing.assert_allclose(mean, chain.notebook_scaler.global_mean_)
    assert chain.predict_windows(X, "nobody")["participant_status"] == "global fallback"
    # a notebook training participant without an own model uses the notebook's per-participant model
    other = next(p for p in chain.notebook_models.models_ if p != q)
    assert chain.participant_status(other) == "training participant" and chain.resolve(other)[2] is chain.notebook_models.models_[other]
    # too few windows: statistics are stored, the model falls back, and more sessions accumulate
    short = chain.train_from_recordings("newcomer", [(train_paths[0], folder_label(Path(train_paths[0]).parent.name)),
                                                      (train_paths[-1], folder_label(Path(train_paths[-1]).parent.name))],
                                        session_id="s1")
    assert short.model is None and short.status().startswith("too few labelled windows") and short.n_windows < S.MIN_WINDOWS
    assert chain.resolve("newcomer")[3] == "global fallback (own statistics)"
    more = chain.train_from_recordings("newcomer", [(p, folder_label(Path(p).parent.name)) for p in train_paths[:12]], session_id="s2")
    assert more.n_windows == short.n_windows + sum(len(chain.features_from_capture(read_openbci(p), assessor())[0]) for p in train_paths[:12])
    assert [s["id"] for s in more.meta["sessions"]] == ["s1", "s2"]
    assert (more.model is not None) == (more.n_windows >= S.MIN_WINDOWS and len(more.meta["commands"]) >= 2)
    chain.forget_user("newcomer")
    assert not chain.user_dir("newcomer").exists() and chain.participant_status("newcomer").startswith("no model yet")
    with pytest.raises(ValueError):
        chain.train_user("", X[:5], np.zeros(5, int), ["c"] * 5, dict(id="x"))
    with pytest.raises(ValueError):
        chain.train_user("bad", X[:5], np.full(5, 9), ["c"] * 5, dict(id="x"))


# ----------------------------------------------------------------------------------------------- the guided session
class ReplayStream:
    """Stands in for BrainFlowStream: serves the samples of Data_clean recordings at 125 Hz against a fake clock."""

    def __init__(self, paths, clock):
        caps = [read_openbci(p) for p in paths]
        self.raw = np.concatenate([c["raw"] for c in caps], axis=1)
        self.counters = np.concatenate([c["counters"] for c in caps])
        self.timestamps = np.concatenate([c["timestamps"] for c in caps])
        self.template = {k: v for k, v in caps[0].items() if k not in ("raw", "counters", "timestamps")}
        self.clock, self.pos, self.t0, self.started, self.stopped, self.reads = clock, 0, None, 0, 0, 0

    def start(self):
        self.started += 1
        self.t0 = self.clock()

    def _available(self):
        return min(int((self.clock() - self.t0) * 125), self.raw.shape[1])

    def discard(self):
        self.pos = self._available()

    def read(self):
        self.reads += 1
        a, b = self.pos, self._available()
        self.pos = b
        return dict(self.template, raw=self.raw[:, a:b], counters=self.counters[a:b], timestamps=self.timestamps[a:b],
                    captured_monotonic=self.clock(), stream_id=f"replay:{self.reads}")

    def stop(self):
        self.stopped += 1


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


def test_guided_session_collects_labelled_windows_and_trains_the_user(chain, participant):
    q, train_paths, _, _, _ = participant
    clock = FakeClock()
    stream = ReplayStream(train_paths[:30], clock)
    plan = S.TrainingPlan(rounds=2, cue_seconds=10, rest_seconds=1, seed=5, settle_seconds=0.5)
    events = []
    session = S.TrainingSession(chain, "tester", plan, stream, present=events.append, sleep=clock.sleep, clock=clock,
                                session_id="unit-session")
    summary = session.run()
    assert stream.started == 1 and stream.stopped == 1 and stream.reads == plan.n_cues == 12
    phases = [e["phase"] for e in events]
    assert phases[0] == "settle" and phases[-1] == "finished" and phases.count("cue_done") == 12
    cue_events = [e for e in events if e["phase"] == "cue"]
    assert [e["command"] for e in cue_events if e["remaining"] == 10] == [c["command"] for c in plan.schedule]
    assert all(e["remaining"] >= 1 for e in cue_events) and {e["remaining"] for e in cue_events} == set(range(1, 11))
    assert all("n_cues" in e and "windows_total" in e and e["session_id"] == "unit-session" for e in events)
    # every cue: 10 s = 5 blocks, the first dropped -> up to 4 kept blocks -> up to 15 windows labelled with the cue
    assert summary["cues_done"] == 12 and not summary["stopped_early"]
    assert all(c["n_blocks"] == 5 and 0 <= c["n_windows"] <= 15 for c in summary["windows_per_cue"])
    assert summary["windows_total"] == sum(c["n_windows"] for c in summary["windows_per_cue"]) > S.MIN_WINDOWS
    assert summary["trained"] and chain.has_own_model("tester")
    user = chain.users["tester"]
    assert user.n_windows == summary["windows_total"] and user.cues_per_command == {c: 2 for c in S.COMMANDS}
    assert summary["expected_accuracy"] == pytest.approx(S.expected_accuracy(2))
    X, y, cue_ids, session_ids, recorded = chain.training_windows("tester")
    assert len(X) == user.n_windows and set(session_ids) == {"unit-session"} and len(np.unique(cue_ids)) == 12
    for c in summary["windows_per_cue"]:
        assert (y[cue_ids == f"unit-session/cue{c['index']:03d}-{c['command']}"] == S.COMMANDS.index(c["command"])).all()
    # the raw EEG of every cue is kept for re-training / audits
    raw_dir = Path(summary["raw_dir"])
    assert raw_dir == chain.user_dir("tester") / "sessions" / "unit-session"
    index = json.loads((raw_dir / "cues.json").read_text())
    z = np.load(raw_dir / "cues.npz")
    assert len(index) == 12 and all(z[f"raw_{k:03d}"].shape == (16, 1250) for k in range(12))
    assert [i["command"] for i in index] == [c["command"] for c in plan.schedule]
    # a second session accumulates, and stopping early keeps what was collected
    stop_after = {"n": 4}
    events2 = []
    stream2 = ReplayStream(train_paths[30:], clock)
    session2 = S.TrainingSession(chain, "tester", S.TrainingPlan(rounds=1, cue_seconds=10, rest_seconds=0, seed=6, settle_seconds=0),
                                 stream2, present=events2.append, sleep=clock.sleep, clock=clock, session_id="unit-session-2",
                                 should_stop=lambda: sum(e["phase"] == "cue_done" for e in events2) >= stop_after["n"])
    summary2 = session2.run()
    assert summary2["stopped_early"] and summary2["cues_done"] == 4 and stream2.stopped == 1
    assert chain.users["tester"].n_windows == user.n_windows + summary2["windows_total"]
    assert [s["id"] for s in chain.users["tester"].meta["sessions"]] == ["unit-session", "unit-session-2"]
    # a session with nothing usable stores nothing new
    empty = ReplayStream(train_paths[:1], clock)
    empty.raw = np.zeros_like(empty.raw)   # flat channels fail the quality checks
    summary3 = S.TrainingSession(chain, "tester", S.TrainingPlan(rounds=1, cue_seconds=4, rest_seconds=0, settle_seconds=0),
                                 empty, sleep=clock.sleep, clock=clock, session_id="unit-session-3", keep_raw=False).run()
    assert not summary3["trained"] and summary3["windows_total"] == 0 and "nothing was stored" in summary3["message"]
    assert [s["id"] for s in chain.users["tester"].meta["sessions"]] == ["unit-session", "unit-session-2"]
    chain.forget_user("tester")


def test_cli_train_folder_status_and_replay(participant, tmp_path, capsys):
    q, train_paths, test_paths, _, _ = participant
    folder = tmp_path / "recordings"
    for p in train_paths[:18]:
        (folder / Path(p).parent.name).mkdir(parents=True, exist_ok=True)
        (folder / Path(p).parent.name / Path(p).name).symlink_to(p)
    users = tmp_path / "users"
    S.main(["--users-dir", str(users), "train-folder", "--participant", "cli-user", "--folder", str(folder), "--no-self-check"])
    out = json.loads(capsys.readouterr().out)
    assert out["participant"] == "cli-user" and out["meta"]["n_windows"] > 0
    S.main(["--users-dir", str(users), "status", "--participant", "cli-user"])
    info = json.loads(capsys.readouterr().out)
    assert info["n_windows"] == out["meta"]["n_windows"] and info["sessions"][0]["source"].startswith("folder:")
    S.main(["--users-dir", str(users), "replay", "--file", test_paths[0], "--participant", "cli-user"])
    replay = json.loads(capsys.readouterr().out)
    assert replay["segments"][0]["windows"] >= 1 and replay["segments"][0]["label"] in S.COMMANDS + ["uncertain"]
    S.main(["--users-dir", str(users), "forget", "--participant", "cli-user"])
    assert "forgot" in capsys.readouterr().out and not (users / "cli-user").exists()
