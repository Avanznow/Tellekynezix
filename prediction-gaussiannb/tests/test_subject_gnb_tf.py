"""The TensorFlow backend of the GaussianNB live path: same flow as PyTorch, GaussianNBTF as the decision model.

Checks against prediction-gaussiannb/tensorflow/gaussiannb_train.ipynb (its artifacts gaussiannb_tf_improved_global.pkl
/ gaussiannb_tf_subject_specific.pkl, its 125-sample / 62-stride windows without tail window, its 60/20/20 file split
and gaussiannb_tf_subject_per_participant.csv):
  * a Data_clean participant's TEST recordings through the TensorFlow route give the notebook's per-participant accuracy;
  * the training flow with backend="tensorflow" on the notebook's train + validation recordings reproduces the
    pickled subject model (statistics, projection, posteriors);
  * the PyTorch and TensorFlow backends agree on the decisions for the same windows;
  * users of the two backends are stored apart (their windows differ) and a chain ignores the other's users;
  * the global TensorFlow chain replays with the notebook's geometry, decides by argmax and calibrates new users.
Skipped when TensorFlow, Data_clean or the notebook's artifacts are not present.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
GNB_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = GNB_DIR / "pytorch"
TF_DIR = GNB_DIR / "tensorflow"
REPO_ROOT = GNB_DIR.parent
for _p in (MODEL_DIR, GNB_DIR, REPO_ROOT):      # pytorch/ first: `gaussiannb_model` must be the PyTorch model
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
pytest.importorskip("tensorflow")

import improved_gnb_inference as I  # noqa: E402
import subject_gnb_inference as S  # noqa: E402
from eeg_data import LoadConfig, load_recordings  # noqa: E402
from improved_gnb_features import participant_of  # noqa: E402
from signal_quality import QualityAssessor, QualityGatedPredictor  # noqa: E402
from signal_quality.acquisition import read_openbci  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402
from signal_quality.salvage import folder_label  # noqa: E402

PER_PARTICIPANT_CSV = TF_DIR / "gaussiannb_tf_subject_per_participant.csv"
PREFERRED = "G03/person03"
WS, STRIDE = S.TF_WINDOW_SIZE, S.TF_WINDOW_STRIDE


def _assessor():
    return QualityAssessor(load_default_config())


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tf_split():
    """The TensorFlow notebook's split (experiment cell): sorted usable recordings, train_test_split 80/20 by file
    stratified on the file label with seed 42 (the 80 % is its train + validation part)."""
    for name in (S.TF_SUBJECT_FILE, I.TF_GLOBAL_FILE):
        if not (TF_DIR / name).exists():
            pytest.skip(f"{name} not present")
    if not PER_PARTICIPANT_CSV.exists() or not (REPO_ROOT / "Data_clean").is_dir():
        pytest.skip("TensorFlow notebook results or Data_clean not present")
    from sklearn.model_selection import train_test_split
    cfg = LoadConfig(board="16ch_125hz", exclude_channels=("EXG Channel 15",), quality_filter=True,
                     bandpass_hz=(1.0, 40.0), notch_hz=60.0)
    recordings, _ = load_recordings(REPO_ROOT / "Data_clean", cfg, verbose=False)
    files = np.array(sorted(r["path"] for r in recordings if r["segments"]))
    labels = np.array([folder_label(Path(f).parent.name) for f in files])
    names = sorted(set(labels))
    ids = np.array([names.index(l) for l in labels], np.int32)
    trainval, test = train_test_split(files, test_size=0.2, random_state=42, stratify=ids)
    return dict(trainval=sorted(trainval), test=sorted(test))


@pytest.fixture(scope="module")
def participant(tf_split):
    """(id, train+validation paths, test paths, notebook accuracy, notebook test windows) of a participant with an own model."""
    with PER_PARTICIPANT_CSV.open() as stream:
        rows = {r["Participant"]: r for r in csv.DictReader(stream)}
    for q in [PREFERRED] + sorted(k for k in rows if k != PREFERRED):
        if rows.get(q, {}).get("Model") != "own":
            continue
        trainval = [f for f in tf_split["trainval"] if participant_of(f) == q]
        test = [f for f in tf_split["test"] if participant_of(f) == q]
        if trainval and test and all(Path(f).exists() for f in trainval + test):
            return q, trainval, test, float(rows[q]["Accuracy"]), int(rows[q]["Test windows"])
    pytest.skip("no participant with all recordings present")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return S.SubjectSpecificGaussianNBChain(backend="tensorflow", users_dir=tmp_path_factory.mktemp("gnb_users_tf"))


@pytest.fixture(scope="module")
def trained(chain, participant):
    q, trainval, _, _, _ = participant
    labelled = [(p, folder_label(Path(p).parent.name)) for p in trainval]
    return chain.train_from_recordings("tf-user", labelled, session_id="tf-notebook-trainval", self_check=False)


# ----------------------------------------------------------------------------------------------- windows and artifacts
def test_tensorflow_window_rule_has_no_tail_window():
    seg = np.random.default_rng(0).normal(size=(250, 15))
    assert I.cut_windows(seg, WS, STRIDE, tail=False).shape == (3, 15, WS)              # starts 0, 62, 124
    assert list(I.window_starts(250, WS, STRIDE, tail=False)) == [0, 62, 124]
    assert I.cut_windows(seg[:124], WS, STRIDE, tail=False).shape[0] == 0                # shorter than one window
    assert list(I.window_starts(250, 128, 64)) == [0, 64, 122]                           # the PyTorch rule keeps its tail


def test_tensorflow_chain_geometry_and_notebook_artifacts(chain):
    assert chain.backend == "tensorflow"
    assert (chain.window_size, chain.window_stride, chain.window_tail) == (WS, STRIDE, False)
    assert chain.capture_samples == WS + 4 * STRIDE == 373
    assert chain.n_features == 435 and chain.classes == S.COMMANDS
    assert chain.notebook_models is not None and chain.notebook_models.backend == "tensorflow"
    assert len(chain.notebook_models.models_) == 17 and set(chain.notebook_models.skipped_) == {"G03/person01"}
    assert set(chain.notebook_scaler.stats_) >= set(chain.notebook_models.models_)
    assert type(chain.notebook_models.global_.nb_).__name__ == "GaussianNBTF"
    assert chain.participant_status("G03/person03") == "training participant"
    assert chain.participant_status("nobody") == "no model yet: global fallback"
    assert chain.users_dir.name.startswith("gnb_users_tf") and S.DEFAULT_USERS_DIR_TF != S.DEFAULT_USERS_DIR


def test_tensorflow_live_path_gives_the_notebook_per_participant_accuracy(chain, participant):
    q, _, test_paths, accuracy_csv, n_test_csv = participant
    correct = total = 0
    for path in test_paths:
        truth = folder_label(Path(path).parent.name)
        out = chain.replay_recording(path, q)
        assert out["participant_status"] == "training participant" and out["truth"] == truth
        segments, _ = chain.segments_from_capture(read_openbci(path), _assessor())
        assert [seg["windows"] for seg in out["segments"]] == [(s.shape[1] - WS) // STRIDE + 1 for s in segments if s.shape[1] >= WS]
        for seg in out["segments"]:
            correct += sum(w == truth for w in seg["window_labels"])
            total += seg["windows"]
    assert total == n_test_csv
    assert correct / total == pytest.approx(accuracy_csv, abs=1e-12)


def test_tensorflow_training_flow_reproduces_the_notebook_model(chain, participant, trained):
    q, trainval, _, _, _ = participant
    assert trained.model is not None and trained.status().startswith("own model (")
    assert trained.meta["framework"] == "tensorflow" and (trained.meta["window_size"], trained.meta["window_stride"]) == (WS, STRIDE)
    assert sum(trained.cues_per_command.values()) == len(trainval)
    # statistics = the notebook scaler's statistics of this participant (its train + validation windows)
    mean_nb, std_nb = chain.notebook_scaler.stats_[q]
    np.testing.assert_allclose(trained.mean, mean_nb, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(trained.std, std_nb, rtol=1e-6, atol=1e-6)
    # model = the pickled subject model: same projection (up to sign), same posteriors on the training windows
    model_nb = chain.notebook_models.models_[q]
    assert type(trained.model.nb_).__name__ == "GaussianNBTF" and list(trained.model.classes_) == list(model_nb.classes_)
    np.testing.assert_allclose(np.abs(trained.model.projection_), np.abs(model_nb.projection_), rtol=1e-6, atol=1e-8)
    X, y, cue_ids, session_ids, _ = chain.training_windows("tf-user")
    assert len(np.unique(cue_ids)) == len(trainval) and set(session_ids) == {"tf-notebook-trainval"}
    Z = (X.astype(np.float64) - trained.mean) / trained.std
    lp_user, lp_nb = trained.model.predict_log_proba(Z), model_nb.predict_log_proba(Z)
    np.testing.assert_allclose(lp_user, lp_nb, rtol=1e-3, atol=2e-2)
    assert np.mean(lp_user.argmax(1) == lp_nb.argmax(1)) > 0.999
    assert np.mean(lp_user.argmax(1) == y) > 0.8
    d = chain.user_dir("tf-user")
    meta = json.loads((d / "meta.json").read_text())
    assert meta["framework"] == "tensorflow" and meta["feature_version"] == S.SUBJECT_FEATURE_VERSION
    assert set(np.load(d / "model.npz").files) >= {"projection", "classes", "priors", "means", "variances"}


def test_backends_agree_on_the_same_model_and_on_separately_fitted_models(chain, participant, trained):
    q, _, test_paths, _, _ = participant
    X_test = np.concatenate([chain.features_from_capture(read_openbci(p), _assessor())[0] for p in test_paths])
    Z = (X_test.astype(np.float64) - trained.mean) / trained.std
    z = np.load(chain.user_dir("tf-user") / "model.npz", allow_pickle=False)
    torch_model, tf_model = S.model_from_arrays(z, "torch"), S.model_from_arrays(z, "tensorflow")
    lp_t, lp_f = torch_model.predict_log_proba(Z), tf_model.predict_log_proba(Z)
    np.testing.assert_allclose(lp_t, lp_f, rtol=1e-4, atol=1e-3)
    assert np.array_equal(lp_t.argmax(1), lp_f.argmax(1))
    # the same labelled windows fitted by each backend (PyTorch: unbiased variance; TensorFlow: population variance)
    X_train, y_train, _, _, _ = chain.training_windows("tf-user")
    mean_t, std_t, model_t, _ = S.fit_user_model(X_train, y_train, q, backend="torch")
    mean_f, std_f, model_f, _ = S.fit_user_model(X_train, y_train, q, backend="tensorflow")
    np.testing.assert_allclose(mean_t, mean_f) and np.testing.assert_allclose(std_t, std_f)
    pred_t = model_t.predict_log_proba((X_test - mean_t) / std_t).argmax(1)
    pred_f = model_f.predict_log_proba((X_test - mean_f) / std_f).argmax(1)
    assert np.mean(pred_t == pred_f) >= 0.99


def test_users_of_the_two_backends_are_kept_apart(chain, participant, trained, capsys):
    q, _, test_paths, _, _ = participant
    torch_chain = S.SubjectSpecificGaussianNBChain(backend="torch", users_dir=chain.users_dir)
    assert "tf-user" not in torch_chain.users and "125-sample windows" in capsys.readouterr().out
    again = S.SubjectSpecificGaussianNBChain(backend="tensorflow", users_dir=chain.users_dir)
    assert again.has_own_model("tf-user") and again.participant_status("tf-user") == trained.status()
    X = chain.features_from_capture(read_openbci(test_paths[0]), _assessor())[0]
    np.testing.assert_allclose(again.log_posteriors(X, "tf-user"), chain.log_posteriors(X, "tf-user"), atol=1e-6)
    decision = chain.predict_windows(X[:5], "tf-user")
    assert decision["participant_status"] == "own model" and decision["votes"] == 5 and decision["label"] in S.COMMANDS + ["uncertain"]
    S.main(["--backend", "tensorflow", "--users-dir", str(chain.users_dir), "status", "--participant", "tf-user"])
    info = json.loads(capsys.readouterr().out)
    assert info["has_model"] and info["n_windows"] == trained.n_windows


def test_tensorflow_global_chain_replays_calibrates_and_decides_by_argmax(participant, tmp_path):
    q, _, test_paths, _, _ = participant
    g = I.ImprovedGaussianNBChain(backend="tensorflow", calibration_dir=tmp_path / "cal_tf")
    assert (g.window_size, g.window_stride, g.window_tail) == (WS, STRIDE, False) and g.capture_samples == 373
    assert g.thresholds is None and g.classes == S.COMMANDS and type(g.model).__name__ == "GaussianNBTF"
    assert g.participant_status(q) == "training participant" and g.participant_status("nobody") == "global fallback"
    cap = read_openbci(test_paths[0])
    out = g.replay_recording(test_paths[0], q)
    segments, _ = g.segments_from_capture(cap, _assessor())
    assert [seg["windows"] for seg in out["segments"]] == [(s.shape[0] - WS) // STRIDE + 1 for s in segments if s.shape[0] >= WS]
    X = g.features_from_segment(segments[0])
    logp = g.log_posteriors(X, q)
    assert out["segments"][0]["window_labels"] == [S.COMMANDS[i] for i in logp.argmax(1)]
    np.testing.assert_allclose(np.exp(logp), g.predict_proba(X, q), atol=1e-6)
    # calibration of an unknown user is stored in the TensorFlow calibration folder only
    entry = g.calibrate_from_recording("new-user", test_paths[0], assessor=_assessor())
    assert entry["feature_version"] == I.FEATURE_VERSION and g.participant_status("new-user") == "calibrated"
    assert (tmp_path / "cal_tf" / "new-user.npz").exists()
    assert I.ImprovedGaussianNBChain(backend="tensorflow", calibration_dir=tmp_path / "cal_tf").participant_status("new-user") == "calibrated"
    # the quality-gated live path on one 5-window capture
    gate = QualityGatedPredictor(QualityAssessor(load_default_config()))
    live = dict(cap)
    for key in ("raw", "counters", "timestamps"):
        live[key] = cap[key][:, 250:250 + 373] if key == "raw" else cap[key][250:250 + 373]
    live.update(captured_monotonic=time.monotonic(), stream_id="replay-tf")
    result = g.predict_capture(live, q, gate)
    assert result["status"] in ("accepted", "withheld")
    if result["status"] == "accepted":
        assert g.last_decision["windows"] == 5 and result["prediction_label"] in S.COMMANDS + ["land"]
