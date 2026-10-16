"""The JAX backend of the GaussianNB live path: same flow as PyTorch and TensorFlow, GaussianNBJAX as the decision model.

Checks against prediction-gaussiannb/jax/instructions_log_gnb_jax.ipynb (its artifacts gaussiannb_jax_improved_global.pkl
/ gaussiannb_jax_subject_specific.pkl, its 125-sample / 50-stride windows without tail window, its 60/20/20 file split,
gaussiannb_jax_subject_per_participant.csv and the chain results it stored in gaussiannb_jax_model.pkl):
  * JAX runs on the CPU and there is one GaussianNBJAX class in the process;
  * a Data_clean participant's TEST recordings through the JAX route give the notebook's per-participant accuracy
    (G03/person03: 175 windows, 0.834286);
  * all TEST recordings through the global JAX chain give the notebook's whole-split window accuracy (2810 windows);
  * the training flow with backend="jax" on the notebook's train + validation recordings reproduces the pickled
    subject model (statistics, projection, posteriors);
  * the three backends agree on the decisions for the same model arrays and on separately fitted models;
  * users of the three backends are stored apart and every chain ignores the others' users;
  * the global JAX chain replays with the notebook's geometry, decides by argmax and calibrates new users.
Skipped when JAX, Data_clean or the notebook's artifacts are not present.
"""
import csv
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")           # the backend module pins the CPU too; here before anything imports jax
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
GNB_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = GNB_DIR / "pytorch"
JAX_DIR = GNB_DIR / "jax"
REPO_ROOT = GNB_DIR.parent
for _p in (MODEL_DIR, GNB_DIR, REPO_ROOT):      # pytorch/ first: `gaussiannb_model` must be the PyTorch model
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if importlib.util.find_spec("jax") is None:
    pytest.skip("jax not installed", allow_module_level=True)

import improved_gnb_inference as I  # noqa: E402
import subject_gnb_inference as S  # noqa: E402
from eeg_data import LoadConfig, load_recordings  # noqa: E402
from improved_gnb_features import participant_of  # noqa: E402
from signal_quality import QualityAssessor, QualityGatedPredictor  # noqa: E402
from signal_quality.acquisition import read_openbci  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402
from signal_quality.salvage import folder_label  # noqa: E402

PER_PARTICIPANT_CSV = JAX_DIR / "gaussiannb_jax_subject_per_participant.csv"
CLASSIC_PICKLE = JAX_DIR / "gaussiannb_jax_model.pkl"        # its meta["results_df"] holds the chains' test accuracies
GLOBAL_CHAIN_NAME = "GaussianNBJAX improved global chain (570 feats + participant z + LDA5)"
# the notebook's chain cell: "GaussianNBJAX improved global chain ...: val=0.4743, test_win=0.5253, test_file=0.6328"
# on 2810 test windows from 177 files (= 1476 / 2810)
GLOBAL_TEST_WINDOWS, GLOBAL_TEST_ACCURACY = 2810, 0.5252669039145907
PREFERRED = "G03/person03"
WS, STRIDE = S.JAX_WINDOW_SIZE, S.JAX_WINDOW_STRIDE


def _assessor():
    return QualityAssessor(load_default_config())


def _notebook_global_accuracy():
    """The global chain's whole-split test accuracy as the notebook stored it (gaussiannb_jax_model.pkl, cell 10 meta),
    else the number printed by its chain cell."""
    try:
        with CLASSIC_PICKLE.open("rb") as stream:
            rows = pickle.load(stream).get("meta", {}).get("results_df", [])
    except Exception:  # noqa: BLE001  (an older classic pickle without the chains' rows)
        rows = []
    for row in rows:
        if row.get("Experiment") == GLOBAL_CHAIN_NAME:
            return float(row["Test_Acc_Window"])
    return GLOBAL_TEST_ACCURACY


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def jax_split():
    """The JAX notebook's split (cell 1): sorted usable recordings, train_test_split 80/20 by file stratified on the
    file label with seed 42 (the 80 % is its train + validation part)."""
    for name in (S.JAX_SUBJECT_FILE, I.JAX_GLOBAL_FILE):
        if not (JAX_DIR / name).exists():
            pytest.skip(f"{name} not present")
    if not PER_PARTICIPANT_CSV.exists() or not (REPO_ROOT / "Data_clean").is_dir():
        pytest.skip("JAX notebook results or Data_clean not present")
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
def participant(jax_split):
    """(id, train+validation paths, test paths, notebook accuracy, notebook test windows) of a participant with an own model."""
    with PER_PARTICIPANT_CSV.open() as stream:
        rows = {r["Participant"]: r for r in csv.DictReader(stream)}
    for q in [PREFERRED] + sorted(k for k in rows if k != PREFERRED):
        if rows.get(q, {}).get("Model") != "own":
            continue
        trainval = [f for f in jax_split["trainval"] if participant_of(f) == q]
        test = [f for f in jax_split["test"] if participant_of(f) == q]
        if trainval and test and all(Path(f).exists() for f in trainval + test):
            return q, trainval, test, float(rows[q]["Accuracy"]), int(rows[q]["Test windows"])
    pytest.skip("no participant with all recordings present")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return S.SubjectSpecificGaussianNBChain(backend="jax", users_dir=tmp_path_factory.mktemp("gnb_users_jax"))


@pytest.fixture(scope="module")
def global_chain(tmp_path_factory):
    return I.ImprovedGaussianNBChain(backend="jax", calibration_dir=tmp_path_factory.mktemp("gnb_calibration_jax"))


@pytest.fixture(scope="module")
def trained(chain, participant):
    q, trainval, _, _, _ = participant
    labelled = [(p, folder_label(Path(p).parent.name)) for p in trainval]
    return chain.train_from_recordings("jax-user", labelled, session_id="jax-notebook-trainval", self_check=False)


# ----------------------------------------------------------------------------------------------- windows and artifacts
def test_jax_window_rule_has_no_tail_window():
    seg = np.random.default_rng(0).normal(size=(250, 15))
    assert I.cut_windows(seg, WS, STRIDE, tail=False).shape == (3, 15, WS)              # starts 0, 50, 100
    assert list(I.window_starts(250, WS, STRIDE, tail=False)) == [0, 50, 100]
    assert list(I.window_starts(274, WS, STRIDE, tail=False)) == [0, 50, 100]            # 125..149 samples left over: no tail
    assert list(I.window_starts(275, WS, STRIDE, tail=False)) == [0, 50, 100, 150]
    assert I.cut_windows(seg[:124], WS, STRIDE, tail=False).shape[0] == 0                # shorter than one window
    assert list(I.window_starts(250, 125, 62, tail=False)) == [0, 62, 124]              # the TensorFlow rule
    assert list(I.window_starts(250, 128, 64)) == [0, 64, 122]                           # the PyTorch rule keeps its tail


def test_jax_runs_on_the_cpu_with_one_model_class(chain, global_chain):
    import jax
    from subject_gnb import jax_backend
    assert jax.default_backend() == "cpu" and all(d.platform == "cpu" for d in jax.devices())
    module = jax_backend()
    cls = module.jax_model_module().GaussianNBJAX
    assert sys.modules["algorithm_gnb_jax"].GaussianNBJAX is cls
    assert type(global_chain.model) is cls and type(chain.notebook_models.global_.nb_) is cls
    assert type(module.ProjectedGaussianNB().fit(np.random.default_rng(1).normal(size=(40, 6)), np.repeat([0, 2], 20), 6).nb_) is cls
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_jax_chain_geometry_and_notebook_artifacts(chain):
    assert chain.backend == "jax"
    assert (chain.window_size, chain.window_stride, chain.window_tail) == (WS, STRIDE, False)
    assert chain.capture_samples == WS + 4 * STRIDE == 325
    assert chain.n_features == 435 and chain.classes == S.COMMANDS
    assert chain.notebook_models is not None and chain.notebook_models.backend == "jax"
    assert len(chain.notebook_models.models_) == 17 and set(chain.notebook_models.skipped_) == {"G03/person01"}
    assert set(chain.notebook_scaler.stats_) >= set(chain.notebook_models.models_)
    assert type(chain.notebook_models.global_.nb_).__name__ == "GaussianNBJAX"
    assert chain.participant_status("G03/person03") == "training participant"
    assert chain.participant_status("nobody") == "no model yet: global fallback"
    assert chain.users_dir.name.startswith("gnb_users_jax")
    # three default stores (AVATAR_GNB_USERS_DIR[_TF|_JAX] / AVATAR_GNB_CALIBRATION_DIR[_TF|_JAX] may relocate them)
    assert len({S.DEFAULT_USERS_DIR, S.DEFAULT_USERS_DIR_TF, S.DEFAULT_USERS_DIR_JAX}) == 3
    assert len({I.DEFAULT_CALIBRATION_DIR, I.DEFAULT_CALIBRATION_DIR_TF, I.DEFAULT_CALIBRATION_DIR_JAX}) == 3
    if "AVATAR_GNB_USERS_DIR_JAX" not in os.environ and "AVATAR_GNB_CALIBRATION_DIR_JAX" not in os.environ:
        assert S.DEFAULT_USERS_DIR_JAX == REPO_ROOT / "predictions_local" / "gnb_users_jax"
        assert I.DEFAULT_CALIBRATION_DIR_JAX == REPO_ROOT / "predictions_local" / "gnb_calibration_jax"


def test_jax_live_path_gives_the_notebook_per_participant_accuracy(chain, participant):
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
    if q == PREFERRED:
        assert (total, round(accuracy_csv, 6)) == (175, 0.834286)
    assert total == n_test_csv
    assert correct / total == pytest.approx(accuracy_csv, abs=1e-12)


def test_jax_global_chain_gives_the_notebook_whole_split_accuracy(global_chain, jax_split):
    g = global_chain
    correct = total = 0
    for path in jax_split["test"]:
        truth = folder_label(Path(path).parent.name)
        out = g.replay_recording(path, participant_of(path))
        assert out["participant_status"] == "training participant"
        for seg in out["segments"]:
            correct += sum(w == truth for w in seg["window_labels"])
            total += seg["windows"]
    assert total == GLOBAL_TEST_WINDOWS
    assert correct / total == pytest.approx(_notebook_global_accuracy(), abs=1e-12)
    assert correct / total == pytest.approx(GLOBAL_TEST_ACCURACY, abs=1e-12)


def test_jax_training_flow_reproduces_the_notebook_model(chain, participant, trained):
    q, trainval, _, _, _ = participant
    assert trained.model is not None and trained.status().startswith("own model (")
    assert trained.meta["framework"] == "jax" and (trained.meta["window_size"], trained.meta["window_stride"]) == (WS, STRIDE)
    assert sum(trained.cues_per_command.values()) == len(trainval)
    # statistics = the notebook scaler's statistics of this participant (its train + validation windows)
    mean_nb, std_nb = chain.notebook_scaler.stats_[q]
    np.testing.assert_allclose(trained.mean, mean_nb, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(trained.std, std_nb, rtol=1e-6, atol=1e-6)
    # model = the pickled subject model: same projection (up to sign), same posteriors on the training windows
    model_nb = chain.notebook_models.models_[q]
    assert type(trained.model.nb_).__name__ == "GaussianNBJAX" and list(trained.model.classes_) == list(model_nb.classes_)
    np.testing.assert_allclose(np.abs(trained.model.projection_), np.abs(model_nb.projection_), rtol=1e-6, atol=1e-8)
    X, y, cue_ids, session_ids, _ = chain.training_windows("jax-user")
    assert len(np.unique(cue_ids)) == len(trainval) and set(session_ids) == {"jax-notebook-trainval"}
    Z = (X.astype(np.float64) - trained.mean) / trained.std
    lp_user, lp_nb = trained.model.predict_log_proba(Z), model_nb.predict_log_proba(Z)
    np.testing.assert_allclose(lp_user, lp_nb, rtol=1e-3, atol=2e-2)
    assert np.mean(lp_user.argmax(1) == lp_nb.argmax(1)) > 0.999
    assert np.mean(lp_user.argmax(1) == y) > 0.8
    d = chain.user_dir("jax-user")
    meta = json.loads((d / "meta.json").read_text())
    assert meta["framework"] == "jax" and meta["feature_version"] == S.SUBJECT_FEATURE_VERSION
    assert set(np.load(d / "model.npz").files) >= {"projection", "classes", "priors", "means", "variances"}


def test_three_backends_agree_on_the_same_model_arrays_and_on_separately_fitted_models(chain, participant, trained):
    pytest.importorskip("tensorflow")
    q, _, test_paths, _, _ = participant
    X_test = np.concatenate([chain.features_from_capture(read_openbci(p), _assessor())[0] for p in test_paths])
    Z = (X_test.astype(np.float64) - trained.mean) / trained.std
    z = np.load(chain.user_dir("jax-user") / "model.npz", allow_pickle=False)
    models = {b: S.model_from_arrays(z, b) for b in S.BACKENDS}
    assert [type(m).__name__ for m in models.values()] == ["ProjectedTorchGaussianNB", "ProjectedGaussianNBTF", "ProjectedGaussianNBJAX"]
    lp = {b: m.predict_log_proba(Z) for b, m in models.items()}
    np.testing.assert_allclose(lp["jax"], trained.model.predict_log_proba(Z), atol=1e-9)        # the arrays round-trip exactly
    for b in ("torch", "tensorflow"):
        np.testing.assert_allclose(lp[b], lp["jax"], rtol=1e-4, atol=1e-3)
        assert np.array_equal(lp[b].argmax(1), lp["jax"].argmax(1))
    # the same labelled windows fitted by each backend (PyTorch: unbiased variance; TensorFlow / JAX: population variance)
    X_train, y_train, _, _, _ = chain.training_windows("jax-user")
    fits = {b: S.fit_user_model(X_train, y_train, q, backend=b) for b in S.BACKENDS}
    preds = {}
    for b, (mean, std, model, skipped) in fits.items():
        assert skipped is None and model is not None
        np.testing.assert_allclose(mean, fits["jax"][0]) and np.testing.assert_allclose(std, fits["jax"][1])
        preds[b] = model.predict_log_proba((X_test - mean) / std).argmax(1)
    assert np.mean(preds["tensorflow"] == preds["jax"]) >= 0.995
    assert np.mean(preds["torch"] == preds["jax"]) >= 0.99


def test_users_of_the_three_backends_are_kept_apart(chain, participant, trained, tmp_path, capsys):
    pytest.importorskip("tensorflow")
    q, trainval, test_paths, _, _ = participant
    # a TensorFlow user in the same store, trained from a few of the same recordings
    tf_chain = S.SubjectSpecificGaussianNBChain(backend="tensorflow", users_dir=chain.users_dir)
    assert "jax-user" not in tf_chain.users and "(stride 50)" in capsys.readouterr().out
    labelled = [(p, folder_label(Path(p).parent.name)) for p in trainval[:12]]
    tf_user = tf_chain.train_from_recordings("tf-user", labelled, session_id="tf-few", self_check=False)
    assert tf_user.meta["framework"] == "tensorflow" and (tf_user.meta["window_size"], tf_user.meta["window_stride"]) == (125, 62)
    # every chain sees only the users of its own framework
    torch_chain = S.SubjectSpecificGaussianNBChain(backend="torch", users_dir=chain.users_dir)
    assert not torch_chain.users and "125-sample windows" in capsys.readouterr().out
    again = S.SubjectSpecificGaussianNBChain(backend="jax", users_dir=chain.users_dir)
    assert set(again.users) == {"jax-user"} and "(stride 62)" in capsys.readouterr().out
    assert set(S.SubjectSpecificGaussianNBChain(backend="tensorflow", users_dir=chain.users_dir).users) == {"tf-user"}
    assert again.has_own_model("jax-user") and again.participant_status("jax-user") == trained.status()
    X = chain.features_from_capture(read_openbci(test_paths[0]), _assessor())[0]
    np.testing.assert_allclose(again.log_posteriors(X, "jax-user"), chain.log_posteriors(X, "jax-user"), atol=1e-6)
    decision = chain.predict_windows(X[:5], "jax-user")
    assert decision["participant_status"] == "own model" and decision["votes"] == 5 and decision["label"] in S.COMMANDS + ["uncertain"]
    # the default stores of the three backends are three different folders, and the CLI reaches the JAX store
    assert len({S.DEFAULT_USERS_DIR.resolve(), S.DEFAULT_USERS_DIR_TF.resolve(), S.DEFAULT_USERS_DIR_JAX.resolve()}) == 3
    S.main(["--backend", "jax", "--users-dir", str(chain.users_dir), "status", "--participant", "jax-user"])
    out = capsys.readouterr().out                      # the chain first reports the ignored tf-user folder, then the JSON
    assert "ignoring user folder tf-user" in out
    info = json.loads(out[out.index("{"):])
    assert info["has_model"] and info["n_windows"] == trained.n_windows
    S.main(["--backend", "tensorflow", "--users-dir", str(chain.users_dir), "status", "--participant", "jax-user"])
    out = capsys.readouterr().out
    assert "ignoring user folder jax-user" in out and not json.loads(out[out.index("{"):])["has_model"]


def test_jax_global_chain_replays_calibrates_and_decides_by_argmax(global_chain, participant, tmp_path):
    q, _, test_paths, _, _ = participant
    g = global_chain
    assert (g.window_size, g.window_stride, g.window_tail) == (WS, STRIDE, False) and g.capture_samples == 325
    assert g.thresholds is None and g.classes == S.COMMANDS and type(g.model).__name__ == "GaussianNBJAX"
    assert g.projection.endswith("JAX notebook)") and g.calibration_dir.name.startswith("gnb_calibration_jax")
    assert g.participant_status(q) == "training participant" and g.participant_status("nobody") == "global fallback"
    cap = read_openbci(test_paths[0])
    out = g.replay_recording(test_paths[0], q)
    segments, _ = g.segments_from_capture(cap, _assessor())
    assert [seg["windows"] for seg in out["segments"]] == [(s.shape[0] - WS) // STRIDE + 1 for s in segments if s.shape[0] >= WS]
    X = g.features_from_segment(segments[0])
    logp = g.log_posteriors(X, q)
    assert out["segments"][0]["window_labels"] == [S.COMMANDS[i] for i in logp.argmax(1)]
    np.testing.assert_allclose(np.exp(logp), g.predict_proba(X, q), atol=1e-6)
    # calibration of an unknown user is stored in the JAX calibration folder only
    entry = g.calibrate_from_recording("new-user", test_paths[0], assessor=_assessor())
    assert entry["feature_version"] == I.FEATURE_VERSION and g.participant_status("new-user") == "calibrated"
    assert (g.calibration_dir / "new-user.npz").exists()
    assert I.ImprovedGaussianNBChain(backend="jax", calibration_dir=g.calibration_dir).participant_status("new-user") == "calibrated"
    assert I.ImprovedGaussianNBChain(backend="torch", calibration_dir=tmp_path / "cal_torch").participant_status("new-user") == "global fallback"
    # the quality-gated live path on one 5-window capture
    gate = QualityGatedPredictor(QualityAssessor(load_default_config()))
    live = dict(cap)
    for key in ("raw", "counters", "timestamps"):
        live[key] = cap[key][:, 250:250 + 325] if key == "raw" else cap[key][250:250 + 325]
    live.update(captured_monotonic=time.monotonic(), stream_id="replay-jax")
    result = g.predict_capture(live, q, gate)
    assert result["status"] in ("accepted", "withheld")
    if result["status"] == "accepted":
        assert g.last_decision["windows"] == 5 and result["prediction_label"] in S.COMMANDS + ["land"]
    # the CLI replays with the JAX backend
    I.main(["--backend", "jax", "--calibration-dir", str(g.calibration_dir), "--file", test_paths[0], "--participant", q])
