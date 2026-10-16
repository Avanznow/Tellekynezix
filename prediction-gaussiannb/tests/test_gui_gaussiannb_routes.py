"""GUI5.BrainwavesBackend's GaussianNB routes, PyTorch and TensorFlow, exercised with stubbed Qt and BrainFlow.

PySide6 and brainflow are not needed: the modules the GUI imports are replaced by minimal stubs before GUI5 is
imported, the BrainFlow processor is replaced by one that serves a Data_clean capture the quality gate accepts, and
the training-session stream is replaced by a replay of Data_clean recordings against the real clock. Checks: both
frameworks load their own chains and stores, predictions reach a decision, calibration from a recording is stored in
the framework's folder, and a short guided session trains and persists a user model with the selected framework.
Skipped when Data_clean or the notebooks' artifacts are not present.
"""
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pytest

GNB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GNB_DIR.parent
if not (REPO_ROOT / "Data_clean" / "G03" / "person03").is_dir():
    pytest.skip("Data_clean not present", allow_module_level=True)
for name in ("pytorch/torch_gaussiannb_subject_models.joblib", "tensorflow/gaussiannb_tf_subject_specific.pkl",
             "tensorflow/gaussiannb_tf_improved_global.pkl", "pytorch/torch_gaussiannb_improved_features.pth"):
    if not (GNB_DIR / name).exists():
        pytest.skip(f"{name} not present", allow_module_level=True)
pytest.importorskip("tensorflow")

TMP = Path(tempfile.mkdtemp(prefix="gui_gnb_"))
os.environ.update(TF_CPP_MIN_LOG_LEVEL="3", AVATAR_GNB_USERS_DIR=str(TMP / "users"), AVATAR_GNB_USERS_DIR_TF=str(TMP / "users_tf"),
                  AVATAR_GNB_CALIBRATION_DIR=str(TMP / "cal"), AVATAR_GNB_CALIBRATION_DIR_TF=str(TMP / "cal_tf"),
                  AVATAR_GNB_TRAIN_ROUNDS="1", AVATAR_GNB_TRAIN_CUE_SECONDS="4", AVATAR_GNB_TRAIN_REST_SECONDS="0",
                  AVATAR_PARTICIPANT_ID="gui-tester")


# ----------------------------------------------------------------------------------------------- stubs
class Signal:
    def __init__(self, *a, **k):
        self.calls = []

    def emit(self, *a):
        self.calls.append(a[0] if len(a) == 1 else a)

    def connect(self, f):
        pass


def Slot(*a, **k):
    def deco(f):
        return f
    return a[0] if (len(a) == 1 and isinstance(a[0], types.FunctionType)) else deco


def Property(*a, **k):
    def deco(f):
        return f
    return deco


class QObject:
    def __init__(self, *a, **k):
        pass


class QTimer:
    def __init__(self, *a, **k):
        self.timeout = Signal()

    def __getattr__(self, n):
        return lambda *a, **k: None


class Any_:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, n):
        return Any_()

    def __call__(self, *a, **k):
        return Any_()


CHOSEN = {"path": None}
CAPTURE = {"path": None, "start": 250, "n": 625}


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _install_stubs():
    if "GUI5" in sys.modules:
        return
    from signal_quality.acquisition import read_openbci

    class FakeProcessor:
        def __init__(self, board_id=None, sample_time=None):
            self.sample_time = sample_time

        def get_quality_capture(self):
            cap = read_openbci(CAPTURE["path"])
            a, n = CAPTURE["start"], CAPTURE["n"]
            out = dict(cap)
            out["raw"], out["counters"], out["timestamps"] = cap["raw"][:, a:a + n], cap["counters"][a:a + n], cap["timestamps"][a:a + n]
            out.update(captured_monotonic=time.monotonic(), stream_id="fake")
            return out

    class BoardIds:
        CYTON_DAISY_BOARD = types.SimpleNamespace(value=2)
        CYTON_BOARD = types.SimpleNamespace(value=0)

    _module("dotenv", load_dotenv=lambda *a, **k: None)
    _module("PySide6")
    _module("PySide6.QtWidgets", QApplication=Any_, QMessageBox=Any_,
            QFileDialog=types.SimpleNamespace(getOpenFileName=lambda *a, **k: (CHOSEN["path"], "x"),
                                              getExistingDirectory=lambda *a, **k: ""))
    _module("PySide6.QtQml", QQmlApplicationEngine=Any_)
    _module("PySide6.QtGui", QDesktopServices=Any_)
    _module("PySide6.QtCore", QObject=QObject, Signal=Signal, Slot=Slot, Property=Property, QProcess=Any_, QUrl=Any_, QTimer=QTimer)
    _module("pdf2image", convert_from_path=lambda *a, **k: [])
    _module("djitellopy", Tello=Any_)
    _module("brainflow")
    _module("brainflow.board_shim", BoardShim=Any_, BrainFlowInputParams=Any_, BoardIds=BoardIds)
    _module("predictions_local.brainflowprocessor", BrainFlowDataProcessor=FakeProcessor)
    _module("predictions_local.deeplearningpytorchpredictor", DeeplearningPytorchPredictor=Any_)
    _module("predictions_local.neurosityprocessor", NeurosityDataProcessor=Any_)
    _module("cameraview")
    _module("cameraview.camera_controller", CameraController=Any_)
    _module("NAO6")
    _module("NAO6.nao_connection", send_command=lambda *a, **k: None)
    _module("developers_api", DevelopersAPI=Any_)
    nao = _module("NA06_Manual_Control", ManualNaoController=Any_)
    nao.__path__ = []
    _module("NA06_Manual_Control.camera_view", DroneCameraController=Any_)
    _module("cloud_api", CloudAPI=Any_)
    _module("shuffler_api", ShufflerAPI=Any_)


@pytest.fixture(scope="module")
def gui():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        _install_stubs()
        import GUI5
    finally:
        os.chdir(cwd)
    return GUI5


@pytest.fixture(scope="module")
def files():
    return sorted((REPO_ROOT / "Data_clean" / "G03" / "person03").rglob("OpenBCI-RAW-*.txt"))


@pytest.fixture(scope="module")
def backend(gui, files):
    from signal_quality import QualityAssessor, QualityGatedPredictor
    from signal_quality.acquisition import read_openbci
    from signal_quality.config import load_default_config
    b = gui.BrainwavesBackend()
    b.current_data_mode, b.current_bci_source = "live", "openbci"
    # a capture slice the quality gate accepts
    gate = QualityGatedPredictor(QualityAssessor(load_default_config()))
    for f in files[:8]:
        cap = read_openbci(str(f))
        for start in range(250, cap["raw"].shape[1] - 625, 125):
            CAPTURE.update(path=str(f), start=start, n=625)
            c = sys.modules["predictions_local.brainflowprocessor"].BrainFlowDataProcessor().get_quality_capture()
            if gate.predict(predict=lambda samples: dict(prediction_label="left", confidence=1.0), **c)["status"] == "accepted":
                CHOSEN["path"] = str(files[1] if files[1] != f else files[2])
                return b
    pytest.skip("no capture slice accepted by the quality gate")


def _log(gui, needle):
    return [m for m in gui.BrainwavesBackend.logMessage.calls if needle in m]


def test_tensorflow_route_predicts_with_global_and_training_participant_models(gui, backend):
    b = backend
    b.selectModel("GaussianNB")
    b.selectFramework("TensorFlow")
    assert gui.BrainwavesBackend.gnbParticipantStatusChanged.calls[-1] == "gui-tester [TensorFlow]: not loaded"
    result = b.run_gaussiannb_tensorflow()
    assert result["status"] == "accepted", result
    chain = b.gnb_chains["tensorflow"]
    assert chain.backend == "tensorflow" and chain.window_size == 125 and chain.thresholds is None
    assert _log(gui, "GaussianNB chain loaded (TensorFlow)") and _log(gui, "GaussianNB (TensorFlow): ")
    assert gui.BrainwavesBackend.gnbParticipantStatusChanged.calls[-1] == "gui-tester [TensorFlow]: global fallback"
    b.setGaussianNBParticipant("G03/person03")
    b.setGaussianNBMode("subject")
    assert gui.BrainwavesBackend.gnbParticipantStatusChanged.calls[-1] == "G03/person03 [TensorFlow]: my model: training participant"
    result = b.run_gaussiannb_tensorflow()
    subject = b.gnb_subject_chains["tensorflow"]
    assert subject.backend == "tensorflow" and subject.resolve("G03/person03")[3] == "training participant"
    assert result["status"] in ("accepted", "withheld") and subject.last_decision is not None
    assert _log(gui, "GaussianNB (my model, TensorFlow)")
    b.setGaussianNBMode("global")
    b.setGaussianNBParticipant("gui-tester")


def test_calibration_is_stored_per_framework_and_pytorch_route_keeps_its_own_chains(gui, backend):
    b = backend
    b.selectFramework("TensorFlow")
    b.calibrateGaussianNBFromRecording()
    assert b.gnb_chains["tensorflow"].participant_status("gui-tester") == "calibrated"
    assert sorted(p.name for p in (TMP / "cal_tf").iterdir()) == ["gui-tester.npz"] and not (TMP / "cal").exists()
    b.selectFramework("PyTorch")
    assert gui.BrainwavesBackend.gnbParticipantStatusChanged.calls[-1].startswith("gui-tester [PyTorch]: ")
    result = b.run_gaussiannb_pytorch()
    assert result["status"] in ("accepted", "withheld") and b.gnb_chains["torch"].backend == "torch"
    assert b.gnb_chains["torch"].participant_status("gui-tester") == "global fallback"      # calibrated only for TensorFlow
    assert b.gnb_chains["torch"].window_size == 128 and b.gnb_chains["torch"].thresholds is not None
    b.selectFramework("JAX")
    assert gui.BrainwavesBackend.gnbParticipantStatusChanged.calls[-1] == "gui-tester: not available for the JAX framework"
    assert b._run_gaussiannb("JAX")["status"] == "withheld"
    b.selectFramework("TensorFlow")


def test_guided_training_session_trains_a_user_with_the_selected_framework(gui, backend, files):
    import subject_gnb_inference as S
    from signal_quality.acquisition import read_openbci
    b = backend
    b.selectFramework("TensorFlow")
    b.setGaussianNBMode("subject")
    b.setGaussianNBParticipant("gui-tester")

    class ReplayStream:
        def __init__(self, board_id=None, serial_port=None, ring_buffer_samples=None):
            caps = [read_openbci(str(p)) for p in files[:12]]
            self.raw = np.concatenate([c["raw"] for c in caps], axis=1)
            self.counters = np.concatenate([c["counters"] for c in caps])
            self.timestamps = np.concatenate([c["timestamps"] for c in caps])
            self.template = {k: v for k, v in caps[0].items() if k not in ("raw", "counters", "timestamps")}
            self.pos, self.t0 = 0, None

        def start(self):
            self.t0 = time.monotonic()

        def _avail(self):
            return min(int((time.monotonic() - self.t0) * 125), self.raw.shape[1])

        def discard(self):
            self.pos = self._avail()

        def read(self):
            a, z = self.pos, self._avail()
            self.pos = z
            return dict(self.template, raw=self.raw[:, a:z], counters=self.counters[a:z], timestamps=self.timestamps[a:z],
                        captured_monotonic=time.monotonic(), stream_id="replay")

        def stop(self):
            pass

    original = S.BrainFlowStream
    S.BrainFlowStream = ReplayStream
    try:
        t0 = time.time()
        b.startGaussianNBTraining()
        while b._gnb_training is not None and time.time() - t0 < 120:
            time.sleep(0.5)
    finally:
        S.BrainFlowStream = original
    assert b._gnb_training is None
    summary = gui.BrainwavesBackend.gnbTrainingFinished.calls[-1]
    assert summary["cues_done"] == 6 and summary["windows_total"] > 0 and summary["participant"] == "gui-tester"
    meta = json.loads((TMP / "users_tf" / "gui-tester" / "meta.json").read_text())
    assert meta["framework"] == "tensorflow" and (meta["window_size"], meta["window_stride"]) == (125, 62)
    assert not (TMP / "users").exists()
    assert _log(gui, "GaussianNB training session (TensorFlow)")
    chain = b.gnb_subject_chains["tensorflow"]
    assert "gui-tester" in chain.users and chain.users["gui-tester"].n_windows == summary["windows_total"]
    result = b.run_gaussiannb_tensorflow()
    assert result["status"] in ("accepted", "withheld") and chain.last_decision["participant"] == "gui-tester"
