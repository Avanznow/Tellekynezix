import sys
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load Neurosity credentials from the bundled env file before anything reads os.environ.
load_dotenv(Path(__file__).resolve().parent / ".env.txt")

from PySide6.QtWidgets import QApplication,QFileDialog, QMessageBox
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, Property, QProcess, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
from pdf2image import convert_from_path
from djitellopy import Tello
import random
import threading
import re
import pandas as pd
import torch
import time
from collections import defaultdict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from predictions_local.brainflowprocessor import BrainFlowDataProcessor
from predictions_local.deeplearningpytorchpredictor import DeeplearningPytorchPredictor
from predictions_local.neurosityprocessor import NeurosityDataProcessor
from signal_quality import QualityAssessor, QualityGatedPredictor, may_execute, withheld_result
from signal_quality.config import load_default_config
from signal_quality.dashboard import build as build_quality_dashboard
from signal_quality.gui_view import preflight_view, quality_view
from signal_quality.report import preflight as preflight_recording
from sklearn.preprocessing import StandardScaler
from cameraview.camera_controller import CameraController
from NAO6.nao_connection import send_command
import asyncio
import copy
import tempfile
import queue
# from Developers.hofCharts import main as hofCharts, ticketsByDev_text NA

from developers_api import DevelopersAPI


								
from NA06_Manual_Control import ManualNaoController
from NA06_Manual_Control.camera_view import DroneCameraController

# Import BCI connection for brainwave prediction
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'prediction-random-forest', 'tensorflow', 'client')))
    from brainflow1 import bciConnection, DataMode
    BCI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BCI connection not available: {e}")
    BCI_AVAILABLE = False


from cloud_api import CloudAPI

from shuffler_api import ShufflerAPI

class TabController(QObject):
    def __init__(self):
        super().__init__()
        self.nao_process = None




class BrainwavesBackend(QObject):
    # Define signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)
    imagesReady = Signal(list)
    logMessage = Signal(str)
    naoStarted = Signal()
    naoEnded = Signal()
    enqueueMoveRequested = Signal(str)
    neurosityStatusChanged = Signal(str)
    signalQualityChanged = Signal(str)
    qualityViewChanged = Signal(dict)  # withholding indicator + channel grid (signal_quality.gui_view)
    gnbParticipantStatusChanged = Signal(str)  # improved GaussianNB chain: participant id and calibration state
    gnbTrainingCueChanged = Signal(dict)  # subject-specific GaussianNB: guided session progress (phase, command, seconds)
    gnbTrainingFinished = Signal(dict)  # subject-specific GaussianNB: session summary (windows, model state, expectation)

    @Slot()
    def startNaoManual(self):
        print("Nao6 Manual session started")
        self.naoStarted.emit("Nao6 Manual session started")

    @Slot()
    def stopNaoManual(self):
        print("Nao6 Manual session ended")
        self.naoEnded.emit("Nao6 Manual session ended")

    @Slot(str, str)
    def connectNao(self, ip="192.168.23.53", port="9559"):
        """Connect to NAO robot with specified IP and Port"""
        try:
            # Log the connection attempt
            self.flight_log.insert(0, f"Attempting to connect to NAO at {ip}:{port}...")
            self.flightLogUpdated.emit(self.flight_log)

            # Send connect command
            if send_command("connect"):
                self.flight_log.insert(0, f"Nao connected successfully at {ip}:{port}.")
            else:
                self.flight_log.insert(0, f"Nao failed to connect at {ip}:{port}.")
        except Exception as e:
            self.flight_log.insert(0, f"Error connecting to NAO: {str(e)}")

        self.flightLogUpdated.emit(self.flight_log)
    
    @Slot()
    def nao_sit_down(self):
        if send_command("sit_down"):
            self.flight_log.insert(0, "Sitting down.")
        else:
            self.flight_log.insert(0, "Nao failed to sit.")
        self.flightLogUpdated.emit(self.flight_log)
    
    @Slot()
    def nao_stand_up(self):
        if send_command("stand_up"):
            self.flight_log.insert(0, "Standing Up.")
        else:
            self.fligt_log.insert(0, "Nao failed to stand up.")
        self.flightLogUpdated.emit(self.flight_log)



    def __init__(self):
        super().__init__()
        self.action_log = [] # List used to store the actions performed by the drone
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("no_assessed_input")
        # Deployed policy from signal_quality/default_config.json (verified 60 Hz mains).
        self.quality_gate = QualityGatedPredictor(QualityAssessor(load_default_config()))
        self.current_data_mode = "synthetic"
        self.current_bci_source = "openbci"
        self.current_model = "Random Forest"  # Default model
        self.current_framework = "PyTorch"  # Default framework
        # Improved GaussianNB chain (prediction-gaussiannb/pytorch/improved_gnb_inference.py): loaded on first use.
        # The participant id selects the per-participant feature statistics; a new user calibrates once.
        self.gnb_chain = None          # improved global chain of the selected framework (see _gnb_chain)
        self.gnb_chains = {}           # backend -> ImprovedGaussianNBChain ("torch" = PyTorch, "tensorflow" = TensorFlow, "jax" = JAX)
        self.gnb_subject_chains = {}   # backend -> SubjectSpecificGaussianNBChain
        self.gnb_chain_lock = threading.Lock()
        self.gnb_participant_id = os.environ.get("AVATAR_PARTICIPANT_ID", "user")
        self._gnb_calibrating = False
        # Subject-specific GaussianNB (prediction-gaussiannb/pytorch/subject_gnb_inference.py): "subject" runs the
        # user's own model, fitted on their labelled windows from the guided interleaved cue session; "global" keeps
        # the improved global chain. AVATAR_GNB_MODE selects the start mode.
        self.gnb_mode = (os.environ.get("AVATAR_GNB_MODE", "global").strip().lower() or "global")
        if self.gnb_mode not in ("global", "subject"):
            self.gnb_mode = "global"
        self.gnb_subject_chain = None
        self._gnb_training = None             # the running TrainingSession (worker thread) or None
        self._gnb_training_stop = threading.Event()
        self.image_paths = []  # Store converted image paths
        self.neurosity_processor = None
        self.neurosity_connected = False
        self.plots_dir = os.path.abspath("plotscode/plots")  # Base plots directory
        self.current_dataset = "refresh"  # Default dataset to display
        self.connected = False
        self.drone_lock = threading.RLock()  # <-- reentrant lock avoids deadlock

        # This timer instance will fire off the parameterized method each timer fire
        self.hover_timer = QTimer()
        self.hover_timer.timeout.connect(self.hover_callback)

        self.is_flying = False


        #Movement clumping
        self.step_cm = 30                     # each "move" command = 30 cm
        self.clump_dir = None                 # current direction being clumped
        self.clump_count = 0                  # how many commands in this batch
        self.clump_window_ms = 1000           # 1 second clump window

        self.clump_timer = QTimer()
        self.clump_timer.setSingleShot(True)
        self.clump_timer.timeout.connect(self._flush_clumped_move)
        self.enqueueMoveRequested.connect(self._enqueue_move)

        # optional override for movement distance (used by clumper)
        self._movement_distance_override = None

        #Tello command queue 
        self.cmd_queue = queue.Queue()
        self._drone_worker = threading.Thread(
            target=self._drone_loop,
            daemon=True,
        )
        self._drone_worker.start()

        try:
            self.tello = Tello(retry_count=1)
        except Exception as e:
            print(f"Warning: Failed to initialize Tello drone: {e}")
            self.logMessage.emit(f"Warning: Failed to initialize Tello drone: {e}")
        
        # Initialize camera controller with tello instance
        self.camera_controller = CameraController()
        if hasattr(self, 'tello'):
            self.camera_controller.set_tello_instance(self.tello)

        # Initialize BCI connection for brainwave prediction
        if BCI_AVAILABLE:
            try:
                self.bcicon = bciConnection.get_instance()
                print("BCI connection initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize BCI connection: {e}")
                self.bcicon = None
        else:
            self.bcicon = None

    # Command queue helpers
    def _queue_action(self, action: str, dist: int | None = None, *, prediction_result=None):
        """
        Put a command into the Tello queue.
        dist is an optional distance override (cm) for movement commands.
        """
        self.cmd_queue.put((action, dist, prediction_result))

    def _queue_prediction(self, result):
        """Keep approval attached to the command until the worker dispatches it."""
        if result is self.current_prediction_result and may_execute(result):
            self._queue_action(result["prediction_label"], prediction_result=result)

    def _drone_loop(self):
        """
        Single worker thread that owns the Tello.
        It pulls commands from the queue and executes them in FIFO order.
        """
        while True:
            action, dist, approval = self.cmd_queue.get()
            try:
                # Manual controls have no EEG approval; automatic commands must
                # still match the current fresh decision when leaving the queue.
                if approval is not None and (approval is not self.current_prediction_result or not may_execute(approval)):
                    self.logMessage.emit(f"Queued prediction withheld: {action}")
                    continue
                # Set distance override if provided
                old = self._movement_distance_override
                if dist is not None:
                    self._movement_distance_override = dist
                try:
                    # All low-level execution is here
                    self.getDroneAction(action)
                finally:
                    self._movement_distance_override = old
            except Exception as e:
                error_msg = f"Worker error during {action}: {e}"
                print(error_msg)
                self.logMessage.emit(error_msg)
            finally:
                self.cmd_queue.task_done()

    @Slot()
    def takeoff(self):
        self.tello.takeoff()
        self.connected = True
        self.is_flying = True
        self.hover_timer.start(200) #every 200 ms, the hover callback will be invoked
        self.logMessage.emit("Hovering")

    def hover_callback(self):
        if self.is_flying:
            self.tello.send_rc_control(0, 0, 0, 0)

    @Slot(str)
    def selectModel(self, model_name):
        """ Select the machine learning model """
        self.logMessage.emit(f"Model selected: {model_name}")
        self.current_model = model_name
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("configuration_changed")
        self.qualityViewChanged.emit(quality_view(self.current_prediction_result, self.quality_gate.assessor.config))
        self.flight_log.insert(0, f"Selected Model: {model_name}")
        self.flightLogUpdated.emit(self.flight_log)
        if model_name == "GaussianNB":
            self._emit_gnb_status()

    @Slot(str)
    def selectFramework(self, framework_name):
        """ Select the machine learning framework """
        self.logMessage.emit(f"Framework selected: {framework_name}")
        self.current_framework = framework_name
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("configuration_changed")
        self.qualityViewChanged.emit(quality_view(self.current_prediction_result, self.quality_gate.assessor.config))
        self.flight_log.insert(0, f"Selected Framework: {framework_name}")
        self.flightLogUpdated.emit(self.flight_log)
        if self.current_model == "GaussianNB":
            self._emit_gnb_status()

    @Slot()
    def readMyMind(self):
        """Assess the actual capture; withheld outcomes never become commands."""
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("assessment_pending")
        self.signalQualityChanged.emit("Assessing EEG input...")
        try:
            if self.current_model == "Band Power + SVM":
                result = self.run_bandpower_svm()
            elif self.current_model == "Riemannian + LR":
                result = self.run_riemannian()
            elif self.current_model == "Random Forest":
                result = self.run_random_forest_pytorch() if self.current_framework == "PyTorch" else self.run_random_forest_tensorflow()
            elif self.current_model == "GaussianNB":
                result = self._run_gaussiannb(self.current_framework)      # PyTorch, TensorFlow or JAX chains
            else:
                result = self.run_deep_learning_pytorch() if self.current_framework == "PyTorch" else self.run_deep_learning_tensorflow()
        except Exception as error:
            result = self.quality_gate.withhold(f"prediction_failed: {error}")
        self.current_prediction_result = result
        quality = result.get("quality") if isinstance(result, dict) else None
        if not may_execute(result):
            self.current_prediction_label = ""
            reason = result.get("withholding_reason", "missing_quality_decision") if isinstance(result, dict) else "missing_quality_decision"
            message = f"Prediction withheld: {reason}"
            if quality and quality.get("bad_channels"):
                message += "; channels: " + ", ".join(quality["bad_channels"])
            self.signalQualityChanged.emit(message)
            self.logMessage.emit(message)
            self.flight_log.insert(0, message)
            self.predictions_log.append({"count": str(len(self.predictions_log) + 1), "server": "Quality gate", "label": "WITHHELD"})
        else:
            prediction = result["prediction_label"]
            self.current_prediction_label = prediction
            self.signalQualityChanged.emit("EEG passed configured quality checks")
            self.logMessage.emit(f"Prediction received: {prediction}")
            self._queue_prediction(result)
            self.predictions_log.append({"count": str(len(self.predictions_log) + 1), "server": "Brainwave AI", "label": prediction})
            self.flight_log.insert(0, f"Queued: {prediction} (Model: {self.current_model}, Framework: {self.current_framework})")
        self.qualityViewChanged.emit(quality_view(result, self.quality_gate.assessor.config))
        self.predictionsTableUpdated.emit(self.predictions_log)
        self.flightLogUpdated.emit(self.flight_log)

    def _choose_recording(self, title):
        path, _ = QFileDialog.getOpenFileName(None, title, str(Path.home()), "OpenBCI recordings (*.txt)")
        return path or None

    @Slot()
    def openQualityDashboard(self):
        """Build the interactive quality dashboard for a recording and open it in the browser."""
        path = self._choose_recording("Open EEG recording in the quality dashboard")
        if path is None:
            return
        out_dir = Path(tempfile.gettempdir()) / "avatar_quality_dashboards"
        try:
            out, payload = build_quality_dashboard(path, out_dir / (Path(path).stem + ".quality.html"),
                                                   self.quality_gate.assessor.config)
        except (ValueError, OSError) as error:
            self.logMessage.emit(f"Quality dashboard not built for {Path(path).name}: {error}")
            return
        session = payload["session"]
        self.logMessage.emit(f"Quality dashboard: {session['usable_windows']}/{session['windows']} windows pass "
                             f"({out})")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    @Slot()
    def checkRecordingQuality(self):
        """Preflight a recording file (e.g. right after a collection session) and show its channels."""
        path = self._choose_recording("Check an EEG recording")
        if path is None:
            return
        try:
            result = preflight_recording(path, self.quality_gate.assessor.config)
        except (ValueError, OSError) as error:
            result = dict(path=path, error=str(error))
        view = preflight_view(result, self.quality_gate.assessor.config)
        self.qualityViewChanged.emit(view)
        self.logMessage.emit(f"{Path(path).name}: {view['headline']}")
        for reason in view["reasons"]:
            self.logMessage.emit(f"  {reason}")

    def run_riemannian(self):
        # Load lazily: existing routes do not require pyRiemann installed.
        from prediction_riemannian.predictor import AvatarRiemannianPredictor
        model_path = os.environ.get("AVATAR_RIEMANNIAN_MODEL")
        if not model_path:
            return self.quality_gate.withhold("Set AVATAR_RIEMANNIAN_MODEL to an independently validated artifact")
        predictor = AvatarRiemannianPredictor(model_path)
        if predictor.artifact.get("task_labels_verified") is not True:
            return self.quality_gate.withhold("task_labels_unverified_experimental_model")
        if self.current_bci_source != "openbci" or self.current_data_mode != "live":
            return self.quality_gate.withhold("Riemannian inference requires verified live OpenBCI acquisition")
        if predictor.assessor.config.policy_id != self.quality_gate.assessor.config.policy_id:
            return self.quality_gate.withhold("Riemannian quality policy differs from GUI policy")
        predictor.gate = self.quality_gate
        processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value,
                                           sample_time=predictor.config.seconds + 1)
        capture = processor.get_quality_capture()
        n = predictor.config.samples
        if capture["raw"].shape[1] < n:
            return self.quality_gate.withhold("Insufficient samples for Riemannian window")
        for name in ("raw", "counters", "timestamps"):
            capture[name] = capture[name][:, -n:] if name == "raw" else capture[name][-n:]
        return predictor.predict_capture(capture)

    def run_bandpower_svm(self):
        # Load lazily: existing routes do not require extra classifier dependencies installed.
        from prediction_bandpower.predictor import AvatarBandPowerSVMPredictor
        model_path = os.environ.get("AVATAR_BANDPOWER_SVM_MODEL")
        if not model_path:
            return self.quality_gate.withhold("Set AVATAR_BANDPOWER_SVM_MODEL to an independently validated artifact")
        predictor = AvatarBandPowerSVMPredictor(model_path)
        if predictor.artifact.get("task_labels_verified") is not True:
            return self.quality_gate.withhold("task_labels_unverified_experimental_model")
        if self.current_bci_source != "openbci" or self.current_data_mode != "live":
            return self.quality_gate.withhold("Band-power SVM inference requires verified live OpenBCI acquisition")
        if predictor.assessor.config.policy_id != self.quality_gate.assessor.config.policy_id:
            return self.quality_gate.withhold("Band-power SVM quality policy differs from GUI policy")
        predictor.gate = self.quality_gate
        processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value,
                                           sample_time=predictor.config.seconds + 1)
        capture = processor.get_quality_capture()
        n = predictor.config.samples
        if capture["raw"].shape[1] < n:
            return self.quality_gate.withhold("Insufficient samples for Band-power SVM window")
        for name in ("raw", "counters", "timestamps"):
            capture[name] = capture[name][:, -n:] if name == "raw" else capture[name][-n:]
        return predictor.predict_capture(capture)

    def _run_quality_gated_remote_rf(self):
        if self.current_bci_source != "openbci":
            return self.quality_gate.withhold("Neurosity quality contract needs raw timestamps, rate and physical channel metadata")
        if not self.bcicon:
            return self.quality_gate.withhold("BCI connection unavailable")
        self.bcicon.set_mode(DataMode.LIVE if self.current_data_mode != "synthetic" else DataMode.SYNTHETIC)
        return self.bcicon.bciConnectionController(quality_gate=self.quality_gate)

    def run_random_forest_pytorch(self):
        # Existing RF selection uses the remote RF service; assessment runs on
        # its original capture before the HTTP request, not a second recording.
        return self._run_quality_gated_remote_rf()

    def run_random_forest_tensorflow(self):
        return self._run_quality_gated_remote_rf()

    def run_deep_learning_pytorch(self):
        if self.current_bci_source != "openbci":
            return self.quality_gate.withhold("Neurosity quality acquisition metadata unavailable")
        board_id = BoardIds.SYNTHETIC_BOARD.value if self.current_data_mode == "synthetic" else BoardIds.CYTON_DAISY_BOARD.value
        self.brainwave_processor = BrainFlowDataProcessor(board_id=board_id, sample_time=3)
        capture = self.brainwave_processor.get_quality_capture()
        def predict_same_capture(raw):
            # Preserve the legacy CNN's scaling interface after raw assessment.
            self.brainwave_data = torch.tensor(StandardScaler().fit_transform(raw.T), dtype=torch.float32)
            return DeeplearningPytorchPredictor()(self.brainwave_data)
        return self.quality_gate.predict(predict=predict_same_capture, **capture)

    def get_brainwave_data(self):
        if self.current_bci_source == "neurosity":
            return self.get_neurosity_brainwave_data()
        board_id = BoardIds.SYNTHETIC_BOARD.value if self.current_data_mode == "synthetic" else BoardIds.CYTON_DAISY_BOARD.value
        self.brainwave_processor = BrainFlowDataProcessor(board_id=board_id)
        self.brainwave_data = self.brainwave_processor.get_tensor()
        return self.brainwave_data

    def run_deep_learning_tensorflow(self):
        return self.quality_gate.withhold("TensorFlow CNN inference is not implemented; no simulated command substituted")

    # ---- GaussianNB: the improved chains of prediction-gaussiannb (PyTorch, TensorFlow or JAX decision model) ----
    GNB_BACKENDS = {"PyTorch": "torch", "TensorFlow": "tensorflow", "JAX": "jax"}
    GNB_LABELS = {"torch": "PyTorch", "tensorflow": "TensorFlow", "jax": "JAX"}

    def _gnb_backend(self, framework=None):
        """The GaussianNB implementation behind the selected framework ('torch', 'tensorflow' or 'jax')."""
        framework = self.current_framework if framework is None else framework
        backend = self.GNB_BACKENDS.get(framework)
        if backend is None:
            raise ValueError(f"GaussianNB is not implemented for the {framework} framework (PyTorch, TensorFlow or JAX)")
        return backend

    def _gnb_chain(self, framework=None):
        """Load the improved GaussianNB chain of the selected framework once (features, participant scaler, LDA
        projection, PyTorch GaussianNB, TensorFlow GaussianNBTF or JAX GaussianNBJAX)."""
        backend = self._gnb_backend(framework)
        with self.gnb_chain_lock:
            if backend not in self.gnb_chains:
                gnb_dir = Path(__file__).resolve().parent / "prediction-gaussiannb" / "pytorch"
                if str(gnb_dir) not in sys.path:
                    sys.path.insert(0, str(gnb_dir))
                from improved_gnb_inference import ImprovedGaussianNBChain
                chain = ImprovedGaussianNBChain(backend=backend)
                self.gnb_chains[backend] = chain
                self.logMessage.emit(f"GaussianNB chain loaded ({self._gnb_label(backend)}): {len(chain.feature_names)} features -> "
                                     f"participant z-score -> {chain.projection} -> GaussianNB, {chain.window_size}-sample windows; "
                                     f"{len(chain.calibrations)} stored calibration(s)")
            self.gnb_chain = self.gnb_chains[backend]
            return self.gnb_chain

    @classmethod
    def _gnb_label(cls, backend):
        return cls.GNB_LABELS.get(backend, str(backend))

    def _emit_gnb_status(self):
        backend = self.GNB_BACKENDS.get(self.current_framework)
        subject_chain = self.gnb_subject_chains.get(backend)
        global_chain = self.gnb_chains.get(backend)
        if backend is None:
            status = f"not available for the {self.current_framework} framework"
        elif self._gnb_training is not None:
            status = "training session running..."
        elif self._gnb_calibrating:
            status = "calibrating..."
        elif self.gnb_mode == "subject":
            status = ("my model: not loaded" if subject_chain is None
                      else "my model: " + subject_chain.participant_status(self.gnb_participant_id))
        elif global_chain is None:
            status = "not loaded"
        else:
            status = global_chain.participant_status(self.gnb_participant_id)
        tag = f" [{self.current_framework}]" if backend is not None else ""
        self.gnbParticipantStatusChanged.emit(f"{self.gnb_participant_id}{tag}: {status}")

    @Slot(result=str)
    def gaussianNBParticipant(self):
        return self.gnb_participant_id

    @Slot(str)
    def setGaussianNBParticipant(self, participant):
        """Participant id used by the improved GaussianNB chain (own statistics after calibration, else global)."""
        participant = str(participant).strip()
        if not participant:
            self.logMessage.emit("GaussianNB participant id must not be empty")
            self._emit_gnb_status()
            return
        self.gnb_participant_id = participant
        self.logMessage.emit(f"GaussianNB participant: {participant}")
        self._emit_gnb_status()

    def _log_gnb_calibration(self, entry):
        self.logMessage.emit(f"GaussianNB calibrated '{entry['participant']}': {entry['n_windows']} windows from "
                             f"{entry.get('n_blocks_kept', '?')}/{entry.get('n_blocks', '?')} usable 2 s blocks "
                             f"({entry['source']})")

    @Slot()
    def calibrateGaussianNB(self):
        """Record AVATAR_GNB_CALIBRATION_SECONDS (default 30) s of unlabelled EEG from the current participant and
        register their feature statistics; runs in a worker thread so the window stays responsive."""
        if self.current_bci_source != "openbci" or self.current_data_mode != "live":
            self.logMessage.emit("GaussianNB calibration requires live OpenBCI acquisition")
            return
        if self._gnb_calibrating:
            self.logMessage.emit("GaussianNB calibration already running")
            return
        participant = self.gnb_participant_id
        try:
            seconds = max(10, int(os.environ.get("AVATAR_GNB_CALIBRATION_SECONDS", "30")))
        except ValueError:
            seconds = 30
        self._gnb_calibrating = True
        self._emit_gnb_status()
        self.logMessage.emit(f"GaussianNB calibration ({self.current_framework}): recording {seconds} s of EEG for '{participant}' "
                             "(sit still, eyes open, headset settled)")

        def work():
            try:
                chain = self._gnb_chain()
                processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value, sample_time=seconds)
                capture = processor.get_quality_capture()
                entry = chain.calibrate_from_capture(participant, capture, assessor=self.quality_gate.assessor)
                self._log_gnb_calibration(entry)
            except Exception as error:
                self.logMessage.emit(f"GaussianNB calibration failed: {error}")
            finally:
                self._gnb_calibrating = False
                self._emit_gnb_status()

        threading.Thread(target=work, daemon=True).start()

    @Slot()
    def calibrateGaussianNBFromRecording(self):
        """Calibrate the current participant from one of their OpenBCI GUI recordings (no headset needed)."""
        path = self._choose_recording("Calibrate GaussianNB from an EEG recording of this participant")
        if path is None:
            return
        try:
            entry = self._gnb_chain().calibrate_from_recording(self.gnb_participant_id, path,
                                                               assessor=self.quality_gate.assessor)
            self._log_gnb_calibration(entry)
        except Exception as error:
            self.logMessage.emit(f"GaussianNB calibration from {Path(path).name} failed: {error}")
        self._emit_gnb_status()

    # ---- GaussianNB subject-specific mode: the user's own model from a guided labelled session -------------------
    def _gnb_subject_chain(self, framework=None):
        """Load the subject-specific chain of the selected framework once (three views -> 435 features -> the user's
        statistics and model, PyTorch GaussianNB, TensorFlow GaussianNBTF or JAX GaussianNBJAX)."""
        backend = self._gnb_backend(framework)
        with self.gnb_chain_lock:
            if backend not in self.gnb_subject_chains:
                gnb_dir = Path(__file__).resolve().parent / "prediction-gaussiannb" / "pytorch"
                if str(gnb_dir) not in sys.path:
                    sys.path.insert(0, str(gnb_dir))
                from subject_gnb_inference import SubjectSpecificGaussianNBChain
                chain = SubjectSpecificGaussianNBChain(backend=backend)
                self.gnb_subject_chains[backend] = chain
                self.logMessage.emit(f"GaussianNB subject-specific chain loaded ({self._gnb_label(backend)}): {chain.n_features} features "
                                     f"-> own statistics -> own LDA + GaussianNB -> 5-window context -> vote gate; "
                                     f"{chain.window_size}-sample windows; {len(chain.users)} stored user model(s) in {chain.users_dir}")
            self.gnb_subject_chain = self.gnb_subject_chains[backend]
            return self.gnb_subject_chain

    @Slot(result=str)
    def gaussianNBMode(self):
        return self.gnb_mode

    @Slot(str)
    def setGaussianNBMode(self, mode):
        """'global': the improved global chain (any user, 49 % on Data_clean); 'subject': the user's own model
        (84 % per participant on Data_clean), which needs this user's guided training session first."""
        mode = str(mode).strip().lower()
        if mode not in ("global", "subject"):
            self.logMessage.emit(f"Unknown GaussianNB mode: {mode}")
            return
        self.gnb_mode = mode
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("configuration_changed")
        self.qualityViewChanged.emit(quality_view(self.current_prediction_result, self.quality_gate.assessor.config))
        if mode == "subject":
            try:
                self._gnb_subject_chain()
            except Exception as error:
                self.logMessage.emit(f"GaussianNB subject-specific chain failed to load: {error}")
            self.logMessage.emit("GaussianNB mode: my model (subject-specific; trained from this user's own labelled "
                                 "windows, acts only when 4 of the last 5 window votes agree)")
        else:
            self.logMessage.emit("GaussianNB mode: global chain (one model for every user)")
        self._emit_gnb_status()

    @Slot(result="QVariant")
    def gaussianNBTrainingPlan(self):
        """The guided session the GUI runs (AVATAR_GNB_TRAIN_ROUNDS / _CUE_SECONDS / _REST_SECONDS): schedule
        length, duration and the notebook's learning-curve expectation."""
        gnb_dir = Path(__file__).resolve().parent / "prediction-gaussiannb" / "pytorch"
        if str(gnb_dir) not in sys.path:
            sys.path.insert(0, str(gnb_dir))
        from subject_gnb_inference import session_plan_from_env
        return session_plan_from_env().as_dict()

    @Slot(result="QVariant")
    def gaussianNBUserInfo(self):
        """What is stored for the current participant: windows and cues per command, expectation, self-check."""
        try:
            return self._gnb_subject_chain().participant_info(self.gnb_participant_id)
        except Exception as error:
            return dict(participant=self.gnb_participant_id, status=f"unavailable: {error}", has_model=False)

    @Slot()
    def startGaussianNBTraining(self):
        """Guided labelled session for the current participant: interleaved cues (every round presents every command
        once in a fresh random order) with rest gaps, the EEG of every cue quality-checked per 2 s block, windowed and
        labelled; then the user's statistics and model are fitted from all their stored windows and saved under
        predictions_local/gnb_users/<id>/. Runs in a worker thread; progress arrives on gnbTrainingCueChanged."""
        if self.current_bci_source != "openbci" or self.current_data_mode != "live":
            self.logMessage.emit("GaussianNB training requires live OpenBCI acquisition")
            return
        if self._gnb_training is not None or self._gnb_calibrating:
            self.logMessage.emit("GaussianNB: a training session or calibration is already running")
            return
        participant = self.gnb_participant_id
        try:
            chain = self._gnb_subject_chain()
            from subject_gnb_inference import BrainFlowStream, TrainingSession, session_plan_from_env
            plan = session_plan_from_env()
        except Exception as error:
            self.logMessage.emit(f"GaussianNB training could not start: {error}")
            return
        self._gnb_training_stop.clear()
        stream = BrainFlowStream(board_id=BoardIds.CYTON_DAISY_BOARD.value)
        session = TrainingSession(chain, participant, plan, stream, present=self.gnbTrainingCueChanged.emit,
                                  should_stop=self._gnb_training_stop.is_set, assessor=self.quality_gate.assessor)
        self._gnb_training = session
        self._emit_gnb_status()
        info = plan.as_dict()
        self.logMessage.emit(f"GaussianNB training session ({self._gnb_label(chain.backend)}) for '{participant}': {plan.rounds} rounds x {len(plan.commands)} "
                             f"commands x {plan.cue_seconds:.0f} s cues with {plan.rest_seconds:.0f} s rest, about "
                             f"{info['total_minutes']} min. Think the cued command for the whole cue; {info['expectation']}")

        def work():
            try:
                summary = session.run()
                self.logMessage.emit(f"GaussianNB training session {summary['session_id']}: {summary['cues_done']}/{summary['n_cues']} "
                                     f"cues, {summary['windows_total']} labelled windows; {summary['message']}"
                                     + (f"; cue-grouped self-check {100 * summary['cue_cv']['accuracy']:.0f} % "
                                        f"(5-window context {100 * summary['cue_cv']['context_accuracy']:.0f} %)"
                                        if summary.get("cue_cv") else ""))
                self.gnbTrainingFinished.emit(summary)
            except Exception as error:
                self.logMessage.emit(f"GaussianNB training session failed: {error}")
                self.gnbTrainingFinished.emit(dict(trained=False, message=str(error), participant=participant,
                                                   session_id=session.session_id))
            finally:
                self._gnb_training = None
                self._emit_gnb_status()

        threading.Thread(target=work, daemon=True).start()

    @Slot()
    def stopGaussianNBTraining(self):
        """Stop the running session after the current second; the cues recorded so far are kept and fitted."""
        if self._gnb_training is None:
            self.logMessage.emit("GaussianNB: no training session is running")
            return
        self._gnb_training_stop.set()
        self.logMessage.emit("GaussianNB training: stopping; the windows collected so far are kept")

    @Slot()
    def trainGaussianNBFromFolder(self):
        """Train (or extend) the current participant's model from a folder of their labelled OpenBCI GUI recordings
        laid out like Data_clean: <folder>/<command>/OpenBCI-RAW-*.txt. Every recording counts as one cue."""
        folder = QFileDialog.getExistingDirectory(None, "Folder with this user's labelled recordings (<command>/OpenBCI-RAW-*.txt)",
                                                  str(Path.home()))
        if not folder:
            return
        try:
            user = self._gnb_subject_chain().train_from_folder(self.gnb_participant_id, folder, assessor=self.quality_gate.assessor)
            cv = user.meta.get("cue_cv")
            self.logMessage.emit(f"GaussianNB trained '{user.participant}' from {Path(folder).name}: {user.status()}"
                                 + (f"; cue-grouped self-check {100 * cv['accuracy']:.0f} %" if cv else ""))
        except Exception as error:
            self.logMessage.emit(f"GaussianNB training from {Path(folder).name} failed: {error}")
        self._emit_gnb_status()

    def _run_gaussiannb_subject(self, framework=None):
        """Subject-specific mode: three views -> 435 features -> the user's statistics and model -> 5-window context
        -> vote-agreement gate (>= 4 of the last 5 window votes). Only accepted raw captures reach it."""
        chain = self._gnb_subject_chain(framework)
        self._emit_gnb_status()
        processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value,
                                           sample_time=int(chain.capture_seconds) + 2)
        capture = processor.get_quality_capture()
        n = chain.capture_samples
        if capture["raw"].shape[1] < n:
            return self.quality_gate.withhold("Insufficient samples for the GaussianNB 5-window context")
        for name in ("raw", "counters", "timestamps"):
            capture[name] = capture[name][:, -n:] if name == "raw" else capture[name][-n:]
        result = chain.predict_capture(capture, self.gnb_participant_id, gate=self.quality_gate)
        decision = chain.last_decision
        if decision is not None:
            probs = ", ".join(f"{k} {v:.2f}" for k, v in decision["probabilities"].items())
            self.logMessage.emit(f"GaussianNB (my model, {self._gnb_label(chain.backend)}): {decision['label']} - {decision['agreement']}/{decision['votes']} window "
                                 f"votes for '{decision['command']}' (need {decision['votes_needed']}); participant "
                                 f"'{decision['participant']}' {decision['participant_status']}; {probs}")
        return result

    def run_gaussiannb_pytorch(self):
        """Improved chain with the PyTorch GaussianNB: loader filtering -> 128-sample windows -> 570 features ->
        participant z-score -> LDA -> GaussianNB -> 5-window causal context -> per-class thresholds."""
        return self._run_gaussiannb("PyTorch")

    def run_gaussiannb_tensorflow(self):
        """The same chains with the TensorFlow GaussianNBTF as the decision model (artifacts of
        prediction-gaussiannb/tensorflow/gaussiannb_train.ipynb: 125-sample windows, argmax decision)."""
        return self._run_gaussiannb("TensorFlow")

    def run_gaussiannb_jax(self):
        """The same chains with the JAX GaussianNBJAX as the decision model, on the CPU (artifacts of
        prediction-gaussiannb/jax/instructions_log_gnb_jax.ipynb: 125-sample windows with a 50-sample stride, argmax
        decision); calibrations and user models live in predictions_local/gnb_calibration_jax and gnb_users_jax."""
        return self._run_gaussiannb("JAX")

    def _run_gaussiannb(self, framework):
        """Only accepted raw captures reach the chain. In subject-specific mode (setGaussianNBMode("subject")) the
        user's own model of the selected framework runs instead of the global chain."""
        if self.current_bci_source != "openbci" or self.current_data_mode != "live":
            return self.quality_gate.withhold("GaussianNB inference requires verified live OpenBCI acquisition")
        if framework not in self.GNB_BACKENDS:
            return self.quality_gate.withhold(f"GaussianNB is not implemented for the {framework} framework")
        if self._gnb_calibrating:
            return self.quality_gate.withhold("GaussianNB calibration in progress")
        if self._gnb_training is not None:
            return self.quality_gate.withhold("GaussianNB training session in progress")
        if self.gnb_mode == "subject":
            return self._run_gaussiannb_subject(framework)
        chain = self._gnb_chain(framework)
        self._emit_gnb_status()
        processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value,
                                           sample_time=int(chain.capture_seconds) + 2)
        capture = processor.get_quality_capture()
        n = chain.capture_samples
        if capture["raw"].shape[1] < n:
            return self.quality_gate.withhold("Insufficient samples for the GaussianNB 5-window context")
        for name in ("raw", "counters", "timestamps"):
            capture[name] = capture[name][:, -n:] if name == "raw" else capture[name][-n:]
        result = chain.predict_capture(capture, self.gnb_participant_id, gate=self.quality_gate)
        decision = chain.last_decision
        if decision is not None:
            probs = ", ".join(f"{k} {v:.2f}" for k, v in decision["probabilities"].items())
            self.logMessage.emit(f"GaussianNB ({framework}): {decision['label']} ({decision['context_windows']}-window context; "
                                 f"participant '{decision['participant']}' {decision['participant_status']}; {probs})")
        return result

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Handle manual action input
        self.predictions_log.append({
            "count": "manual",
            "server": "manual",
            "label": manual_action
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Also update flight log
        self.flight_log.insert(0, f"Manual Action: {manual_action}")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit(f"Manual input: {manual_action}") 

    @Slot()
    def executeAction(self):
        result = self.current_prediction_result
        if not may_execute(result):
            self.current_prediction_label = ""
            self.current_prediction_result = withheld_result("prediction_missing_or_expired")
            self.signalQualityChanged.emit("Prediction withheld: obtain a fresh accepted EEG capture")
            self.logMessage.emit("No fresh quality-approved prediction to execute")
            return
        self._queue_prediction(result)
        self.flight_log.insert(0, f"Queued: {result['prediction_label']}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def connectDrone(self):
        self.doDroneTAction('connect')
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit("Drone connected.")

    @Slot(str)
    def keepDroneAlive(self,text):
        # Mock function to simulate sending keep-alive signal
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

        #remove null at end and make all lowercase
        text = text.strip().lower()

        #exicut cmd
        self.doDroneTAction(text)

        
        

    @Slot(str)
    def doDroneTAction(self, action):
        if action in ('up', 'down', 'forward', 'backward', 'left', 'right'):
            # Clumped in main thread; clumper will enqueue final chunks.
            self.enqueueMoveRequested.emit(action)
        else:
            # Non-movement actions go straight into the queue.
            self._queue_action(action)



    def _enqueue_move(self, direction: str):
        with self.drone_lock:
            if self.clump_dir is None:
                # start new batch
                self.clump_dir = direction
                self.clump_count = 1
            elif self.clump_dir == direction:
                # same direction → just increase count
                self.clump_count += 1
            else:
                # different direction → flush existing batch immediately
                d = self.clump_dir
                c = self.clump_count
                self.clump_dir = direction
                self.clump_count = 1
                # flush previous batch (this only enqueues work)
                self._execute_clumped_move(d, c)

            # (re)start 1s timer (Qt main thread)
            self.clump_timer.stop()
            self.clump_timer.start(self.clump_window_ms)


    def _flush_clumped_move(self):
        with self.drone_lock:
            if self.clump_dir is None or self.clump_count == 0:
                return
            direction = self.clump_dir
            count = self.clump_count
            self.clump_dir = None
            self.clump_count = 0

        # Just enqueue the appropriate chunks; execution is in worker thread
        self._execute_clumped_move(direction, count)


    def _split_distance(self, total, max_step=500):
        """
        Split a total distance into chunks of at most max_step.
        Example: 1100 -> [500, 500, 100]
        """
        chunks = []
        remaining = total
        while remaining > 0:
            step = min(remaining, max_step)
            chunks.append(step)
            remaining -= step
        return chunks

    def _execute_clumped_move(self, direction: str, count: int):
        total_distance = count * self.step_cm
        chunks = self._split_distance(total_distance, max_step=500)

        for dist in chunks:
            self._queue_action(direction, dist)



    @Slot(str)
    def getDroneAction(self, action):
        with self.drone_lock:
            try:
                if action == 'connect':
                    self.tello.connect(wait_for_state=False)
                    battery = self.tello.get_battery()
                    self.connected = True
                    self.logMessage.emit(f"Connected to Tello Drone (Battery: {battery}%)")
                    self.flight_log.insert(0, f"Drone connected (Battery: {battery}%)")
                    self.flightLogUpdated.emit(self.flight_log)
                    return
                elif not self.connected:
                    self.logMessage.emit("Drone not connected. Please connect first.")
                    self.flight_log.insert(0, "Command failed: Drone not connected")
                    self.flightLogUpdated.emit(self.flight_log)
                    return

                def record_action(name, value=None):
                    self.action_log.append((name, value))

                # movement actions: distance is normally 30cm,
                # but can be overridden by the clumper
                if action == 'up':
                    dist = self._movement_distance_override or 30
                    self.tello.move_up(dist)
                    record_action('up', dist)
                    self.logMessage.emit("Moving up")
                    self.flight_log.insert(0, f"Moving up {dist}cm")

                elif action == 'down':
                    dist = self._movement_distance_override or 30
                    self.tello.move_down(dist)
                    record_action('down', dist)
                    self.logMessage.emit("Moving down")
                    self.flight_log.insert(0, f"Moving down {dist}cm")

                elif action == 'forward':
                    dist = self._movement_distance_override or 30
                    self.tello.move_forward(dist)
                    record_action('forward', dist)
                    self.logMessage.emit("Moving forward")
                    self.flight_log.insert(0, f"Moving forward {dist}cm")

                elif action == 'backward':
                    dist = self._movement_distance_override or 30
                    self.tello.move_back(dist)
                    record_action('backward', dist)
                    self.logMessage.emit("Moving backward")
                    self.flight_log.insert(0, f"Moving backward {dist}cm")

                elif action == 'left':
                    dist = self._movement_distance_override or 30
                    self.tello.move_left(dist)
                    record_action('left', dist)
                    self.logMessage.emit("Moving left")
                    self.flight_log.insert(0, f"Moving left {dist}cm")

                elif action == 'right':
                    dist = self._movement_distance_override or 30
                    self.tello.move_right(dist)
                    record_action('right', dist)
                    self.logMessage.emit("Moving right")
                    self.flight_log.insert(0, f"Moving right {dist}cm")

					
                elif action == 'turn_left':
                    self.tello.rotate_counter_clockwise(45)
                    record_action('turn_left', 45)
                    self.logMessage.emit("Rotating left")
                    self.flight_log.insert(0, "Rotating left 45°")
					
                elif action == 'turn_right':
                    self.tello.rotate_clockwise(45)
                    record_action('turn_right', 45)
                    self.logMessage.emit("Rotating right")
                    self.flight_log.insert(0, "Rotating right 45°")
					
                elif action == 'flip_forward':
                    self.tello.flip_forward()
                    record_action('flip_forward')
                    self.logMessage.emit("Flipping forward")
                    self.flight_log.insert(0, "Flipping forward")
					
                elif action == 'flip_back':
                    self.tello.flip_back()
                    record_action('flip_back')
                    self.logMessage.emit("Flipping backward")
                    self.flight_log.insert(0, "Flipping backward")
					
                elif action == 'flip_left':
                    self.tello.flip_left()
                    record_action('flip_left')
                    self.logMessage.emit("Flipping left")
                    self.flight_log.insert(0, "Flipping left")
					
                elif action == 'flip_right':
                    self.tello.flip_right()
                    record_action('flip_right')
                    self.logMessage.emit("Flipping right")
                    self.flight_log.insert(0, "Flipping right")

                elif action == 'takeoff':
                    self.tello.takeoff()
                    self.logMessage.emit("Taking off")
                    self.flight_log.insert(0, "Taking off")
                    self.action_log.clear()
				
                elif action == 'land':
                    self.tello.land()
                    self.logMessage.emit("Landing")
                    self.flight_log.insert(0, "Landing")
                    self.action_log.clear()
					
                elif action == 'go_home':
                    self.go_home()
                    self.logMessage.emit("Going home")
                    self.flight_log.insert(0, "Going home")
					
                elif action == 'stream':
                    if hasattr(self, 'camera_controller'):
                        self.camera_controller.start_camera_stream()
                        self.logMessage.emit("Starting camera stream")
                        self.flight_log.insert(0, "Starting camera stream")
						
                    else:
                        self.logMessage.emit("Camera controller not available")
                        self.flight_log.insert(0, "Camera controller not available")
						
                else:
                    self.logMessage.emit("Unknown action")
                    self.flight_log.insert(0, "Unknown action")

                self.flightLogUpdated.emit(self.flight_log)

            except Exception as e:
                error_msg = f"Error during {action}: {str(e)}"
                self.logMessage.emit(error_msg)
                self.flight_log.insert(0, error_msg)
                self.flightLogUpdated.emit(self.flight_log)
                # If critical error, mark as disconnected
                if "Tello" in str(e) or "timeout" in str(e).lower():
                    self.connected = False

    def go_home(self):
        try:
            self.logMessage.emit("Returning to home by reversing actions...")
            self.flight_log.insert(0, "Returning to home by reversing actions")

            # Take a snapshot so we don't race with new actions
            history = list(self.action_log)  # list of (name, value)

            current_dir = None
            current_total = 0

            def flush_segment():
                nonlocal current_dir, current_total
                if current_dir is None or current_total <= 0:
                    return
                # Use the same 500cm safety limit
                for dist in self._split_distance(current_total, max_step=500):
                    # Queue a single clumped command (direction + distance)
                    self._queue_action(current_dir, dist)
                current_dir = None
                current_total = 0

            # Walk the history in reverse order
            for action, value in reversed(history):
                opp = None

                # Map movement actions to their opposites
                if action == "up":
                    opp = "down"
                elif action == "down":
                    opp = "up"
                elif action == "forward":
                    opp = "backward"
                elif action == "backward":
                    opp = "forward"
                elif action == "left":
                    opp = "right"
                elif action == "right":
                    opp = "left"

                if opp is not None:
                    # Use recorded distance if available, otherwise default step size
                    dist = value if (value is not None) else self.step_cm

                    # If same direction as current segment, accumulate; otherwise flush + start new
                    if current_dir == opp:
                        current_total += dist
                    else:
                        flush_segment()
                        current_dir = opp
                        current_total = dist
                else:
                    # For non-movement actions, flush the current segment and enqueue them individually
                    flush_segment()
                    if action == "turn_left":
                        self._queue_action("turn_right")
                    elif action == "turn_right":
                        self._queue_action("turn_left")
                    elif action == "flip_forward":
                        self._queue_action("flip_back")
                    elif action == "flip_back":
                        self._queue_action("flip_forward")
                    elif action == "flip_left":
                        self._queue_action("flip_right")
                    elif action == "flip_right":
                        self._queue_action("flip_left")
                    # ignore 'takeoff', 'land', 'go_home' themselves, etc.

            # Flush any remaining movement in the last segment
            flush_segment()

            # Finally, land when we get "home"
            self._queue_action("land")

            self.flight_log.insert(0, "Go-home path queued")
            self.logMessage.emit("Go-home path queued.")
            self.flightLogUpdated.emit(self.flight_log)

            # Reset action log for the next flight
            self.action_log.clear()

        except Exception as e:
            error_msg = f"Error during go_home: {str(e)}"
            self.logMessage.emit(error_msg)
            self.flight_log.insert(0, error_msg)
            self.flightLogUpdated.emit(self.flight_log)


    @Slot()
    def check_plots_exist(self):
        """
        Check if all necessary plot PDFs exist in both Rollback and Refresh directories.
        If not, run controller.py to generate them.
        """
        print("\n=== CHECKING IF PLOTS EXIST ===")

        # Create plots base directory if it doesn't exist
        plots_base_dir = Path(self.plots_dir)
        if not plots_base_dir.exists():
            print(f"Creating plots base directory: {plots_base_dir}")
            plots_base_dir.mkdir(parents=True, exist_ok=True)

        # List of datasets to check
        datasets = ["rollback", "refresh"]

        # List of PDF files that should exist for each dataset
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        # Check if all directories and PDFs exist
        missing_pdfs = False
        for dataset in datasets:
            dataset_dir = plots_base_dir / dataset
            if not dataset_dir.exists():
                print(f"Creating dataset directory: {dataset_dir}")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                missing_pdfs = True
                continue

            print(f"Checking PDFs in {dataset_dir}...")
            for pdf_file in pdf_files:
                pdf_path = dataset_dir / pdf_file
                if not pdf_path.exists():
                    print(f"Missing file: {pdf_path}")
                    missing_pdfs = True
                    break

        # If any PDFs are missing, run the controller.py script
        if missing_pdfs:
            print("Some plot files are missing. Running controller.py to generate them...")

            # Get the path to controller.py in the plotscode directory
            controller_path = Path(self.plots_dir).parent / "controller.py"  # plotscode/controller.py
            print(f"Controller path: {controller_path}")
            print(f"Controller exists: {controller_path.exists()}")

            if controller_path.exists():
                try:
                    # Change to the plotscode directory before running the script
                    original_dir = os.getcwd()
                    os.chdir(controller_path.parent)

                    # Run the controller.py script to generate plots for both datasets
                    print(f"Executing: {sys.executable} {controller_path}")
                    result = subprocess.run(
                        [sys.executable, str(controller_path)],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    # Go back to the original directory
                    os.chdir(original_dir)

                    # Print output for debugging
                    print(f"Output: {result.stdout}")
                    if result.stderr:
                        print(f"Errors: {result.stderr}")

                    print("Successfully generated plot files.")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Error running controller.py: {e}")
                    if hasattr(e, 'stderr'):
                        print(f"Error output: {e.stderr}")
                    return False
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    return False
            else:
                print(f"Controller script not found: {controller_path}")
                return False

        return True  # All files exist

    @Slot(str)
    def setDataset(self, dataset_name):
        """
        Set the current dataset to display (refresh or rollback).
        :param dataset_name: Name of the dataset ('refresh' or 'rollback')
        """
        if dataset_name.lower() in ["refresh", "rollback"]:
            self.current_dataset = dataset_name.lower()
            print(f"Switched to {self.current_dataset} dataset")
            # Update the displayed images
            self.convert_pdfs_to_images()
        else:
            print(f"Invalid dataset name: {dataset_name}")

    @Slot()
    def convert_pdfs_to_images(self):
        """
        Convert PDF files from the current dataset to images and send to QML.
        """
        print(f"\n=== STARTING CONVERT PDFS TO IMAGES FOR {self.current_dataset.upper()} ===")

        # First check if all plot PDFs exist, and generate them if needed
        success = self.check_plots_exist()
        print(f"Result of check_plots_exist: {success}")

        # Current dataset directory
        dataset_dir = Path(self.plots_dir) / self.current_dataset

        # Convert PDF files to images and send image paths + graph names to QML.
        self.image_paths = []
        graph_titles = ["Takeoff", "Forward", "Right",
                        "Landing", "Backward", "Left"]

        # Load files in the correct order
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = dataset_dir / pdf_file
            if not pdf_path.exists():
                print(f"Missing file: {pdf_path}")  # Debugging: Check missing PDFs
                continue  # Skip if file does not exist

            images = convert_from_path(str(pdf_path), dpi=150)  # Convert PDF to image
            image_path = dataset_dir / f"{pdf_file.replace('.pdf', '.png')}"
            images[0].save(str(image_path), "PNG")  # Save first page as an image

            # Debugging: Print the generated image path
            print(f"Generated image: {image_path}")

            self.image_paths.append({
                "graphTitle": graph_titles[i],
                "imagePath": QUrl.fromLocalFile(str(image_path)).toString()
            })

        # Debugging: Print final list of image paths
        print("Final Image Paths Sent to QML:", self.image_paths)

        self.imagesReady.emit(self.image_paths)  # Send data to QML

    #Allows the user to switch between the OpenBCI or Neurosity Crown headsets
    @Slot(str)
    def setBCISource(self, source):
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("acquisition_changed")
        self.logMessage.emit(f"Switching BCI source to: {source}")

        if source == "neurosity":
            self.current_bci_source = "neurosity"

        elif source == "openbci":
            self.current_bci_source = "openbci"

        else:
            self.logMessage.emit("Unknown BCI source selected")
            
    @Slot(str)
    def setDataMode(self, mode):
        """
        Set data mode to either synthetic or live based on radio button selection.
        """
        
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("acquisition_changed")
        self.current_data_mode = mode

        if mode == "synthetic":
            self.init_synthetic_board()
            self.logMessage.emit("Switched to Synthetic Data Mode")
        elif mode == "live":
            self.init_live_board()
            self.logMessage.emit("Switched to Live Data Mode")
        else:
            print(f"Unknown data mode: {mode}")

    def init_synthetic_board(self):
        """ Initialize BrainFlow with synthetic board for testing """
        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        print("\nSynthetic board initialized.")

    def init_live_board(self):
        """ Initialize BrainFlow with a real headset """
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-D200PMA1"  # Update if different on your system
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        print("\nLive headset board initialized.")

    @Slot(str)
    def setBCISource(self, source):
        self.current_prediction_label = ""
        self.current_prediction_result = withheld_result("acquisition_changed")
        self.logMessage.emit(f"Switching BCI source to: {source}")

        if source == "neurosity":
            self.current_bci_source = "neurosity"
            self.current_data_mode = "live"
            # print ("setBCISource bug")
            try:
                self.init_neurosity_processor()
                state = self.neurosity_processor.get_device_state_once()
                self.emitNeurosityStatus(state)
            except Exception as e:
                self.neurosity_connected = False
                self.neurosity_processor = None
                self.logMessage.emit(f"Failed to initialize Neurosity: {e}")

        elif source == "openbci":
            self.current_bci_source = "openbci"
            self.neurosity_connected = False
            self.neurosity_processor = None
            if self.current_data_mode == "live":
                self.init_live_board()

        else:
            self.logMessage.emit("Unknown BCI source selected")

    def init_neurosity_processor(self):
        """Initialize the Neurosity processor using environment variables."""
        if self.neurosity_processor is not None and self.neurosity_connected:
            return

        email = os.getenv("NEUROSITY_EMAIL")
        password = os.getenv("NEUROSITY_PASSWORD")
        device_id = os.getenv("NEUROSITY_DEVICE_ID")

        missing = [k for k, v in (
            ("NEUROSITY_EMAIL", email),
            ("NEUROSITY_PASSWORD", password),
            ("NEUROSITY_DEVICE_ID", device_id),
        ) if not v]
        if missing:
            raise RuntimeError(
                "Missing Neurosity credentials in .env.txt: " + ", ".join(missing)
            )

        self.neurosity_processor = NeurosityDataProcessor(email, password, device_id, status_callback=self.emitNeurosityStatus)
        self.neurosity_connected = True
        self.logMessage.emit("Neurosity connector initialized")

    def emitNeurosityStatus(self, state):
        if state == "online":
            self.neurosityStatusChanged.emit("Online")
        else:
            self.neurosityStatusChanged.emit("Offline")

    def get_neurosity_brainwave_data(self):
        """Capture and preprocess EEG data from the Neurosity headset."""
        if not self.neurosity_connected:
            self.init_neurosity_processor()
        if self.neurosity_processor is None:
            raise RuntimeError("Neurosity processor is not configured")
        state = self.neurosity_processor.get_device_state_once()
        self.emitNeurosityStatus(state)
        if state != "online":
            self.logMessage.emit(
                f"Warning: Neurosity device is {state}."
            )
            raise RuntimeError(
                f"Neurosity device is {state}."
            )
        
        self.logMessage.emit("Capturing EEG from Neurosity headset...")
        self.brainwave_data = self.neurosity_processor.get_tensor()
        self.logMessage.emit("Neurosity EEG data captured")
        # print(self.brainwave_data)
        return self.brainwave_data


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Create our controllers
    tab_controller = TabController()
    print("TabController created")
    manual_nao_controller = ManualNaoController()
    drone_camera_controller = DroneCameraController()

    # Initialize backend before loading QML
    # Queue holds just directions now: e.g. "forward", "left", etc.
    cloud_api = CloudAPI()
    backend = BrainwavesBackend()
    developers = DevelopersAPI()
    shuffler_api = ShufflerAPI()
    engine.rootContext().setContextProperty("tabController", tab_controller)
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("developersBackend", developers)
    engine.rootContext().setContextProperty("cloudAPI", cloud_api)
    engine.rootContext().setContextProperty("imageModel", [])  # Initialize empty model
    engine.rootContext().setContextProperty("cameraController", backend.camera_controller)
    print("Controllers exposed to QML")
    engine.rootContext().setContextProperty("fileShufflerGui", shuffler_api)  # For file shuffler
    engine.rootContext().setContextProperty("manualNaoController", manual_nao_controller)
    engine.rootContext().setContextProperty("droneCameraController", drone_camera_controller)

    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"

    engine.load(str(qml_file))

    # Start of change : Added Cloud Computing (Transfer Data) functionality 

    if engine.rootObjects():
        cloud_api.set_root_object(engine.rootObjects()[0])
        cloud_api.connect_buttons()
    else:
        print("Error: QML not loaded properly.")

    # End of change : Added Cloud Computing (Transfer Data) functionality 


    # Convert PDFs after engine load
    try:
        backend.convert_pdfs_to_images()
    except Exception as e:
        print(f"Error converting PDFs: {str(e)}")

    # Ensure image model updates correctly
    backend.imagesReady.connect(lambda images: engine.rootContext().setContextProperty("imageModel", images))

    sys.exit(app.exec())