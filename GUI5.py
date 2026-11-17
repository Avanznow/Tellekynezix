import sys
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load Neurosity credentials from the bundled env file before anything reads os.environ.
load_dotenv(Path(__file__).resolve().parent / ".env.txt")

import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication,QFileDialog, QMessageBox
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, Property, QProcess, QUrl, QTimer
from pdf2image import convert_from_path
from djitellopy import Tello
import random
import threading
import re
import torch
import time
from collections import defaultdict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from predictions_local.brainflowprocessor import BrainFlowDataProcessor
from predictions_local.deeplearningpytorchpredictor import DeeplearningPytorchPredictor
from predictions_local.neurosityprocessor import NeurosityDataProcessor
from cameraview.camera_controller import CameraController
from NAO6.nao_connection import send_command
from robot_services.qt_simulation import SimulationController
import asyncio
import copy
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

class SimulatedTello:
    """Stand-in used when no real Tello is connected. Prints each command
    and waits a moment, so the whole app (log, 30 s hold, auto-return) works."""
    def _cmd(self, name, *args):
        print(f"[SIM] {name} {' '.join(str(a) for a in args)}".strip())
        time.sleep(0.5)

    def move_up(self, d): self._cmd("move_up", d)
    def move_down(self, d): self._cmd("move_down", d)
    def move_forward(self, d): self._cmd("move_forward", d)
    def move_back(self, d): self._cmd("move_back", d)
    def move_left(self, d): self._cmd("move_left", d)
    def move_right(self, d): self._cmd("move_right", d)
    def rotate_clockwise(self, deg): self._cmd("rotate_clockwise", deg)
    def rotate_counter_clockwise(self, deg): self._cmd("rotate_counter_clockwise", deg)
    def flip_forward(self): self._cmd("flip_forward")
    def flip_back(self): self._cmd("flip_back")
    def flip_left(self): self._cmd("flip_left")
    def flip_right(self): self._cmd("flip_right")
    def takeoff(self): self._cmd("takeoff")
    def land(self): self._cmd("land")
    def send_rc_control(self, *a): pass


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
    loginSucceeded = Signal("QVariantList")
    loginFailed = Signal(str)
    deviceConnected = Signal()
    deviceOffline = Signal(str)
    moveCompleted = Signal()        # emitted by worker thread after a user move finishes
    flyingStateChanged = Signal(bool)  # emitted by worker thread on takeoff/land

    # 3D drone view in ManualDroneControl.qml
    drone3DPreview = Signal(str)   # every button / BCI command -> instant animation
    drone3DAction = Signal(str)    # backend-initiated actions (auto-return home)
    drone3DUpdated = Signal(float, float, float, float)  # x, y, z, rotation (pose sync)

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
        self.current_data_mode = "synthetic"
        self.current_bci_source = "openbci"
        self.current_model = "Random Forest"  # Default model
        self.current_framework = "PyTorch"  # Default framework
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

        # Auto-return-home: after a directional move, hold position for 30 s.
        # If no other button is pressed in that time, fly back home and hover.
        self.auto_return_ms = 30000
        self.auto_return_timer = QTimer()
        self.auto_return_timer.setSingleShot(True)
        self.auto_return_timer.timeout.connect(self._auto_return_home)
        self.moveCompleted.connect(self._arm_auto_return)
        self.flyingStateChanged.connect(self._on_flying_state_changed)

        self.sim_tello = SimulatedTello()  # used when no real drone is connected
        self._busy = False             # True while the worker is running a command
        self._current_is_return = False  # True while running a return-home step


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
    def _queue_action(self, action: str, dist: int | None = None, is_return: bool = False):
        """
        Put a command into the Tello queue.
        dist is an optional distance override (cm) for movement commands.
        is_return marks steps of a return-home path (not recorded in action_log).
        """
        self.cmd_queue.put((action, dist, is_return))

    def _drone_loop(self):
        """
        Single worker thread that owns the Tello.
        It pulls commands from the queue and executes them in FIFO order.
        """
        while True:
            action, dist, is_return = self.cmd_queue.get()
            self._busy = True
            self._current_is_return = is_return
            try:
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
                self._current_is_return = False
                self._busy = False
                self.cmd_queue.task_done()

    def _drone(self):
        """Real Tello when connected, otherwise the simulator."""
        if self.connected and hasattr(self, 'tello'):
            return self.tello
        return self.sim_tello

    @Slot()
    def takeoff(self):
        self._queue_action('takeoff')
        self.logMessage.emit("Hovering")

    def hover_callback(self):
        # Keeps the Tello hovering (and stops its 15 s no-command auto-land)
        # while it holds position. Skipped while a command is running.
        if self.is_flying and not self._busy and self.connected:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
            except Exception as e:
                print(f"Hover keepalive failed: {e}")

    @Slot(bool)
    def _on_flying_state_changed(self, flying):
        self.is_flying = flying
        if flying:
            self.hover_timer.start(200)
        else:
            self.hover_timer.stop()
            self.auto_return_timer.stop()

    @Slot()
    def _arm_auto_return(self):
        """(Re)start the 30 s hold timer after a directional move finishes."""
        self.auto_return_timer.start(self.auto_return_ms)
        self.logMessage.emit("Holding position for 30 s...")

    @Slot()
    def _auto_return_home(self):
        """30 s passed with no new button press: go home and hover."""
        if not self.is_flying or not self.action_log:
            return
        self.logMessage.emit("No input for 30 s - returning home")
        self.flight_log.insert(0, "Auto-return: no input for 30 s")
        self.flightLogUpdated.emit(self.flight_log)
        self.drone3DAction.emit("go_home")   # bring the 3D drone home too
        self.go_home(land=False)

    @Slot(str)
    def selectModel(self, model_name):
        """ Select the machine learning model """
        self.logMessage.emit(f"Model selected: {model_name}")
        self.current_model = model_name
        self.flight_log.insert(0, f"Selected Model: {model_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def selectFramework(self, framework_name):
        """ Select the machine learning framework """
        self.logMessage.emit(f"Framework selected: {framework_name}")
        self.current_framework = framework_name
        self.flight_log.insert(0, f"Selected Framework: {framework_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def readMyMind(self):
        """ Runs the selected model and processes the brainwave data. """
        if self.current_model == "Random Forest":
            if self.current_framework == "PyTorch":
                prediction = self.run_random_forest_pytorch()
            else:
                prediction = self.run_random_forest_tensorflow()
        elif self.current_model == "GaussianNB":
            if self.current_framework == "PyTorch":
                prediction = self.run_gaussiannb_pytorch()
            else:
                prediction = self.run_gaussiannb_tensorflow()
        else:  # Deep Learning
            if self.current_framework == "PyTorch":
                prediction = self.run_deep_learning_pytorch()
                self.logMessage.emit("Getting prediction from pytorch using the deep learning model.")
            else:
                prediction = self.run_deep_learning_tensorflow()
        self.logMessage.emit(f"Prediction received: {prediction}")
        # Set current prediction
        self.current_prediction_label = prediction

        #auto mode is desired
        self.doDroneTAction(prediction)

        # Log the prediction
        self.predictions_log.append({
            "count": str(len(self.predictions_log) + 1),
            "server": "Brainwave AI",
            "label": prediction
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Update Flight Log
        self.flight_log.insert(0, f"Executed: {prediction} (Model: {self.current_model}, Framework: {self.current_framework})")
        self.flightLogUpdated.emit(self.flight_log)

    def run_random_forest_pytorch(self):
        """ Random Forest model processing with PyTorch backend """
        print("Running Random Forest Model with PyTorch...")
        try:
            if self.current_bci_source == "neurosity":
                self.logMessage.emit("Using Neurosity for Random Forest prediction")
                self.get_neurosity_brainwave_data()
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

            # Use the BCI connection to get real brainwave data from OpenBCI
            if hasattr(self, 'bcicon') and self.bcicon:
                prediction_response = self.bcicon.bciConnectionController()
                if prediction_response:
                    return prediction_response.get('prediction_label', 'forward')
        except Exception as e:
            print(f"Error with PyTorch Random Forest: {e}")

        # Fallback to simulation with PyTorch-specific labels
        time.sleep(1)
        return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

    def run_random_forest_tensorflow(self):
        """ Random Forest model processing with TensorFlow backend """
        print("Running Random Forest Model with TensorFlow...")
        try:
            if self.current_bci_source == "neurosity":
                self.logMessage.emit("Using Neurosity for TensorFlow Random Forest prediction")
                self.get_neurosity_brainwave_data()
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

            # Use the BCI connection to get real brainwave data from OpenBCI
            if hasattr(self, 'bcicon') and self.bcicon:
                prediction_response = self.bcicon.bciConnectionController()
                if prediction_response:
                    return prediction_response.get('prediction_label', 'forward')
        except Exception as e:
            print(f"Error with TensorFlow Random Forest: {e}")

        # Fallback to simulation with TensorFlow-specific labels
        time.sleep(1)
        return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

    def run_deep_learning_pytorch(self):
        """ Deep Learning model processing with PyTorch backend """
        print("Running Deep Learning Model with PyTorch...")       
        self.get_brainwave_data()
        try:
            model = DeeplearningPytorchPredictor()
            pred_label = model(self.brainwave_data)
            return pred_label
        except Exception as e:
            print(f"Error with PyTorch Deep Learning: {e}")
            return "Error"
    
    def get_brainwave_data(self):
        if self.current_bci_source == 'neurosity':
            # print ("get_brainwave_data bug")
            try:
                return self.get_neurosity_brainwave_data()
            except Exception as e:
                self.logMessage.emit(f"Neurosity capture failed, falling back to OpenBCI/OpenBCI synthetic: {e}")
                self.current_bci_source = 'openbci'

        if self.current_data_mode == 'synthetic':
            self.brainwave_processor = BrainFlowDataProcessor(board_id=BoardIds.SYNTHETIC_BOARD.value)
            self.brainwave_data = self.brainwave_processor.get_tensor()
            print("synethetic data retrieved")
            return self.brainwave_data
        else:
            self.brainwave_processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value)
            self.brainwave_data = self.brainwave_processor.get_tensor()
            print("live Cyton Daisy data retrieved")
            return self.brainwave_data

    def run_deep_learning_tensorflow(self):
        """ Deep Learning model processing with TensorFlow backend """
        print("Running Deep Learning Model with TensorFlow...")
        try:
            # Simulate TensorFlow deep learning model processing
            # In a real implementation, this would load and run a TensorFlow CNN model
            time.sleep(2)  # Simulate longer processing time for deep learning
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land", "up", "down"])
        except Exception as e:
            print(f"Error with TensorFlow Deep Learning: {e}")
            return "forward"
    def run_gaussiannb_pytorch(self):
        """ GaussianNB model processing with PyTorch backend """
        print("Running GaussianNB Model with PyTorch...")
        try:
            if self.current_bci_source == "neurosity":
                self.logMessage.emit("Using Neurosity for GaussianNB prediction")
                self.get_neurosity_brainwave_data()
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

            # Import the GaussianNB model
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'prediction-gaussiannb', 'pytorch'))
            from gaussiannb_model import GaussianNB
            
            # Try to load trained model
            model_path = os.path.join(os.path.dirname(__file__), 'prediction-gaussiannb', 'pytorch', 'gaussiannb_trained.pth')
            
            if os.path.exists(model_path):
                # Load the trained model
                checkpoint = torch.load(model_path)
                model = GaussianNB(
                    num_features=checkpoint['num_features'],
                    num_classes=checkpoint['num_classes']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Get prediction from BCI connection
                if hasattr(self, 'bcicon'):
                    prediction_response = self.bcicon.bciConnectionController()
                    if prediction_response:
                        return prediction_response.get('prediction_label', 'forward')
                
                # Fallback
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
            else:
                print(f"GaussianNB model not found at {model_path}. Using simulation.")
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
                
        except Exception as e:
            print(f"Error with PyTorch GaussianNB: {e}")
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
    
    def run_gaussiannb_tensorflow(self):
        """ GaussianNB model processing with TensorFlow backend """
        print("Running GaussianNB Model with TensorFlow...")
        try:
            # Note: GaussianNB with TensorFlow is not implemented yet
            # For now, fallback to simulation
            print("TensorFlow backend for GaussianNB not implemented. Using simulation.")
            time.sleep(1)
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
        except Exception as e:
            print(f"Error with TensorFlow GaussianNB: {e}")
            return "forward"
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
        # Execute the current prediction
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)
            self.logMessage.emit(f"Executed action: {self.current_prediction_label}")

            # send it to the drone
            self.doDroneTAction(self.current_prediction_label)
            
    @Slot()
    def connectDrone(self):
        self.doDroneTAction('connect')
        self.logMessage.emit("Connecting to drone...")

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
        # Any new button press cancels a pending auto-return.
        # Directional moves re-arm it once they finish (see moveCompleted).
        self.auto_return_timer.stop()

        # Animate the 3D drone right away (QML buttons and BCI predictions
        # both come through here, so one signal covers both).
        self.drone3DPreview.emit(action)

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
                    try:
                        self.tello.connect(wait_for_state=False)
                        battery = self.tello.get_battery()
                        self.connected = True
                        self.logMessage.emit(f"Connected to Tello Drone (Battery: {battery}%)")
                        self.flight_log.insert(0, f"Drone connected (Battery: {battery}%)")
                    except Exception as e:
                        self.connected = False
                        self.logMessage.emit(f"No drone found - using simulation mode ({e})")
                        self.flight_log.insert(0, "No drone found - simulation mode")
                    self.flightLogUpdated.emit(self.flight_log)
                    return

                # Connection is optional: without a drone, commands run on the simulator
                drone = self._drone()
                sim = "" if self.connected else "[SIM] "

                is_return = self._current_is_return

                def record_action(name, value=None):
                    # Return-home steps are not recorded, so home stays fixed
                    if not is_return:
                        self.action_log.append((name, value))

                # movement actions: distance is normally 30cm,
                # but can be overridden by the clumper
                if action == 'up':
                    dist = self._movement_distance_override or 30
                    drone.move_up(dist)
                    record_action('up', dist)
                    self.logMessage.emit("Moving up")
                    self.flight_log.insert(0, sim + f"Moving up {dist}cm")

                elif action == 'down':
                    dist = self._movement_distance_override or 30
                    drone.move_down(dist)
                    record_action('down', dist)
                    self.logMessage.emit("Moving down")
                    self.flight_log.insert(0, sim + f"Moving down {dist}cm")

                elif action == 'forward':
                    dist = self._movement_distance_override or 30
                    drone.move_forward(dist)
                    record_action('forward', dist)
                    self.logMessage.emit("Moving forward")
                    self.flight_log.insert(0, sim + f"Moving forward {dist}cm")

                elif action == 'backward':
                    dist = self._movement_distance_override or 30
                    drone.move_back(dist)
                    record_action('backward', dist)
                    self.logMessage.emit("Moving backward")
                    self.flight_log.insert(0, sim + f"Moving backward {dist}cm")

                elif action == 'left':
                    dist = self._movement_distance_override or 30
                    drone.move_left(dist)
                    record_action('left', dist)
                    self.logMessage.emit("Moving left")
                    self.flight_log.insert(0, sim + f"Moving left {dist}cm")

                elif action == 'right':
                    dist = self._movement_distance_override or 30
                    drone.move_right(dist)
                    record_action('right', dist)
                    self.logMessage.emit("Moving right")
                    self.flight_log.insert(0, sim + f"Moving right {dist}cm")

                if action in ('up', 'down', 'forward', 'backward', 'left', 'right') and not is_return:
                    # Start the 30 s hold timer (on the main thread)
                    self.moveCompleted.emit()

                if action in ('up', 'down', 'forward', 'backward', 'left', 'right'):
                    pass
                elif action == 'turn_left':
                    drone.rotate_counter_clockwise(45)
                    record_action('turn_left', 45)
                    self.logMessage.emit("Rotating left")
                    self.flight_log.insert(0, sim + f"Rotating left 45°")
					
                elif action == 'turn_right':
                    drone.rotate_clockwise(45)
                    record_action('turn_right', 45)
                    self.logMessage.emit("Rotating right")
                    self.flight_log.insert(0, sim + f"Rotating right 45°")
					
                elif action == 'flip_forward':
                    drone.flip_forward()
                    record_action('flip_forward')
                    self.logMessage.emit("Flipping forward")
                    self.flight_log.insert(0, sim + f"Flipping forward")
					
                elif action == 'flip_back':
                    drone.flip_back()
                    record_action('flip_back')
                    self.logMessage.emit("Flipping backward")
                    self.flight_log.insert(0, sim + f"Flipping backward")
					
                elif action == 'flip_left':
                    drone.flip_left()
                    record_action('flip_left')
                    self.logMessage.emit("Flipping left")
                    self.flight_log.insert(0, sim + f"Flipping left")
					
                elif action == 'flip_right':
                    drone.flip_right()
                    record_action('flip_right')
                    self.logMessage.emit("Flipping right")
                    self.flight_log.insert(0, sim + f"Flipping right")

                elif action == 'takeoff':
                    drone.takeoff()
                    self.flyingStateChanged.emit(True)
                    self.logMessage.emit("Taking off")
                    self.flight_log.insert(0, sim + f"Taking off")
                    self.action_log.clear()
				
                elif action == 'land':
                    drone.land()
                    self.flyingStateChanged.emit(False)
                    self.logMessage.emit("Landing")
                    self.flight_log.insert(0, sim + f"Landing")
                    self.action_log.clear()
					
                elif action == 'go_home':
                    self.go_home(land=False)
                    self.logMessage.emit("Going home")
                    self.flight_log.insert(0, sim + f"Going home")
					
                elif action == 'stream':
                    if hasattr(self, 'camera_controller'):
                        self.camera_controller.start_camera_stream()
                        self.logMessage.emit("Starting camera stream")
                        self.flight_log.insert(0, sim + f"Starting camera stream")
						
                    else:
                        self.logMessage.emit("Camera controller not available")
                        self.flight_log.insert(0, sim + f"Camera controller not available")
						
                else:
                    self.logMessage.emit("Unknown action")
                    self.flight_log.insert(0, sim + f"Unknown action")

                self.flightLogUpdated.emit(self.flight_log)

            except Exception as e:
                error_msg = f"Error during {action}: {str(e)}"
                self.logMessage.emit(error_msg)
                self.flight_log.insert(0, error_msg)
                self.flightLogUpdated.emit(self.flight_log)
                # If critical error, mark as disconnected
                if "Tello" in str(e) or "timeout" in str(e).lower():
                    self.connected = False

    def go_home(self, land=True):
        """Reverse all recorded moves back to home. Lands if land=True, else hovers."""
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
                    self._queue_action(current_dir, dist, is_return=True)
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
                        self._queue_action("turn_right", is_return=True)
                    elif action == "turn_right":
                        self._queue_action("turn_left", is_return=True)
                    elif action == "flip_forward":
                        self._queue_action("flip_back", is_return=True)
                    elif action == "flip_back":
                        self._queue_action("flip_forward", is_return=True)
                    elif action == "flip_left":
                        self._queue_action("flip_right", is_return=True)
                    elif action == "flip_right":
                        self._queue_action("flip_left", is_return=True)
                    # ignore 'takeoff', 'land', 'go_home' themselves, etc.

            # Flush any remaining movement in the last segment
            flush_segment()

            if land:
                # Land when we get "home"
                self._queue_action("land")
                self.flight_log.insert(0, "Go-home path queued (will land)")
            else:
                # Stay in the air; hover keepalive holds the drone in place
                self.flight_log.insert(0, "Go-home path queued (will hover)")
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
        self.logMessage.emit(f"Switching BCI source to: {source}")

        if source == "neurosity":
            self.current_bci_source = "neurosity"
            self.current_data_mode = "live"
            # print ("setBCISource bug")
            try:
                self.ensure_neurosity_processor()
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

    def ensure_neurosity_processor(self):
        """Initialize the Neurosity processor using environment variables."""
        if self.neurosity_processor is not None and self.neurosity_connected:
            return

        # lines below are only used when the email, password, device id is read from file .env
        # email = os.getenv("NEUROSITY_EMAIL")
        # password = os.getenv("NEUROSITY_PASSWORD")
        # device_id = os.getenv("NEUROSITY_DEVICE_ID")

        # missing = [k for k, v in (
        #     ("NEUROSITY_EMAIL", email),
        #     ("NEUROSITY_PASSWORD", password),
        #     ("NEUROSITY_DEVICE_ID", device_id),
        # ) if not v]
        # if missing:
        #     raise RuntimeError(
        #         "Missing Neurosity credentials in .env.txt: " + ", ".join(missing)
        #     )

        # self.neurosity_processor = NeurosityDataProcessor(email, password, device_id, status_callback=self.emitNeurosityStatus)
        
        self.neurosity_processor = NeurosityDataProcessor(
            status_callback=self.emitNeurosityStatus
        )
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
            self.ensure_neurosity_processor()
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
    
    @Slot(result=bool)
    def isNeurosityLoggedIn(self):
        if self.neurosity_processor is None:
            return False
        return self.neurosity_processor.is_logged_in()
    
    @Slot(str, str)
    def neurosityLogin(self, email, password):
        # print(email)
        # print(password)
        self.ensure_neurosity_processor()
        ok = self.neurosity_processor.login(email, password)

        if ok:
            devices = self.neurosity_processor.get_devices()
            if not devices:
                devices = ["No devices paired"]
                self.loginSucceeded.emit(devices)
                self.neurosityStatusChanged.emit("No Device")
                return
            else:
                self.loginSucceeded.emit(devices)
                self.neurosityStatusChanged.emit("offline")
        else:
            self.loginFailed.emit("Invalid email or password.")
    
    @Slot()
    def neurosityLogout(self):
        self.neurosity_processor.logout()
        self.neurosityStatusChanged.emit("Signed out")
    
    @Slot(str)
    def selectNeurosityDevice(self, device):

        state = self.neurosity_processor.select_device(device)
        # print("backend" + state)

        if state == "online":
            self.deviceConnected.emit()
        else:
            self.deviceOffline.emit(
                "Device is offline. Please connect it in the Neurosity app."
            )


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Create our controllers
    tab_controller = TabController()
    print("TabController created")
    manual_nao_controller = ManualNaoController()
    nao_simulation = SimulationController()
    app.aboutToQuit.connect(nao_simulation.close)
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
    engine.rootContext().setContextProperty("naoSimulation", nao_simulation)
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

    exit_code = app.exec()

    engine.clearComponentCache()
    for root in engine.rootObjects():
        root.deleteLater()
    del engine
    sys.exit(exit_code)