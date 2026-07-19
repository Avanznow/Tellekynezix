import sys
import os
import socket
import threading
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

from PySide6.QtCore import (
    QObject, QUrl, Qt, Signal, QSize, QTimer, QThread
)
from PySide6.QtGui import QColor, QVector3D, QPixmap, QImage, QQuaternion
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QGridLayout, QSplitter, QSizePolicy,
    QSpacerItem, QTextEdit, QPushButton
)
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender

# OpenCV for Tello video 
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent


def resource_path(*parts) -> str:
    return str(BASE_DIR.joinpath(*parts))


# =================================================
# ==============  MTL-AWARE LOADING  ==============
# =================================================

@dataclass
class MtlData:
    name: str = "default"
    Kd: Optional[QColor] = None    # Diffuse color
    Ks: Optional[QColor] = None    # Specular color
    Ns: Optional[float] = None     # Shininess (spec exponent)
    map_Kd: Optional[Path] = None  # Diffuse texture map


def _to_qcolor_from_floats(rgb):
    r, g, b = rgb
    return QColor(
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255)
    )


def parse_mtl(mtl_path: Path) -> Dict[str, MtlData]:
    """Parse a simple subset of MTL: newmtl, Kd, Ks, Ns, map_Kd."""
    materials: Dict[str, MtlData] = {}
    if not mtl_path.exists():
        return materials

    current: Optional[MtlData] = None
    try:
        for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "newmtl" and len(parts) > 1:
                name = " ".join(parts[1:])
                current = MtlData(name=name)
                materials[name] = current

            elif current is not None:
                if tag == "Kd" and len(parts) >= 4:
                    current.Kd = _to_qcolor_from_floats([
                        float(parts[1]), float(parts[2]), float(parts[3])
                    ])
                elif tag == "Ks" and len(parts) >= 4:
                    current.Ks = _to_qcolor_from_floats([
                        float(parts[1]), float(parts[2]), float(parts[3])
                    ])
                elif tag == "Ns" and len(parts) >= 2:
                    try:
                        current.Ns = float(parts[1])
                    except ValueError:
                        current.Ns = None
                elif tag == "map_Kd" and len(parts) >= 2:
                    tex_rel = " ".join(parts[1:])
                    current.map_Kd = (mtl_path.parent / tex_rel).resolve()
    except Exception as e:
        print(f"[WARN] Failed to parse MTL {mtl_path}: {e}")

    return materials


def _read_obj_mtllib(obj_path: Path) -> Optional[Path]:
    """Find mtllib line in OBJ to locate sibling MTL file."""
    try:
        for raw in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if line.lower().startswith("mtllib"):
                mtl_name = line.split(maxsplit=1)[1].strip()
                return (obj_path.parent / mtl_name).resolve()
    except Exception as e:
        print(f"[WARN] Could not read OBJ for mtllib: {e}")
    return None


def apply_mtl_fallback_material(entity: Qt3DCore.QEntity, obj_file: Path):
    """
    If OBJ has MTL, apply a representative material:
      - If map_Kd exists -> QDiffuseMapMaterial with texture
      - else -> QDiffuseSpecularMaterial from Kd/Ks/Ns
    Otherwise keep existing materials.
    """
    mtl_file = _read_obj_mtllib(obj_file)
    if not mtl_file or not mtl_file.exists():
        print("[INFO] No MTL found or not accessible; keeping existing material.")
        return

    mats = parse_mtl(mtl_file)
    if not mats:
        print("[INFO] MTL parsed empty; keeping existing material.")
        return

    first = next(iter(mats.values()))

    # Remove existing materials on entity
    to_remove = []
    for comp in entity.components():
        if isinstance(comp, Qt3DRender.QMaterial):
            to_remove.append(comp)
    for comp in to_remove:
        entity.removeComponent(comp)

    if first.map_Kd and first.map_Kd.exists():
        tex_loader = Qt3DRender.QTextureLoader(entity)
        tex_loader.setSource(QUrl.fromLocalFile(str(first.map_Kd)))
        tex_mat = Qt3DExtras.QDiffuseMapMaterial(entity)
        tex_mat.setDiffuse(tex_loader)
        entity.addComponent(tex_mat)
        print(f"[INFO] Applied texture from map_Kd: {first.map_Kd}")
        return

    ds_mat = Qt3DExtras.QDiffuseSpecularMaterial(entity)
    if first.Kd:
        ds_mat.setDiffuse(first.Kd)
    if first.Ks:
        ds_mat.setSpecular(first.Ks)
    if first.Ns is not None:
        ds_mat.setShininess(max(1.0, min(128.0, first.Ns * 0.1)))

    entity.addComponent(ds_mat)
    print("[INFO] Applied color/specular from MTL (no map_Kd).")


def build_mesh_entity(parent: Qt3DCore.QEntity, model_path: str) -> Qt3DCore.QEntity:
    """
    Build a mesh entity under `parent` using QMesh for `model_path`.
    Tries to apply MTL fallback if it's an OBJ.
    Returns the newly created entity.
    """
    ent = Qt3DCore.QEntity(parent)
    mesh = Qt3DRender.QMesh(ent)
    mesh.setSource(QUrl.fromLocalFile(str(Path(model_path).resolve())))

    xform = Qt3DCore.QTransform(ent)
    ent.addComponent(mesh)
    ent.addComponent(xform)

    default_phong = Qt3DExtras.QPhongMaterial(ent)
    default_phong.setDiffuse(QColor(190, 190, 190))
    default_phong.setAmbient(QColor(70, 70, 70))
    default_phong.setSpecular(QColor(255, 255, 255))
    default_phong.setShininess(80.0)
    ent.addComponent(default_phong)

    p = Path(model_path)
    if p.suffix.lower() == ".obj":
        apply_mtl_fallback_material(ent, p)

    return ent


def get_transform(entity: Qt3DCore.QEntity) -> Optional[Qt3DCore.QTransform]:
    """Helper: return the first QTransform component of an entity."""
    for comp in entity.components():
        if isinstance(comp, Qt3DCore.QTransform):
            return comp
    return None


# =================================================
# =============== Tello UDP Controller ============
# =================================================

class TelloController(QObject):
    status = Signal(str)
    attitudeChanged = Signal(float, float, float)  # pitch, roll, yaw

    def __init__(self, host="192.168.10.1", cmd_port=8889, local_port=9000, state_port=8890, parent=None):
        super().__init__(parent)
        self.host = host
        self.cmd_port = cmd_port
        self.local_port = local_port
        self.state_port = state_port

        self.sock = None
        self.state_sock = None
        self._connected = False
        self._state_running = False
        self._state_thread = None
        self._cmd_lock = threading.Lock()

    def connect_to_tello(self):
        """Bind local UDP socket, enter SDK mode, and start state listener."""
        if self._connected:
            self.status.emit("[Tello] Already connected.")
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", self.local_port))
            self.sock.settimeout(2.0)
            self._connected = True
            self.status.emit(f"[Tello] Local UDP bound to :{self.local_port}")
        except Exception as e:
            self.status.emit(f"[Tello] Socket bind error: {e}")
            self._connected = False
            return

        # Enter SDK mode first
        self._send_async("command")

        # Start state listener
        self._start_state_listener()

    def is_connected(self):
        return self._connected

    def _send(self, cmd: str):
        if not self._connected or self.sock is None:
            self.status.emit("[Tello] Not connected (Wi‑Fi?). Command skipped: " + cmd)
            return

        with self._cmd_lock:
            try:
                self.sock.sendto(cmd.encode("utf-8"), (self.host, self.cmd_port))
                self.status.emit(f"[Tello →] {cmd}")
                try:
                    resp, _ = self.sock.recvfrom(1024)
                    self.status.emit(f"[Tello ←] {resp.decode('utf-8', errors='ignore').strip()}")
                except socket.timeout:
                    self.status.emit("[Tello] (no response)")
            except OSError as e:
                self.status.emit(f"[Tello] Send error: {e}")

    def _send_async(self, cmd: str):
        t = threading.Thread(target=self._send, args=(cmd,), daemon=True)
        t.start()

    def _start_state_listener(self):
        if self._state_running:
            return

        try:
            self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.state_sock.bind(("", self.state_port))
            self.state_sock.settimeout(1.0)
            self._state_running = True
            self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
            self._state_thread.start()
            self.status.emit(f"[Tello] State listener bound to :{self.state_port}")
        except Exception as e:
            self.status.emit(f"[Tello] State listener error: {e}")
            self._state_running = False

    def _parse_state_packet(self, packet: str):
        """
        Example packet:
        pitch:0;roll:0;yaw:87;vgx:0;vgy:0;vgz:0;templ:64;temph:67;tof:10;h:0;bat:87;baro:12.34;time:0;agx:...;agy:...;agz:...;
        """
        result = {}
        for item in packet.strip().split(";"):
            if not item or ":" not in item:
                continue
            k, v = item.split(":", 1)
            result[k.strip()] = v.strip()
        return result

    def _state_loop(self):
        while self._state_running:
            try:
                data, _ = self.state_sock.recvfrom(2048)
                text = data.decode("utf-8", errors="ignore").strip()
                state = self._parse_state_packet(text)

                pitch = float(state.get("pitch", 0))
                roll  = float(state.get("roll", 0))
                yaw   = float(state.get("yaw", 0))

                self.attitudeChanged.emit(pitch, roll, yaw)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                self.status.emit(f"[Tello] State parse error: {e}")

    # Convenience methods
    def takeoff(self): self._send_async("takeoff")
    def land(self): self._send_async("land")
    def forward(self, cm=50): self._send_async(f"forward {int(cm)}")
    def back(self, cm=50): self._send_async(f"back {int(cm)}")
    def left(self, cm=50): self._send_async(f"left {int(cm)}")
    def right(self, cm=50): self._send_async(f"right {int(cm)}")
    def cw(self, deg=90): self._send_async(f"cw {int(deg)}")
    def ccw(self, deg=90): self._send_async(f"ccw {int(deg)}")
    def up(self, cm=50): self._send_async(f"up {int(cm)}")
    def down(self, cm=50): self._send_async(f"down {int(cm)}")

    # Flip commands
    def flip_left(self):    self._send_async("flip l")
    def flip_right(self):   self._send_async("flip r")
    def flip_forward(self): self._send_async("flip f")
    def flip_back(self):    self._send_async("flip b")

    # Video stream control
    def streamon(self): self._send_async("streamon")
    def streamoff(self): self._send_async("streamoff")

    # "Go Home" helper (Tello lacks GPS/RTH)
    def go_home(self):
        self.status.emit("[Tello] 'Go Home' not supported on Tello (no GPS). Executing safe landing.")
        self.land()

    def close(self):
        self._connected = False
        self._state_running = False

        if self.state_sock:
            try:
                self.state_sock.close()
            except Exception:
                pass
            self.state_sock = None

        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# =================================================
# =============== 3D Viewer Widget ================
# =================================================

class ObjectViewer(QWidget):
    def __init__(self, logger_callback=None):
        super().__init__()
        self.logger = logger_callback or (lambda s: None)
        self.setWindowTitle("Manual Drone Controller")
        self.resize(800, 600)

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # 3D Window
        self.view = Qt3DExtras.Qt3DWindow()
        self.view.defaultFrameGraph().setClearColor(QColor(135, 206, 235))
        self.container = QWidget.createWindowContainer(self.view, self)
        main_layout.addWidget(self.container, 1)

        # Root entity
        self.rootEntity = Qt3DCore.QEntity()

        # Camera
        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16 / 9, 0.1, 1400)
        self.camera.setPosition(QVector3D(10, 0, 30))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

        # Simple orbit controller
        self.orbit = Qt3DExtras.QOrbitCameraController(self.rootEntity)
        self.orbit.setCamera(self.camera)
        self.orbit.setLinearSpeed(50.0)
        self.orbit.setLookSpeed(180.0)

        # Lighting rig
        keyLightEntity = Qt3DCore.QEntity(self.rootEntity)
        keyLight = Qt3DRender.QDirectionalLight(keyLightEntity)
        keyLight.setColor(QColor(255, 255, 255))
        keyLight.setIntensity(1.2)
        keyLight.setWorldDirection(QVector3D(-0.6, -1.0, -0.5).normalized())
        keyLightEntity.addComponent(keyLight)

        fillLightEntity = Qt3DCore.QEntity(self.rootEntity)
        fillLight = Qt3DRender.QPointLight(fillLightEntity)
        fillLight.setColor(QColor(220, 235, 255))
        fillLight.setIntensity(0.4)
        fillLightXf = Qt3DCore.QTransform()
        fillLightXf.setTranslation(QVector3D(-20.0, 10.0, -10.0))
        fillLightEntity.addComponent(fillLight)
        fillLightEntity.addComponent(fillLightXf)

        self.headLightEntity = Qt3DCore.QEntity(self.rootEntity)
        self.headLight = Qt3DRender.QPointLight(self.headLightEntity)
        self.headLight.setColor("white")
        self.headLight.setIntensity(0.25)
        self.headLightXf = Qt3DCore.QTransform()
        self.headLightEntity.addComponent(self.headLight)
        self.headLightEntity.addComponent(self.headLightXf)

        def _sync_headlight():
            self.headLightXf.setTranslation(self.camera.position())

        self._headlight_timer = QTimer(self)
        self._headlight_timer.timeout.connect(_sync_headlight)
        self._headlight_timer.start(16)

        # Drone group: all visible drone geometry lives under this parent
        # so one transform can orient the whole model from live telemetry.
        self.droneGroup = Qt3DCore.QEntity(self.rootEntity)
        self.droneGroupTransform = Qt3DCore.QTransform(self.droneGroup)
        self.droneGroup.addComponent(self.droneGroupTransform)

        # Base model alignment (adjust if needed for your mesh)
        self.baseModelRotation = QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), -70)
        self.droneGroupTransform.setRotation(self.baseModelRotation)

        # Main drone model
        default_model_path = resource_path("Manual_Drone_Controller", "drone.obj")
        if os.path.exists(default_model_path):
            self.drone = build_mesh_entity(self.droneGroup, default_model_path)
        else:
            self.drone = Qt3DCore.QEntity(self.droneGroup)
            cube = Qt3DExtras.QCuboidMesh(self.drone)
            xf = Qt3DCore.QTransform(self.drone)
            mat = Qt3DExtras.QPhongMaterial(self.drone)
            mat.setDiffuse(QColor(190, 190, 190))
            mat.setAmbient(QColor(70, 70, 70))
            mat.setSpecular(QColor(255, 255, 255))
            mat.setShininess(80.0)
            self.drone.addComponent(cube)
            self.drone.addComponent(xf)
            self.drone.addComponent(mat)

        # Optional frame sequence under the same droneGroup
        self.frame_entities: List[Qt3DCore.QEntity] = []
        frames_folder = resource_path("frames")
        if os.path.isdir(frames_folder):
            obj_files = sorted(
                [
                    os.path.join(frames_folder, f)
                    for f in os.listdir(frames_folder)
                    if f.lower().endswith(".obj")
                ]
            )
            for path in obj_files:
                ent = build_mesh_entity(self.droneGroup, path)
                ent.setEnabled(False)
                self.frame_entities.append(ent)
            if self.frame_entities:
                self.frame_entities[0].setEnabled(True)

        self.view.setRootEntity(self.rootEntity)

        # Frame animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_obj_frame)
        self.current_frame_index = 0

    def log(self, msg):
        self.logger(msg)

    def zoom_out(self):
        cam_pos = self.camera.position()
        view_center = self.camera.viewCenter()
        direction = cam_pos - view_center
        new_pos = cam_pos + direction.normalized() * 5.0
        self.camera.setPosition(new_pos)

    def zoom_in(self):
        cam_pos = self.camera.position()
        view_center = self.camera.viewCenter()
        direction = cam_pos - view_center
        zoom_step = 5.0
        new_pos = cam_pos - direction.normalized() * zoom_step
        self.camera.setPosition(new_pos)

    def play_obj_animation(self):
        if not self.frame_entities:
            self.log("No .obj frames found.")
            return
        self.current_frame_index = 0
        self.animation_timer.start(33)  # ~30 FPS

    def stop_obj_animation(self):
        if not self.frame_entities:
            self.log("No .obj frames found.")
            return
        self.animation_timer.stop()

    def next_obj_frame(self):
        if not self.frame_entities:
            return
        self.frame_entities[self.current_frame_index].setEnabled(False)
        self.current_frame_index = (self.current_frame_index + 1) % len(self.frame_entities)
        self.frame_entities[self.current_frame_index].setEnabled(True)

    def set_drone_attitude(self, pitch: float, roll: float, yaw: float):
        """
        Apply live Tello attitude to the 3D model.

        NOTE:
        Depending on the OBJ's local axis conventions, you may need to flip
        a sign below (e.g. -roll or -yaw) or swap one axis.
        """
        if self.droneGroupTransform is None:
            return

        # Quaternion composition from telemetry
        # These axis choices are a practical starting point for many drone meshes.
        q_pitch = QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), pitch)
        q_roll  = QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), roll)
        q_yaw   = QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), yaw)

        target = self.baseModelRotation * (q_yaw * q_pitch * q_roll)

        # Light smoothing to reduce jitter
        current = self.droneGroupTransform.rotation()
        smoothed = QQuaternion.slerp(current, target, 0.2)
        self.droneGroupTransform.setRotation(smoothed)


# =================================================
# ========== Tello Video: worker + widget =========
# =================================================

class _VideoThread(QThread):
    frame = Signal(QImage)
    status = Signal(str)

    def __init__(self, url='udp://0.0.0.0:11111', parent=None):
        super().__init__(parent)
        self.url = url
        self._running = False
        self._cap = None

    def run(self):
        if not _CV2_AVAILABLE:
            self.status.emit("OpenCV not installed. Install with: pip install opencv-python")
            return

        self._running = True

        # Try to open with FFMPEG backend first
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.url)

        if not self._cap or not self._cap.isOpened():
            self.status.emit("Video: failed to open UDP stream on 11111")
            return

        self.status.emit("Video: stream opened")

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.msleep(5)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            # Copy to detach from numpy memory
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame.emit(qimg)

        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass

        self.status.emit("Video: stream closed")

    def stop(self):
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass


class TelloVideoWidget(QWidget):
    """QLabel-based viewer that shows frames from the Tello UDP stream, with start/stop/capture controls."""
    def __init__(self, tello: TelloController, logger_callback=None, parent=None):
        super().__init__(parent)
        self.tello = tello
        self.logger = logger_callback or (lambda s: None)

        self._thread = _VideoThread()
        self._thread.frame.connect(self._on_frame)
        self._thread.status.connect(self._on_status)

        self._streaming = False
        self._last_frame = QImage()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ---- Header ----
        title = QLabel("CAMERA VIEW")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-weight: 700;
            padding: 6px;
            color: white;
            background: #1c2733;
            border-radius: 6px;
        """)
        layout.addWidget(title)

        # ---- Video display ----
        self.label = QLabel("No video")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            background-color: #000;
            color: #fff;
            border-radius: 6px;
        """)
        self.label.setMinimumHeight(0)
        layout.addWidget(self.label, 1)

        # ---- Button row (now at bottom) ----
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(6, 0, 6, 6)
        btn_row.setSpacing(8)

        self.start_btn = QPushButton("Start Stream")
        self.stop_btn = QPushButton("Stop Stream")
        self.capture_btn = QPushButton("Capture")

        button_style = """
            QPushButton {
                background-color: #223142;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2c4157;
            }
            QPushButton:disabled {
                background-color: #4a5661;
                color: #cfd8dc;
            }
        """

        self.start_btn.setStyleSheet(button_style)
        self.stop_btn.setStyleSheet(button_style)
        self.capture_btn.setStyleSheet(button_style)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.capture_btn.clicked.connect(self.capture_frame)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.capture_btn)

        layout.addLayout(btn_row)

        self._set_streaming_ui(False)

        if not _CV2_AVAILABLE:
            self._on_status("OpenCV not installed. Install with: pip install opencv-python")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)

    def is_streaming(self) -> bool:
        return self._streaming

    def _set_streaming_ui(self, streaming: bool):
        self._streaming = streaming
        self.start_btn.setEnabled(not streaming)
        self.stop_btn.setEnabled(streaming)
        self.capture_btn.setEnabled(not self._last_frame.isNull())

    def _on_frame(self, qimg: QImage):
        self._last_frame = qimg.copy()

        pix = QPixmap.fromImage(qimg)
        self.label.setPixmap(
            pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        if not self.capture_btn.isEnabled():
            self.capture_btn.setEnabled(True)

    def resizeEvent(self, event):
        if self.label.pixmap():
            self.label.setPixmap(
                self.label.pixmap().scaled(
                    self.label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        super().resizeEvent(event)

    def _on_status(self, msg: str):
        self.logger(msg)

    def _start_thread(self):
        if not self._thread.isRunning():
            self._thread.start()
        self._set_streaming_ui(True)

    def start(self):
        if not self.tello.is_connected():
            self.logger("[Video] Connect to Tello first.")
            return

        if self._streaming:
            self.logger("[Video] Stream already running.")
            return

        self.logger("[Video] Starting stream...")
        self.tello.streamon()

        # Slight delay helps the Tello start sending frames before OpenCV opens the stream
        QTimer.singleShot(500, self._start_thread)

    def stop(self):
        if not self._streaming and not self._thread.isRunning():
            self.logger("[Video] Stream already stopped.")
            return

        self.logger("[Video] Stopping stream...")

        if self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(1000)

        self.tello.streamoff()
        self._set_streaming_ui(False)

    def capture_frame(self):
        if self._last_frame.isNull():
            self.logger("[Video] No frame available to capture.")
            return

        captures_dir = BASE_DIR / "captures"
        captures_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = captures_dir / f"tello_capture_{ts}.png"

        ok = self._last_frame.save(str(out_file), "PNG")
        if ok:
            self.logger(f"[Video] Capture saved: {out_file}")
        else:
            self.logger("[Video] Failed to save capture.")

# =================================================
# ======= UI wrapper with Tello bindings ==========
# =================================================

class DroneControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Drone Controller")
        self.resize(1200, 700)

        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # =========================
        # Left panel (controls)
        # =========================
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # -------------------- COLORS / SIZES --------------------
        PANEL_BG = "#607d8b"
        CARD_BG = "#1c2733"
        CARD_BG_HOVER = "#223142"
        ACCENT = "#ffffff"
        TEXT_MAIN = "#ffffff"

        TILE_W = 140
        TILE_H = 110
        GRID_SPACING = 12
        BIG_W = TILE_W * 3 + GRID_SPACING * 2
        BIG_H = 140

        left_panel.setStyleSheet(f"background-color: {PANEL_BG};")

        # ---- Header ----
        header = QLabel("Drone Controls")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(
            "font-size: 18px; font-weight: 600; margin: 4px; color: white;"
        )
        left_layout.addWidget(header)

        # -------------------- Button factory --------------------
        def make_button(label: str, on_click, icon_path: str = "", w=TILE_W, h=TILE_H, icon_scale=0.55):
            btn = QFrame()
            btn.setStyleSheet(f"""
                QFrame {{
                    background-color: {CARD_BG};
                    border-radius: 8px;
                }}
                QFrame:hover {{
                    background-color: {CARD_BG_HOVER};
                }}
                QLabel {{
                    color: {ACCENT};
                }}
            """)
            btn.setFixedSize(QSize(w, h))

            v = QVBoxLayout(btn)
            v.setContentsMargins(8, 8, 8, 8)
            v.setSpacing(6)

            if icon_path and os.path.exists(icon_path):
                icon_lbl = QLabel()
                icon_lbl.setAlignment(Qt.AlignCenter)
                icon_w = int(w * icon_scale)
                icon_h = int(h * icon_scale * 0.7)
                icon_lbl.setPixmap(
                    QPixmap(icon_path).scaled(
                        icon_w,
                        icon_h,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
                v.addWidget(icon_lbl)
            else:
                v.addSpacerItem(QSpacerItem(1, 6))

            lab = QLabel(label)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
            v.addWidget(lab)
            v.addStretch(1)

            def _mousePressEvent(_):
                on_click()

            btn.mousePressEvent = _mousePressEvent
            return btn

        def make_big_button(label: str, on_click, icon_path: str = "", h=BIG_H):
            return make_button(label, on_click, icon_path, w=BIG_W, h=h, icon_scale=0.5)

        # ---- Flight log card ----
        log_card = QFrame()
        log_card.setMinimumHeight(170)
        log_card.setStyleSheet("""
            QFrame {
                background: #f0f2f5;
                border-radius: 8px;
            }
        """)
        log_v = QVBoxLayout(log_card)
        log_v.setContentsMargins(8, 8, 8, 8)
        log_v.setSpacing(6)

        log_label = QLabel("FLIGHT LOG")
        log_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        log_label.setStyleSheet("font-weight: 700; color: #37474f;")
        log_v.addWidget(log_label)

        self.text_field = QTextEdit()
        self.text_field.setReadOnly(True)
        self.text_field.setStyleSheet(
            "background-color: white; padding: 6px; border-radius: 6px; color: black;"
        )
        self.text_field.setMinimumHeight(120)
        self.text_field.setPlaceholderText("Flight log...")
        log_v.addWidget(self.text_field)

        def log_fn(msg: str):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.text_field.append(f"{ts} {msg}")

        # =========================
        # Tello controller
        # =========================
        self.tello = TelloController()
        self.tello.status.connect(log_fn)

        # Right-side widgets created early so callbacks can reference them
        self.viewer = ObjectViewer(logger_callback=log_fn)
        self.camera_widget = TelloVideoWidget(self.tello, logger_callback=log_fn)

        # Tie live drone attitude to the 3D model
        self.tello.attitudeChanged.connect(self.viewer.set_drone_attitude)

        # =========================
        # Log Callbacks
        # =========================
        def do_home():
            log_fn("Home pressed")
            self.tello.go_home()

        def do_up():
            log_fn("Up 50 cm")
            self.tello.up(50)

        def do_down():
            log_fn("Down 50 cm")
            self.tello.down(50)

        def do_forward():
            log_fn("Forward 50 cm")
            self.tello.forward(50)

        def do_backward():
            log_fn("Backward 50 cm")
            self.tello.back(50)

        def do_left():
            log_fn("Left 50 cm")
            self.tello.left(50)

        def do_right():
            log_fn("Right 50 cm")
            self.tello.right(50)

        def do_turn_left():
            log_fn("Turn Left 90°")
            self.tello.ccw(90)

        def do_turn_right():
            log_fn("Turn Right 90°")
            self.tello.cw(90)

        def do_connect():
            log_fn("Connect")
            threading.Thread(target=self.tello.connect_to_tello, daemon=True).start()

        def do_takeoff():
            log_fn("Takeoff")
            if self.viewer:
                self.viewer.play_obj_animation()
            self.tello.takeoff()

        def do_land():
            log_fn("Land")
            if self.viewer:
                self.viewer.stop_obj_animation()
            self.tello.land()

        def do_stream():
            if not hasattr(self, "camera_widget"):
                log_fn("Stream: camera widget not initialized")
                return

            if not self.camera_widget.is_streaming():
                log_fn("Stream: start")
                self.camera_widget.start()
            else:
                log_fn("Stream: stop")
                self.camera_widget.stop()

        def do_flip_left():
            log_fn("Flip Left")
            self.tello.flip_left()

        def do_flip_right():
            log_fn("Flip Right")
            self.tello.flip_right()

        def do_flip_forward():
            log_fn("Flip Forward")
            self.tello.flip_forward()

        def do_flip_back():
            log_fn("Flip Back")
            self.tello.flip_back()

        # ---- ICON PATHS ----
        ICONS = {
            "connect":  resource_path("images", "connect.svg"),
            "takeoff":  resource_path("images", "Drone Take off.svg"),
            "land":     resource_path("images", "Drone Landing.svg"),
            "home":     resource_path("images", "home.svg"),
            "up":       resource_path("images", "climb.svg"),
            "down":     resource_path("images", "Descend.svg"),
            "forward":  resource_path("images", "forward arrow.svg"),
            "back":     resource_path("images", "back arrow.svg"),
            "left":     resource_path("images", "left arrow.svg"),
            "right":    resource_path("images", "right arrow.svg"),
            "turn_l":   resource_path("images", "Yaw left.svg"),
            "turn_r":   resource_path("images", "Yaw right.svg"),
            "flip_l":   resource_path("images", "Flip Left.svg"),
            "flip_r":   resource_path("images", "Flip Right.svg"),
            "flip_f":   resource_path("images", "Forward Flip.svg"),
            "flip_b":   resource_path("images", "Backward Flip.svg"),
            "stream":   resource_path("images", "stream.svg"),
        }


        # =========================
        # Controls grid
        # =========================
        grid = QGridLayout()
        grid.setHorizontalSpacing(GRID_SPACING)
        grid.setVerticalSpacing(GRID_SPACING)
        grid.setContentsMargins(0, 0, 0, 0)

        def add(btn_or_item, r, c, rs=1, cs=1):
            if isinstance(btn_or_item, QSpacerItem):
                grid.addItem(btn_or_item, r, c, rs, cs)
            else:
                grid.addWidget(btn_or_item, r, c, rs, cs)

        # Row 0
        add(make_button("Home", do_home, ICONS.get("home"), TILE_W, TILE_H), 0, 0)
        add(make_big_button("Up", do_up, ICONS.get("up"), BIG_H), 0, 1, 1, 3)
        add(make_button("Connect", do_connect, ICONS.get("connect"), TILE_W, TILE_H), 0, 4)

        # Row 1
        add(make_button("Flip Forward", do_flip_forward, ICONS.get("flip_f"), TILE_W, TILE_H), 1, 0)
        add(make_big_button("Forward", do_forward, ICONS.get("forward"), BIG_H), 1, 1, 1, 3)
        add(make_button("Flip Right", do_flip_right, ICONS.get("flip_r"), TILE_W, TILE_H), 1, 4)

        # Row 2
        add(make_button("Turn Left", do_turn_left, ICONS.get("turn_l"), TILE_W, TILE_H), 2, 0)
        add(make_button("Left", do_left, ICONS.get("left"), TILE_W, TILE_H), 2, 1)
        add(make_button("Stream", do_stream, ICONS.get("stream"), TILE_W, TILE_H), 2, 2)
        add(make_button("Right", do_right, ICONS.get("right"), TILE_W, TILE_H), 2, 3)
        add(make_button("Turn Right", do_turn_right, ICONS.get("turn_r"), TILE_W, TILE_H), 2, 4)

        # Row 3
        add(make_button("Flip Back", do_flip_back, ICONS.get("flip_b"), TILE_W, TILE_H), 3, 0)
        add(make_big_button("Back", do_backward, ICONS.get("back"), BIG_H), 3, 1, 1, 3)
        add(make_button("Flip Left", do_flip_left, ICONS.get("flip_l"), TILE_W, TILE_H), 3, 4)

        # Row 4
        add(make_button("Takeoff", do_takeoff, ICONS.get("takeoff"), TILE_W, TILE_H), 4, 0)
        add(make_big_button("Down", do_down, ICONS.get("down"), BIG_H), 4, 1, 1, 3)
        add(make_button("Land", do_land, ICONS.get("land"), TILE_W, TILE_H), 4, 4)

        # Stretch middle columns a bit if needed
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(4, 0)

        left_layout.addLayout(grid)
        # Push flight log toward the bottom of the left panel
        left_layout.addStretch(1)
        left_layout.addWidget(log_card)

        # =========================
        # Right panel
        # =========================
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.camera_widget)

        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        QTimer.singleShot(0, lambda: splitter.setSizes([320, 420]))

        right_layout.addWidget(splitter)

        # =========================
        # Main left/right splitter
        # =========================
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)

        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)

        QTimer.singleShot(0, lambda: self.main_splitter.setSizes([620, 500]))

        main_layout.addWidget(self.main_splitter)


    def closeEvent(self, event):
        try:
            if hasattr(self, "camera_widget"):
                self.camera_widget.stop()
        except Exception:
            pass

        try:
            if hasattr(self, "tello"):
                self.tello.close()
        except Exception:
            pass

        super().closeEvent(event)


# =================================================
# ================= App entrypoint ================
# =================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { background-color: #1c2733; }")
    win = DroneControlWindow()
    win.show()
    sys.exit(app.exec())
