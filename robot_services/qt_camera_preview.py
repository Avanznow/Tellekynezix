"""Local webcam and test-video preview for the Manual NAO camera panel."""

from pathlib import Path
import time

from PySide6.QtCore import QObject, Property, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QMediaPlayer, QVideoFrame


class CameraPreviewController(QObject):
    changed = Signal()
    devicesChanged = Signal()
    logMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = QMediaCaptureSession(self)
        self._player = QMediaPlayer(self)
        self._camera = None
        self._sink = None
        self._kind = "webcam"
        self._file = QUrl()
        self._active = False
        self._status = "Choose a source, then click Start Stream."
        self._image = QImage()
        self._capture_image = QImage()
        self._frames = 0
        self._started = 0.0
        self._fps = 0.0
        self._player.errorOccurred.connect(
            lambda *_: self._error(self._player.errorString()) if self._kind == "video" else None)
        self._player.mediaStatusChanged.connect(self._media_status)
        self._stats = QTimer(self)
        self._stats.setInterval(1000)
        self._stats.timeout.connect(self._update_stats)
        self._startup = QTimer(self)
        self._startup.setSingleShot(True)
        self._startup.setInterval(10000)
        self._startup.timeout.connect(lambda: self._error("No video frames received. Check the source and try again."))
        self._devices = []
        self._selected_camera = b""
        self._media_devices = QMediaDevices(self)
        self._media_devices.videoInputsChanged.connect(self._refresh_devices)
        self._refresh_devices()

    @Property("QStringList", notify=devicesChanged)
    def cameras(self):
        return [device.description() for device in self._devices]

    @Property(int, notify=devicesChanged)
    def selectedCameraIndex(self):
        return next((index for index, device in enumerate(self._devices)
                     if bytes(device.id()) == self._selected_camera), -1)

    @Slot()
    def _refresh_devices(self):
        devices = QMediaDevices.videoInputs()
        if self._selected_camera and not any(bytes(device.id()) == self._selected_camera for device in devices):
            if self._kind == "webcam":
                self.stop()
                self._status = "Selected camera disconnected. Choose another webcam."
            self._selected_camera = b""
        elif not self._selected_camera and devices:
            default = next((device for device in devices if device.isDefault()), devices[0])
            self._selected_camera = bytes(default.id())
        self._devices = devices
        self.devicesChanged.emit()
        self.changed.emit()

    @Slot(int)
    def chooseCamera(self, index):
        if self._active or not 0 <= index < len(self._devices):
            return
        self._selected_camera = bytes(self._devices[index].id())
        self._status = "Camera selected. Click Start Stream."
        self.devicesChanged.emit()
        self.changed.emit()

    @Property(bool, notify=changed)
    def active(self):
        return self._active

    @Property(bool, notify=changed)
    def hasFrame(self):
        return not self._image.isNull()

    @Property(str, notify=changed)
    def status(self):
        return self._status

    @Property(str, notify=changed)
    def sourceLabel(self):
        if self._kind == "webcam":
            index = self.selectedCameraIndex
            return "Webcam: " + self._devices[index].description() if index >= 0 else "No webcam selected"
        return (
            "Test video: " + Path(self._file.toLocalFile()).name if not self._file.isEmpty() else "Choose a test video")

    @Property(str, notify=changed)
    def frameInfo(self):
        if self._image.isNull():
            return ""
        return f"{self._image.width()} × {self._image.height()} · {self._fps:.1f} frames/s"

    @Slot(QObject)
    def attachOutput(self, output):
        self.stop()
        if self._sink is not None:
            self._sink.videoFrameChanged.disconnect(self._frame)
            self._sink.destroyed.disconnect(self._output_lost)
        self._sink = output.property("videoSink")
        if self._sink is not None:
            self._sink.videoFrameChanged.connect(self._frame)
            self._sink.destroyed.connect(self._output_lost)

    @Slot()
    def _output_lost(self):
        self._sink = None
        self.stop()

    @Slot(str)
    def chooseSource(self, kind):
        if kind not in ("webcam", "video"):
            return
        self.stop()
        self._kind = kind
        self._status = "Choose a source, then click Start Stream."
        self.changed.emit()

    @Slot(QUrl)
    def chooseVideo(self, url):
        self.stop()
        if not url.isLocalFile() or not Path(url.toLocalFile()).is_file():
            self._status = "Choose an existing local video file."
        else:
            self._kind = "video"
            self._file = url
            self._status = "Video selected. Click Start Stream."
        self.changed.emit()

    @Slot()
    def start(self):
        if self._active:
            return
        if self._sink is None:
            self._status = "Video display is unavailable."
            self.changed.emit()
            return
        if self._kind == "video" and (self._file.isEmpty() or not Path(self._file.toLocalFile()).is_file()):
            self._status = "Choose a test video first."
            self.changed.emit()
            return
        if self._kind == "webcam":
            index = self.selectedCameraIndex
            if index < 0:
                self._status = "Choose an available local webcam."
                self.changed.emit()
                return
            device = self._devices[index]
            if self._camera is not None:
                self._session.setCamera(None)
                self._camera.deleteLater()
            self._camera = QCamera(device, self)
            camera = self._camera
            camera.errorOccurred.connect(
                lambda *_: self._error(camera.errorString()) if self._camera is camera and self._kind == "webcam" else None)
            self._session.setCamera(self._camera)
            self._session.setVideoSink(self._sink)
        else:
            self._player.setVideoSink(self._sink)
            self._player.setSource(self._file)
        self._active = True
        self._frames, self._started, self._fps = 0, time.monotonic(), 0.0
        self._status = "Starting preview..."
        self._stats.start()
        self._startup.start()
        self.changed.emit()
        self.logMessage.emit("Camera preview starting: " + self.sourceLabel)
        if self._kind == "webcam":
            self._camera.start()
        else:
            self._player.setPosition(0)
            self._player.play()

    def _frame(self, frame):
        if not self._active or not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        first = self._image.isNull()
        self._image = image
        self._frames += 1
        self._startup.stop()
        self._status = "Preview running"
        if first:
            self.changed.emit()

    def _update_stats(self):
        elapsed = time.monotonic() - self._started
        self._fps = self._frames / elapsed if elapsed > 0 else 0.0
        self.changed.emit()

    def _media_status(self, status):
        if self._active and status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.stop()
            self._status = "Test video ended. Click Start Stream to replay."
            self.changed.emit()

    def _error(self, message):
        if not self._active:
            return
        self.stop()
        self._status = message or "Camera preview failed."
        self.logMessage.emit("Camera preview error: " + self._status)
        self.changed.emit()

    @Slot()
    def stop(self):
        was_active = self._active
        self._active = False
        self._stats.stop()
        self._startup.stop()
        self._player.stop()
        self._player.setSource(QUrl())
        self._player.setVideoSink(None)
        if self._camera is not None:
            self._camera.stop()
        self._session.setVideoSink(None)
        self._image = QImage()
        self._fps = 0.0
        if self._sink is not None:
            self._sink.setVideoFrame(QVideoFrame())
        self._status = "Preview stopped."
        self.changed.emit()
        if was_active:
            self.logMessage.emit("Camera preview stopped.")

    @Slot(result=bool)
    def prepareCapture(self):
        if not self._active or self._image.isNull():
            return False
        self._capture_image = self._image.copy()
        return True

    @Slot()
    def cancelCapture(self):
        self._capture_image = QImage()

    @Slot(QUrl, result=bool)
    def capture(self, url):
        image = self._capture_image if not self._capture_image.isNull() else (
            self._image if self._active else QImage())
        if image.isNull() or not url.isLocalFile():
            self._status = "No preview frame available to save."
            self.changed.emit()
            return False
        saved = image.save(url.toLocalFile(), "PNG")
        self._capture_image = QImage()
        self._status = "Frame saved." if saved else "Could not save the frame."
        self.logMessage.emit(self._status)
        self.changed.emit()
        return saved
