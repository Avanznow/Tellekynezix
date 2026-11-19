"""Decode a local video through Qt and exercise preview lifecycle and capture."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtCore import QObject, QUrl
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickView
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from robot_services.qt_camera_preview import CameraPreviewController


class CameraPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.controller = CameraPreviewController()
        self.addCleanup(self.controller.stop)
        self.view = QQuickView()
        self.addCleanup(self.view.close)
        self.view.setInitialProperties({"cameraController": self.controller})
        path = Path(__file__).resolve().parents[1] / "NA06_Manual_Control/camera_view/CameraPreview.qml"
        self.view.setSource(QUrl.fromLocalFile(str(path)))
        self.assertEqual(self.view.status(), QQuickView.Ready, str(self.view.errors()))

    def wait_for(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTest.qWait(20)
        self.fail("Preview did not reach expected state: " + self.controller.status)

    def video(self, frame_count=30):
        path = Path(self.folder.name) / "preview.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        try:
            for i in range(frame_count):
                writer.write(np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8))
        finally:
            writer.release()
        return QUrl.fromLocalFile(str(path))

    def test_video_frames_capture_stop_and_restart(self):
        self.assertFalse(self.controller.active)
        self.controller.chooseVideo(self.video())
        self.assertFalse(self.controller.active)
        self.controller.start()
        self.wait_for(lambda: self.controller.hasFrame)
        output = self.view.rootObject().findChild(QObject, "naoPreviewVideo")
        self.app.processEvents()
        self.assertTrue(output.property("visible"))
        self.assertIn("Test video", self.controller.sourceLabel)
        image_path = Path(self.folder.name) / "capture.png"
        self.assertTrue(self.controller.capture(QUrl.fromLocalFile(str(image_path))))
        image = QImage(str(image_path))
        self.assertEqual((image.width(), image.height()), (64, 48))
        self.controller.stop()
        self.assertFalse(self.controller.active)
        self.assertFalse(self.controller.hasFrame)
        self.app.processEvents()
        self.assertFalse(output.property("visible"))
        self.assertFalse(self.controller.capture(QUrl.fromLocalFile(str(image_path))))
        self.controller.start()
        self.wait_for(lambda: self.controller.hasFrame)
        self.view.close()
        self.view.setSource(QUrl())
        self.app.processEvents()
        self.assertFalse(self.controller.active)

    def test_camera_selection_uses_device_id_and_handles_unplug(self):
        laptop, logitech = Mock(), Mock()
        for device, identity, description, default in (
            (laptop, b"laptop", "Laptop camera", True),
            (logitech, b"logitech", "Logitech webcam", False),
        ):
            device.id.return_value = identity
            device.description.return_value = description
            device.isDefault.return_value = default
        self.controller._selected_camera = b""
        with patch("robot_services.qt_camera_preview.QMediaDevices.videoInputs",
                   return_value=[laptop, logitech]) as inputs:
            self.controller._refresh_devices()
            self.assertEqual(self.controller.cameras, ["Laptop camera", "Logitech webcam"])
            self.assertEqual(self.controller.selectedCameraIndex, 0)
            self.controller.chooseCamera(1)
            self.assertEqual(self.controller.sourceLabel, "Webcam: Logitech webcam")
            inputs.return_value = [logitech, laptop]
            self.controller._refresh_devices()
            self.assertEqual(self.controller.selectedCameraIndex, 0)
            # Intercept QCamera creation to verify the chosen device without
            # opening either physical webcam in this automated test.
            self.controller._session = Mock()
            with patch("robot_services.qt_camera_preview.QCamera") as camera:
                self.controller.start()
                camera.assert_called_once_with(logitech, self.controller)
                camera.return_value.start.assert_called_once()
                self.controller.chooseCamera(1)
                self.assertEqual(self.controller.selectedCameraIndex, 0)
                inputs.return_value = [laptop]
                self.controller._refresh_devices()
                self.assertFalse(self.controller.active)
                self.assertEqual(self.controller.selectedCameraIndex, -1)
                self.assertIn("disconnected", self.controller.status)
            self.controller.chooseCamera(0)
            self.assertEqual(self.controller.sourceLabel, "Webcam: Laptop camera")

    def test_video_end_and_invalid_source_recovery(self):
        self.controller.chooseSource("video")
        self.controller.start()
        self.assertFalse(self.controller.active)
        self.assertIn("Choose a test video", self.controller.status)
        self.controller.chooseVideo(self.video(5))
        self.controller.start()
        self.wait_for(lambda: self.controller.hasFrame)
        self.wait_for(lambda: not self.controller.active)
        self.assertIn("ended", self.controller.status)
        self.controller.chooseVideo(QUrl("https://example.com/video.mp4"))
        self.assertIn("existing local video", self.controller.status)


if __name__ == "__main__":
    unittest.main()
