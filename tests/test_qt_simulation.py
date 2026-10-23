"""Qt/controller integration checks without launching the hardware-heavy app."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time
import unittest

from PySide6.QtCore import QObject, QMetaObject, QUrl
from PySide6.QtQuick import QQuickView
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from robot_services.qt_simulation import SimulationController
from robot_services.simulation import SimulationRobotService


class QtSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = SimulationController(
            service=SimulationRobotService(posture_duration=.01), telemetry_interval=.02
        )
        self.logs = []
        self.controller.logMessage.connect(self.logs.append)
        self.addCleanup(self.controller.close)

    def wait_for(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        self.fail("Timed out waiting for Qt/controller state")

    def test_connection_monitoring_disconnect_and_reconnect(self):
        self.assertEqual(self.controller.statusText, "Disconnected")
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected and bool(self.controller.telemetry))
        self.assertEqual(self.controller.telemetry["mode"], "simulation")
        self.controller.disconnectSimulation()
        self.wait_for(lambda: not self.controller.connected and not self.controller.pending)
        self.assertEqual(self.controller.telemetry, {})
        QTest.qWait(50)
        self.app.processEvents()
        self.assertEqual(self.controller.telemetry, {})
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected and bool(self.controller.telemetry))

    def test_posture_completion_error_and_busy_recovery(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Stand")
        self.assertTrue(self.controller.busy)
        self.wait_for(lambda: not self.controller.busy)
        self.assertEqual(self.controller.posture, "Stand")
        self.controller.requestPosture("Unsupported")
        self.wait_for(lambda: not self.controller.busy and any("error" in x for x in self.logs))
        self.assertEqual(self.controller.posture, "Stand")

    def test_disconnect_during_posture_drops_old_updates(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Stand")
        self.controller.disconnectSimulation()
        self.wait_for(lambda: not self.controller.pending)
        self.assertFalse(self.controller.connected)
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.posture, "Unknown")

    def test_close_stops_worker_and_is_repeatable(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Stand")
        self.controller.close()
        self.controller.close()
        self.assertFalse(self.controller._thread.is_alive())

    def test_invalid_monitor_interval_rejected_before_starting_worker(self):
        for interval in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                SimulationController(telemetry_interval=interval)

    def test_manual_tab_loads_and_buttons_drive_simulation(self):
        view = QQuickView()
        view.rootContext().setContextProperty("naoSimulation", self.controller)
        view.setResizeMode(QQuickView.SizeRootObjectToView)
        view.resize(1100, 740)
        view.setSource(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ManualNaoControl.qml")))
        self.assertEqual(view.status(), QQuickView.Ready, str(view.errors()))
        view.show()
        root = view.rootObject()
        connect = root.findChild(QObject, "simulationConnectButton")
        stand = root.findChild(QObject, "simulationStandButton")
        sit = root.findChild(QObject, "simulationSitButton")
        self.assertFalse(stand.property("enabled"))
        QTest.qWait(50)
        self.app.processEvents()
        model = root.findChild(QObject, "simulationNaoModel")
        initial_position = model.property("position")
        def visual_child(item, name):
            if item.objectName() == name:
                return item
            for child in item.childItems():
                found = visual_child(child, name)
                if found is not None:
                    return found
            return None
        preview = visual_child(root, "simulationPreview_forward")
        self.assertIsNotNone(preview)
        QMetaObject.invokeMethod(preview, "clicked")
        self.wait_for(lambda: model.property("position") != initial_position)
        self.wait_for(lambda: not root.property("animationInProgress"))
        self.assertNotEqual(model.property("position"), initial_position)
        self.assertFalse(self.controller.connected)
        QMetaObject.invokeMethod(connect, "clicked")
        self.wait_for(lambda: stand.property("enabled") and bool(self.controller.telemetry))
        QMetaObject.invokeMethod(stand, "clicked")
        self.wait_for(lambda: self.controller.posture == "Stand" and not self.controller.busy)
        QMetaObject.invokeMethod(sit, "clicked")
        self.wait_for(lambda: self.controller.posture == "Sit" and not self.controller.busy)
        QMetaObject.invokeMethod(connect, "clicked")
        self.wait_for(lambda: not self.controller.connected and not self.controller.pending)
        self.assertFalse(stand.property("enabled"))
        view.close()
        view.setSource(QUrl())


if __name__ == "__main__":
    unittest.main()
