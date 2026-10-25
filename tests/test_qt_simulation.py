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
        hardware = root.findChild(QObject, "naoHardwareConnectButton")
        self.assertFalse(hardware.property("enabled"))
        self.assertEqual(root.findChild(QObject, "naoHardwareIp").property("text"), "192.168.23.53")
        self.assertEqual(root.findChild(QObject, "naoHardwarePort").property("text"), "9559")
        camera = root.findChild(QObject, "naoCameraPanel")
        self.assertIsNotNone(camera)
        self.assertIsNone(camera.property("cameraController"))
        for name in ("naoCameraStartButton", "naoCameraStopButton", "naoCameraCaptureButton"):
            self.assertFalse(camera.findChild(QObject, name).property("enabled"))
        tabs = root.findChild(QObject, "naoInformationTabs")
        self.assertEqual(tabs.property("currentIndex"), 0)
        tabs.setProperty("currentIndex", 1)
        self.app.processEvents()
        self.assertFalse(camera.property("visible"))
        tabs.setProperty("currentIndex", 0)
        self.app.processEvents()
        self.assertTrue(camera.property("visible"))
        connect = root.findChild(QObject, "simulationConnectButton")
        stand = root.findChild(QObject, "simulationStandButton")
        sit = root.findChild(QObject, "simulationSitButton")
        self.assertFalse(stand.property("enabled"))
        QTest.qWait(50)
        self.app.processEvents()
        model = root.findChild(QObject, "simulationNaoModel")
        view_camera = root.findChild(QObject, "simulationViewCamera")
        start_marker = root.findChild(QObject, "simulationStartMarker")
        follow = root.findChild(QObject, "simulationFollowCamera")
        self.assertFalse(follow.property("checked"))
        initial_camera = view_camera.property("position")
        initial_marker = start_marker.property("position")
        def visual_child(item, name):
            if item.objectName() == name:
                return item
            for child in item.childItems():
                found = visual_child(child, name)
                if found is not None:
                    return found
            return None
        forward = visual_child(root, "simulationMotion_forward")
        stop = root.findChild(QObject, "simulationStopButton")
        self.assertIsNotNone(forward)
        self.assertFalse(forward.property("enabled"))
        self.assertFalse(self.controller.connected)
        QMetaObject.invokeMethod(connect, "clicked")
        self.wait_for(lambda: stand.property("enabled") and bool(self.controller.telemetry))
        self.assertFalse(hardware.property("enabled"))
        self.assertFalse(camera.findChild(QObject, "naoCameraStartButton").property("enabled"))
        initial_position = model.property("position")
        self.wait_for(lambda: forward.property("enabled"))
        QMetaObject.invokeMethod(forward, "clicked")
        self.wait_for(lambda: model.property("position") != initial_position)
        self.assertEqual(view_camera.property("position"), initial_camera)
        self.assertEqual(start_marker.property("position"), initial_marker)
        follow.setProperty("checked", True)
        self.app.processEvents()
        self.assertAlmostEqual(view_camera.property("position").z(), model.property("position").z() + 7.2, places=4)
        follow.setProperty("checked", False)
        self.app.processEvents()
        fixed_camera = view_camera.property("position")
        QTest.qWait(60)
        self.app.processEvents()
        self.assertEqual(view_camera.property("position"), fixed_camera)
        self.assertEqual(self.controller.posture, "Stand")
        self.assertTrue(forward.property("enabled"))
        right = visual_child(root, "simulationMotion_turn_right")
        self.assertTrue(right.property("enabled"))
        QMetaObject.invokeMethod(right, "clicked")
        self.wait_for(lambda: self.controller.motion["action"] == "turn_right")
        self.wait_for(lambda: self.controller.motion["heading"] > 0)
        self.assertFalse(stand.property("enabled"))
        self.assertTrue(stop.property("enabled"))
        QMetaObject.invokeMethod(stop, "clicked")
        self.wait_for(lambda: not self.controller.busy and not self.controller.stopping)
        stopped_position = model.property("position")
        self.assertEqual(model.property("walking"), 0)
        QTest.qWait(100)
        self.app.processEvents()
        self.assertEqual(model.property("position"), stopped_position)
        QMetaObject.invokeMethod(sit, "clicked")
        self.wait_for(lambda: self.controller.posture == "Sit" and not self.controller.busy)
        self.assertEqual(model.property("sitAmount"), 1)
        QMetaObject.invokeMethod(connect, "clicked")
        self.wait_for(lambda: not self.controller.connected and not self.controller.pending)
        self.assertFalse(stand.property("enabled"))
        view.close()
        view.setSource(QUrl())

    def test_stop_during_posture_freezes_view_state_and_recovers(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Sit")
        self.controller.stopMotion()
        self.wait_for(lambda: not self.controller.stopping and not self.controller.busy)
        self.assertEqual(self.controller.posture, "Unknown")
        frozen = self.controller.motion["sit_amount"]
        QTest.qWait(100)
        self.app.processEvents()
        self.assertEqual(self.controller.motion["sit_amount"], frozen)
        self.controller.requestPosture("Stand")
        self.wait_for(lambda: self.controller.posture == "Stand" and not self.controller.busy)

    def test_disconnect_stops_continuous_motion_and_shutdown_joins_worker(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Stand")
        self.wait_for(lambda: self.controller.posture == "Stand" and not self.controller.busy)
        self.controller.requestMotion("turn_left")
        self.wait_for(lambda: self.controller.motion["heading"] < 0)
        self.controller.disconnectSimulation()
        self.wait_for(lambda: not self.controller.pending)
        frozen = dict(self.controller.motion)
        QTest.qWait(100)
        self.app.processEvents()
        self.assertEqual(self.controller.motion, frozen)
        self.controller.close()
        self.assertFalse(self.controller._thread.is_alive())

    def start_walking(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Stand")
        self.wait_for(lambda: self.controller.posture == "Stand" and not self.controller.busy)
        self.controller.requestMotion("forward")
        self.wait_for(lambda: self.controller.motion["action"] == "forward")

    def test_automatic_switch_settles_before_starting_selected_direction(self):
        self.start_walking()
        self.assertTrue(self.controller.canSwitchMotion)
        self.controller.requestMotion("backward")
        self.wait_for(lambda: self.controller.motion["action"] == "settling")
        position = (self.controller.motion["x"], self.controller.motion["z"])
        self.assertTrue(self.controller.canSwitchMotion)
        QTest.qWait(80)
        self.app.processEvents()
        self.assertEqual((self.controller.motion["x"], self.controller.motion["z"]), position)
        self.wait_for(lambda: self.controller.motion["action"] == "backward")
        self.wait_for(lambda: self.controller.motion["z"] < position[1])
        self.controller.requestMotion("turn_right")
        self.wait_for(lambda: self.controller.motion["action"] == "turn_right")
        self.wait_for(lambda: self.controller.motion["heading"] > 0)

    def test_latest_direction_wins_during_automatic_settling(self):
        self.start_walking()
        self.controller.requestMotion("backward")
        self.wait_for(lambda: self.controller.motion["action"] == "settling")
        self.controller.requestMotion("turn_left")
        self.controller.requestMotion("turn_right")
        self.wait_for(lambda: self.controller.motion["action"] == "turn_right")
        self.assertFalse(any("backward started" in log or "turn_left started" in log for log in self.logs))

    def test_stop_cancels_pending_switch(self):
        self.start_walking()
        self.controller.requestMotion("backward")
        self.wait_for(lambda: self.controller.motion["action"] == "settling")
        self.controller.stopMotion()
        self.wait_for(lambda: not self.controller.busy and not self.controller.stopping)
        frozen = dict(self.controller.motion)
        QTest.qWait(450)
        self.app.processEvents()
        self.assertEqual(self.controller.motion, frozen)
        self.assertEqual(frozen["action"], "idle")
        self.assertFalse(any("backward started" in log for log in self.logs))

    def test_disconnect_cancels_pending_switch_across_reconnect(self):
        self.start_walking()
        self.controller.requestMotion("turn_left")
        self.wait_for(lambda: self.controller.motion["action"] == "settling")
        self.controller.disconnectSimulation()
        self.wait_for(lambda: not self.controller.pending)
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        QTest.qWait(450)
        self.app.processEvents()
        self.assertEqual(self.controller.motion["action"], "idle")
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.controller.posture, "Unknown")

    def test_reselecting_active_direction_does_not_settle_or_restart(self):
        self.start_walking()
        count = len(self.logs)
        self.controller.requestMotion("forward")
        self.wait_for(lambda: not self.controller._motion_switching)
        self.assertEqual(self.controller.motion["action"], "forward")
        self.assertFalse(any("switching" in log for log in self.logs[count:]))

    def test_motion_immediately_after_connect_prepares_stand_automatically(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.assertEqual(self.controller.posture, "Unknown")
        self.assertTrue(self.controller.canSwitchMotion)
        self.controller.requestMotion("forward")
        self.wait_for(lambda: self.controller.motion["action"] == "forward")
        self.assertEqual(self.controller.posture, "Stand")
        self.assertTrue(any("preparing Stand before forward" in log for log in self.logs))

    def test_motion_from_sitting_prepares_stand_automatically(self):
        self.controller.connectSimulation()
        self.wait_for(lambda: self.controller.connected)
        self.controller.requestPosture("Sit")
        self.wait_for(lambda: self.controller.posture == "Sit" and not self.controller.busy)
        self.assertTrue(self.controller.canSwitchMotion)
        self.controller.requestMotion("turn_left")
        self.wait_for(lambda: self.controller.motion["action"] == "turn_left")
        self.assertEqual(self.controller.posture, "Stand")

    def test_stop_cancels_automatic_stand_preparation(self):
        controller = SimulationController(service=SimulationRobotService(posture_duration=.2))
        self.addCleanup(controller.close)
        controller.connectSimulation()
        self.wait_for(lambda: controller.connected)
        controller.requestMotion("backward")
        self.wait_for(lambda: controller.motion["action"] == "posture")
        controller.stopMotion()
        self.wait_for(lambda: not controller.busy and not controller.stopping)
        QTest.qWait(450)
        self.app.processEvents()
        self.assertEqual(controller.motion["action"], "idle")
        self.assertEqual(controller.motion["z"], 0)
        self.assertEqual(controller.posture, "Unknown")

    def test_latest_direction_wins_during_automatic_stand_preparation(self):
        controller = SimulationController(service=SimulationRobotService(posture_duration=.08))
        self.addCleanup(controller.close)
        controller.connectSimulation()
        self.wait_for(lambda: controller.connected)
        controller.requestMotion("forward")
        self.wait_for(lambda: controller.motion["action"] == "posture")
        self.assertTrue(controller.canSwitchMotion)
        controller.requestMotion("turn_right")
        self.wait_for(lambda: controller.motion["action"] == "turn_right")
        self.assertEqual(controller.motion["z"], 0)
        self.assertEqual(controller.posture, "Stand")


if __name__ == "__main__":
    unittest.main()
