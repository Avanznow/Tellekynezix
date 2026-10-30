"""Diagnostic health formatting, CLI and asynchronous resource lifecycle tests."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock

from robot_services.diagnostics import DiagnosticOptions, format_report, monitor
from robot_services.errors import RobotConnectionError, TelemetryValidationError
from robot_services.interface import RobotService
from robot_services.models import (
    Measurement, OperatingMode, RobotStatus, SampleQuality, SensorSample, TelemetrySnapshot,
)


def snapshot(charge=100, charging=0, **groups):
    now = datetime.now(timezone.utc)
    battery = groups.pop("battery", SensorSample(SampleQuality.VALID, now, {
        "charge": Measurement(charge, "%"), "charging": Measurement(charging, "boolean")
    }))
    return TelemetrySnapshot(0, now, OperatingMode.SIMULATION, "simulation:test",
                             battery=battery, **groups)


STATUS = RobotStatus(OperatingMode.SIMULATION, True, False)


class DiagnosticFormattingTests(unittest.TestCase):
    def test_battery_levels_and_exact_thresholds(self):
        for charge, label in ((100, "OK"), (20, "OK"), (19, "WARNING"),
                              (10, "WARNING"), (9, "CRITICAL"), (0, "CRITICAL")):
            with self.subTest(charge=charge):
                report = format_report(STATUS, snapshot(charge), DiagnosticOptions())
                self.assertIn(f"Battery: {charge}% [{label}]", report)
                self.assertIn("[SIMULATION] Connected", report)
                self.assertIn("Charging: No", report)

    def test_charging_does_not_hide_low_battery_alert(self):
        report = format_report(STATUS, snapshot(5, 1), DiagnosticOptions())
        self.assertIn("CRITICAL", report)
        self.assertIn("Charging: Yes", report)

    def test_custom_battery_thresholds(self):
        report = format_report(STATUS, snapshot(22), DiagnosticOptions(warn_below=25, critical_below=12))
        self.assertIn("WARNING", report)

    def test_unavailable_battery_is_not_reported_as_zero(self):
        missing = SensorSample(SampleQuality.UNAVAILABLE, error="Battery not accessible")
        report = format_report(STATUS, snapshot(battery=missing), DiagnosticOptions())
        self.assertIn("Battery: Unavailable", report)
        self.assertIn("Charging: Unknown", report)
        self.assertIn("Battery not accessible", report)
        self.assertNotIn("CRITICAL", report)

    def test_invalid_battery_units_range_and_charging(self):
        for value, unit in ((101, "%"), (-1, "%"), (0.5, "fraction")):
            with self.subTest(value=value, unit=unit):
                now = datetime.now(timezone.utc)
                battery = SensorSample(SampleQuality.VALID, now, {
                    "charge": Measurement(value, unit), "charging": Measurement(2, "boolean")
                })
                report = format_report(STATUS, snapshot(battery=battery), DiagnosticOptions())
                self.assertIn("Battery: INVALID", report)
                self.assertIn("Charging: Unknown", report)

    def test_partial_and_invalid_sensors_leave_other_health_visible(self):
        now = datetime.now(timezone.utc)
        partial = SensorSample(SampleQuality.PARTIAL, now, {
            "left": Measurement(1, "m"), "right": Measurement(None, "m")
        }, "Right sensor missing")
        invalid = SensorSample(SampleQuality.INVALID, error="Bad IMU packet")
        report = format_report(STATUS, snapshot(imu=invalid, ultrasonic=partial), DiagnosticOptions())
        self.assertIn("IMU: INVALID", report)
        self.assertIn("Ultrasonic: PARTIAL", report)
        self.assertIn("Right sensor missing", report)
        self.assertIn("Battery: 100% [OK]", report)
        self.assertIn("FSR: UNAVAILABLE", report)

    def test_valid_sensor_and_mode_mismatch(self):
        now = datetime.now(timezone.utc)
        imu = SensorSample(SampleQuality.VALID, now, {"x": Measurement(0, "m/s^2")})
        self.assertIn("IMU: OK", format_report(STATUS, snapshot(imu=imu), DiagnosticOptions()))
        hardware = RobotStatus(OperatingMode.HARDWARE, True, False)
        with self.assertRaises(TelemetryValidationError):
            format_report(hardware, snapshot(), DiagnosticOptions())

    def test_invalid_configuration(self):
        for kwargs in ({"interval": 0}, {"interval": float("nan")}, {"timeout": -1},
                       {"warn_below": 5, "critical_below": 10}, {"warn_below": 101},
                       {"critical_below": -1}, {"count": 0}, {"count": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DiagnosticOptions(**kwargs)


class DiagnosticMonitorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = Mock(spec=RobotService)
        self.service.connect = AsyncMock(return_value=STATUS)
        self.service.get_status = AsyncMock(return_value=STATUS)
        self.service.disconnect = AsyncMock()
        self.closed = False
        self.output = []

    def use_stream(self, readings):
        async def stream():
            try:
                for reading in readings:
                    yield reading
            finally:
                self.closed = True
        self.service.stream_telemetry.return_value = stream()

    async def test_finite_run_reports_every_sample_and_cleans_up_without_motion(self):
        self.use_stream([snapshot(100), snapshot(15)])
        result = await monitor(self.service, DiagnosticOptions(count=2), self.output.append)
        self.assertEqual(result, 0)
        self.assertEqual(len(self.output), 3)  # Startup plus two reports.
        self.assertIn("WARNING", self.output[-1])
        self.assertTrue(self.closed)
        self.service.disconnect.assert_awaited_once()
        self.service.set_posture.assert_not_called()
        self.service.start_motion.assert_not_called()
        self.service.stop_motion.assert_not_called()

    async def test_connect_failure_still_disconnects(self):
        self.service.connect.side_effect = RobotConnectionError("connection failed")
        result = await monitor(self.service, DiagnosticOptions(count=1), self.output.append)
        self.assertEqual(result, 1)
        self.assertIn("connection failed", self.output[-1])
        self.service.disconnect.assert_awaited_once()
        self.service.stream_telemetry.assert_not_called()

    async def test_false_connection_status_is_failure(self):
        self.service.connect.return_value = RobotStatus(OperatingMode.SIMULATION, False, False)
        self.assertEqual(await monitor(self.service, DiagnosticOptions(count=1), self.output.append), 1)
        self.service.disconnect.assert_awaited_once()

    async def test_stream_error_and_unexpected_end_are_failures(self):
        self.use_stream([])
        self.assertEqual(await monitor(self.service, DiagnosticOptions(count=1), self.output.append), 1)
        self.assertIn("stream ended", self.output[-1])

    async def test_connection_loss_during_monitoring(self):
        self.use_stream([snapshot()])
        self.service.get_status.side_effect = RobotConnectionError("lost connection")
        self.assertEqual(await monitor(self.service, DiagnosticOptions(count=1), self.output.append), 1)
        self.assertTrue(self.closed)
        self.service.disconnect.assert_awaited_once()

    async def test_connect_timeout_is_bounded(self):
        async def blocked():
            await asyncio.Event().wait()
        self.service.connect.side_effect = blocked
        result = await monitor(self.service, DiagnosticOptions(count=1, timeout=.02), self.output.append)
        self.assertEqual(result, 1)
        self.assertIn("TimeoutError", self.output[-1])
        self.service.disconnect.assert_awaited_once()

    async def test_cancellation_closes_stream_and_disconnects(self):
        waiting = asyncio.Event()
        async def stream():
            try:
                waiting.set()
                await asyncio.Event().wait()
                yield snapshot()
            finally:
                self.closed = True
        self.service.stream_telemetry.return_value = stream()
        task = asyncio.create_task(monitor(self.service, DiagnosticOptions(), self.output.append))
        await asyncio.wait_for(waiting.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.closed)
        self.service.disconnect.assert_awaited_once()

    async def test_cleanup_error_is_reported_as_failure(self):
        self.use_stream([snapshot()])
        self.service.disconnect.side_effect = RobotConnectionError("cleanup failed")
        result = await monitor(self.service, DiagnosticOptions(count=1), self.output.append)
        self.assertEqual(result, 1)
        self.assertIn("Disconnect failed", self.output[-1])


class DiagnosticCliTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        script = Path(__file__).resolve().parents[1] / "tools/robot_diagnostics.py"
        return subprocess.run([sys.executable, "-B", str(script), *args],
                              capture_output=True, text=True, timeout=10, cwd=cwd)

    def test_standalone_script_runs_outside_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli("--count", "2", "--interval", ".01", cwd=folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Battery: 100% [OK]"), 2)
        self.assertIn("[SIMULATION]", result.stdout)
        self.assertIn("FSR: UNAVAILABLE", result.stdout)

    def test_simulated_critical_battery_and_charging(self):
        result = self.run_cli("--count", "1", "--battery-percent", "5", "--charging")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Battery: 5% [CRITICAL]", result.stdout)
        self.assertIn("Charging: Yes", result.stdout)

    def test_invalid_interval_and_hardware_mode_fail_explicitly(self):
        for args in (("--interval", "0"), ("--mode", "hardware"), ("--battery-percent", "101")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
