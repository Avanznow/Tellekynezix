"""Async telemetry lifecycle and fixture tests, without hardware or GUI."""

import asyncio
from datetime import datetime, timezone
import json
import unittest
from unittest.mock import MagicMock, patch

from robot_services.models import Measurement, OperatingMode, SampleQuality, SensorSample
from robot_services.simulation import SimulationRobotService


class SimulationTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = SimulationRobotService(posture_duration=0.001)
        await self.service.connect()
        self.addAsyncCleanup(self.service.disconnect)

    async def test_default_readings_are_labeled_constant_fixtures(self):
        first = await self.service.read_telemetry()
        second = await self.service.read_telemetry()
        self.assertIs(first.mode, OperatingMode.SIMULATION)
        self.assertEqual(first.source, "simulation:constant-fixtures")
        self.assertEqual(first.imu.values, second.imu.values)
        self.assertEqual(first.imu.values["acceleration_z"], Measurement(9.80665, "m/s^2"))
        self.assertEqual(first.ultrasonic.values["left"], Measurement(1.0, "m"))
        self.assertEqual(first.battery.values["charge"], Measurement(100, "%"))
        self.assertIs(first.fsr.quality, SampleQuality.UNAVAILABLE)
        self.assertIs(first.joints.quality, SampleQuality.UNAVAILABLE)
        self.assertTrue(first.fsr.error)
        self.assertLessEqual(first.imu.sampled_at, first.created_at)
        self.assertGreaterEqual(second.imu.sampled_at, first.imu.sampled_at)
        json.loads(first.to_json())

    async def test_configured_distances_battery_and_partial_sensor(self):
        partial = SensorSample(SampleQuality.PARTIAL, datetime.now(timezone.utc),
                               {"left": Measurement(0.5, "m"), "right": Measurement(None, "m")},
                               "Right sensor unavailable")
        service = SimulationRobotService(left_distance=2, right_distance=3,
                                         battery_percent=42, charging=True,
                                         sensor_overrides={"ultrasonic": partial})
        await service.connect()
        partial.values.clear()  # Constructor must own a defensive copy.
        first = await service.read_telemetry()
        self.assertIs(first.ultrasonic.quality, SampleQuality.PARTIAL)
        self.assertIsNone(first.ultrasonic.values["right"].value)
        self.assertEqual(first.battery.values["charge"].value, 42)
        self.assertEqual(first.battery.values["charging"].value, 1)
        self.assertIs(first.imu.quality, SampleQuality.VALID)
        first.ultrasonic.values.clear()  # Snapshots must also own their values.
        second = await service.read_telemetry()
        self.assertEqual(second.ultrasonic.values["left"].value, 0.5)
        await service.disconnect()

    async def test_custom_distance_and_unavailable_override(self):
        missing = SensorSample(SampleQuality.UNAVAILABLE, error="Injected IMU failure")
        service = SimulationRobotService(left_distance=2, right_distance=3,
                                         sensor_overrides={"imu": missing})
        await service.connect()
        snapshot = await service.read_telemetry()
        self.assertEqual(snapshot.ultrasonic.values["left"].value, 2)
        self.assertEqual(snapshot.ultrasonic.values["right"].value, 3)
        self.assertIs(snapshot.imu.quality, SampleQuality.UNAVAILABLE)
        self.assertIsNone(snapshot.imu.sampled_at)
        await service.disconnect()

    async def test_reads_and_streams_share_sequence_and_repeated_connect_preserves_it(self):
        one, two = self.service.stream_telemetry(.001), self.service.stream_telemetry(.001)
        try:
            self.assertEqual((await anext(one)).sequence, 0)
            self.assertEqual((await self.service.read_telemetry()).sequence, 1)
            await self.service.connect()
            self.assertEqual((await anext(two)).sequence, 2)
            self.assertEqual((await anext(one)).sequence, 3)
        finally:
            await one.aclose()
            await two.aclose()

    async def test_stream_spacing(self):
        stream = self.service.stream_telemetry(.02)
        try:
            await asyncio.wait_for(anext(stream), .5)
            started = asyncio.get_running_loop().time()
            await anext(stream)
            self.assertGreaterEqual(asyncio.get_running_loop().time() - started, .02)
        finally:
            await stream.aclose()

    async def test_early_timer_wakeup_waits_until_minimum_deadline(self):
        stream = self.service.stream_telemetry(.02)
        await anext(stream)
        loop = MagicMock()
        loop.time.side_effect = [0, 0, .005, .02]
        waits = []

        async def early_timeout(awaitable, timeout):
            awaitable.close()
            waits.append(timeout)
            raise TimeoutError

        try:
            with patch("robot_services.simulation.asyncio.get_running_loop", return_value=loop), \
                 patch("robot_services.simulation.asyncio.wait_for", side_effect=early_timeout):
                await anext(stream)
            self.assertEqual(len(waits), 2)
            self.assertAlmostEqual(waits[0], .02)
            self.assertAlmostEqual(waits[1], .015)
        finally:
            await stream.aclose()

    async def test_disconnect_wakes_pending_read_and_old_stream_stays_closed(self):
        stream = self.service.stream_telemetry(3600)
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await self.service.disconnect()
        await self.service.connect()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(pending, .5)
        self.assertEqual((await self.service.read_telemetry()).sequence, 0)
        await stream.aclose()

    async def test_unstarted_and_paused_streams_cannot_restart_after_reconnect(self):
        unstarted = self.service.stream_telemetry(.001)
        paused = self.service.stream_telemetry(.001)
        await anext(paused)
        await self.service.disconnect()
        await self.service.connect()
        for stream in (unstarted, paused):
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
            await stream.aclose()

    async def test_cancelled_and_closed_streams_stop_without_disconnect(self):
        stream = self.service.stream_telemetry(3600)
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        await stream.aclose()
        other = self.service.stream_telemetry(.001)
        await anext(other)
        await other.aclose()
        with self.assertRaises(StopAsyncIteration):
            await anext(other)
        self.assertTrue((await self.service.get_status()).connected)

    async def test_monitoring_continues_during_posture_without_changing_busy(self):
        stream = self.service.stream_telemetry(.001)
        await anext(stream)
        self.assertFalse((await self.service.get_status()).busy)
        action = asyncio.create_task(self.service.set_posture("Stand", .000001))
        await asyncio.sleep(0)
        try:
            self.assertTrue((await self.service.get_status()).busy)
            await anext(stream)
            self.assertTrue((await self.service.get_status()).busy)
        finally:
            await self.service.disconnect()
            await stream.aclose()
            with self.assertRaises(asyncio.CancelledError):
                await action

    async def test_invalid_intervals_and_configuration(self):
        for interval in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.service.stream_telemetry(interval)
        for kwargs in ({"left_distance": -1}, {"right_distance": float("nan")},
                       {"battery_percent": 101}, {"charging": 1},
                       {"sensor_overrides": {"unknown": None}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimulationRobotService(**kwargs)

    async def test_telemetry_does_not_open_network_connections(self):
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            snapshot = await self.service.read_telemetry()
            self.assertEqual(snapshot.mode, OperatingMode.SIMULATION)
            stream = self.service.stream_telemetry(.001)
            await anext(stream)
            await stream.aclose()


if __name__ == "__main__":
    unittest.main()
