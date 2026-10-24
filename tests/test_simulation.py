"""Connection/status contract tests requiring no robot or GUI."""

from dataclasses import FrozenInstanceError
import asyncio
import unittest
from unittest.mock import patch

from robot_services.errors import RobotBusyError, RobotConnectionError, UnsupportedOperationError
from robot_services.interface import RobotService
from robot_services.models import OperatingMode, PostureOutcome
from robot_services.simulation import SimulationRobotService


class SimulationConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = SimulationRobotService()

    async def test_initial_state_is_disconnected_idle_and_unknown(self):
        self.assertIsInstance(self.service, RobotService)
        status = await self.service.get_status()
        self.assertIs(status.mode, OperatingMode.SIMULATION)
        self.assertFalse(status.connected)
        self.assertFalse(status.busy)
        self.assertIsNone(status.posture)

    async def test_connect_returns_connected_status_without_assuming_posture(self):
        status = await self.service.connect()
        self.assertTrue(status.connected)
        self.assertFalse(status.busy)
        self.assertIsNone(status.posture)
        self.assertIs(status.mode, OperatingMode.SIMULATION)

    async def test_repeated_connect_preserves_status(self):
        first = await self.service.connect()
        second = await self.service.connect()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)

    async def test_disconnect_is_safe_before_connection_and_when_repeated(self):
        self.assertIsNone(await self.service.disconnect())
        await self.service.connect()
        await self.service.disconnect()
        await self.service.disconnect()
        status = await self.service.get_status()
        self.assertFalse(status.connected)
        self.assertFalse(status.busy)
        self.assertIsNone(status.posture)

    async def test_reconnect_starts_idle_with_unknown_posture(self):
        await self.service.connect()
        await self.service.disconnect()
        status = await self.service.connect()
        self.assertTrue(status.connected)
        self.assertFalse(status.busy)
        self.assertIsNone(status.posture)

    async def test_old_status_is_an_immutable_snapshot(self):
        previous = await self.service.get_status()
        await self.service.connect()
        self.assertFalse(previous.connected)
        with self.assertRaises(FrozenInstanceError):
            previous.connected = True

    async def test_instances_have_independent_connections(self):
        other = SimulationRobotService()
        await self.service.connect()
        self.assertFalse((await other.get_status()).connected)

    async def test_telemetry_requires_connection(self):
        with self.assertRaises(RobotConnectionError):
            await self.service.read_telemetry()
        with self.assertRaises(RobotConnectionError):
            self.service.stream_telemetry(0.1)

    async def test_posture_operations_require_connection(self):
        with self.assertRaises(RobotConnectionError):
            await self.service.set_posture("Stand", 0.5)
        with self.assertRaises(RobotConnectionError):
            await self.service.get_posture()

    async def test_posture_completion_and_same_posture_is_immediate(self):
        service = SimulationRobotService(posture_duration=0.001)
        await service.connect()
        self.assertIsNone(await service.get_posture())
        for name in ("Stand", "Sit"):
            result = await service.set_posture(name, 1)
            self.assertIs(result.outcome, PostureOutcome.COMPLETED_VERIFIED)
            self.assertEqual(result.observed_posture, name)
            self.assertEqual(await service.get_posture(), name)
            with patch("robot_services.simulation.asyncio.sleep", side_effect=AssertionError("Unexpected delay")):
                await service.set_posture(name, 1)
        self.assertFalse((await service.get_status()).busy)

    async def test_busy_state_and_previous_posture_preserved_until_completion(self):
        service = SimulationRobotService(posture_duration=0.001)
        await service.connect()
        await service.set_posture("Sit", 1)
        started, release = asyncio.Event(), asyncio.Event()
        async def delay(seconds):
            self.assertEqual(seconds, 0.002)
            started.set()
            await release.wait()
        with patch("robot_services.simulation.asyncio.sleep", side_effect=delay):
            task = asyncio.create_task(service.set_posture("Stand", 0.5))
            try:
                await asyncio.wait_for(started.wait(), 1)
                self.assertTrue((await service.get_status()).busy)
                self.assertEqual(await service.get_posture(), "Sit")
                self.assertTrue((await service.connect()).busy)
                with self.assertRaises(RobotBusyError):
                    await service.set_posture("Sit", 1)
                release.set()
                await task
            finally:
                release.set()
                await service.disconnect()

    async def test_caller_cancellation_clears_busy_and_posture(self):
        service = SimulationRobotService(posture_duration=0.001)
        await service.connect()
        await service.set_posture("Sit", 1)
        task = asyncio.create_task(service.set_posture("Stand", 0.000001))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        status = await service.get_status()
        self.assertFalse(status.busy)
        self.assertIsNone(status.posture)
        self.assertTrue(status.connected)

    async def test_disconnect_cancels_action_and_reconnect_has_no_stale_completion(self):
        service = SimulationRobotService(posture_duration=0.001)
        await service.connect()
        task = asyncio.create_task(service.set_posture("Stand", 0.000001))
        await asyncio.sleep(0)
        await service.disconnect()
        await service.connect()
        await service.set_posture("Sit", 1)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await service.get_posture(), "Sit")
        self.assertFalse((await service.get_status()).busy)

    async def test_invalid_arguments_leave_status_unchanged(self):
        await self.service.connect()
        for speed in (0, -1, 2, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                await self.service.set_posture("Stand", speed)
        with self.assertRaises(UnsupportedOperationError):
            await self.service.set_posture("Fly", 1)
        self.assertFalse((await self.service.get_status()).busy)
        self.assertIsNone(await self.service.get_posture())
        for duration in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                SimulationRobotService(posture_duration=duration)

    async def test_connection_lifecycle_does_not_open_network_connections(self):
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            service = SimulationRobotService()
            await service.connect()
            await service.get_status()
            await service.disconnect()


if __name__ == "__main__":
    unittest.main()
