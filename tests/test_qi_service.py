"""SDK-shape mocks only: these tests do not import qi or contact a robot."""

import asyncio
import json
import threading
import unittest
from unittest.mock import Mock, patch

from robot_services.errors import RobotConnectionError, UnsupportedOperationError
from robot_services.models import OperatingMode, SampleQuality
from robot_services.qi_service import QiRobotService


class FakeFuture:
    def __init__(self, value=None, *, error=None, pending=False):
        self.result = value
        self.error = error
        self.ready = not pending
        self.callbacks = []
        self.cancellations = 0

    def addCallback(self, callback):
        self.callbacks.append(callback)
        if self.ready:
            callback(self)

    def value(self, timeout):
        assert timeout == 0, "Adapter must never block on future.value()"
        assert self.ready
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.result

    def cancel(self):
        self.cancellations += 1  # Intentionally advisory, like native cancellation.

    def finish(self, value=None):
        self.result = value
        self.ready = True
        for callback in self.callbacks:
            callback(self)


class FakeBattery:
    def __init__(self, value=72):
        self.calls = []
        self.read = Mock(side_effect=lambda: FakeFuture(value))

    def getBatteryCharge(self, *, _async):
        assert _async is True
        self.calls.append("getBatteryCharge")
        return self.read()


class FakeSession:
    def __init__(self, battery=None):
        self.connected = False
        self.calls = []
        self.battery = battery or FakeBattery()
        self.connect_future = FakeFuture()
        self.close_future = FakeFuture()
        self.status_future = None
        self.service_error = None

    def connect(self, endpoint, *, _async):
        assert _async is True
        self.calls.append(("connect", endpoint))
        self.connected = True
        return self.connect_future

    def isConnected(self, *, _async):
        assert _async is True
        self.calls.append(("isConnected",))
        return self.status_future or FakeFuture(self.connected)

    def service(self, name, *, _async):
        assert _async is True
        self.calls.append(("service", name))
        return FakeFuture(self.battery, error=self.service_error)

    def close(self, *, _async):
        assert _async is True
        self.calls.append(("close",))
        self.connected = False
        return self.close_future


class QiServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = FakeSession()
        self.factory = Mock(return_value=self.session)
        self.service = QiRobotService("robot.test", port=9559, timeout=.05,
                                      session_factory=self.factory)

    async def asyncTearDown(self):
        await self.service.disconnect()

    async def wait_registered(self, future):
        async def registered():
            while not future.callbacks:
                await asyncio.sleep(0)
        await asyncio.wait_for(registered(), 1)

    async def test_sdk_is_lazy_and_missing_sdk_has_clear_error(self):
        with patch("robot_services.qi_service.importlib.import_module", side_effect=ImportError) as load:
            service = QiRobotService("robot.test")
            load.assert_not_called()
            self.assertFalse((await service.get_status()).connected)
            with self.assertRaisesRegex(RobotConnectionError, "Compatible qi SDK"):
                await service.connect()
            load.assert_called_once_with("qi")
        await service.disconnect()

    async def test_connect_and_read_charge_use_only_read_only_sdk_calls(self):
        with patch("robot_services.qi_service.importlib.import_module", side_effect=AssertionError("No SDK import")):
            status = await self.service.connect()
            self.assertTrue(status.connected)
            self.assertIs(status.mode, OperatingMode.HARDWARE)
            self.assertIsNone(status.posture)
            snapshot = await self.service.read_telemetry()
        self.assertIs(snapshot.mode, OperatingMode.HARDWARE)
        self.assertIs(snapshot.battery.quality, SampleQuality.PARTIAL)
        self.assertEqual(snapshot.battery.values["charge"].value, 72)
        self.assertEqual(snapshot.battery.values["charge"].unit, "%")
        self.assertIsNone(snapshot.battery.values["charging"].value)
        self.assertIn("hardware:qi:", snapshot.source)
        for sample in (snapshot.imu, snapshot.ultrasonic, snapshot.fsr, snapshot.joints):
            self.assertIs(sample.quality, SampleQuality.UNAVAILABLE)
        json.loads(snapshot.to_json())
        self.assertIn(("connect", "tcp://robot.test:9559"), self.session.calls)
        self.assertEqual(self.session.battery.calls, ["getBatteryCharge"])
        self.assertTrue(all(call[0] in ("connect", "isConnected", "service") for call in self.session.calls))

    async def test_reads_require_connection_and_actuation_is_explicitly_unsupported(self):
        with self.assertRaises(RobotConnectionError):
            await self.service.read_telemetry()
        await self.service.connect()
        self.assertIsNone(await self.service.get_posture())
        self.assertEqual(await self.service.get_motion_capabilities(), ())
        with self.assertRaises(UnsupportedOperationError):
            await self.service.set_posture("Stand", .5)
        with self.assertRaises(UnsupportedOperationError):
            await self.service.start_motion("forward")
        with self.assertRaises(UnsupportedOperationError):
            await self.service.stop_motion()
        with self.assertRaises(UnsupportedOperationError):
            self.service.stream_telemetry(1)

    async def test_concurrent_and_repeated_connect_share_one_session(self):
        await asyncio.gather(self.service.connect(), self.service.connect())
        await self.service.connect()
        self.factory.assert_called_once()
        self.assertEqual(sum(call[0] == "connect" for call in self.session.calls), 1)
        await self.service.disconnect()
        await self.service.disconnect()
        self.assertEqual(self.session.calls.count(("close",)), 1)

    async def test_repeated_connect_returns_first_confirmed_status(self):
        await self.service.connect()
        self.session.isConnected = Mock(side_effect=[FakeFuture(True), FakeFuture(False)])
        status = await self.service.connect()
        self.assertTrue(status.connected)
        self.session.isConnected.assert_called_once_with(_async=True)
        self.factory.assert_called_once()

    async def test_connection_failure_closes_partial_session(self):
        self.session.connect_future = FakeFuture(error="connection refused")
        with self.assertRaisesRegex(RobotConnectionError, "connection refused"):
            await self.service.connect()
        self.assertIn(("close",), self.session.calls)
        self.assertFalse((await self.service.get_status()).connected)

    async def test_connection_requires_confirmation(self):
        self.session.status_future = FakeFuture(False)
        with self.assertRaisesRegex(RobotConnectionError, "did not confirm"):
            await self.service.connect()
        self.assertIn(("close",), self.session.calls)

    async def test_connection_timeout_cancels_future_and_ignores_late_reply(self):
        pending = FakeFuture(pending=True)
        self.session.connect_future = pending
        with self.assertRaisesRegex(RobotConnectionError, "timed out"):
            await self.service.connect()
        self.assertGreater(pending.cancellations, 0)
        pending.finish()
        await asyncio.sleep(0)
        self.assertFalse((await self.service.get_status()).connected)
        self.assertFalse(self.service._pending)
        self.assertIn(("close",), self.session.calls)

    async def test_cancelled_connect_closes_partial_session(self):
        pending = FakeFuture(pending=True)
        self.session.connect_future = pending
        task = asyncio.create_task(self.service.connect())
        await self.wait_registered(pending)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn(("close",), self.session.calls)
        self.assertFalse(self.service._pending)

    async def test_missing_battery_service_is_unavailable_and_can_be_retried(self):
        await self.service.connect()
        self.session.service_error = "ALBattery missing"
        missing = await self.service.read_telemetry()
        self.assertIs(missing.battery.quality, SampleQuality.UNAVAILABLE)
        self.assertIn("ALBattery missing", missing.battery.error)
        self.session.service_error = None
        recovered = await self.service.read_telemetry()
        self.assertEqual(recovered.sequence, 1)
        self.assertEqual(recovered.battery.values["charge"].value, 72)

    async def test_invalid_battery_values_are_marked_not_fabricated(self):
        await self.service.connect()
        for value in (None, True, "72", -1, 101, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.session.battery.read.return_value = None
                self.session.battery.read.side_effect = lambda: FakeFuture(value)
                snapshot = await self.service.read_telemetry()
                self.assertIs(snapshot.battery.quality, SampleQuality.INVALID)
                self.assertEqual(snapshot.battery.values, {})
                json.loads(snapshot.to_json())

    async def test_battery_timeout_is_unavailable_and_late_value_is_ignored(self):
        await self.service.connect()
        pending = FakeFuture(pending=True)
        self.session.battery.read.side_effect = [pending, FakeFuture(25)]
        first = await self.service.read_telemetry()
        self.assertIs(first.battery.quality, SampleQuality.UNAVAILABLE)
        self.assertGreater(pending.cancellations, 0)
        pending.finish(99)
        await asyncio.sleep(0)
        second = await self.service.read_telemetry()
        self.assertEqual(second.battery.values["charge"].value, 25)
        self.assertFalse(self.service._pending)

    async def test_lost_connection_during_battery_read_raises_connection_error(self):
        await self.service.connect()
        def fail():
            self.session.connected = False
            return FakeFuture(error="transport lost")
        self.session.battery.read.side_effect = fail
        with self.assertRaises(RobotConnectionError):
            await self.service.read_telemetry()
        self.assertFalse((await self.service.get_status()).connected)

    async def test_disconnect_invalidates_read_before_fresh_session(self):
        await self.service.connect()
        pending = FakeFuture(pending=True)
        self.session.battery.read.side_effect = [pending]
        task = asyncio.create_task(self.service.read_telemetry())
        await self.wait_registered(pending)
        await self.service.disconnect()
        fresh = FakeSession(FakeBattery(35))
        self.factory.return_value = fresh
        await self.service.connect()
        with self.assertRaises(RobotConnectionError):
            await task
        pending.finish(99)
        await asyncio.sleep(0)
        snapshot = await self.service.read_telemetry()
        self.assertEqual(snapshot.sequence, 0)
        self.assertEqual(snapshot.battery.values["charge"].value, 35)

    async def test_callback_from_sdk_thread_is_delivered_to_asyncio(self):
        await self.service.connect()
        pending = FakeFuture(pending=True)
        self.session.battery.read.side_effect = [pending]
        task = asyncio.create_task(self.service.read_telemetry())
        await self.wait_registered(pending)
        thread = threading.Thread(target=pending.finish, args=(64,))
        thread.start()
        snapshot = await task
        thread.join(1)
        self.assertEqual(snapshot.battery.values["charge"].value, 64)

    async def test_close_failure_still_clears_local_connection(self):
        await self.service.connect()
        self.session.close_future = FakeFuture(error="close failed")
        with self.assertRaisesRegex(RobotConnectionError, "close failed"):
            await self.service.disconnect()
        self.assertFalse((await self.service.get_status()).connected)
        await self.service.disconnect()

    async def test_status_timeout_marks_backend_disconnected(self):
        await self.service.connect()
        self.session.status_future = FakeFuture(pending=True)
        with self.assertRaisesRegex(RobotConnectionError, "timed out"):
            await self.service.get_status()
        self.assertFalse((await self.service.get_status()).connected)

    async def test_injected_adapter_never_uses_network_connection_helpers(self):
        with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), \
             patch("asyncio.open_connection", side_effect=AssertionError("Network forbidden")):
            await self.service.connect()
            await self.service.read_telemetry()
            await self.service.disconnect()

    async def test_read_cancellation_leaves_connection_open_and_ignores_late_callback(self):
        await self.service.connect()
        pending = FakeFuture(pending=True)
        self.session.battery.read.side_effect = [pending]
        task = asyncio.create_task(self.service.read_telemetry())
        await self.wait_registered(pending)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        pending.finish(99)
        await asyncio.sleep(0)
        self.assertGreater(pending.cancellations, 0)
        self.assertFalse(self.service._pending)
        self.assertTrue((await self.service.get_status()).connected)

    async def test_disconnect_during_connect_closes_exactly_once(self):
        pending = FakeFuture(pending=True)
        self.session.connect_future = pending
        task = asyncio.create_task(self.service.connect())
        await self.wait_registered(pending)
        await self.service.disconnect()
        with self.assertRaises(RobotConnectionError):
            await task
        pending.finish()
        await asyncio.sleep(0)
        self.assertEqual(self.session.calls.count(("close",)), 1)
        self.assertFalse((await self.service.get_status()).connected)

    async def test_close_timeout_is_reported_and_session_is_detached(self):
        await self.service.connect()
        self.session.close_future = FakeFuture(pending=True)
        with self.assertRaisesRegex(RobotConnectionError, "timed out"):
            await self.service.disconnect()
        self.assertFalse((await self.service.get_status()).connected)
        self.assertFalse(self.service._pending)

    def test_configuration_validation_and_ipv6(self):
        for kwargs in ({"host": ""}, {"host": "tcp://robot"}, {"host": "user@robot"},
                       {"host": "robot/path"}, {"host": "robot test"}, {"port": 0},
                       {"port": True}, {"timeout": 0}, {"timeout": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QiRobotService(**({"host": "robot.test"} | kwargs))
        self.assertEqual(QiRobotService("::1", 9558).endpoint, "tcp://[::1]:9558")


if __name__ == "__main__":
    unittest.main()
