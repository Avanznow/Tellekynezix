"""SDK-shape mocks only: these tests do not import qi or contact a robot."""

import asyncio
import json
import threading
import unittest
from unittest.mock import Mock, patch

from robot_services.errors import RobotBusyError, RobotConnectionError, UnsupportedOperationError
from robot_services.models import OperatingMode, PostureOutcome, SampleQuality
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
        self.memory = None
        self.posture = None

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
        if name == "ALMemory":
            return FakeFuture(self.memory, error="ALMemory missing" if self.memory is None else None)
        if name == "ALRobotPosture":
            return FakeFuture(self.posture, error="Posture service missing" if self.posture is None else None)
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

    async def test_imu_batch_preserves_valid_axes_and_battery(self):
        keys = [f"Device/SubDeviceList/InertialSensor/Accelerometer{axis}/Sensor/Value" for axis in "XYZ"]
        memory = Mock()
        readings = iter([
            FakeFuture([.25, -.5, 9.8]),
            FakeFuture([None, float("nan"), 9.8]),
            FakeFuture([True, "bad", None]),
            FakeFuture([None, None, None]),
            FakeFuture([1, 2]),
            FakeFuture(error="read failed"),
        ])
        memory.getListData.side_effect = lambda requested, **_: (
            next(readings) if requested == keys else FakeFuture([.4, .6])
        )
        self.session.memory = memory
        await self.service.connect()
        results = [await self.service.read_telemetry() for _ in range(6)]
        self.assertEqual([item.imu.quality for item in results], [
            SampleQuality.VALID, SampleQuality.PARTIAL, SampleQuality.INVALID,
            SampleQuality.UNAVAILABLE, SampleQuality.INVALID, SampleQuality.UNAVAILABLE,
        ])
        self.assertEqual(results[0].imu.values["acceleration_y"].value, -.5)
        self.assertEqual(results[0].imu.values["acceleration_y"].unit, "m/s^2")
        self.assertIsNone(results[1].imu.values["acceleration_x"].value)
        self.assertIsNone(results[1].imu.values["acceleration_y"].value)
        self.assertEqual(results[1].imu.values["acceleration_z"].value, 9.8)
        for result in results:
            self.assertEqual(result.battery.values["charge"].value, 72)
            json.loads(result.to_json())
        memory.getListData.assert_any_call(keys, _async=True)
        self.assertEqual(results[-1].ultrasonic.quality, SampleQuality.VALID)

    async def test_sonar_maps_sides_and_preserves_other_groups(self):
        keys = [f"Device/SubDeviceList/US/{side}/Sensor/Value" for side in ("Left", "Right")]
        readings = iter([
            FakeFuture([.42, .7]),
            FakeFuture([5.0, .5]),
            FakeFuture([0, -1]),
            FakeFuture([None, 5.0]),
            FakeFuture([.4]),
            FakeFuture(error="sonar read failed"),
        ])
        memory = Mock()
        def get_data(requested, *, _async):
            self.assertTrue(_async)
            if requested == keys:
                return next(readings)
            if len(requested) == 8:
                return FakeFuture([0] * 8)
            return FakeFuture([.25, -.5, 9.8])

        memory.getListData.side_effect = get_data
        self.session.memory = memory
        await self.service.connect()
        results = [await self.service.read_telemetry() for _ in range(6)]
        self.assertEqual([item.ultrasonic.quality for item in results], [
            SampleQuality.VALID, SampleQuality.PARTIAL, SampleQuality.INVALID,
            SampleQuality.UNAVAILABLE, SampleQuality.INVALID, SampleQuality.UNAVAILABLE,
        ])
        left, right = (results[0].ultrasonic.values[side] for side in ("left", "right"))
        self.assertEqual((left.value, left.unit, right.value, right.unit), (.42, "m", .7, "m"))
        self.assertIsNone(results[1].ultrasonic.values["left"].value)
        self.assertEqual(results[1].ultrasonic.values["right"].value, .5)
        self.assertIn("no echo", results[1].ultrasonic.error)
        for result in results:
            self.assertEqual(result.imu.quality, SampleQuality.VALID)
            self.assertEqual(result.battery.values["charge"].value, 72)
            json.loads(result.to_json())
        memory.getListData.assert_any_call(keys, _async=True)
        self.assertEqual(results[-1].fsr.quality, SampleQuality.VALID)
        self.assertEqual(self.session.calls.count(("service", "ALMemory")), 2)

    async def test_fsr_maps_eight_loads_and_preserves_other_groups(self):
        keys = [
            f"Device/SubDeviceList/{foot}/FSR/{corner}/Sensor/Value"
            for foot in ("LFoot", "RFoot")
            for corner in ("FrontLeft", "FrontRight", "RearLeft", "RearRight")
        ]
        loads = [0, .1, .2, .3, .4, .5, .6, .7]
        readings = iter([
            FakeFuture(loads),
            FakeFuture([None, -1, float("inf"), True, .4, .5, .6, .7]),
            FakeFuture([-1] * 8),
            FakeFuture([None] * 8),
            FakeFuture(loads[:-1]),
            FakeFuture(error="FSR read failed"),
        ])

        def get_data(requested, *, _async):
            self.assertTrue(_async)
            if requested == keys:
                return next(readings)
            if len(requested) == 3:
                return FakeFuture([.25, -.5, 9.8])
            return FakeFuture([.4, .6])

        self.session.memory = Mock()
        self.session.memory.getListData.side_effect = get_data
        await self.service.connect()
        results = [await self.service.read_telemetry() for _ in range(6)]
        self.assertEqual([result.fsr.quality for result in results], [
            SampleQuality.VALID, SampleQuality.PARTIAL, SampleQuality.INVALID,
            SampleQuality.UNAVAILABLE, SampleQuality.INVALID, SampleQuality.UNAVAILABLE,
        ])
        names = (
            "left_front_left", "left_front_right", "left_rear_left", "left_rear_right",
            "right_front_left", "right_front_right", "right_rear_left", "right_rear_right",
        )
        self.assertEqual(
            [(results[0].fsr.values[name].value, results[0].fsr.values[name].unit) for name in names],
            [(value, "kg") for value in loads],
        )
        self.assertEqual([results[1].fsr.values[name].value for name in names],
                         [None, None, None, None, .4, .5, .6, .7])
        for result in results:
            self.assertEqual(result.imu.quality, SampleQuality.VALID)
            self.assertEqual(result.ultrasonic.quality, SampleQuality.VALID)
            self.assertEqual(result.battery.values["charge"].value, 72)
            json.loads(result.to_json())

    async def test_joint_positions_preserve_angles_and_validate_hand_aperture(self):
        names = (
            "HeadYaw", "HeadPitch",
            "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
            "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
            "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
            "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
        )
        keys = [f"Device/SubDeviceList/{name}/Position/Sensor/Value" for name in names]
        values = [-.25] * 25
        values[7], values[13] = .25, 1
        partial = values.copy()
        partial[0], partial[7], partial[13] = None, 1.1, -.1
        readings = iter([
            FakeFuture(values), FakeFuture(partial), FakeFuture([True] * 25),
            FakeFuture([None] * 25), FakeFuture(error="joint read failed"),
        ])
        other_readings = {2: [.4, .6], 3: [0, 0, 9.8], 8: [0] * 8}

        def get_data(requested, *, _async):
            self.assertTrue(_async)
            if requested == keys:
                return next(readings)
            return FakeFuture(other_readings[len(requested)])

        self.session.memory = Mock()
        self.session.memory.getListData.side_effect = get_data
        await self.service.connect()
        results = [await self.service.read_telemetry() for _ in range(5)]
        self.assertEqual([result.joints.quality for result in results], [
            SampleQuality.VALID, SampleQuality.PARTIAL, SampleQuality.INVALID,
            SampleQuality.UNAVAILABLE, SampleQuality.UNAVAILABLE,
        ])
        self.assertEqual(
            [(results[0].joints.values[name].value, results[0].joints.values[name].unit) for name in names],
            [(value, "ratio" if name in ("LHand", "RHand") else "rad") for name, value in zip(names, values)],
        )
        for name in ("HeadYaw", "LHand", "RHand"):
            self.assertIsNone(results[1].joints.values[name].value)
        self.assertEqual(results[1].joints.values["LHipYawPitch"].value, -.25)
        for result in results:
            self.assertEqual(result.fsr.quality, SampleQuality.VALID)
            self.assertEqual(result.battery.values["charge"].value, 72)
            json.loads(result.to_json())

    async def test_posture_requires_success_and_matching_observation(self):
        service = QiRobotService("robot.test", session_factory=self.factory, enable_posture=True)
        self.addAsyncCleanup(service.disconnect)
        self.session.posture = proxy = Mock()
        proxy.goToPosture.side_effect = [FakeFuture(True), FakeFuture(True), FakeFuture(False), FakeFuture(True)]
        proxy.getPosture.side_effect = [FakeFuture("Stand"), FakeFuture("Stand"), FakeFuture(error="read failed")]
        await service.connect()
        with self.assertRaises(ValueError):
            await service.set_posture("Stand", float("nan"))
        with self.assertRaises(UnsupportedOperationError):
            await service.set_posture("StandZero", .5)
        results = [
            await service.set_posture(name, .5) for name in ("Stand", "Sit", "Sit", "Sit")
        ]
        self.assertEqual([result.outcome for result in results], [
            PostureOutcome.COMPLETED_VERIFIED, PostureOutcome.UNVERIFIED,
            PostureOutcome.FAILED, PostureOutcome.UNVERIFIED,
        ])
        self.assertEqual([result.observed_posture for result in results], ["Stand", "Stand", None, None])
        self.assertEqual(proxy.goToPosture.call_count, 4)
        proxy.goToPosture.assert_called_with("Sit", .5, _async=True)
        self.assertFalse((await service.get_status()).busy)

    async def test_posture_timeout_blocks_another_request_in_same_session(self):
        service = QiRobotService("robot.test", timeout=.05, session_factory=self.factory, enable_posture=True)
        self.addAsyncCleanup(service.disconnect)
        self.session.posture = proxy = Mock()
        pending = FakeFuture(pending=True)
        proxy.goToPosture.return_value = pending
        await service.connect()
        request = asyncio.create_task(service.set_posture("Stand", .5))
        await self.wait_registered(pending)
        self.assertTrue((await service.get_status()).busy)
        with self.assertRaises(RobotBusyError):
            await service.set_posture("Sit", .5)
        self.assertIs((await request).outcome, PostureOutcome.UNVERIFIED)
        with self.assertRaises(RobotBusyError):
            await service.set_posture("Sit", .5)
        proxy.goToPosture.assert_called_once_with("Stand", .5, _async=True)
        proxy.getPosture.assert_not_called()

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
    async def test_stream_wakes_on_disconnect_and_does_not_resume_after_reconnect(self):
        with self.assertRaises(RobotConnectionError):
            self.service.stream_telemetry(1)
        await self.service.connect()
        stream = self.service.stream_telemetry(3600)
        self.assertEqual((await anext(stream)).sequence, 0)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await self.service.disconnect()
        self.factory.return_value = FakeSession()
        await self.service.connect()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(pending, 1)
        self.assertEqual((await self.service.read_telemetry()).sequence, 0)

    async def test_stream_reports_transport_loss(self):
        await self.service.connect()
        stream = self.service.stream_telemetry(.001)
        try:
            await anext(stream)
            self.session.connected = False
            with self.assertRaises(RobotConnectionError):
                await anext(stream)
        finally:
            await stream.aclose()

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
