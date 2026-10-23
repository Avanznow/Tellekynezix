"""Continuous visual motion, settling, cancellation and session isolation."""

import asyncio
import unittest

from robot_services.errors import RobotBusyError, RobotConnectionError, UnsupportedOperationError
from robot_services.simulation import SimulationRobotService


class SimulationMotionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = SimulationRobotService(posture_duration=.001, settle_duration=.06)
        await self.service.connect()
        await self.service.set_posture("Stand", 1)
        self.addAsyncCleanup(self.service.disconnect)

    async def test_motion_requires_connection_standing_and_valid_arguments(self):
        for command in ("forward", "backward", "turn_left", "turn_right"):
            self.assertIn(command, await self.service.get_motion_capabilities())
        with self.assertRaises(UnsupportedOperationError):
            await self.service.start_motion("fly")
        for speed in (0, -1, 2, True, "1", float("nan"), float("inf")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                await self.service.start_motion("forward", speed)
        await self.service.set_posture("Sit", 1)
        with self.assertRaises(ValueError):
            await self.service.start_motion("forward")
        await self.service.disconnect()
        with self.assertRaises(RobotConnectionError):
            await self.service.start_motion("forward")
        with self.assertRaises(RobotConnectionError):
            await self.service.stop_motion()

    async def test_forward_backward_and_busy_rejection(self):
        await self.service.start_motion("forward")
        await asyncio.sleep(.08)
        forward = await self.service.get_motion_state()
        self.assertGreater(forward.z, 0)
        self.assertTrue((await self.service.get_status()).busy)
        with self.assertRaises(RobotBusyError):
            await self.service.start_motion("backward")
        with self.assertRaises(RobotBusyError):
            await self.service.set_posture("Sit", 1)
        await self.service.stop_motion()
        await self.service.start_motion("backward")
        await asyncio.sleep(.08)
        self.assertLess((await self.service.get_motion_state()).z, forward.z)

    async def test_turns_change_heading_without_travel(self):
        await self.service.start_motion("turn_left")
        await asyncio.sleep(.08)
        left = await self.service.get_motion_state()
        self.assertLess(left.heading, 0)
        self.assertEqual((left.x, left.z), (0, 0))
        await self.service.stop_motion()
        await self.service.start_motion("turn_right")
        await asyncio.sleep(.08)
        self.assertGreater((await self.service.get_motion_state()).heading, left.heading)

    async def test_stop_freezes_travel_then_settles_legs(self):
        await self.service.start_motion("forward")
        await asyncio.sleep(.08)
        stop = asyncio.create_task(self.service.stop_motion())
        await asyncio.sleep(0)
        halted = await self.service.get_motion_state()
        self.assertEqual(halted.action, "settling")
        self.assertTrue((await self.service.get_status()).busy)
        await asyncio.sleep(.02)
        during = await self.service.get_motion_state()
        self.assertEqual((during.x, during.z, during.heading, during.gait_phase),
                         (halted.x, halted.z, halted.heading, halted.gait_phase))
        self.assertLess(during.gait_weight, halted.gait_weight)
        settled = await stop
        self.assertEqual((settled.x, settled.z, settled.heading),
                         (halted.x, halted.z, halted.heading))
        self.assertEqual(settled.gait_weight, 0)
        self.assertEqual(settled.action, "idle")
        self.assertFalse((await self.service.get_status()).busy)
        self.assertEqual(await self.service.get_posture(), "Stand")

    async def test_stop_during_posture_freezes_visual_pose_and_marks_unknown(self):
        service = SimulationRobotService(posture_duration=.2)
        self.addAsyncCleanup(service.disconnect)
        await service.connect()
        posture = asyncio.create_task(service.set_posture("Sit", 1))
        await asyncio.sleep(.06)
        stopped = await service.stop_motion()
        with self.assertRaises(asyncio.CancelledError):
            await posture
        self.assertGreater(stopped.sit_amount, 0)
        self.assertLess(stopped.sit_amount, 1)
        await asyncio.sleep(.05)
        self.assertEqual((await service.get_motion_state()).sit_amount, stopped.sit_amount)
        self.assertIsNone(await service.get_posture())
        self.assertFalse((await service.get_status()).busy)

    async def test_repeated_stop_and_idle_sit_stop(self):
        await self.service.start_motion("turn_right")
        first = asyncio.create_task(self.service.stop_motion())
        await asyncio.sleep(0)
        second = asyncio.create_task(self.service.stop_motion())
        self.assertEqual(await first, await second)
        await self.service.set_posture("Sit", 1)
        before = await self.service.get_motion_state()
        self.assertEqual(await self.service.stop_motion(), before)
        self.assertEqual(await self.service.get_posture(), "Sit")

    async def test_disconnect_freezes_motion_and_old_session_cannot_move_new_session(self):
        await self.service.start_motion("forward")
        await asyncio.sleep(.04)
        await self.service.disconnect()
        stopped = await self.service.get_motion_state()
        await self.service.connect()
        await asyncio.sleep(.08)
        self.assertEqual(await self.service.get_motion_state(), stopped)
        self.assertIsNone(await self.service.get_posture())
        self.assertFalse((await self.service.get_status()).busy)

    async def test_disconnect_cancels_settling_and_prevents_stale_stand(self):
        await self.service.start_motion("forward")
        stop = asyncio.create_task(self.service.stop_motion())
        await asyncio.sleep(0)
        await self.service.disconnect()
        with self.assertRaises(asyncio.CancelledError):
            await stop
        await self.service.connect()
        await asyncio.sleep(.08)
        self.assertIsNone(await self.service.get_posture())
        self.assertEqual((await self.service.get_motion_state()).action, "idle")

    async def test_telemetry_continues_during_motion_and_settling(self):
        await self.service.start_motion("forward")
        first = await self.service.read_telemetry()
        stop = asyncio.create_task(self.service.stop_motion())
        await asyncio.sleep(0)
        second = await self.service.read_telemetry()
        self.assertGreater(second.sequence, first.sequence)
        await stop


if __name__ == "__main__":
    unittest.main()
