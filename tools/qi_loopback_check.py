"""Opt-in libqi integration check. Run with qi installed in the active environment."""

import asyncio
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
from threading import Thread
import unittest

import qi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_services.models import PostureOutcome
from robot_services.qi_service import QiRobotService


class TestPosture:
    def __init__(self):
        self.current = "Unknown"

    def goToPosture(self, name, speed):
        if speed != .5:
            return False
        if name == "Stand":
            self.current = "Stand"
        return True

    def getPosture(self):
        return self.current


class TestBattery:
    def __init__(self):
        self.charge = 21

    def getBatteryCharge(self):
        self.charge = max(5, self.charge - 8)
        return self.charge


class TestMemory:
    def __init__(self):
        self.imu = iter(([.25, -.5, 9.8], [None, -.5, 9.8], ["bad", "bad", "bad"]))
        self.sonar = iter(([.4, .6], [5.0, .5], [0, -1]))
        self.fsr = iter(([0, .1, .2, .3, .4, .5, .6, .7], [None] * 4 + [.4] * 4, [-1] * 8))
        self.joints = iter(([.25] * 25, [None] + [.25] * 24, ["bad"] * 25))

    def getListData(self, keys):
        expected = [f"Device/SubDeviceList/InertialSensor/Accelerometer{axis}/Sensor/Value" for axis in "XYZ"]
        if keys == expected:
            return next(self.imu, [None, None, None])
        if keys == [f"Device/SubDeviceList/US/{side}/Sensor/Value" for side in ("Left", "Right")]:
            return next(self.sonar, [None, None])
        fsr_keys = [
            f"Device/SubDeviceList/{foot}/FSR/{corner}/Sensor/Value"
            for foot in ("LFoot", "RFoot")
            for corner in ("FrontLeft", "FrontRight", "RearLeft", "RearRight")
        ]
        if keys == fsr_keys:
            return next(self.fsr, [None] * 8)
        if len(set(keys)) == 25 and all(key.endswith("/Position/Sensor/Value") for key in keys):
            return next(self.joints, [None] * 25)
        raise ValueError(f"Unexpected ALMemory keys: {keys}")


class QiLoopbackTests(unittest.TestCase):
    def setUp(self):
        self.server = qi.Session()
        self.addCleanup(self.server.close)
        self.server.listenStandalone("tcp://127.0.0.1:0")
        self.server.registerService("ALBattery", TestBattery())
        self.server.registerService("ALMemory", TestMemory())
        self.port = int(str(self.server.endpoints()[0]).rsplit(":", 1)[1])
        script = Path(__file__).with_name("robot_diagnostics.py").resolve()
        self.command = [
            sys.executable, "-B", str(script), "--mode", "hardware",
            "--host", "127.0.0.1", "--port", str(self.port), "--interval", ".1",
        ]

    def test_posture_completion_requires_matching_sdk_observation(self):
        self.server.registerService("ALRobotPosture", TestPosture())

        async def check():
            service = QiRobotService("127.0.0.1", port=self.port, enable_posture=True)
            await service.connect()
            try:
                self.assertIsNone(await service.get_posture())
                stand = await service.set_posture("Stand", .5)
                self.assertIs(stand.outcome, PostureOutcome.COMPLETED_VERIFIED)
                self.assertEqual(stand.observed_posture, "Stand")
                sit = await service.set_posture("Sit", .5)
                self.assertIs(sit.outcome, PostureOutcome.UNVERIFIED)
                self.assertEqual(sit.observed_posture, "Stand")
                rejected = await service.set_posture("Sit", .25)
                self.assertIs(rejected.outcome, PostureOutcome.FAILED)
            finally:
                await service.disconnect()

        asyncio.run(check())

    def test_cli_reports_battery_and_joint_position_quality(self):
        result = subprocess.run(self.command + ["--count", "3"], capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[HARDWARE] Connected | Battery: 13% [WARNING]", result.stdout)
        self.assertIn("[HARDWARE] Connected | Battery: 5% [CRITICAL]", result.stdout)
        self.assertIn("Charging: Unknown", result.stdout)
        self.assertIn("IMU: OK |", result.stdout)
        self.assertIn("IMU: PARTIAL |", result.stdout)
        self.assertIn("IMU: INVALID |", result.stdout)
        self.assertIn("Ultrasonic: OK |", result.stdout)
        self.assertIn("Ultrasonic: PARTIAL |", result.stdout)
        self.assertIn("Ultrasonic: INVALID |", result.stdout)
        self.assertIn("FSR: OK |", result.stdout)
        self.assertIn("FSR: PARTIAL |", result.stdout)
        self.assertIn("FSR: INVALID |", result.stdout)
        self.assertIn(" | Joints: OK | ", result.stdout)
        self.assertIn(" | Joints: PARTIAL | ", result.stdout)
        self.assertIn(" | Joints: INVALID | ", result.stdout)
        self.assertEqual(result.stdout.count("Connected | Battery:"), 3)

    def run_until_first_report(self, action):
        with subprocess.Popen(self.command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as client:
            lines = Queue()

            def read_output():
                for line in client.stdout:
                    lines.put(line)
                lines.put(None)

            reader = Thread(target=read_output, daemon=True)
            reader.start()
            output = []
            try:
                while True:
                    line = lines.get(timeout=10)
                    self.assertIsNotNone(line, "".join(output))
                    output.append(line)
                    if "[HARDWARE] Connected | Battery:" in line:
                        break
                action(client)
                client.wait(timeout=10)
            finally:
                if client.poll() is None:
                    client.kill()
                    client.wait(timeout=5)
                reader.join(timeout=5)
            while not lines.empty():
                line = lines.get_nowait()
                if line is not None:
                    output.append(line)
            return client.returncode, "".join(output)

    def test_cli_reports_connection_loss(self):
        code, output = self.run_until_first_report(lambda _: self.server.close())
        self.assertEqual(code, 1, output)
        self.assertIn("ERROR: Monitoring failed:", output)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Uses Linux SIGINT delivery")
    def test_ctrl_c_exits_without_cleanup_error(self):
        code, output = self.run_until_first_report(lambda client: client.send_signal(signal.SIGINT))
        self.assertEqual(code, 130, output)
        self.assertIn("Monitoring stopped by user", output)
        self.assertNotIn("cleanup failed", output)
        self.assertNotIn("Disconnect failed", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
