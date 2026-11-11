"""Opt-in libqi integration check. Run with qi installed in the active environment."""

from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
from threading import Thread
import unittest

import qi


class TestBattery:
    def __init__(self):
        self.charge = 21

    def getBatteryCharge(self):
        self.charge = max(5, self.charge - 8)
        return self.charge


class TestMemory:
    def __init__(self):
        self.samples = iter(([.25, -.5, 9.8], [None, -.5, 9.8], ["bad", "bad", "bad"]))

    def getListData(self, keys):
        expected = [f"Device/SubDeviceList/InertialSensor/Accelerometer{axis}/Sensor/Value" for axis in "XYZ"]
        if keys != expected:
            raise ValueError(f"Unexpected accelerometer keys: {keys}")
        return next(self.samples, [None, None, None])


class QiLoopbackTests(unittest.TestCase):
    def setUp(self):
        self.server = qi.Session()
        self.addCleanup(self.server.close)
        self.server.listenStandalone("tcp://127.0.0.1:0")
        self.server.registerService("ALBattery", TestBattery())
        self.server.registerService("ALMemory", TestMemory())
        port = str(self.server.endpoints()[0]).rsplit(":", 1)[1]
        script = Path(__file__).with_name("robot_diagnostics.py").resolve()
        self.command = [
            sys.executable, "-B", str(script), "--mode", "hardware",
            "--host", "127.0.0.1", "--port", port, "--interval", ".1",
        ]

    def test_cli_reports_successive_battery_values_and_imu_quality(self):
        result = subprocess.run(self.command + ["--count", "3"], capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[HARDWARE] Connected | Battery: 13% [WARNING]", result.stdout)
        self.assertIn("[HARDWARE] Connected | Battery: 5% [CRITICAL]", result.stdout)
        self.assertIn("Charging: Unknown", result.stdout)
        self.assertIn("IMU: OK |", result.stdout)
        self.assertIn("IMU: PARTIAL |", result.stdout)
        self.assertIn("IMU: INVALID |", result.stdout)
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
