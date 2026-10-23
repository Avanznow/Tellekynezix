"""Hardware-free checks for the common interface and telemetry schema."""

from datetime import datetime, timedelta, timezone
import inspect
import json
import unittest

from robot_services.errors import TelemetryValidationError
from robot_services.interface import RobotService
from robot_services.models import (
    Measurement, OperatingMode, PostureOutcome, PostureResult,
    SampleQuality, SensorSample, TelemetrySnapshot,
)


class ServiceContractTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)

    def snapshot(self, **kwargs):
        return TelemetrySnapshot(
            sequence=0, created_at=self.now, mode=OperatingMode.SIMULATION,
            source="simulation:test", **kwargs
        )

    def test_missing_sensors_are_explicit_and_serializable(self):
        payload = json.loads(self.snapshot().to_json())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["mode"], "simulation")
        self.assertTrue(payload["created_at"].endswith("Z"))
        for group in ("imu", "ultrasonic", "fsr", "joints", "battery"):
            self.assertEqual(payload[group]["quality"], "unavailable")
            self.assertIsNone(payload[group]["sampled_at"])
            self.assertEqual(payload[group]["values"], {})

    def test_one_failed_sensor_preserves_other_readings(self):
        imu = SensorSample(SampleQuality.VALID, self.now,
                           {"acceleration_x": Measurement(0.0, "m/s^2")})
        snapshot = self.snapshot(imu=imu)
        payload = snapshot.to_dict()
        self.assertEqual(payload["imu"]["values"]["acceleration_x"]["value"], 0.0)
        self.assertEqual(payload["fsr"]["quality"], "unavailable")

    def test_partial_sample_preserves_null_and_acquisition_time(self):
        acquired = self.now - timedelta(seconds=1)
        sample = SensorSample(SampleQuality.PARTIAL, acquired, {
            "left": Measurement(0.5, "m"), "right": Measurement(None, "m")
        }, "Right sensor unavailable")
        payload = json.loads(self.snapshot(ultrasonic=sample).to_json())
        self.assertIsNone(payload["ultrasonic"]["values"]["right"]["value"])
        self.assertNotEqual(payload["ultrasonic"]["sampled_at"], payload["created_at"])

    def test_nonfinite_and_malformed_measurements_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, "1"):
            with self.subTest(value=value), self.assertRaises(TelemetryValidationError):
                Measurement(value, "m")
        with self.assertRaises(TelemetryValidationError):
            Measurement(1, "")

    def test_quality_cannot_misrepresent_readings(self):
        numeric = {"x": Measurement(1, "m")}
        for quality, timestamp, values, error in (
            (SampleQuality.VALID, self.now, {}, None),
            (SampleQuality.VALID, None, numeric, None),
            (SampleQuality.VALID, self.now, {"x": Measurement(None, "m")}, None),
            (SampleQuality.PARTIAL, self.now, numeric, None),
            (SampleQuality.UNAVAILABLE, self.now, numeric, "unavailable"),
            (SampleQuality.INVALID, self.now, {}, None),
        ):
            with self.subTest(quality=quality), self.assertRaises(TelemetryValidationError):
                SensorSample(quality, timestamp, values, error)

    def test_timestamp_and_sequence_validation(self):
        with self.assertRaises(TelemetryValidationError):
            TelemetrySnapshot(0, datetime.now(), OperatingMode.HARDWARE, "robot")
        with self.assertRaises(TelemetryValidationError):
            TelemetrySnapshot(-1, self.now, OperatingMode.HARDWARE, "robot")
        future = SensorSample(SampleQuality.VALID, self.now + timedelta(seconds=1),
                              {"x": Measurement(1, "m")})
        with self.assertRaises(TelemetryValidationError):
            self.snapshot(imu=future)

    def test_verified_posture_requires_matching_observation(self):
        accepted = PostureResult("Stand", PostureOutcome.ACCEPTED)
        self.assertIsNone(accepted.observed_posture)
        with self.assertRaises(ValueError):
            PostureResult("Stand", PostureOutcome.COMPLETED_VERIFIED)
        result = PostureResult("Stand", PostureOutcome.COMPLETED_VERIFIED, "Stand")
        self.assertEqual(result.observed_posture, "Stand")

    def test_interface_requires_backend_implementation(self):
        with self.assertRaises(TypeError):
            RobotService()
        for name in ("connect", "disconnect", "get_status", "set_posture",
                     "get_posture", "read_telemetry"):
            self.assertTrue(inspect.iscoroutinefunction(getattr(RobotService, name)))
        self.assertIn("stream_telemetry", RobotService.__abstractmethods__)

    def test_snapshot_rejects_readings_mutated_after_validation(self):
        sample = SensorSample(SampleQuality.VALID, self.now,
                              {"x": Measurement(1, "m")})
        sample.values.clear()
        with self.assertRaises(TelemetryValidationError):
            self.snapshot(imu=sample)

    def test_serialization_revalidates_mutable_readings(self):
        sample = SensorSample(SampleQuality.VALID, self.now,
                              {"x": Measurement(1, "m")})
        snapshot = self.snapshot(imu=sample)
        sample.values["x"] = "malformed"
        with self.assertRaises(TelemetryValidationError):
            snapshot.to_json()


if __name__ == "__main__":
    unittest.main()
