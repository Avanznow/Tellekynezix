"""Version 1 service data contract, independent of Qt and robot SDKs.

Sensor values are named scalar measurements with explicit units. Adapters must
verify SDK units before converting readings. Unknown measurements use None,
never a fabricated zero. Hardware and simulation use the same schema.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math

from .errors import TelemetryValidationError


class OperatingMode(str, Enum):
    SIMULATION = "simulation"
    HARDWARE = "hardware"


class SampleQuality(str, Enum):
    VALID = "valid"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class PostureOutcome(str, Enum):
    ACCEPTED = "accepted"
    COMPLETED_VERIFIED = "completed_verified"
    FAILED = "failed"
    UNVERIFIED = "unverified"


def _utc(value: datetime) -> None:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise TelemetryValidationError("Timestamps must be timezone-aware UTC")


@dataclass(frozen=True)
class Measurement:
    """A finite numeric value in explicit units, or None when missing."""

    value: float | None
    unit: str

    def __post_init__(self):
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise TelemetryValidationError("Measurement unit is required")
        if self.value is not None and (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
        ):
            raise TelemetryValidationError("Measurement must be finite or None")


@dataclass(frozen=True)
class SensorSample:
    """One sensor group; timestamp is acquisition time, None if unavailable.

    Named values allow SDK adapters to describe available axes, feet or joints.
    PARTIAL means some requested readings are missing; error explains why.
    INVALID and UNAVAILABLE samples cannot contain usable numeric values.
    """

    quality: SampleQuality
    sampled_at: datetime | None = None
    values: dict[str, Measurement] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self):
        if not isinstance(self.quality, SampleQuality):
            raise TelemetryValidationError("Unknown sample quality")
        if self.sampled_at is not None:
            _utc(self.sampled_at)
        for name, measurement in self.values.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(measurement, Measurement):
                raise TelemetryValidationError("Samples require named Measurement values")
        usable = any(item.value is not None for item in self.values.values())
        missing = any(item.value is None for item in self.values.values())
        if self.quality is SampleQuality.VALID:
            if not usable or missing or self.sampled_at is None or self.error is not None:
                raise TelemetryValidationError("Valid samples require timestamp and complete values")
        elif self.quality is SampleQuality.PARTIAL:
            if not usable or self.sampled_at is None or not self.error:
                raise TelemetryValidationError("Partial samples require readings, timestamp and reason")
        elif usable or not self.error:
            raise TelemetryValidationError("Unavailable/invalid samples need a reason and no usable values")


def _missing() -> SensorSample:
    return SensorSample(SampleQuality.UNAVAILABLE, error="Not sampled")


@dataclass(frozen=True)
class RobotStatus:
    """Backend status; posture None means unknown, not standing or sitting."""

    mode: OperatingMode
    connected: bool
    busy: bool
    posture: str | None = None


@dataclass(frozen=True)
class PostureResult:
    """Command outcome; only COMPLETED_VERIFIED confirms the observed posture."""

    requested_posture: str
    outcome: PostureOutcome
    observed_posture: str | None = None
    message: str | None = None

    def __post_init__(self):
        if not self.requested_posture or not isinstance(self.outcome, PostureOutcome):
            raise ValueError("A requested posture and known outcome are required")
        if self.outcome is PostureOutcome.COMPLETED_VERIFIED and self.observed_posture != self.requested_posture:
            raise ValueError("Verified completion requires a matching observed posture")


@dataclass(frozen=True)
class TelemetrySnapshot:
    """JSON-ready snapshot with independent acquisition times for sensor groups.

    created_at is assembly time, not a claim of simultaneous acquisition.
    source identifies the backend/device, never credentials. Sequence is a
    nonnegative counter within one connection session. Schema version is 1.
    Battery charging state is represented by a named 0/1 measurement with unit
    'boolean'; adapters may omit it when unsupported. Unit names and measurement
    keys must be documented by adapters before real sensor integration.
    """

    sequence: int
    created_at: datetime
    mode: OperatingMode
    source: str
    imu: SensorSample = field(default_factory=_missing)
    ultrasonic: SensorSample = field(default_factory=_missing)
    fsr: SensorSample = field(default_factory=_missing)
    joints: SensorSample = field(default_factory=_missing)
    battery: SensorSample = field(default_factory=_missing)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self):
        _utc(self.created_at)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise TelemetryValidationError("Sequence must be a nonnegative integer")
        if not isinstance(self.mode, OperatingMode) or not isinstance(self.source, str) or not self.source.strip():
            raise TelemetryValidationError("Mode and source are required")
        for sample in (self.imu, self.ultrasonic, self.fsr, self.joints, self.battery):
            if not isinstance(sample, SensorSample):
                raise TelemetryValidationError("Sensor groups must be SensorSample instances")
            sample.__post_init__()
            if sample.sampled_at is not None and sample.sampled_at > self.created_at:
                raise TelemetryValidationError("Acquisition time cannot follow assembly time")

    def to_dict(self) -> dict:
        """Return JSON-compatible values, preserving missing values as null."""
        # Frozen dataclasses still contain mutable dictionaries. Revalidate at
        # the serialization boundary rather than exporting malformed readings.
        self.__post_init__()
        def convert(value):
            if isinstance(value, datetime):
                return value.isoformat().replace("+00:00", "Z")
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value
        return convert(asdict(self))

    def to_json(self) -> str:
        """Serialize without nonstandard NaN or Infinity JSON values."""
        return json.dumps(self.to_dict(), allow_nan=False)
