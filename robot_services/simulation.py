"""In-process simulation backend with connection and timed posture operations.

No sockets, robot SDK or Qt objects are created. Readings are test fixtures,
not physics or measurements of the viewer scene.
"""

import asyncio
import math
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import replace
from collections.abc import AsyncIterator

from .errors import RobotBusyError, RobotConnectionError, UnsupportedOperationError
from .interface import RobotService
from .models import (
    Measurement, OperatingMode, PostureOutcome, PostureResult, RobotStatus,
    SampleQuality, SensorSample, TelemetrySnapshot,
)


class SimulationRobotService(RobotService):
    """Maintain independent simulated connection sessions.

    Status is readable while disconnected. Repeated connect calls preserve the
    current session; disconnect clears posture and busy state. Busy will reflect
    posture activity only, never telemetry streaming. Use from one event loop.
    """

    def __init__(
        self, posture_duration: float = 1.0, *, left_distance: float = 1.0,
        right_distance: float = 1.0, battery_percent: float = 100.0,
        charging: bool = False,
        sensor_overrides: dict[str, SensorSample] | None = None,
    ) -> None:
        """Configure constant fixtures; values do not drain or track scene motion.

        Distances are nonnegative meters; charge is 0..100 percent. Overrides
        replace complete groups (imu/ultrasonic/fsr/joints/battery), supporting
        partial/unavailable failure fixtures. Their readings are copied and
        acquisition times refreshed per snapshot. FSR/joints default unavailable.
        Posture duration is seconds at speed 1, not a physical timing model.
        """
        if (isinstance(posture_duration, bool)
                or not isinstance(posture_duration, (int, float))
                or not math.isfinite(posture_duration) or posture_duration <= 0):
            raise ValueError("posture_duration must be positive and finite")
        self._posture_duration = posture_duration
        self._action: asyncio.Task | None = None
        self._connected = False
        self._busy = False
        self._posture: str | None = None
        self._sequence = 0
        self._session_end: asyncio.Event | None = None
        for name, value in (("left_distance", left_distance),
                            ("right_distance", right_distance),
                            ("battery_percent", battery_percent)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be nonnegative and finite")
        if battery_percent > 100 or not isinstance(charging, bool):
            raise ValueError("Charge must be 0..100 and charging must be bool")
        now = datetime.now(timezone.utc)
        def valid(values):
            return SensorSample(SampleQuality.VALID, now, values)
        self._sensors = {
            "imu": valid({
                "acceleration_x": Measurement(0.0, "m/s^2"),
                "acceleration_y": Measurement(0.0, "m/s^2"),
                "acceleration_z": Measurement(9.80665, "m/s^2"),
                "angular_velocity_x": Measurement(0.0, "rad/s"),
                "angular_velocity_y": Measurement(0.0, "rad/s"),
                "angular_velocity_z": Measurement(0.0, "rad/s"),
                "roll": Measurement(0.0, "rad"),
                "pitch": Measurement(0.0, "rad"),
                "yaw": Measurement(0.0, "rad"),
            }),
            "ultrasonic": valid({"left": Measurement(left_distance, "m"),
                                  "right": Measurement(right_distance, "m")}),
            "battery": valid({"charge": Measurement(battery_percent, "%"),
                               "charging": Measurement(int(charging), "boolean")}),
            "fsr": SensorSample(SampleQuality.UNAVAILABLE,
                                error="Simulated FSR values/units not defined"),
            "joints": SensorSample(SampleQuality.UNAVAILABLE,
                                   error="Simulated joint mapping not defined"),
        }
        for name, sample in (sensor_overrides or {}).items():
            if name not in self._sensors or not isinstance(sample, SensorSample):
                raise ValueError("Overrides require known sensor groups and SensorSample values")
            sample.__post_init__()
            self._sensors[name] = deepcopy(sample)

    async def connect(self) -> RobotStatus:
        """Connect locally, preserving an existing session; return a snapshot."""
        if not self._connected:
            self._sequence = 0
            self._session_end = asyncio.Event()
            self._connected = True
        return await self.get_status()

    async def disconnect(self) -> None:
        """Clear state and cancel/await active posture work; safe repeatedly."""
        action = self._action
        self._action = None
        if self._session_end is not None:
            self._session_end.set()
        self._connected = False
        self._busy = False
        self._posture = None
        self._sequence = 0
        if action is not None:
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)

    async def get_status(self) -> RobotStatus:
        """Return a fresh immutable status snapshot in either connection state."""
        return RobotStatus(
            mode=OperatingMode.SIMULATION,
            connected=self._connected,
            busy=self._busy,
            posture=self._posture,
        )

    def _require_connection(self) -> None:
        if not self._connected:
            raise RobotConnectionError("Simulation backend is disconnected")

    async def set_posture(self, name: str, speed: float) -> PostureResult:
        """Await Stand/Sit completion at speed 0 < speed <= 1.

        Slower speeds take duration/speed seconds. Concurrent requests raise
        RobotBusyError. Caller cancellation propagates and makes posture unknown.
        Verification confirms simulated state only, never physical motion.
        """
        self._require_connection()
        if name not in ("Stand", "Sit"):
            raise UnsupportedOperationError(f"Unsupported simulated posture: {name!r}")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not math.isfinite(speed) or not 0 < speed <= 1):
            raise ValueError("speed must be finite and greater than 0, at most 1")
        duration = self._posture_duration / speed
        if not math.isfinite(duration):
            raise ValueError("speed produces an unrepresentable duration")
        if self._busy:
            raise RobotBusyError("A simulated posture operation is already running")
        if self._posture == name:
            return self._posture_result(name)
        self._busy = True
        action = asyncio.create_task(self._perform_posture(name, duration))
        self._action = action
        try:
            return await action
        finally:
            # Also covers cancellation before the child coroutine first runs.
            if self._action is action:
                if action.cancelled():
                    self._posture = None
                self._action = None
                self._busy = False

    def _posture_result(self, name: str) -> PostureResult:
        return PostureResult(name, PostureOutcome.COMPLETED_VERIFIED, name,
                             "Verified simulated state only")

    async def _perform_posture(self, name: str, duration: float) -> PostureResult:
        action = asyncio.current_task()
        try:
            await asyncio.sleep(duration)
            self._posture = name
            return self._posture_result(name)
        except asyncio.CancelledError:
            if self._action is action:
                self._posture = None
            raise
        finally:
            if self._action is action:
                self._busy = False

    async def get_posture(self) -> str | None:
        """Return the last completed simulated posture, or None if unknown."""
        self._require_connection()
        return self._posture

    async def read_telemetry(self) -> TelemetrySnapshot:
        """Return freshly timestamped constant simulated readings; offline raises.

        IMU fixture axes use a nominal stationary +Z-up frame with acceleration
        including gravity; roll/pitch/yaw are zero. This is not an SDK mapping.
        Sequence numbers are shared by manual reads and all streams in a session.
        """
        self._require_connection()
        sampled_at = datetime.now(timezone.utc)
        groups = {
            name: replace(sample, values=deepcopy(sample.values),
                          sampled_at=(sampled_at if sample.sampled_at is not None else None))
            for name, sample in self._sensors.items()
        }
        snapshot = TelemetrySnapshot(
            sequence=self._sequence, created_at=datetime.now(timezone.utc),
            mode=OperatingMode.SIMULATION, source="simulation:constant-fixtures",
            **groups,
        )
        self._sequence += 1
        return snapshot

    def stream_telemetry(self, interval: float) -> AsyncIterator[TelemetrySnapshot]:
        """Return a stream bound to the current session; validate on creation.

        First snapshot is immediate. Subsequent snapshots wait at least interval
        seconds after consumer resumption (no buffering/catch-up). Disconnect
        wakes pending reads and ends old streams even after reconnect. Cancellation
        propagates; consumers must aclose() abandoned streams. No producer task
        runs in the background and streaming does not change busy state.
        """
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or interval <= 0):
            raise ValueError("interval must be positive and finite")
        self._require_connection()
        session_end = self._session_end

        async def snapshots():
            while not session_end.is_set():
                yield await self.read_telemetry()
                # Async timers may wake up one clock tick early on Windows.
                # Recheck the deadline to honor minimum sample spacing.
                loop = asyncio.get_running_loop()
                deadline = loop.time() + interval
                while not session_end.is_set():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(session_end.wait(), timeout=remaining)
                    except TimeoutError:
                        pass

        return snapshots()
