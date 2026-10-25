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
    Measurement, MotionState, OperatingMode, PostureOutcome, PostureResult, RobotStatus,
    SampleQuality, SensorSample, TelemetrySnapshot,
)


class SimulationRobotService(RobotService):
    """Maintain independent simulated connection sessions.

    Status is readable while disconnected. Repeated connect calls preserve the
    current session; disconnect clears posture and busy state. Busy reflects
    posture, travel, turning or settling, never telemetry streaming. Use from
    one event loop. Visual position is preserved across connection sessions.
    """

    def __init__(
        self, posture_duration: float = 1.0, *, left_distance: float = 1.0,
        right_distance: float = 1.0, battery_percent: float = 100.0,
        charging: bool = False,
        sensor_overrides: dict[str, SensorSample] | None = None,
        settle_duration: float = 0.35,
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
        if (isinstance(settle_duration, bool) or not isinstance(settle_duration, (int, float))
                or not math.isfinite(settle_duration) or settle_duration <= 0):
            raise ValueError("settle_duration must be positive and finite")
        self._settle_duration = settle_duration
        self._motion_task: asyncio.Task | None = None
        self._motion = MotionState()
        self._transition = None
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
        self._update_pose()
        action = self._action
        motion_task = self._motion_task
        self._action = None
        self._motion_task = None
        self._transition = None
        self._motion = replace(self._motion, action="idle")
        if self._session_end is not None:
            self._session_end.set()
        self._connected = False
        self._busy = False
        self._posture = None
        self._sequence = 0
        tasks = [task for task in (action, motion_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
        self._begin_transition(duration, 1.0 if name == "Sit" else 0.0, 0.0, "posture")
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
                self._transition = None
                self._motion = replace(self._motion, action="idle")

    def _posture_result(self, name: str) -> PostureResult:
        return PostureResult(name, PostureOutcome.COMPLETED_VERIFIED, name,
                             "Verified simulated state only")

    async def _perform_posture(self, name: str, duration: float) -> PostureResult:
        action = asyncio.current_task()
        try:
            await asyncio.sleep(duration)
            self._motion = replace(self._motion, sit_amount=1.0 if name == "Sit" else 0.0,
                                   gait_weight=0.0)
            self._posture = name
            return self._posture_result(name)
        except asyncio.CancelledError:
            if self._action is action:
                self._update_pose()
                self._posture = None
            raise
        finally:
            if self._action is action:
                self._busy = False

    async def get_posture(self) -> str | None:
        """Return the last completed simulated posture, or None if unknown."""
        self._require_connection()
        return self._posture

    def _begin_transition(self, duration, sit_target, gait_target, action):
        self._update_pose()
        self._transition = (asyncio.get_running_loop().time(), duration,
                            self._motion.sit_amount, sit_target,
                            self._motion.gait_weight, gait_target)
        self._motion = replace(self._motion, action=action)

    def _update_pose(self):
        if self._transition is None:
            return
        start, duration, sit_from, sit_to, gait_from, gait_to = self._transition
        fraction = min(1.0, max(0.0, (asyncio.get_running_loop().time() - start) / duration))
        blend = 0.5 - 0.5 * math.cos(math.pi * fraction)
        self._motion = replace(self._motion,
                               sit_amount=sit_from + (sit_to - sit_from) * blend,
                               gait_weight=gait_from + (gait_to - gait_from) * blend)

    async def get_motion_capabilities(self) -> tuple[str, ...]:
        """Supported visual continuous motions, without hardware claims."""
        return ("forward", "backward", "turn_left", "turn_right")

    async def get_motion_state(self) -> MotionState:
        """Return a fresh visual state, also available after disconnect."""
        self._update_pose()
        return self._motion

    async def start_motion(self, command: str, speed: float = 0.5) -> MotionState:
        """Start continuous visual movement from verified simulated Stand.

        Rates at full speed: 0.8 scene units/sec and 60 degrees/sec. This is a
        procedural visual gait, not a physically balanced trajectory. Competing
        commands raise RobotBusyError; callers must Stop before changing motion.
        """
        self._require_connection()
        if command not in await self.get_motion_capabilities():
            raise UnsupportedOperationError(f"Unsupported simulated motion: {command!r}")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not math.isfinite(speed) or not 0 < speed <= 1):
            raise ValueError("speed must be finite and greater than 0, at most 1")
        if self._busy:
            raise RobotBusyError("Stop the active operation before starting another")
        if self._posture != "Stand":
            raise ValueError("Request Stand before walking or turning")
        self._busy = True
        self._motion = replace(self._motion, action=command, gait_weight=1.0, sit_amount=0.0)
        self._motion_task = asyncio.create_task(self._run_motion(command, speed))
        return self._motion

    async def _run_motion(self, command, speed):
        loop = asyncio.get_running_loop()
        previous = loop.time()
        try:
            while True:
                await asyncio.sleep(0.02)
                now = loop.time()
                # Avoid jumping after a stalled worker; this is a visual model.
                dt = min(0.1, max(0, now - previous))
                previous = now
                state = self._motion
                direction = -1 if command == "backward" else 1
                phase = state.gait_phase + direction * dt * 5 * speed
                if command.startswith("turn_"):
                    sign = -1 if command == "turn_left" else 1
                    self._motion = replace(state, heading=state.heading + sign * dt * 60 * speed,
                                           gait_phase=phase)
                else:
                    angle = math.radians(state.heading)
                    distance = direction * dt * .8 * speed
                    self._motion = replace(state, x=state.x + distance * math.sin(angle),
                                           z=state.z + distance * math.cos(angle), gait_phase=phase)
        except asyncio.CancelledError:
            raise

    async def stop_motion(self) -> MotionState:
        """Halt travel now; settle walking legs to Stand, or freeze posture work.

        Repeated Stop during settling awaits the same operation. Disconnect
        cancels settling and leaves posture unknown. Idle Stop is a no-op.
        """
        self._require_connection()
        self._update_pose()
        if self._motion.action == "settling" and self._action is not None:
            await asyncio.shield(self._action)
            return await self.get_motion_state()
        if self._motion_task is not None:
            task = self._motion_task
            self._motion_task = None
            task.cancel()
            # Claim the settling state before yielding, preventing new commands.
            self._begin_transition(self._settle_duration, 0.0, 0.0, "settling")
            action = asyncio.create_task(self._settle(task))
            self._action = action
            await asyncio.shield(action)
        elif self._action is not None:
            action = self._action
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
        return await self.get_motion_state()

    async def _settle(self, movement):
        action = asyncio.current_task()
        try:
            await asyncio.gather(movement, return_exceptions=True)
            await asyncio.sleep(self._settle_duration)
            self._motion = replace(self._motion, gait_weight=0.0, sit_amount=0.0, action="idle")
            self._posture = "Stand"
        except asyncio.CancelledError:
            if self._action is action:
                self._update_pose()
                self._posture = None
            raise
        finally:
            if self._action is action:
                self._action = None
                self._transition = None
                self._busy = False
                self._motion = replace(self._motion, action="idle")

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
