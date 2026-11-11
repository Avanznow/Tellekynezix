"""Libqi service adapter. Importing this module never loads the SDK.

Sensor reads use ALBattery and ALMemory. The native SDK
must support _async=True and qi.Future callbacks. Tests inject sessions instead
of importing qi or making network connections. Physical compatibility is unverified.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
import importlib
import ipaddress
import math
import re

from .errors import (
    RobotBusyError, RobotConnectionError, RobotServiceError, TelemetryValidationError,
    UnsupportedOperationError,
)
from .interface import RobotService
from .models import (
    Measurement, OperatingMode, PostureOutcome, PostureResult, RobotStatus, SampleQuality,
    SensorSample, TelemetrySnapshot,
)


_ACCELEROMETER_KEYS = {
    f"acceleration_{axis.lower()}": f"Device/SubDeviceList/InertialSensor/Accelerometer{axis}/Sensor/Value"
    for axis in "XYZ"
}
_SONAR_KEYS = {
    side.lower(): f"Device/SubDeviceList/US/{side}/Sensor/Value"
    for side in ("Left", "Right")
}
_SONAR_NO_ECHO_METRES = 5.0
_JOINT_NAMES = (
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
)
_JOINT_KEYS = {name: f"Device/SubDeviceList/{name}/Position/Sensor/Value" for name in _JOINT_NAMES}
_JOINT_UNITS = {name: "ratio" if name in ("LHand", "RHand") else "rad" for name in _JOINT_NAMES}

_FSR_KEYS = {
    f"{side}_{position}_{corner}": (
        f"Device/SubDeviceList/{foot}/FSR/{position.title()}{corner.title()}/Sensor/Value"
    )
    for side, foot in (("left", "LFoot"), ("right", "RFoot"))
    for position in ("front", "rear")
    for corner in ("left", "right")
}


class _QiCallError(RobotServiceError):
    """An SDK call failed; the caller checks whether the session was lost."""


class QiRobotService(RobotService):
    """Injectable, read-only hardware adapter, confined to one asyncio event loop.

    host is a hostname/IP (without scheme, credentials or brackets); port defaults
    to 9559. timeout bounds each SDK future in seconds. session_factory must return
    a new qi.Session-compatible object promptly; None lazily imports qi.Session.
    Constructing the service does not connect. Concurrent connect calls share one
    connection attempt. Disconnect invalidates pending work before closing its
    session; reconnect creates a fresh session. Native cancellation is best effort.
    """

    def __init__(self, host: str, port: int = 9559, timeout: float = 5.0,
                 *, session_factory: Callable[[], object] | None = None,
                 enable_posture: bool = False):
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a hostname or IP address")
        if ":" in host:
            try:
                address = ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError("Invalid IPv6 address") from exc
            endpoint_host = f"[{address}]"
        elif re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host):
            endpoint_host = host
        else:
            raise ValueError("host must not contain a scheme, credentials, path or whitespace")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer from 1 to 65535")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive and finite")
        if session_factory is not None and not callable(session_factory):
            raise ValueError("session_factory must be callable")
        self.endpoint = f"tcp://{endpoint_host}:{port}"
        self.timeout = timeout
        self._factory = session_factory
        if not isinstance(enable_posture, bool):
            raise ValueError("enable_posture must be a boolean")
        self._enable_posture = enable_posture
        self._posture_request = None
        self._session = None
        self._session_end = asyncio.Event()
        self._battery = None
        self._memory = None
        self._connected = False
        self._sequence = 0
        self._connect_lock = asyncio.Lock()
        self._pending = {}

    @staticmethod
    def _cancel(future):
        try:
            future.cancel()
        except Exception:
            pass  # Native cancellation is advisory; late replies are ignored.

    async def _call(self, session, method, *args):
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        native = None

        def completed(done):
            if waiter.done():
                return
            try:
                # The callback runs only when ready; a zero timeout never blocks.
                value = done.value(0)
            except Exception as exc:
                waiter.set_exception(_QiCallError(str(exc) or type(exc).__name__))
            else:
                waiter.set_result(value)

        def callback(done):
            try:
                loop.call_soon_threadsafe(completed, done)
            except RuntimeError:
                pass  # The owning loop has already shut down.

        try:
            native = method(*args, _async=True)
            self._pending[waiter] = (session, native)
            native.addCallback(callback)
            return await asyncio.wait_for(waiter, self.timeout)
        except TimeoutError as exc:
            if native is not None:
                self._cancel(native)
            raise _QiCallError(f"SDK call timed out after {self.timeout:g} seconds") from exc
        except asyncio.CancelledError:
            if native is not None:
                self._cancel(native)
            raise
        except _QiCallError:
            raise
        except Exception as exc:
            if native is not None:
                self._cancel(native)
            raise _QiCallError(str(exc) or type(exc).__name__) from exc
        finally:
            self._pending.pop(waiter, None)
            if not waiter.done():
                waiter.cancel()

    def _check_session(self, session):
        if self._session is not session:
            raise RobotConnectionError("Session was disconnected or replaced")

    def _require_connection(self):
        if self._session is None or not self._connected:
            raise RobotConnectionError("Hardware backend is disconnected")
        return self._session

    def _invalidate(self, session):
        for waiter, (owner, native) in list(self._pending.items()):
            if owner is session:
                self._cancel(native)
                if not waiter.done():
                    waiter.set_exception(_QiCallError("Session disconnected"))

    async def connect(self) -> RobotStatus:
        """Explicitly connect; fail clearly if the optional SDK is unavailable."""
        async with self._connect_lock:
            if self._session is not None:
                status = await self.get_status()
                if status.connected:
                    return status
                await self.disconnect()
            try:
                factory = self._factory
                if factory is None:
                    try:
                        factory = importlib.import_module("qi").Session
                    except (ImportError, AttributeError) as exc:
                        raise RobotConnectionError(
                            "Compatible qi SDK is not available; install it in a verified runtime "
                            "or inject a session factory for tests"
                        ) from exc
                session = factory()
            except RobotConnectionError:
                raise
            except Exception as exc:
                raise RobotConnectionError(f"Cannot create SDK session: {exc}") from exc
            self._session = session
            self._session_end = asyncio.Event()
            self._battery = None
            self._memory = None
            self._connected = False
            self._sequence = 0
            try:
                await self._call(session, session.connect, self.endpoint)
                self._check_session(session)
                connected = await self._call(session, session.isConnected)
                self._check_session(session)
                if connected is not True:
                    raise RobotConnectionError("SDK did not confirm a connected session")
                self._connected = True
                return RobotStatus(OperatingMode.HARDWARE, True, False)
            except (Exception, asyncio.CancelledError) as exc:
                if self._session is session:
                    try:
                        await self.disconnect()
                    except RobotConnectionError as cleanup:
                        exc.add_note(f"Session cleanup also failed: {cleanup}")
                if isinstance(exc, (RobotConnectionError, asyncio.CancelledError)):
                    raise
                raise RobotConnectionError(f"SDK connection failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Invalidate reads immediately, then close; repeat calls are harmless.

        A close timeout still leaves local status disconnected and is reported
        as RobotConnectionError. It cannot guarantee that remote cleanup finished.
        """
        session = self._session
        self._session_end.set()
        self._session = None
        self._posture_request = None
        self._battery = None
        self._memory = None
        self._connected = False
        if session is None:
            return
        self._invalidate(session)
        try:
            await self._call(session, session.close)
        except _QiCallError as exc:
            raise RobotConnectionError(f"SDK session close failed: {exc}") from exc

    async def get_status(self) -> RobotStatus:
        """Check session connectivity; posture and actuation are not queried."""
        session = self._session
        if session is None or not self._connected:
            return RobotStatus(OperatingMode.HARDWARE, False, False)
        try:
            connected = await self._call(session, session.isConnected)
            self._check_session(session)
            if not isinstance(connected, bool):
                raise _QiCallError("SDK returned an invalid connection status")
            self._connected = connected
            return RobotStatus(OperatingMode.HARDWARE, connected,
                               connected and self._posture_request is not None)
        except _QiCallError as exc:
            if self._session is session:
                self._connected = False
            raise RobotConnectionError(f"SDK status check failed: {exc}") from exc

    async def _read_memory_sample(
        self, session, group: str, keys: dict[str, str], units: str | dict[str, str], *,
        sonar: bool = False, minimum: float | None = None,
    ) -> SensorSample:
        try:
            if self._memory is None:
                memory = await self._call(session, session.service, "ALMemory")
                self._check_session(session)
                self._memory = memory
            raw = await self._call(session, self._memory.getListData, list(keys.values()))
        except (_QiCallError, AttributeError) as exc:
            self._check_session(session)
            if not (await self.get_status()).connected:
                raise RobotConnectionError(f"Robot disconnected during {group} read") from exc
            self._memory = None
            return SensorSample(SampleQuality.UNAVAILABLE, error=f"{group} unavailable: {exc}")

        self._check_session(session)
        sampled_at = datetime.now(timezone.utc)
        if not isinstance(raw, (list, tuple)) or len(raw) != len(keys):
            return SensorSample(SampleQuality.INVALID, sampled_at, error=f"Expected {len(keys)} {group} values")

        values = {}
        problems = []
        invalid = False
        for name, value in zip(keys, raw):
            unit = units if isinstance(units, str) else units[name]
            try:
                measurement = Measurement(value, unit)
                if sonar and value is not None and value <= 0:
                    raise TelemetryValidationError("Sonar distance must be positive")
                if minimum is not None and value is not None and value < minimum:
                    raise TelemetryValidationError(f"{group} reading below minimum")
                if unit == "ratio" and value is not None and not 0 <= value <= 1:
                    raise TelemetryValidationError("Hand aperture must be between 0 and 1")
            except (TelemetryValidationError, OverflowError):
                invalid = True
                problems.append(f"{name} invalid")
                measurement = Measurement(None, unit)
            else:
                if value is None:
                    problems.append(f"{name} missing")
                elif sonar and value >= _SONAR_NO_ECHO_METRES:
                    problems.append(f"{name} no echo")
                    measurement = Measurement(None, unit)
            values[name] = measurement

        if not problems:
            return SensorSample(SampleQuality.VALID, sampled_at, values)
        reason = ", ".join(problems)
        if any(item.value is not None for item in values.values()):
            return SensorSample(SampleQuality.PARTIAL, sampled_at, values, reason)
        if invalid:
            return SensorSample(SampleQuality.INVALID, sampled_at, error=reason)
        return SensorSample(SampleQuality.UNAVAILABLE, error=reason)

    async def read_telemetry(self) -> TelemetrySnapshot:
        """Read battery, acceleration, sonar, foot loads and joint positions.

        Charge is validated as finite 0..100 percent. Charging-state ingestion is
        not implemented, so valid charge produces a PARTIAL battery group. Missing
        services/read failures produce UNAVAILABLE only if connectivity is confirmed.
        """
        session = self._require_connection()
        if not (await self.get_status()).connected:
            raise RobotConnectionError("Robot disconnected before battery read")
        try:
            if self._battery is None:
                battery = await self._call(session, session.service, "ALBattery")
                self._check_session(session)
                self._battery = battery
            charge = await self._call(session, self._battery.getBatteryCharge)
            self._check_session(session)
            now = datetime.now(timezone.utc)
            if (isinstance(charge, bool) or not isinstance(charge, (int, float))
                    or not math.isfinite(charge) or not 0 <= charge <= 100):
                sample = SensorSample(SampleQuality.INVALID, now,
                                      error="Battery charge must be finite and within 0..100 percent")
            else:
                sample = SensorSample(SampleQuality.PARTIAL, now,
                                      {"charge": Measurement(charge, "%"),
                                       "charging": Measurement(None, "boolean")},
                                      "Charging-state ingestion is not implemented")
        except (_QiCallError, AttributeError) as exc:
            self._check_session(session)
            if not (await self.get_status()).connected:
                raise RobotConnectionError("Robot disconnected during battery read") from exc
            sample = SensorSample(SampleQuality.UNAVAILABLE, error=f"Battery unavailable: {exc}")
            self._battery = None  # Permit a later read to resolve the service again.
        self._check_session(session)
        self._require_connection()
        imu = await self._read_memory_sample(session, "IMU", _ACCELEROMETER_KEYS, "m/s^2")
        self._check_session(session)
        self._require_connection()
        ultrasonic = await self._read_memory_sample(session, "ultrasonic", _SONAR_KEYS, "m", sonar=True)
        self._check_session(session)
        self._require_connection()
        fsr = await self._read_memory_sample(session, "FSR", _FSR_KEYS, "kg", minimum=0)
        self._check_session(session)
        self._require_connection()
        joints = await self._read_memory_sample(session, "joint", _JOINT_KEYS, _JOINT_UNITS)
        self._check_session(session)
        self._require_connection()
        snapshot = TelemetrySnapshot(
            self._sequence, datetime.now(timezone.utc), OperatingMode.HARDWARE,
            f"hardware:qi:{self.endpoint}", imu=imu, ultrasonic=ultrasonic,
            fsr=fsr,
            joints=joints,
            battery=sample,
        )
        self._sequence += 1
        return snapshot

    async def get_posture(self) -> str | None:
        """Read the current predefined posture when posture RPCs are enabled."""
        session = self._require_connection()
        if not self._enable_posture:
            return None
        try:
            proxy = await self._call(session, session.service, "ALRobotPosture")
            self._check_session(session)
            return await self._observe_posture(session, proxy)
        except (_QiCallError, AttributeError) as exc:
            self._check_session(session)
            if not (await self.get_status()).connected:
                raise RobotConnectionError("Robot disconnected during posture observation") from exc
            return None

    async def _observe_posture(self, session, proxy) -> str | None:
        observed = await self._call(session, proxy.getPosture)
        self._check_session(session)
        self._require_connection()
        if isinstance(observed, str) and observed.strip() and observed != "Unknown":
            return observed
        return None

    async def set_posture(self, name: str, speed: float) -> PostureResult:
        """Request Stand or Sit once and verify with a separate observation."""
        if not self._enable_posture or name not in ("Stand", "Sit"):
            raise UnsupportedOperationError(f"Hardware posture request unavailable: {name!r}")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not 0 < speed <= 1):
            raise ValueError("speed must be finite and greater than 0, at most 1")
        session = self._require_connection()
        if self._posture_request is not None:
            raise RobotBusyError("A posture request is active or its completion is unknown")

        request = object()
        self._posture_request = request
        command_started = False
        command_pending = False
        try:
            proxy = await self._call(session, session.service, "ALRobotPosture")
            self._check_session(session)
            self._require_connection()
            command = proxy.goToPosture
            command_started = True
            command_pending = True
            reached = await self._call(session, command, name, speed)
            command_pending = reached is not True and reached is not False
            self._check_session(session)
            self._require_connection()
            if reached is False:
                return PostureResult(name, PostureOutcome.FAILED, message="NAOqi did not reach the posture")
            if reached is not True:
                return PostureResult(name, PostureOutcome.UNVERIFIED, message="Unexpected NAOqi completion result")
            observed = await self._observe_posture(session, proxy)
            outcome = (PostureOutcome.COMPLETED_VERIFIED if observed == name else PostureOutcome.UNVERIFIED)
            return PostureResult(name, outcome, observed)
        except (_QiCallError, AttributeError) as exc:
            self._check_session(session)
            if not (await self.get_status()).connected:
                raise RobotConnectionError("Robot disconnected during posture request") from exc
            outcome = PostureOutcome.UNVERIFIED if command_started else PostureOutcome.FAILED
            return PostureResult(name, outcome, message=str(exc))
        finally:
            if not command_pending and self._posture_request is request:
                self._posture_request = None

    def stream_telemetry(self, interval: float) -> AsyncIterator[TelemetrySnapshot]:
        """Sample immediately, then wait at least interval seconds between reads."""
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or interval <= 0 or not math.isfinite(interval)):
            raise ValueError("interval must be positive and finite")
        self._require_connection()
        session_end = self._session_end

        async def snapshots():
            loop = asyncio.get_running_loop()
            while not session_end.is_set():
                try:
                    snapshot = await self.read_telemetry()
                except RobotConnectionError:
                    if session_end.is_set():
                        return
                    raise
                if session_end.is_set():
                    return
                yield snapshot
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
