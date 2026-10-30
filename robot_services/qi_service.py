"""Read-only libqi service adapter. Importing this module never loads the SDK.

Only connect/isConnected/service/getBatteryCharge/close are used. The native SDK
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

from .errors import RobotConnectionError, RobotServiceError, UnsupportedOperationError
from .interface import RobotService
from .models import (
    Measurement, OperatingMode, PostureResult, RobotStatus, SampleQuality,
    SensorSample, TelemetrySnapshot,
)


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
                 *, session_factory: Callable[[], object] | None = None):
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
        self._session = None
        self._battery = None
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
            self._battery = None
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
        self._session = None
        self._battery = None
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
            return RobotStatus(OperatingMode.HARDWARE, connected, False)
        except _QiCallError as exc:
            if self._session is session:
                self._connected = False
            raise RobotConnectionError(f"SDK status check failed: {exc}") from exc

    async def read_telemetry(self) -> TelemetrySnapshot:
        """Read ALBattery.getBatteryCharge; mark other readings unavailable.

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
        missing = lambda name: SensorSample(SampleQuality.UNAVAILABLE,
                                            error=f"Hardware {name} ingestion is not implemented")
        snapshot = TelemetrySnapshot(
            self._sequence, datetime.now(timezone.utc), OperatingMode.HARDWARE,
            f"hardware:qi:{self.endpoint}", imu=missing("IMU"), ultrasonic=missing("ultrasonic"),
            fsr=missing("FSR"), joints=missing("joint"), battery=sample,
        )
        self._sequence += 1
        return snapshot

    async def get_posture(self) -> str | None:
        """Return Unknown (None); posture observation is not implemented yet."""
        self._require_connection()
        return None

    async def set_posture(self, name: str, speed: float) -> PostureResult:
        """Explicitly reject actuation in this read-only increment."""
        raise UnsupportedOperationError("Hardware posture commands are not implemented")

    def stream_telemetry(self, interval: float) -> AsyncIterator[TelemetrySnapshot]:
        """Streaming is intentionally deferred; use individual read_telemetry calls."""
        raise UnsupportedOperationError("Hardware telemetry streaming is not implemented")
