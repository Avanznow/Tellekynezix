"""Async client for the simulation-only JSON Lines server."""

import asyncio
from datetime import datetime
import math

from .errors import RobotConnectionError, RobotServiceError
from .interface import RobotService
from .models import (
    Measurement, MotionState, OperatingMode, PostureOutcome, PostureResult,
    RobotStatus, SampleQuality, SensorSample, TelemetrySnapshot,
)
from .simulation_rpc import MAX_MESSAGE, _decode, _encode


class RemoteSimulationService(RobotService):
    """One TCP session on one event loop. Commands are serialized and never retried."""

    def __init__(self, host="127.0.0.1", port=8765, timeout=5.0):
        if not isinstance(host, str) or not host.strip():
            raise ValueError("Simulation host is required")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Simulation port must be 1..65535")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.host, self.port, self.timeout = host.strip(), port, timeout
        self._reader = self._writer = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    def _drop(self):
        writer = self._writer
        self._reader = self._writer = None
        if writer is not None:
            writer.close()
        return writer

    async def _request(self, method, params=None):
        async with self._lock:
            if self._writer is None:
                raise RobotConnectionError("Remote simulation is disconnected")
            writer, reader = self._writer, self._reader
            self._next_id += 1
            request_id = str(self._next_id)
            try:
                writer.write(_encode({"version": 1, "id": request_id, "method": method, "params": params or {}}))
                await asyncio.wait_for(writer.drain(), self.timeout)
                reply = _decode(await asyncio.wait_for(reader.readline(), self.timeout))
                if self._writer is not writer or reply["id"] != request_id:
                    raise ValueError("Simulation session or response id changed")
                if type(reply.get("ok")) is not bool:
                    raise ValueError("Invalid response outcome")
                if not reply["ok"]:
                    error = reply.get("error")
                    if not isinstance(error, dict):
                        raise ValueError("Invalid error response")
                    raise RobotServiceError(f"{error.get('code')}: {error.get('message')}")
                if not isinstance(reply.get("result"), dict):
                    raise ValueError("Expected an object result")
                return reply["result"]
            except asyncio.CancelledError:
                if self._writer is writer:
                    self._drop()
                raise
            except (OSError, TimeoutError, ValueError) as exc:
                if self._writer is writer:
                    self._drop()
                raise RobotConnectionError(f"Remote simulation communication failed: {exc}") from exc

    def _status(self, value):
        try:
            if (value["mode"] != "simulation" or type(value["connected"]) is not bool
                    or type(value["busy"]) is not bool
                    or value.get("posture") is not None and not isinstance(value["posture"], str)):
                raise ValueError("Invalid simulation status")
            return RobotStatus(OperatingMode.SIMULATION, value["connected"], value["busy"], value.get("posture"))
        except (KeyError, TypeError, ValueError) as exc:
            self._drop()
            raise RobotConnectionError("Invalid remote simulation status") from exc

    async def connect(self):
        if self._writer is not None:
            return await self.get_status()
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, limit=MAX_MESSAGE), self.timeout)
        except (OSError, TimeoutError) as exc:
            raise RobotConnectionError(f"Cannot connect to simulation at {self.host}:{self.port}: {exc}") from exc
        self._next_id = 0
        try:
            status = self._status(await self._request("connect"))
            if not status.connected:
                raise RobotConnectionError("Server did not establish a simulation session")
            return status
        except BaseException:
            await self.disconnect()
            raise

    async def disconnect(self):
        # Closing TCP immediately also interrupts an in-flight request. The server
        # releases its simulation session on EOF or after the bounded operation.
        writer = self._drop()
        if writer is not None:
            try:
                await asyncio.wait_for(writer.wait_closed(), self.timeout)
            except (OSError, TimeoutError):
                pass

    async def get_status(self):
        if self._writer is None:
            return RobotStatus(OperatingMode.SIMULATION, False, False)
        return self._status(await self._request("get_status"))

    async def get_posture(self):
        return (await self.get_status()).posture

    async def set_posture(self, name, speed):
        result = await self._request("set_posture", {"name": name, "speed": speed})
        try:
            if result["requested_posture"] != name:
                raise ValueError("Posture request mismatch")
            return PostureResult(name, PostureOutcome(result["outcome"]),
                                 result.get("observed_posture"), result.get("message"))
        except (KeyError, TypeError, ValueError) as exc:
            self._drop()
            raise RobotConnectionError("Invalid remote posture result") from exc

    async def get_motion_state(self):
        if self._writer is None:
            return MotionState()
        result = await self._request("get_motion_state")
        try:
            state = MotionState(**result)
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for key, value in result.items() if key != "action"):
                raise ValueError("Invalid visual motion values")
            return state
        except (TypeError, ValueError) as exc:
            self._drop()
            raise RobotConnectionError("Invalid remote visual state") from exc

    async def read_telemetry(self):
        result = await self._request("read_telemetry")
        try:
            if result.pop("schema_version") != 1 or result["mode"] != "simulation":
                raise ValueError("Unsupported telemetry schema or mode")
            result["mode"] = OperatingMode.SIMULATION
            result["created_at"] = datetime.fromisoformat(result["created_at"])
            for group in ("imu", "ultrasonic", "fsr", "joints", "battery"):
                data = result[group]
                result[group] = SensorSample(
                    SampleQuality(data["quality"]),
                    datetime.fromisoformat(data["sampled_at"]) if data["sampled_at"] else None,
                    {key: Measurement(**value) for key, value in data["values"].items()},
                    data["error"],
                )
            return TelemetrySnapshot(**result)
        except (KeyError, TypeError, ValueError) as exc:
            self._drop()
            raise RobotConnectionError("Invalid remote telemetry snapshot") from exc

    def stream_telemetry(self, interval):
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        if self._writer is None:
            raise RobotConnectionError("Remote simulation is disconnected")
        writer = self._writer

        async def snapshots():
            while self._writer is writer:
                yield await self.read_telemetry()
                await asyncio.sleep(interval)

        return snapshots()
