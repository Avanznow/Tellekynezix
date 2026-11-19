"""Transport-independent LoRa command and telemetry gateway for the Avatar bus."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import json
import logging
import math
import time
from typing import Any, Protocol


class GatewayError(ValueError):
    """An authenticated packet has an unsupported or malformed application payload."""


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


class DecodedFrame(Protocol):
    message_type: Any
    sequence: int
    timestamp_ms: int
    payload: bytes


class PacketCodec(Protocol):
    """Authenticated packet codec supplied by the selected wire protocol."""

    def decode(self, packet: bytes) -> DecodedFrame:
        """Authenticate and decode a packet; raise on invalid authentication."""

    def encode(
        self, message_type: str, sequence: int, timestamp_ms: int, payload: bytes
    ) -> bytes:
        """Encode and authenticate a gateway response."""


@dataclass(frozen=True)
class CommandRoute:
    """A command-specific validator and async Avatar-bus handler."""

    validate: Callable[[Mapping[str, Any]], None]
    execute: Callable[[Mapping[str, Any]], Awaitable[None]]


class LoraGateway:
    """Validate authenticated packets before routing commands or telemetry.

    Wire encoding/authentication is injected through `codec`; routing is kept
    independent from radio drivers and physical robot implementations.
    """

    def __init__(
        self,
        codec: PacketCodec,
        command_routes: Mapping[str, CommandRoute],
        telemetry_provider: Callable[[], Awaitable[Mapping[str, Any]]],
        *,
        clock_ms: Callable[[], int] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(getattr(codec, "decode", None)) or not callable(
            getattr(codec, "encode", None)
        ):
            raise TypeError("codec must provide decode and encode methods")
        if not isinstance(command_routes, Mapping) or not command_routes:
            raise ValueError("At least one command route is required")
        for name, route in command_routes.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(route, CommandRoute)
                or not callable(route.validate)
                or not callable(route.execute)
            ):
                raise ValueError(
                    "Command routes require names, validators, and handlers"
                )
        if not callable(telemetry_provider):
            raise TypeError("telemetry_provider must be callable")

        self.codec = codec
        self.command_routes = dict(command_routes)
        self.telemetry_provider = telemetry_provider
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _parse_payload(payload: bytes) -> dict[str, Any]:
        try:
            value = json.loads(
                payload.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise GatewayError("Payload must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise GatewayError("Payload must be a JSON object")
        return value

    @staticmethod
    def _json_bytes(value: Mapping[str, Any]) -> bytes:
        try:
            return json.dumps(
                dict(value), separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise GatewayError("Payload is not valid finite JSON data") from exc

    @staticmethod
    def _message_type(frame: DecodedFrame) -> str:
        message_type = frame.message_type
        return message_type.name if hasattr(message_type, "name") else str(message_type)

    def _response(
        self, message_type: str, sequence: int, payload: Mapping[str, Any]
    ) -> bytes:
        timestamp = self.clock_ms()
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError("clock_ms must return a nonnegative integer")
        return self.codec.encode(
            message_type,
            sequence,
            timestamp,
            self._json_bytes(payload),
        )

    async def handle_packet(self, packet: bytes) -> bytes:
        """Authenticate, validate, and route one inbound packet.

        Codec/authentication errors propagate without a reply. Authenticated but
        invalid application commands receive a generic error response.
        """
        frame = self.codec.decode(packet)
        message_type = self._message_type(frame).upper()
        if message_type == "HEARTBEAT":
            if frame.payload:
                self.logger.warning(
                    "Rejected heartbeat with payload",
                    extra={"sequence": frame.sequence},
                )
                return self._response(
                    "ERROR", frame.sequence, {"code": "invalid_heartbeat"}
                )
            self.logger.info(
                "Received LoRa heartbeat", extra={"sequence": frame.sequence}
            )
            return self._response("ACK", frame.sequence, {"status": "alive"})
        if message_type != "COMMAND":
            self.logger.warning(
                "Rejected unsupported LoRa message",
                extra={"message_type": message_type, "sequence": frame.sequence},
            )
            return self._response(
                "ERROR", frame.sequence, {"code": "unsupported_message_type"}
            )

        try:
            payload = self._parse_payload(frame.payload)
            if set(payload) - {"command", "parameters"}:
                raise GatewayError("Command contains unsupported fields")
            command = payload.get("command")
            parameters = payload.get("parameters", {})
            if not isinstance(command, str) or not isinstance(parameters, dict):
                raise GatewayError("Command and parameter object are required")
            route = self.command_routes.get(command)
            if route is None:
                raise GatewayError("Command is not allowed")
            route.validate(parameters)
        except ValueError as exc:
            self.logger.warning(
                "Rejected invalid LoRa command",
                extra={"sequence": frame.sequence, "reason": str(exc)},
            )
            return self._response("ERROR", frame.sequence, {"code": "invalid_command"})

        self.logger.info(
            "Forwarding validated LoRa command",
            extra={"sequence": frame.sequence, "command": command},
        )
        await route.execute(parameters)
        return self._response(
            "ACK", frame.sequence, {"status": "accepted", "command": command}
        )

    async def telemetry_packet(self, sequence: int) -> bytes:
        """Serialize the current Avatar telemetry snapshot into an outbound frame."""
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or not 0 <= sequence <= 0xFFFFFFFF
        ):
            raise ValueError("sequence must be an unsigned 32-bit integer")
        telemetry = await self.telemetry_provider()
        if not isinstance(telemetry, Mapping):
            raise GatewayError("Telemetry provider must return an object")
        payload = self._json_bytes(telemetry)
        timestamp = self.clock_ms()
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError("clock_ms must return a nonnegative integer")
        self.logger.info("Sending Avatar telemetry", extra={"sequence": sequence})
        return self.codec.encode("TELEMETRY", sequence, timestamp, payload)


def validate_motion_command(parameters: Mapping[str, Any]) -> None:
    """Validate the shared move/stop command parameter shape."""
    if set(parameters) - {"direction", "speed"}:
        raise GatewayError("Unsupported motion parameters")
    direction = parameters.get("direction")
    if not isinstance(direction, str) or direction not in {
        "forward",
        "backward",
        "left",
        "right",
        "stop",
    }:
        raise GatewayError("Unsupported movement direction")
    speed = parameters.get("speed", 0.5)
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not math.isfinite(speed)
        or not 0 < speed <= 1
    ):
        raise GatewayError("Movement speed must be finite and between 0 and 1")
    if direction == "stop" and "speed" in parameters:
        raise GatewayError("Stop does not accept a speed")
