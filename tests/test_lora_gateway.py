"""Hardware-free tests for the LoRa-to-Avatar gateway."""

import json
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock

from robot_services.lora_gateway import (
    CommandRoute,
    GatewayError,
    LoraGateway,
    validate_motion_command,
)


@dataclass
class FakeFrame:
    message_type: str
    sequence: int
    timestamp_ms: int
    payload: bytes


class MemoryCodec:
    def decode(self, packet):
        message_type, sequence, timestamp_ms, payload = json.loads(packet)
        return FakeFrame(message_type, sequence, timestamp_ms, bytes.fromhex(payload))

    def encode(self, message_type, sequence, timestamp_ms, payload):
        return json.dumps(
            [message_type, sequence, timestamp_ms, payload.hex()]
        ).encode()

    def packet(self, message_type, sequence, payload=b""):
        return self.encode(message_type, sequence, 123, payload)


class LoraGatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.codec = MemoryCodec()
        self.execute = AsyncMock()
        self.telemetry = AsyncMock(return_value={"battery": 87, "connected": True})
        self.gateway = LoraGateway(
            self.codec,
            {
                "move": CommandRoute(validate_motion_command, self.execute),
                "stop": CommandRoute(validate_motion_command, self.execute),
            },
            self.telemetry,
            clock_ms=lambda: 456,
        )

    def response(self, packet):
        message_type, sequence, timestamp_ms, payload = json.loads(packet)
        return message_type, sequence, timestamp_ms, json.loads(bytes.fromhex(payload))

    async def test_authenticated_heartbeat_returns_ack(self):
        packet = self.codec.packet("HEARTBEAT", 9)
        response = await self.gateway.handle_packet(packet)
        self.assertEqual(self.response(response), ("ACK", 9, 456, {"status": "alive"}))
        self.execute.assert_not_awaited()

    async def test_validated_command_is_forwarded_and_acknowledged(self):
        payload = json.dumps(
            {"command": "move", "parameters": {"direction": "forward", "speed": 0.7}}
        ).encode()
        response = await self.gateway.handle_packet(
            self.codec.packet("COMMAND", 10, payload)
        )
        self.assertEqual(
            self.response(response),
            ("ACK", 10, 456, {"status": "accepted", "command": "move"}),
        )
        self.execute.assert_awaited_once_with({"direction": "forward", "speed": 0.7})

    async def test_invalid_commands_are_rejected_before_forwarding(self):
        cases = (
            b'{"command":"erase","parameters":{}}',
            b'{"command":"move","parameters":{"direction":"forward","speed":3}}',
            b'{"command":"move","parameters":{"direction":"forward","extra":1}}',
            b'{"command":"move","parameters":{"direction":"forward","speed":NaN}}',
            b'{"command":"move","parameters":{"direction":[]}}',
            b"not-json",
        )
        for sequence, payload in enumerate(cases, start=1):
            with self.subTest(payload=payload):
                response = await self.gateway.handle_packet(
                    self.codec.packet("COMMAND", sequence, payload)
                )
                self.assertEqual(
                    self.response(response),
                    ("ERROR", sequence, 456, {"code": "invalid_command"}),
                )
        self.execute.assert_not_awaited()

    async def test_unsupported_messages_and_heartbeat_payload_are_rejected(self):
        for message_type, payload, expected in (
            ("TELEMETRY", b"", "unsupported_message_type"),
            ("HEARTBEAT", b"unexpected", "invalid_heartbeat"),
        ):
            response = await self.gateway.handle_packet(
                self.codec.packet(message_type, 11, payload)
            )
            self.assertEqual(
                self.response(response),
                ("ERROR", 11, 456, {"code": expected}),
            )

    async def test_telemetry_is_serialized_into_an_outbound_frame(self):
        packet = await self.gateway.telemetry_packet(12)
        self.assertEqual(
            self.response(packet),
            ("TELEMETRY", 12, 456, {"battery": 87, "connected": True}),
        )
        self.telemetry.assert_awaited_once()

    async def test_authentication_errors_are_not_converted_to_command_errors(self):
        class InvalidCodec(MemoryCodec):
            def decode(self, packet):
                raise ValueError("authentication failed")

        gateway = LoraGateway(
            InvalidCodec(), self.gateway.command_routes, self.telemetry
        )
        with self.assertRaisesRegex(ValueError, "authentication"):
            await gateway.handle_packet(b"invalid")
        self.execute.assert_not_awaited()

    async def test_non_json_telemetry_is_rejected(self):
        self.telemetry.return_value = {"value": float("inf")}
        with self.assertRaises(GatewayError):
            await self.gateway.telemetry_packet(1)


if __name__ == "__main__":
    unittest.main()
