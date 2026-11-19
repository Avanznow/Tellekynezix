"""Authenticated, versioned binary frames for the Avatar LoRa control bus."""

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import hmac
import struct


MAGIC = b"AV"
PROTOCOL_VERSION = 1
AUTH_TAG_LENGTH = 16
MAX_PAYLOAD_LENGTH = 0xFFFF
HEADER = struct.Struct("!2sBBIQH")


class MessageType(IntEnum):
    HEARTBEAT = 1
    COMMAND = 2
    TELEMETRY = 3
    ACK = 4
    ERROR = 5


@dataclass(frozen=True)
class Frame:
    message_type: MessageType
    sequence: int
    timestamp_ms: int
    payload: bytes = b""
    version: int = PROTOCOL_VERSION


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("Authentication key must be at least 16 bytes")


def _validate_frame(frame: Frame) -> None:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if frame.version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {frame.version}")
    if not isinstance(frame.message_type, MessageType):
        raise ValueError("Unknown message type")
    if (
        isinstance(frame.sequence, bool)
        or not isinstance(frame.sequence, int)
        or not 0 <= frame.sequence <= 0xFFFFFFFF
    ):
        raise ValueError("Sequence must be an unsigned 32-bit integer")
    if (
        isinstance(frame.timestamp_ms, bool)
        or not isinstance(frame.timestamp_ms, int)
        or not 0 <= frame.timestamp_ms <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("Timestamp must be an unsigned 64-bit integer")
    if not isinstance(frame.payload, bytes) or len(frame.payload) > MAX_PAYLOAD_LENGTH:
        raise ValueError("Payload must be bytes of at most 65535 bytes")


def encode_frame(frame: Frame, key: bytes) -> bytes:
    """Encode a frame and append its truncated HMAC-SHA256 authentication tag."""
    _validate_key(key)
    _validate_frame(frame)
    header = HEADER.pack(
        MAGIC,
        frame.version,
        int(frame.message_type),
        frame.sequence,
        frame.timestamp_ms,
        len(frame.payload),
    )
    authenticated = header + frame.payload
    tag = hmac.new(key, authenticated, hashlib.sha256).digest()[:AUTH_TAG_LENGTH]
    return authenticated + tag


def decode_frame(packet: bytes, key: bytes) -> Frame:
    """Validate a complete packet and return its authenticated frame."""
    _validate_key(key)
    if not isinstance(packet, bytes):
        raise TypeError("packet must be bytes")
    minimum_size = HEADER.size + AUTH_TAG_LENGTH
    if len(packet) < minimum_size:
        raise ValueError("Packet is shorter than the minimum frame size")

    magic, version, raw_type, sequence, timestamp_ms, payload_length = (
        HEADER.unpack_from(packet)
    )
    if magic != MAGIC:
        raise ValueError("Invalid frame magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")
    expected_size = HEADER.size + payload_length + AUTH_TAG_LENGTH
    if len(packet) != expected_size:
        raise ValueError("Packet length does not match its payload length")
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise ValueError(f"Unknown message type: {raw_type}") from exc

    authenticated = packet[:-AUTH_TAG_LENGTH]
    received_tag = packet[-AUTH_TAG_LENGTH:]
    expected_tag = hmac.new(key, authenticated, hashlib.sha256).digest()[
        :AUTH_TAG_LENGTH
    ]
    if not hmac.compare_digest(received_tag, expected_tag):
        raise ValueError("Frame authentication failed")

    return Frame(
        message_type=message_type,
        sequence=sequence,
        timestamp_ms=timestamp_ms,
        payload=packet[HEADER.size : -AUTH_TAG_LENGTH],
        version=version,
    )
