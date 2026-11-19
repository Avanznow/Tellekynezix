"""Tests for the authenticated Avatar LoRa frame protocol."""

import unittest

from robot_services.lora_protocol import (
    AUTH_TAG_LENGTH,
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
)


class LoraProtocolTests(unittest.TestCase):
    def setUp(self):
        self.key = b"avatar-test-key-0001"

    def test_round_trip_binary_payload(self):
        frame = Frame(
            MessageType.COMMAND, 0x01020304, 0x0102030405060708, b"\x00move\xff"
        )
        self.assertEqual(decode_frame(encode_frame(frame, self.key), self.key), frame)

    def test_golden_vector(self):
        frame = Frame(MessageType.COMMAND, 1, 0x0102030405060708, b"{}")
        self.assertEqual(
            encode_frame(frame, self.key).hex(),
            "4156010200000001010203040506070800027b7d"
            "7b61c7794c2fddbaa4d8aa2de18c015a",
        )

    def test_rejects_modified_payload_or_authentication_tag(self):
        packet = encode_frame(Frame(MessageType.HEARTBEAT, 1, 2), self.key)
        for index in (18, len(packet) - 1):
            modified = bytearray(packet)
            modified[index] ^= 1
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "authentication"
            ):
                decode_frame(bytes(modified), self.key)

    def test_rejects_wrong_key_and_malformed_packets(self):
        packet = encode_frame(Frame(MessageType.ACK, 1, 2), self.key)
        with self.assertRaisesRegex(ValueError, "authentication"):
            decode_frame(packet, b"another-test-key")
        for malformed in (b"", packet[:-1], packet + b"\x00"):
            with self.subTest(packet=malformed), self.assertRaises(ValueError):
                decode_frame(malformed, self.key)

    def test_rejects_invalid_fields_and_short_keys(self):
        for frame in (
            Frame(MessageType.COMMAND, -1, 0),
            Frame(MessageType.COMMAND, 0x100000000, 0),
            Frame(MessageType.COMMAND, 0, -1),
            Frame(MessageType.COMMAND, 0, 0, b"x" * 65536),
        ):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                encode_frame(frame, self.key)
        with self.assertRaises(ValueError):
            encode_frame(Frame(MessageType.HEARTBEAT, 0, 0), b"short")

    def test_authentication_tag_is_compact(self):
        packet = encode_frame(Frame(MessageType.HEARTBEAT, 0, 0), self.key)
        self.assertEqual(len(packet), 18 + AUTH_TAG_LENGTH)


if __name__ == "__main__":
    unittest.main()
