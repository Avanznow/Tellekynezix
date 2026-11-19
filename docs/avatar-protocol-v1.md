# Avatar Control Bus Protocol v1

The Avatar control bus uses authenticated binary frames intended for constrained
links such as LoRa. Version 1 provides integrity and peer authentication with
HMAC-SHA256; it does not encrypt payloads. Deployments must provision a unique,
secret key of at least 16 bytes to each authorized peer and protect it at rest.
Do not transmit commands or telemetry over an untrusted link without such a key.

## Frame layout

All integers use network byte order (big-endian).

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 2 | Magic | ASCII `AV` (`0x4156`) |
| 2 | 1 | Version | `1` |
| 3 | 1 | Message type | 1 heartbeat, 2 command, 3 telemetry, 4 ACK, 5 error |
| 4 | 4 | Sequence | Unsigned per-peer message counter |
| 8 | 8 | Timestamp | Unsigned Unix time in milliseconds |
| 16 | 2 | Payload length | Unsigned byte count, maximum 65535 |
| 18 | variable | Payload | Opaque application bytes |
| end - 16 | 16 | Authentication tag | First 16 bytes of HMAC-SHA256 over header and payload |

A receiver must reject unknown versions/types, truncated frames, length
mismatches, and invalid authentication tags before parsing or acting on a
payload. Sequence replay tracking is a session/link-management responsibility;
the frame codec does not maintain peer state.

## Python API and golden vector

```python
from robot_services.lora_protocol import Frame, MessageType, encode_frame, decode_frame

key = b"provisioned-key-at-least-16-bytes"
frame = Frame(MessageType.COMMAND, sequence=1, timestamp_ms=1_000, payload=b"{}")
packet = encode_frame(frame, key)
assert decode_frame(packet, key) == frame
```

The unit tests in `tests/test_lora_protocol.py` include a fixed header encoding
vector and tamper/rejection cases. Payload schemas are defined by the gateway
layer, not by the binary codec.
