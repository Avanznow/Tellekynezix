# LoRa-to-Avatar gateway

`robot_services.lora_gateway.LoraGateway` decouples radio transport, authenticated
frame encoding, and Avatar command execution. A transport adapter receives
radio bytes and passes them to `handle_packet()`, then transmits the returned
response bytes. It calls `telemetry_packet()` when a peer requests or is due
telemetry.

The gateway requires an injected `PacketCodec`. Its decoder must authenticate
the packet before returning a frame; authentication failures propagate and
receive no response. The encoder signs outbound ACK, ERROR, and telemetry
frames. This keeps cryptographic keys and wire-protocol details out of the
gateway.

Applications register a finite mapping of command names to `CommandRoute`
instances. Each route validates its parameter object before its async handler
is called. Unknown commands, invalid JSON, unsupported fields, and malformed
parameters are rejected without reaching the Avatar bus. The example
`validate_motion_command` accepts only forward/backward/left/right/stop and a
finite normalized speed in `(0, 1]`; applications should use it only if those
semantics match their Avatar bus.

The gateway logs message type, sequence, command, and rejection reason, never
the authentication key or complete packet payload. It does not open radio
hardware, replay commands, or claim that a physical movement has completed.
The transport and application should use the `LinkManager` separately for
heartbeat timeout, signal quality, safe-state, and reconnect behavior.
