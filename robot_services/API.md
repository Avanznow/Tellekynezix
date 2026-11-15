# NAO service API

Use `robot_services.interface.RobotService` as the backend contract. Create and
use each service on one asyncio event loop. Both backends implement the same
telemetry schema.

| Backend | Constructor parameters |
| --- | --- |
| `robot_services.simulation.SimulationRobotService` | `posture_duration=1.0` and `settle_duration=0.35` in seconds, `left_distance=1.0` and `right_distance=1.0` in metres, `battery_percent=100.0`, `charging=False`, and optional `sensor_overrides` mapping group names to complete `SensorSample` fixtures. Only `posture_duration` may be positional. |
| `robot_services.qi_service.QiRobotService` | Required `host` (hostname or IP without a scheme or IPv6 brackets), `port=9559` (1–65535), `timeout=5.0` seconds per SDK call, optional keyword-only `session_factory` for SDK-compatible test sessions, and keyword-only `enable_posture=False`. See [hardware behavior](HARDWARE.md). |

## Methods

| Method | Return value and behavior |
| --- | --- |
| `await connect()` | `RobotStatus` for the established session. Repeated calls reuse a connected session. |
| `await disconnect()` | `None`. Ends the session and its telemetry streams. Safe to call again. |
| `await get_status()` | `RobotStatus` with `mode`, `connected`, `busy`, and last known `posture` (or `None`). The qi backend reports connectivity and busy state here. Use `get_posture()` for a separate SDK observation. |
| `await get_posture()` | Observed posture name as `str`, or `None` if unknown. Requires a connection. The qi backend exposes observations when posture RPCs are enabled. |
| `await set_posture(name, speed)` | `PostureResult`. Pass `"Stand"` or `"Sit"` and a finite relative speed greater than 0 and at most 1. Hardware requests require `enable_posture=True`. |
| `await read_telemetry()` | One `TelemetrySnapshot` from the active session. Sensor errors are represented per group while the session remains connected. |
| `stream_telemetry(interval)` | `AsyncIterator[TelemetrySnapshot]`. Call without `await`, then iterate with `async for`. Requires a connection and a positive finite interval in seconds. The first sample is immediate. Close with `aclose()` when leaving early. |
| `await get_motion_capabilities()` | Tuple of command names. Simulation supports `forward`, `backward`, `turn_left`, and `turn_right`. The qi backend returns an empty tuple. |
| `await get_motion_state()` | Current `MotionState` in simulation. |
| `await start_motion(command, speed=0.5)` | Initial `MotionState` for the selected simulation command. Speed is finite and greater than 0 and at most 1. |
| `await stop_motion()` | `MotionState` after simulation travel or turning stops and the model settles, or after an active posture transition is frozen. |

`MotionState` describes the viewer: `x` and `z` are scene units, `heading` is
degrees, `gait_phase` is radians, and `gait_weight` and `sit_amount` are animation
blends from 0 to 1. `action` identifies the current animation.

`PostureResult` contains `requested_posture`, `outcome`, optional
`observed_posture`, and optional `message`. Outcomes are `accepted`,
`completed_verified`, `failed`, and `unverified`. Verified completion
requires an exact match between requested and observed posture.

## Errors and cancellation

- `RobotConnectionError` indicates a disconnected backend or a communication failure.
- `RobotBusyError` indicates a conflicting operation. On qi, an uncertain command
  can also keep posture requests blocked for the current session.
- `UnsupportedOperationError` indicates an unsupported or disabled capability.
- `ValueError` indicates an invalid argument. `TelemetryValidationError` is a
  `ValueError` and a `RobotServiceError` for invalid telemetry.
- `asyncio.CancelledError` propagates. See the [posture timeout and cancellation
  behavior](HARDWARE.md#posture-requests) for SDK operations.

## Telemetry schema version 1

`TelemetrySnapshot.to_dict()` returns a JSON-compatible dictionary.
`to_json()` returns a JSON string. Both validate the snapshot. Each record
contains `schema_version`, `sequence` (starting at 0 per connection),
`created_at` (assembly time), `mode` (`simulation` or `hardware`), `source`,
and the five sensor groups below. A group contains `quality`, `sampled_at`,
`values`, and `error`. Timestamps serialize in UTC with a trailing `Z`.
Each measurement has `value` (finite number or `null`) and `unit`.

| Group | Measurements and units |
| --- | --- |
| `imu` | `acceleration_x/y/z` in `m/s^2` on both backends. Default simulation fixtures also provide `angular_velocity_x/y/z` in `rad/s` and `roll`, `pitch`, `yaw` in `rad`. |
| `ultrasonic` | `left` and `right` in `m`. |
| `fsr` | Eight hardware corner loads such as `left_front_left` and `right_rear_right` in `kg`. |
| `joints` | 23 hardware joint angles by NAOqi name in `rad`, plus `LHand` and `RHand` apertures in `ratio` (0–1). |
| `battery` | `charge` in `%` and `charging` in `boolean` (0, 1, or `null`; hardware currently reports `null`). |

A `valid` group contains complete usable values. A `partial` group retains
usable readings and carries a reason. An `unavailable` or `invalid` group
carries a reason and has no usable numeric values. Default simulation FSR
and joint groups are unavailable and can be supplied as fixtures.

## Run a simulation example

From the repository root:

```python
import asyncio
from contextlib import aclosing

from robot_services.simulation import SimulationRobotService


async def main():
    service = SimulationRobotService(battery_percent=75)
    await service.connect()
    try:
        async with aclosing(service.stream_telemetry(1.0)) as snapshots:
            snapshot = await anext(snapshots)
            print(snapshot.to_json())
    finally:
        await service.disconnect()


asyncio.run(main())
```

For the CLI and JSON Lines export, see [diagnostics](DIAGNOSTICS.md).
The JSON schema is available for local consumers. The Avatar system bus topic
and consumer contract are pending.
