# Read-only qi hardware adapter: first increment

`QiRobotService` implements the shared `RobotService` interface for connection,
session status and individual battery snapshots. No qi SDK is imported until
`connect()` is explicitly called without an injected `session_factory`.
Constructing/importing the adapter does not connect to a robot.

## Scope and limits

- `connect()` uses `Session.connect(endpoint, _async=True)` and confirms connectivity.
- `get_status()` checks `Session.isConnected(_async=True)`. Connected means an SDK
  session exists, not that the robot is balanced, ready to move or otherwise healthy.
- `read_telemetry()` resolves `ALBattery` with `Session.service` and requests
  `getBatteryCharge(_async=True)`. Expected charge is a finite number in 0..100 %.
- Battery values outside that contract are INVALID; missing services/read failures
  are UNAVAILABLE only when session connectivity can still be confirmed.
- A valid charge produces a PARTIAL battery sample with charging unknown.
- IMU reads use one `ALMemory.getListData` call for the NAOqi 2.8 torso
  accelerometer X, Y, and Z keys. Telemetry exposes them as `acceleration_x`,
  `acceleration_y`, and `acceleration_z` in `m/s^2`, preserving the SDK axes.
  Valid axes survive missing or malformed readings. A missing service or failed
  read marks the IMU unavailable if the session is still connected.
- Ultrasonic reads use one `ALMemory.getListData` call for
  `Device/SubDeviceList/US/{Left,Right}/Sensor/Value`. The snapshot reports
  `left` and `right` in metres. Zero and negative values are invalid.
  Readings at or above the documented 5 m no-echo limit are unavailable.
  One usable side produces a partial sample when the other side is missing
  or invalid.
- The adapter reads existing sonar values from ALMemory. Sonar capture and
  freshness depend on the robot's configuration and still require physical
  verification. The documented NAO V6 detection range is 0.20–0.80 m.
- Foot FSR reads use one `ALMemory.getListData` call for the four corner
  sensors on each foot. The snapshot reports approximate loads in `kg`,
  with names such as `left_front_left` and `right_rear_right`.
  Zero is valid, negative and nonfinite loads are invalid, and valid corners
  are preserved in partial samples.
- Joint position reads use one `ALMemory.getListData` call for the 25
  `Device/SubDeviceList/<joint>/Position/Sensor/Value` keys. Readings are
  named by their NAOqi joint names. Angles are in radians. `LHand` and
  `RHand` use unit `ratio`, from 0 (closed) to 1 (open).
  `LHipYawPitch` represents the shared hip actuator.
- Posture and motion commands are unsupported.
- `stream_telemetry(interval)` starts with an immediate read, waits at least
  `interval` seconds between reads, and ends when its session is explicitly
  disconnected. Transport errors propagate to the consumer.
- The diagnostic CLI can select this adapter with `--mode hardware --host HOST`.
  See [diagnostic usage and the loopback check](DIAGNOSTICS.md).

The initial connection, battery read, and disconnect passed a local libqi test
on Pop!_OS 24.04 x86-64 with Python 3.12.3 and qi 3.1.5. Physical NAO firmware
compatibility and battery readings still require verification. The
legacy NAOqi 2.8 Windows SDK targets Python 2.7. Modern `qi` Python packages and
their available platforms are a separate compatibility decision; a package being
installable does not establish compatibility with a particular robot firmware.

## Configuration and lifecycle

Constructor: `QiRobotService(host, port=9559, timeout=5.0, session_factory=None)`.
Host is a hostname or IP, without URL scheme, credentials, brackets or path.
IPv6 is formatted with brackets in the connection endpoint. Port must be 1..65535;
timeout must be finite and positive. All timeouts are seconds, per SDK operation.

An injected factory must return a fresh compatible session promptly. This permits
Windows development and tests without installing qi. The adapter is confined to
one asyncio event loop, and uses callback delivery onto that loop to await native
qi futures. Completed future values are read with timeout 0, never a blocking wait.

Repeated/concurrent connects reuse the established session. Disconnect invalidates
pending reads before closing the SDK session and is safe when repeated. Reconnect
creates a fresh session and resets the snapshot sequence. Replies from old sessions
cannot overwrite current readings. Timed-out/cancelled native operations receive a
best-effort cancellation request; late callbacks are ignored. Native cancellation
or a close timeout cannot prove the remote SDK has released its resources.

Connection/status/close failures raise `RobotConnectionError`. Caller cancellation
propagates as `asyncio.CancelledError`. Import/connection failure never falls back
silently to simulation. Unsupported functions raise `UnsupportedOperationError`.

## Tests

```powershell
python -B -m unittest discover -s tests -p test_qi_service.py -v
```

These tests inject SDK-shaped sessions and futures. They cover connection and
cleanup, battery payloads, missing services, timeout/cancellation, foreign-thread
callbacks and stale replies. They do not establish real SDK/robot compatibility.

## Reference points for later verification

- [NAOqi 2.8 sensor keys and units](https://doc.aldebaran.com/2-8/family/nao_technical/lola/actuator_sensor_names.html)
- [NAO V6 sonar specifications](https://doc.aldebaran.com/2-8/family/nao_technical/sonar_naov6.html)

- Official SDK source: https://github.com/aldebaran/libqi-python
- Session API tests: https://github.com/aldebaran/libqi-python/blob/master/qi/test/test_session.py
- Future callback/timeout API: https://github.com/aldebaran/libqi-python/blob/master/src/pyfuture.cpp
- Battery API mapping to verify: https://doc.aldebaran.com/2-8/naoqi/sensors/albattery-api.html
- Legacy installation guide: https://doc.aldebaran.com/2-8/dev/python/install_guide.html
- Modern published package files: https://pypi.org/project/qi/#files
