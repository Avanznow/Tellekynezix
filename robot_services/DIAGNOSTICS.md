# Standalone health diagnostics

From the repository root, activate the existing Python environment and run:

```powershell
python tools/robot_diagnostics.py --mode simulation
```

The tool runs independently of the GUI. It reports connection, battery charge,
charging state, and IMU/ultrasonic/FSR/joint sensor quality once per second.
Every report is timestamped and labeled SIMULATION or HARDWARE. The tool reads
status and telemetry. In simulation mode, readings are fixtures and unavailable
sensor groups include their reasons.

Press Ctrl+C to stop. The monitor closes its telemetry stream and disconnects,
including on cancellation or connection failure. The final interrupt message
says cleanup was attempted; any cleanup error is printed separately.

## Options and examples

```powershell
# Produce three reports, then exit.
python tools/robot_diagnostics.py --count 3

# Exercise the low battery warning with simulated data.
python tools/robot_diagnostics.py --battery-percent 15 --count 1

# Exercise the critical battery alert.
python tools/robot_diagnostics.py --battery-percent 5 --count 1

# Adjust interval, thresholds and service timeout.
python tools/robot_diagnostics.py --interval 2 --warn-below 25 --critical-below 12 --timeout 5
```

Warning applies below 20%; critical applies below 10%. These are configurable
project defaults, not manufacturer-specified operating limits. Charging does
not hide low battery alerts. Unknown charge is never substituted with zero.
Invalid battery units/ranges and partially valid sensor groups are reported.
Missing/invalid sensor groups do not suppress the remaining sensor reports.

`--mode` defaults to `simulation`. Hardware mode requires `--host` and accepts
an optional `--port` (default 9559). `--battery-percent` and `--charging` are
available in simulation mode.
`--charging` enables the simulated charging flag. Intervals/timeouts must be
positive finite numbers; thresholds require 0 <= critical < warning <= 100.
`--count` must be a positive integer. The equivalent module entry point is
`python -m robot_services.diagnostics`.

Exit codes: 0 for a completed finite run (including reported health alerts), 1
for service/reporting/cleanup failure, 2 for invalid CLI arguments, and 130 for
Ctrl+C. Connection/status/cleanup calls have the configured timeout. Waiting
for a stream sample allows the update interval plus that timeout.

## Reuse and remaining work

`monitor(service, options, emit)` accepts any implementation of `RobotService`;
`emit` receives formatted strings and can be replaced in tests or another client.
`DiagnosticOptions` documents configuration, and `format_report` validates the
snapshot before formatting it. The command-line tool creates a separate
service instance for each run.

## Hardware diagnostics on Linux

Activate the Python environment with `qi` installed, then run the following with
the robot's hostname or IP address:

```bash
python tools/robot_diagnostics.py --mode hardware --host ROBOT_HOST --count 3
```

Omit `--count` to monitor continuously. The adapter reads battery charge,
three-axis torso acceleration, published left and right sonar distances, and
eight foot FSR loads. The CLI reports IMU, ultrasonic, and FSR sample quality.
Charging state is unknown, and joint readings are unavailable.
Physical NAO verification remains pending.

To check the CLI against a local libqi service on Pop!_OS, run:

```bash
source ~/nao-qi-test/bin/activate
python -B tools/qi_loopback_check.py
```

The check starts loopback-only battery and ALMemory services. It exercises
changing battery values, valid and degraded IMU, sonar, and FSR samples, a
dropped session, and SIGINT handling on Linux. Its readings are test values.
