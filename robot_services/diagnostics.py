"""Read-only health monitoring through RobotService, independent of the GUI.

Run: python -m robot_services.diagnostics --mode simulation
Battery thresholds are configurable project defaults, not manufacturer limits.
"""

import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math

from .errors import RobotConnectionError, TelemetryValidationError
from .interface import RobotService
from .models import RobotStatus, SampleQuality, TelemetrySnapshot
from .simulation import SimulationRobotService


@dataclass(frozen=True)
class DiagnosticOptions:
    """Sampling interval/timeout in seconds, battery thresholds in percent.

    count=None monitors until cancellation; a positive count limits reports.
    timeout bounds connection/status/cleanup calls. Stream waits allow interval
    plus timeout, since the next snapshot may not yet be due.
    """

    interval: float = 1.0
    warn_below: float = 20.0
    critical_below: float = 10.0
    count: int | None = None
    timeout: float = 5.0

    def __post_init__(self):
        for name in ("interval", "warn_below", "critical_below", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.interval <= 0 or self.timeout <= 0:
            raise ValueError("interval and timeout must be greater than zero")
        if not 0 <= self.critical_below < self.warn_below <= 100:
            raise ValueError("thresholds must satisfy 0 <= critical < warning <= 100")
        if self.count is not None and (
            isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0
        ):
            raise ValueError("count must be a positive integer")


def format_report(
    status: RobotStatus, snapshot: TelemetrySnapshot, options: DiagnosticOptions
) -> str:
    """Return a labeled health report; malformed payloads raise validation errors.

    Missing/invalid/partial sensors are reported explicitly without stopping
    other monitoring. Battery charge must be in percent and within 0..100.
    Charging is a numeric 0/1 measurement with unit 'boolean', or Unknown.
    """
    payload = snapshot.to_dict()  # Revalidate at the reporting boundary.
    if status.mode != snapshot.mode:
        raise TelemetryValidationError("Status and telemetry operating modes disagree")
    battery = snapshot.battery
    charge = battery.values.get("charge")
    charging = battery.values.get("charging")
    usable = battery.quality in (SampleQuality.VALID, SampleQuality.PARTIAL)
    if not usable or charge is None or charge.value is None:
        battery_text = "Unavailable"
    elif charge.unit != "%" or not 0 <= charge.value <= 100:
        battery_text = "INVALID (expected 0..100 %)"
    else:
        severity = ("CRITICAL" if charge.value < options.critical_below else
                    "WARNING" if charge.value < options.warn_below else "OK")
        battery_text = f"{charge.value:g}% [{severity}]"
    charging_text = "Unknown"
    if usable and charging is not None and charging.unit == "boolean" and charging.value in (0, 1):
        charging_text = "Yes" if charging.value == 1 else "No"
    labels = {SampleQuality.VALID: "OK", SampleQuality.PARTIAL: "PARTIAL",
              SampleQuality.UNAVAILABLE: "UNAVAILABLE", SampleQuality.INVALID: "INVALID"}
    sensors = []
    reasons = []
    for name, sample in (("IMU", snapshot.imu), ("Ultrasonic", snapshot.ultrasonic),
                         ("FSR", snapshot.fsr), ("Joints", snapshot.joints),
                         ("Battery sensor", battery)):
        sensors.append(f"{name}: {labels[sample.quality]}")
        if sample.error:
            reasons.append(f"  {name}: {sample.error}")
    connected = "Connected" if status.connected else "Disconnected"
    lines = [f"{payload['created_at']} [{snapshot.mode.value.upper()}] {connected} | "
             f"Battery: {battery_text} | Charging: {charging_text}",
             " | ".join(sensors)]
    return "\n".join(lines + reasons)


async def monitor(
    service: RobotService,
    options: DiagnosticOptions,
    emit: Callable[[str], None] = print,
) -> int:
    """Connect, report snapshots, and always close streams and disconnect.

    Return 0 after a requested count, or 1 on service/cleanup failure. Health
    warnings remain visible in reports; unavailable sensors do not end the run.
    Cancellation propagates after cleanup. Never calls posture or motion methods.
    """
    stream = None
    result = 0
    try:
        status = await asyncio.wait_for(service.connect(), options.timeout)
        if not status.connected:
            raise RobotConnectionError("Backend did not establish a connection")
        emit(f"[{status.mode.value.upper()}] Health monitoring started; press Ctrl+C to stop.")
        stream = service.stream_telemetry(options.interval)
        reports = 0
        while options.count is None or reports < options.count:
            try:
                snapshot = await asyncio.wait_for(anext(stream), options.interval + options.timeout)
            except StopAsyncIteration:
                raise RobotConnectionError("Telemetry stream ended before monitoring completed") from None
            status = await asyncio.wait_for(service.get_status(), options.timeout)
            if not status.connected:
                raise RobotConnectionError("Backend disconnected during monitoring")
            emit(format_report(status, snapshot, options))
            reports += 1
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        emit(f"ERROR: Monitoring failed: {detail}")
        result = 1
    finally:
        if stream is not None:
            try:
                await asyncio.wait_for(stream.aclose(), options.timeout)
            except Exception as exc:
                emit(f"ERROR: Stream cleanup failed: {str(exc) or type(exc).__name__}")
                result = 1
        try:
            await asyncio.wait_for(service.disconnect(), options.timeout)
        except Exception as exc:
            emit(f"ERROR: Disconnect failed: {str(exc) or type(exc).__name__}")
            result = 1
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return 0 normally, 1 on failure, 130 for Ctrl+C.

    argparse rejects invalid arguments with exit code 2. Hardware mode is not
    implemented: the CLI accepts only explicit/default simulation mode.
    """
    parser = argparse.ArgumentParser(description="Standalone NAO health diagnostics (simulation)")
    parser.add_argument("--mode", choices=("simulation",), default="simulation")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings (default: 1)")
    parser.add_argument("--warn-below", type=float, default=20, help="Battery warning threshold in percent")
    parser.add_argument("--critical-below", type=float, default=10, help="Critical battery threshold in percent")
    parser.add_argument("--count", type=int, help="Stop after this many reports; otherwise run continuously")
    parser.add_argument("--timeout", type=float, default=5, help="Service timeout in seconds")
    parser.add_argument("--battery-percent", type=float, default=100, help="Simulated battery charge (0..100)")
    parser.add_argument("--charging", action="store_true", help="Simulate a charging battery")
    args = parser.parse_args(argv)
    try:
        options = DiagnosticOptions(args.interval, args.warn_below, args.critical_below, args.count, args.timeout)
        service = SimulationRobotService(battery_percent=args.battery_percent, charging=args.charging)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return asyncio.run(monitor(service, options, lambda text: print(text, flush=True)))
    except KeyboardInterrupt:
        print("Monitoring stopped by user; cleanup attempted.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
