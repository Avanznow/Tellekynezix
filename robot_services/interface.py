"""Shared asynchronous service contract for simulation and hardware backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .models import MotionState, PostureResult, RobotStatus, TelemetrySnapshot
from .errors import UnsupportedOperationError


class RobotService(ABC):
    """Backend contract, with no Qt, libqi or network dependency.

    Backends raise RobotConnectionError for communication failures and
    UnsupportedOperationError for unsupported requests. Invalid arguments raise
    ValueError. Cancellation must propagate and release streaming resources.
    A failed sensor yields a marked sample; lost transport raises an error.
    Implementations must not automatically retry actuation requests.
    """

    @abstractmethod
    async def connect(self) -> RobotStatus:
        """Establish a connection and return its status; raise on failure."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Stop streams and release resources; repeated calls must be safe."""

    @abstractmethod
    async def get_status(self) -> RobotStatus:
        """Return connection, busy state, mode and last known posture."""

    @abstractmethod
    async def set_posture(self, name: str, speed: float) -> PostureResult:
        """Request a supported posture at finite normalized speed 0 < speed <= 1.

        Names and support are backend-specific until SDK verification. Accepted
        requests must not be reported as verified completion without observation.
        """

    @abstractmethod
    async def get_posture(self) -> str | None:
        """Return observed posture, or None when connected but unknown."""

    async def get_motion_capabilities(self) -> tuple[str, ...]:
        """Return supported continuous motion names; empty for unsupported backends."""
        return ()

    async def get_motion_state(self) -> MotionState:
        """Return visual/odometry state only when the backend supports it."""
        raise UnsupportedOperationError("Motion state is not supported")

    async def start_motion(self, command: str, speed: float = 0.5) -> MotionState:
        """Start a supported continuous motion at normalized speed 0 < speed <= 1.

        Return after acceptance; motion continues until stopped or disconnected.
        Unsupported backends must not silently emulate physical motion.
        """
        raise UnsupportedOperationError("Continuous motion is not supported")

    async def stop_motion(self) -> MotionState:
        """Halt travel/turning, then settle to Stand where supported.

        Position and heading must stay fixed during settling. Stop during a
        posture transition freezes its pose and marks the posture unknown.
        """
        raise UnsupportedOperationError("Motion stopping is not supported")

    @abstractmethod
    async def read_telemetry(self) -> TelemetrySnapshot:
        """Return one snapshot; individual sensor failures permit partial data."""

    @abstractmethod
    def stream_telemetry(self, interval: float) -> AsyncIterator[TelemetrySnapshot]:
        """Yield snapshots at a requested positive finite interval in seconds.

        Consume with 'async for'. Interval is a requested minimum spacing, not a
        real-time guarantee. Consumers close abandoned streams with aclose();
        disconnect and cancellation must also terminate streaming cleanly.
        """
