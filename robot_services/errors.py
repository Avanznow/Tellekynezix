"""Errors shared by hardware and simulation service implementations."""


class RobotServiceError(Exception):
    """Base error for an unsuccessful service operation."""


class RobotConnectionError(RobotServiceError):
    """The backend is disconnected or communication failed."""


class RobotBusyError(RobotServiceError):
    """Another posture operation is already in progress."""


class UnsupportedOperationError(RobotServiceError):
    """The backend does not support the requested operation or posture."""


class TelemetryValidationError(RobotServiceError, ValueError):
    """A telemetry payload violates the shared data contract."""
