"""Transport-independent heartbeat, link-quality, and fail-safe management."""

from collections.abc import Callable
from enum import Enum
import math
import time


class LinkState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    LINK_LOST = "link_lost"
    FAIL_SAFE = "fail_safe"
    RECONNECTING = "reconnecting"


class LinkManager:
    """Track authenticated peer heartbeats and request fail-safe/reconnect actions.

    The caller supplies decoded, authenticated heartbeats and polls this manager
    regularly. Callbacks perform application-specific safe actions and one
    reconnect attempt; this class does not own sockets or retry actuation.
    """

    def __init__(
        self,
        on_fail_safe: Callable[[], None],
        attempt_reconnect: Callable[[], bool],
        *,
        on_state_change: Callable[[LinkState, LinkState], None] | None = None,
        heartbeat_timeout: float = 3.0,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        minimum_rssi_dbm: float = -90.0,
        minimum_snr_db: float = 5.0,
    ) -> None:
        for name, value in (
            ("heartbeat_timeout", heartbeat_timeout),
            ("reconnect_initial_delay", reconnect_initial_delay),
            ("reconnect_max_delay", reconnect_max_delay),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if reconnect_max_delay < reconnect_initial_delay:
            raise ValueError(
                "reconnect_max_delay must be at least reconnect_initial_delay"
            )
        for name, value in (
            ("minimum_rssi_dbm", minimum_rssi_dbm),
            ("minimum_snr_db", minimum_snr_db),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not callable(on_fail_safe) or not callable(attempt_reconnect):
            raise TypeError("Fail-safe and reconnect callbacks are required")

        self.on_fail_safe = on_fail_safe
        self.attempt_reconnect = attempt_reconnect
        if on_state_change is not None and not callable(on_state_change):
            raise TypeError("on_state_change must be callable")
        self.on_state_change = on_state_change
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.reconnect_initial_delay = float(reconnect_initial_delay)
        self.reconnect_max_delay = float(reconnect_max_delay)
        self.minimum_rssi_dbm = float(minimum_rssi_dbm)
        self.minimum_snr_db = float(minimum_snr_db)
        self.state = LinkState.DISCONNECTED
        self.last_heartbeat_at: float | None = None
        self.last_sequence: int | None = None
        self.last_rssi_dbm: float | None = None
        self.last_snr_db: float | None = None
        self._next_reconnect_at: float | None = None
        self._reconnect_delay = self.reconnect_initial_delay
        self._fail_safe_applied = False
        self.missed_heartbeats = 0
        self._packet_loss_detected = False

    def _set_state(self, state: LinkState) -> None:
        if self.state is state:
            return
        previous = self.state
        self.state = state
        if self.on_state_change is not None:
            self.on_state_change(previous, state)

    @staticmethod
    def _time_value(now: float) -> float:
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
        ):
            raise ValueError("now must be finite")
        return float(now)

    def heartbeat(
        self,
        sequence: int,
        *,
        received_at: float | None = None,
        rssi_dbm: float | None = None,
        snr_db: float | None = None,
    ) -> LinkState:
        """Record a fresh authenticated peer heartbeat and return current state."""
        now = self._time_value(time.monotonic() if received_at is None else received_at)
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or not 0 <= sequence <= 0xFFFFFFFF
        ):
            raise ValueError("sequence must be an unsigned 32-bit integer")
        advance = 1
        if self.last_sequence is not None:
            advance = (sequence - self.last_sequence) & 0xFFFFFFFF
            if advance == 0 or advance >= 0x80000000:
                raise ValueError("heartbeat sequence is duplicated or stale")
        for name, value in (("rssi_dbm", rssi_dbm), ("snr_db", snr_db)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite or None")

        self.last_sequence = sequence
        self.last_heartbeat_at = now
        if self.last_sequence is not None:
            self.missed_heartbeats += advance - 1
        self._packet_loss_detected = advance > 1
        self.last_rssi_dbm = float(rssi_dbm) if rssi_dbm is not None else None
        self.last_snr_db = float(snr_db) if snr_db is not None else None
        self._next_reconnect_at = None
        self._reconnect_delay = self.reconnect_initial_delay
        self._fail_safe_applied = False
        self._set_state(LinkState.CONNECTED)
        return self.state

    def poll(self, now: float | None = None) -> LinkState:
        """Refresh state, run due fail-safe/reconnect callbacks, and return state."""
        current_time = self._time_value(time.monotonic() if now is None else now)
        if self.last_heartbeat_at is None:
            return self.state

        heartbeat_age = current_time - self.last_heartbeat_at
        if heartbeat_age < 0:
            raise ValueError("now cannot precede the latest heartbeat")
        if heartbeat_age > self.heartbeat_timeout:
            if self.state not in (
                LinkState.LINK_LOST,
                LinkState.FAIL_SAFE,
                LinkState.RECONNECTING,
            ):
                self._set_state(LinkState.LINK_LOST)
            if not self._fail_safe_applied:
                self._fail_safe_applied = True
                self._next_reconnect_at = current_time + self._reconnect_delay
                self._set_state(LinkState.FAIL_SAFE)
                self.on_fail_safe()
            if (
                self._next_reconnect_at is not None
                and current_time >= self._next_reconnect_at
            ):
                self._next_reconnect_at = current_time + self._reconnect_delay
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.reconnect_max_delay
                )
                self._set_state(LinkState.RECONNECTING)
                reconnected = self.attempt_reconnect()
                if not reconnected:
                    self._set_state(LinkState.FAIL_SAFE)
            return self.state

        if (
            (
                self.last_rssi_dbm is not None
                and self.last_rssi_dbm < self.minimum_rssi_dbm
            )
            or (self.last_snr_db is not None and self.last_snr_db < self.minimum_snr_db)
            or self._packet_loss_detected
        ):
            self._set_state(LinkState.DEGRADED)
        else:
            self._set_state(LinkState.CONNECTED)
        return self.state

    def disconnect(self) -> LinkState:
        """Clear peer session state without invoking hardware actions."""
        self._set_state(LinkState.DISCONNECTED)
        self.last_heartbeat_at = None
        self.last_sequence = None
        self.last_rssi_dbm = None
        self.last_snr_db = None
        self._next_reconnect_at = None
        self._reconnect_delay = self.reconnect_initial_delay
        self._fail_safe_applied = False
        self.missed_heartbeats = 0
        self._packet_loss_detected = False
        return self.state
