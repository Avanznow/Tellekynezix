"""Hardware-free tests for the Avatar link lifecycle manager."""

import unittest

from robot_services.link_manager import LinkManager, LinkState


class LinkManagerTests(unittest.TestCase):
    def setUp(self):
        self.fail_safe_calls = []
        self.reconnect_calls = []
        self.reconnect_result = False
        self.transitions = []
        self.manager = LinkManager(
            lambda: self.fail_safe_calls.append("safe"),
            lambda: self.reconnect_calls.append("attempt") or self.reconnect_result,
            on_state_change=lambda old, new: self.transitions.append(new),
            heartbeat_timeout=3,
            reconnect_initial_delay=1,
            reconnect_max_delay=4,
        )

    def test_heartbeat_connects_and_signal_quality_degrades(self):
        self.assertIs(
            self.manager.heartbeat(1, received_at=10, rssi_dbm=-70, snr_db=10),
            LinkState.CONNECTED,
        )
        self.assertIs(self.manager.poll(12), LinkState.CONNECTED)
        self.manager.heartbeat(2, received_at=12, rssi_dbm=-95, snr_db=10)
        self.assertIs(self.manager.poll(13), LinkState.DEGRADED)
        self.manager.heartbeat(3, received_at=13, rssi_dbm=-80, snr_db=9)
        self.assertIs(self.manager.poll(14), LinkState.CONNECTED)

    def test_packet_loss_is_counted_and_reported_as_degraded(self):
        self.manager.heartbeat(5, received_at=10)
        self.manager.heartbeat(8, received_at=11)
        self.assertEqual(self.manager.missed_heartbeats, 2)
        self.assertIs(self.manager.poll(12), LinkState.DEGRADED)
        self.manager.heartbeat(9, received_at=12)
        self.assertIs(self.manager.poll(13), LinkState.CONNECTED)

    def test_timeout_applies_fail_safe_once_and_reconnects_with_backoff(self):
        self.manager.heartbeat(1, received_at=0)
        self.assertIs(self.manager.poll(3.1), LinkState.FAIL_SAFE)
        self.assertEqual(self.fail_safe_calls, ["safe"])
        self.assertEqual(
            self.transitions,
            [LinkState.CONNECTED, LinkState.LINK_LOST, LinkState.FAIL_SAFE],
        )
        self.assertIs(self.manager.poll(4.0), LinkState.FAIL_SAFE)
        self.assertIs(self.manager.poll(4.1), LinkState.FAIL_SAFE)
        self.assertEqual(self.reconnect_calls, ["attempt"])
        self.assertIs(self.manager.poll(5.2), LinkState.FAIL_SAFE)
        self.assertEqual(self.fail_safe_calls, ["safe"])
        self.assertIs(self.manager.poll(7.2), LinkState.FAIL_SAFE)
        self.assertEqual(self.reconnect_calls, ["attempt", "attempt", "attempt"])
        self.assertEqual(
            self.transitions.count(LinkState.RECONNECTING),
            3,
        )

    def test_heartbeat_recovers_reconnecting_session_and_resets_backoff(self):
        self.manager.heartbeat(10, received_at=0)
        self.manager.poll(4)
        self.reconnect_result = True
        self.assertIs(self.manager.poll(5), LinkState.RECONNECTING)
        self.assertIs(self.manager.heartbeat(11, received_at=5), LinkState.CONNECTED)
        self.assertIs(self.manager.poll(6), LinkState.CONNECTED)
        self.assertEqual(self.fail_safe_calls, ["safe"])

    def test_sequence_wrap_is_allowed_but_replay_is_rejected(self):
        self.manager.heartbeat(0xFFFFFFFF, received_at=1)
        self.manager.heartbeat(0, received_at=2)
        for stale in (0, 0xFFFFFFFF):
            with self.subTest(sequence=stale), self.assertRaisesRegex(
                ValueError, "stale"
            ):
                self.manager.heartbeat(stale, received_at=3)

    def test_validation_and_disconnect(self):
        for timeout in (0, float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                LinkManager(lambda: None, lambda: False, heartbeat_timeout=timeout)
        with self.assertRaises(ValueError):
            self.manager.heartbeat(-1, received_at=0)
        self.manager.heartbeat(1, received_at=0)
        self.assertIs(self.manager.disconnect(), LinkState.DISCONNECTED)
        self.assertIs(self.manager.poll(10), LinkState.DISCONNECTED)


if __name__ == "__main__":
    unittest.main()
