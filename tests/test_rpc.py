"""Hardware-free RPC contract tests.

Run from the repository root:
    python -B -m unittest discover -s tests -p test_rpc.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

from robot_services.rpc import NaoRpcClient, NaoRpcError


class NaoRpcClientTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.connection.recv.side_effect = [b"SUCCESS", b""]
        self.context = MagicMock()
        self.context.__enter__.return_value = self.connection
        self.patcher = patch(
            "robot_services.rpc.socket.create_connection", return_value=self.context
        )
        self.create_connection = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_public_commands_send_existing_bridge_commands(self):
        for method, command in (
            ("connect", b"connect"),
            ("sit_down", b"sit_down"),
            ("stand_up", b"stand_up"),
        ):
            with self.subTest(method=method):
                self.connection.reset_mock()
                self.connection.recv.side_effect = [b"SUCCESS", b""]
                self.context.reset_mock()
                self.create_connection.reset_mock()
                self.assertIs(getattr(NaoRpcClient(), method)(), True)
                self.create_connection.assert_called_once_with(
                    ("localhost", 5000), timeout=5.0
                )
                self.connection.sendall.assert_called_once_with(command)
                self.context.__exit__.assert_called_once_with(None, None, None)

    def test_custom_bridge_settings(self):
        NaoRpcClient(host="bridge.example", port=6000, timeout=2.5).connect()
        self.create_connection.assert_called_once_with(
            ("bridge.example", 6000), timeout=2.5
        )

    def test_failed_response_returns_false(self):
        self.connection.recv.side_effect = [b"FAILED", b""]
        self.assertIs(NaoRpcClient().connect(), False)
        self.context.__exit__.assert_called_once_with(None, None, None)

    def test_fragmented_response_with_whitespace(self):
        self.connection.recv.side_effect = [b" SU", b"CC", b"ESS\r\n", b""]
        self.assertIs(NaoRpcClient().connect(), True)

    def test_invalid_responses_raise_and_close_socket(self):
        for reply in (b"", b"UNKNOWN", b"SUCCESSFAILED", b"\xff"):
            with self.subTest(reply=reply):
                self.context.reset_mock()
                self.create_connection.reset_mock()
                self.connection.recv.side_effect = [reply, b""]
                with self.assertRaisesRegex(NaoRpcError, "Unexpected"):
                    NaoRpcClient().connect()
                self.context.__exit__.assert_called_once()
                self.create_connection.assert_called_once()

    def test_oversized_response_raises_and_closes_socket(self):
        self.connection.recv.side_effect = [b"x" * 1024, b"x"]
        with self.assertRaisesRegex(NaoRpcError, "exceeded"):
            NaoRpcClient().connect()
        self.context.__exit__.assert_called_once()

    def test_connection_failures_are_wrapped_without_retry(self):
        for failure in (ConnectionRefusedError("refused"), TimeoutError("timed out")):
            with self.subTest(failure=type(failure).__name__):
                self.create_connection.reset_mock()
                self.create_connection.side_effect = failure
                with self.assertRaises(NaoRpcError) as caught:
                    NaoRpcClient().connect()
                self.assertIs(caught.exception.__cause__, failure)
                self.create_connection.assert_called_once()
                self.connection.sendall.assert_not_called()

    def test_send_and_receive_failures_close_socket_without_retry(self):
        for operation, failure in (
            ("sendall", BrokenPipeError("broken pipe")),
            ("recv", TimeoutError("timed out")),
            ("recv", ConnectionResetError("reset")),
        ):
            with self.subTest(operation=operation, failure=type(failure).__name__):
                self.connection.reset_mock()
                self.context.reset_mock()
                self.create_connection.reset_mock()
                self.connection.sendall.side_effect = None
                self.connection.recv.side_effect = [b"SUCCESS", b""]
                getattr(self.connection, operation).side_effect = failure
                with self.assertRaises(NaoRpcError) as caught:
                    NaoRpcClient().stand_up()
                self.assertIs(caught.exception.__cause__, failure)
                self.context.__exit__.assert_called_once()
                self.connection.sendall.assert_called_once_with(b"stand_up")
                self.create_connection.assert_called_once()

    def test_each_command_opens_and_closes_a_connection(self):
        self.connection.recv.side_effect = [b"SUCCESS", b"", b"FAILED", b""]
        client = NaoRpcClient()
        self.assertIs(client.connect(), True)
        self.assertIs(client.sit_down(), False)
        self.assertEqual(self.create_connection.call_count, 2)
        self.assertEqual(self.context.__enter__.call_count, 2)
        self.assertEqual(self.context.__exit__.call_count, 2)

    def test_nonpositive_timeout_rejected_before_network_access(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    NaoRpcClient(timeout=timeout)
        self.create_connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
