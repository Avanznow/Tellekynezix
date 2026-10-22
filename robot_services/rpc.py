"""Synchronous client for the existing local NAO command bridge."""

import socket


class NaoRpcError(Exception):
    """The bridge could not be reached or returned an invalid response."""


class NaoRpcClient:
    """Send commands to the bridge, which connects to the physical robot.

    Args:
        host: Bridge hostname or IP address, rather than the robot's address.
        port: Bridge TCP port.
        timeout: Positive connection and socket-operation timeout in seconds.

    Each command uses a fresh socket and is sent exactly once. A transport
    failure does not prove that the robot did not execute the command.
    """

    def __init__(
        self, host: str = "localhost", port: int = 5000, timeout: float = 5.0
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.host = host
        self.port = port
        self.timeout = timeout

    def connect(self) -> bool:
        """Ask the bridge to check its robot connection; return its success."""
        return self._send_command("connect")

    def sit_down(self) -> bool:
        """Request sitting; return whether the bridge accepted the request.

        Success does not verify that the physical posture has been reached.
        """
        return self._send_command("sit_down")

    def stand_up(self) -> bool:
        """Request standing; return whether the bridge accepted the request.

        Success does not verify that the physical posture has been reached.
        """
        return self._send_command("stand_up")

    def _send_command(self, command: str) -> bool:
        """Return False for FAILED; raise NaoRpcError for transport/protocol errors."""
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            ) as connection:
                connection.sendall(command.encode("ascii"))
                # The bridge closes each connection after sending its reply.
                # Read until EOF because TCP may split even a short response.
                response = bytearray()
                while True:
                    chunk = connection.recv(1024)
                    if not chunk:
                        break
                    response.extend(chunk)
                    if len(response) > 1024:
                        raise NaoRpcError("NAO bridge response exceeded 1024 bytes")
        except OSError as exc:
            raise NaoRpcError(
                f"NAO bridge communication failed at {self.host}:{self.port}: {exc}"
            ) from exc

        reply = bytes(response).strip()
        if reply == b"SUCCESS":
            return True
        if reply == b"FAILED":
            return False
        raise NaoRpcError(f"Unexpected NAO bridge response: {reply!r}")
