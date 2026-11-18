"""Version 1 simulation-only TCP protocol and command round-trip probe."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import math

from .errors import RobotServiceError
from .simulation import SimulationRobotService


VERSION = 1
MAX_MESSAGE = 65536


def _encode(message):
    data = json.dumps(message, allow_nan=False).encode("utf-8") + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("Message exceeds 65536 bytes")
    return data


def _decode(data):
    if not data.endswith(b"\n") or len(data) > MAX_MESSAGE:
        raise ValueError("Expected a complete JSON line of at most 65536 bytes")
    message = json.loads(data)
    if not isinstance(message, dict) or type(message.get("version")) is not int or message["version"] != VERSION:
        raise ValueError("Expected protocol version 1")
    request_id = message.get("id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
        raise ValueError("id must be a string of 1 to 64 characters")
    return message


async def _dispatch(service, request):
    if set(request) != {"version", "id", "method", "params"}:
        raise ValueError("Expected version, id, method and params")
    method, params = request["method"], request["params"]
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if method == "set_posture":
        if set(params) != {"name", "speed"}:
            raise ValueError("set_posture requires name and speed")
        result = asdict(await service.set_posture(params["name"], params["speed"]))
    else:
        if params:
            raise ValueError("This method takes no parameters")
        if method == "connect":
            result = asdict(await service.connect())
        elif method == "get_status":
            result = asdict(await service.get_status())
        elif method == "get_motion_state":
            result = asdict(await service.get_motion_state())
        elif method == "read_telemetry":
            result = (await service.read_telemetry()).to_dict()
        elif method == "disconnect":
            await service.disconnect()
            result = asdict(await service.get_status())
        else:
            raise ValueError("Unsupported simulation method")
    return result


@asynccontextmanager
async def simulation_server(host="127.0.0.1", port=8765, timeout=10.0):
    """Own client tasks and simulation sessions until the server context ends."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    clients = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        clients.add(task)
        service = SimulationRobotService()
        last_id = 0
        try:
            while True:
                data = await asyncio.wait_for(reader.readline(), timeout)
                if not data:
                    break
                request_id = None
                try:
                    request = _decode(data)
                    request_id = request["id"]
                    if not request_id.isascii() or not request_id.isdecimal():
                        raise ValueError("Request ids must be positive decimal integers")
                    numeric_id = int(request_id)
                    if numeric_id <= last_id:
                        raise ValueError("Duplicate or out-of-order request id")
                    last_id = numeric_id
                    result = await asyncio.wait_for(_dispatch(service, request), timeout)
                    reply = {"version": VERSION, "id": request_id, "ok": True, "result": result}
                except (ValueError, RobotServiceError, TimeoutError) as exc:
                    reply = {"version": VERSION, "id": request_id, "ok": False,
                             "error": {"code": type(exc).__name__, "message": str(exc) or "Operation timed out"}}
                writer.write(_encode(reply))
                await asyncio.wait_for(writer.drain(), timeout)
                if request_id is None:
                    break
        except (OSError, TimeoutError, ValueError):
            pass
        finally:
            try:
                await service.disconnect()
            finally:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout)
                except (OSError, TimeoutError):
                    pass
                clients.discard(task)

    server = await asyncio.start_server(handle, host, port, limit=MAX_MESSAGE)
    try:
        yield server
    finally:
        server.close()
        for task in list(clients):
            task.cancel()
        if clients:
            await asyncio.gather(*list(clients), return_exceptions=True)
        await server.wait_closed()


async def probe(host, port=8765, timeout=10.0):
    """Send Stand once, verify its simulated result, then disconnect."""
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port, limit=MAX_MESSAGE), timeout)
    try:
        replies = []
        steps = (
            ("connect", {}),
            ("set_posture", {"name": "Stand", "speed": 1.0}),
            ("get_status", {}),
            ("disconnect", {}),
        )
        for index, (method, params) in enumerate(steps):
            request_id = str(index + 1)
            writer.write(_encode({"version": VERSION, "id": request_id, "method": method, "params": params}))
            await asyncio.wait_for(writer.drain(), timeout)
            reply = _decode(await asyncio.wait_for(reader.readline(), timeout))
            if reply["id"] != request_id or reply.get("ok") is not True:
                raise ValueError(f"Unsuccessful {method} response: {reply}")
            result = reply.get("result")
            if not isinstance(result, dict):
                raise ValueError(f"Expected an object result from {method}")
            replies.append(result)
        connected, posture, status, disconnected = replies
        if (connected.get("mode") != "simulation" or connected.get("connected") is not True
                or posture.get("outcome") != "completed_verified" or posture.get("observed_posture") != "Stand"
                or status.get("mode") != "simulation" or status.get("posture") != "Stand"
                or status.get("busy") is not False or disconnected.get("connected") is not False):
            raise ValueError("Responses did not verify the simulated Stand round trip")
        return posture
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout)
        except (OSError, TimeoutError):
            pass


def main():
    parser = argparse.ArgumentParser(description="NAO simulation connection test")
    parser.add_argument("action", choices=("serve", "probe"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("port must be 1..65535 and timeout must be positive and finite")

    async def run():
        if args.action == "serve":
            async with simulation_server(args.host, args.port, args.timeout) as server:
                print(f"SIMULATION server listening on {server.sockets[0].getsockname()}", flush=True)
                await server.serve_forever()
        else:
            result = await probe(args.host, args.port, args.timeout)
            print("PASS: simulation connected, Stand completed and verified, then disconnected")
            print(json.dumps(result))

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"Connection test failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
