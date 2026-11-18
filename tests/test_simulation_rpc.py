"""Real TCP round trips against isolated simulation sessions."""

import asyncio
import json
import unittest

from robot_services.simulation_rpc import simulation_server, probe
from robot_services.remote_simulation import RemoteSimulationService


class SimulationRpcTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = simulation_server(port=0, timeout=3)
        self.server = await self.context.__aenter__()
        self.port = self.server.sockets[0].getsockname()[1]
        self.addAsyncCleanup(self.context.__aexit__, None, None, None)

    async def open_client(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)

        async def close():
            writer.close()
            await writer.wait_closed()

        self.addAsyncCleanup(close)
        return reader, writer

    async def request(self, client, request_id, method, params=None):
        reader, writer = client
        writer.write(json.dumps({"version": 1, "id": request_id, "method": method,
                                 "params": params or {}}).encode() + b"\n")
        await writer.drain()
        return json.loads(await asyncio.wait_for(reader.readline(), 3))

    async def test_stand_round_trip(self):
        result = await probe("127.0.0.1", self.port, timeout=3)
        self.assertEqual(result["requested_posture"], "Stand")
        self.assertEqual(result["observed_posture"], "Stand")
        self.assertEqual(result["outcome"], "completed_verified")
        self.assertIn("simulated", result["message"])

    async def test_remote_client_status_telemetry_posture_and_reconnect(self):
        service = RemoteSimulationService("127.0.0.1", self.port, timeout=3)
        self.addAsyncCleanup(service.disconnect)
        self.assertTrue((await service.connect()).connected)
        await asyncio.gather(service.read_telemetry(), service.get_status())
        result = await service.set_posture("Sit", 1)
        self.assertEqual(result.observed_posture, "Sit")
        self.assertEqual((await service.get_motion_state()).sit_amount, 1)
        self.assertEqual((await service.read_telemetry()).mode.value, "simulation")
        await service.disconnect()
        self.assertFalse((await service.get_status()).connected)
        await service.connect()
        self.assertIsNone((await service.get_status()).posture)

    async def test_remote_client_disconnect_interrupts_command_and_can_reconnect(self):
        service = RemoteSimulationService("127.0.0.1", self.port, timeout=3)
        self.addAsyncCleanup(service.disconnect)
        await service.connect()
        task = asyncio.create_task(service.set_posture("Stand", .5))
        await asyncio.sleep(.01)
        await service.disconnect()
        with self.assertRaises(Exception):
            await asyncio.wait_for(task, 1)
        await service.connect()
        self.assertTrue((await service.get_status()).connected)

    async def test_sessions_are_isolated_and_telemetry_is_versioned(self):
        first, second = await self.open_client(), await self.open_client()
        self.assertTrue((await self.request(first, "1", "connect"))["result"]["connected"])
        self.assertFalse((await self.request(second, "1", "get_status"))["result"]["connected"])
        sample = (await self.request(first, "2", "read_telemetry"))["result"]
        self.assertEqual((sample["schema_version"], sample["mode"], sample["sequence"]),
                         (1, "simulation", 0))
        await self.request(first, "3", "disconnect")
        error = await self.request(first, "4", "read_telemetry")
        self.assertFalse(error["ok"])
        self.assertEqual(error["error"]["code"], "RobotConnectionError")

    async def test_invalid_commands_and_duplicate_ids_never_execute(self):
        client = await self.open_client()
        await self.request(client, "1", "connect")
        bad = await self.request(client, "2", "set_posture", {"name": "Stand", "speed": 0})
        self.assertEqual(bad["error"]["code"], "ValueError")
        duplicate = await self.request(client, "2", "set_posture", {"name": "Stand", "speed": 1})
        self.assertFalse(duplicate["ok"])
        self.assertIn("Duplicate", duplicate["error"]["message"])
        status = await self.request(client, "3", "get_status")
        self.assertIsNone(status["result"]["posture"])
        unsupported = await self.request(client, "4", "execute")
        self.assertFalse(unsupported["ok"])

    async def test_command_timeout_cancels_simulation_and_allows_status_check(self):
        async with simulation_server(port=0, timeout=.1) as server:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            try:
                client = reader, writer
                await self.request(client, "1", "connect")
                reply = await self.request(client, "2", "set_posture", {"name": "Stand", "speed": .5})
                self.assertFalse(reply["ok"])
                self.assertEqual(reply["error"]["code"], "TimeoutError")
                status = (await self.request(client, "3", "get_status"))["result"]
                self.assertFalse(status["busy"])
                self.assertIsNone(status["posture"])
            finally:
                writer.close()
                await writer.wait_closed()

    async def test_fragmented_json_and_malformed_input(self):
        reader, writer = await self.open_client()
        payload = b'{"version":1,"id":"1","method":"connect","params":{}}\n'
        writer.write(payload[:15])
        await writer.drain()
        writer.write(payload[15:])
        await writer.drain()
        self.assertTrue(json.loads(await reader.readline())["ok"])
        writer.write(b"not json\n")
        await writer.drain()
        self.assertFalse(json.loads(await reader.readline())["ok"])
        self.assertEqual(await reader.readline(), b"")


if __name__ == "__main__":
    unittest.main()
