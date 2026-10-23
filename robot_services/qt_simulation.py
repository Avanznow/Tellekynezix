"""Qt adapter for the simulation service; all async work runs on one worker loop."""

import asyncio
import math
import threading
from concurrent.futures import CancelledError

from PySide6.QtCore import QObject, Property, Signal, Slot

from .simulation import SimulationRobotService


class SimulationController(QObject):
    """Expose GUI-thread properties and slots without importing a hardware client.

    Connect starts telemetry automatically at telemetry_interval seconds. The
    generation token discards queued messages from previous connection sessions.
    Call close() when the application quits to cancel work and join the worker.
    """

    changed = Signal()
    logMessage = Signal(str)
    _event = Signal(int, object)

    def __init__(self, parent=None, *, service=None, telemetry_interval=0.5):
        super().__init__(parent)
        if (isinstance(telemetry_interval, bool)
                or not isinstance(telemetry_interval, (int, float))
                or not math.isfinite(telemetry_interval) or telemetry_interval <= 0):
            raise ValueError("telemetry_interval must be positive and finite")
        self._service = service if service is not None else SimulationRobotService()
        self._interval = telemetry_interval
        self._connected = False
        self._busy = False
        self._pending = False
        self._posture = "Unknown"
        self._telemetry = {}
        self._generation = 0
        self._closed = False
        self._monitor = None
        self._event.connect(self._receive)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="NAO simulation", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("Simulation worker failed to start")

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            tasks = asyncio.all_tasks(self._loop)
            for task in tasks:
                task.cancel()
            if tasks:
                self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    @Property(bool, notify=changed)
    def connected(self):
        return self._connected

    @Property(bool, notify=changed)
    def busy(self):
        return self._busy

    @Property(bool, notify=changed)
    def pending(self):
        return self._pending

    @Property(str, notify=changed)
    def posture(self):
        return self._posture

    @Property(str, notify=changed)
    def statusText(self):
        if self._pending:
            return "Connecting..." if not self._connected else "Disconnecting..."
        return "Busy" if self._busy else ("Connected" if self._connected else "Disconnected")

    @Property("QVariantMap", notify=changed)
    def telemetry(self):
        return self._telemetry

    @Slot(int, object)
    def _receive(self, generation, event):
        if self._closed or generation != self._generation:
            return
        if "log" in event:
            self.logMessage.emit(event["log"])
        if "status" in event:
            status = event["status"]
            self._connected = status.connected
            self._busy = status.busy
            self._posture = status.posture or "Unknown"
        if "pending" in event:
            self._pending = event["pending"]
        if "telemetry" in event:
            self._telemetry = event["telemetry"]
        self.changed.emit()

    def _submit(self, coroutine, generation):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        def done(result):
            try:
                result.result()
            except CancelledError:
                pass
            except Exception as exc:
                self._event.emit(generation, {"pending": False, "log": f"Simulation error: {exc}"})
        future.add_done_callback(done)

    @Slot()
    def connectSimulation(self):
        if self._closed or self._pending or self._connected:
            return
        self._generation += 1
        generation = self._generation
        self._pending = True
        self.changed.emit()
        self._submit(self._connect(generation), generation)

    async def _connect(self, generation):
        status = await self._service.connect()
        self._event.emit(generation, {"status": status, "pending": False,
                                     "log": "Simulation connected; no hardware connection."})
        self._monitor = asyncio.create_task(self._watch(generation))

    async def _watch(self, generation):
        stream = None
        try:
            stream = self._service.stream_telemetry(self._interval)
            async for snapshot in stream:
                self._event.emit(generation, {"telemetry": snapshot.to_dict()})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._event.emit(generation, {"telemetry": {}, "log": f"Telemetry stopped: {exc}"})
        finally:
            if stream is not None:
                await stream.aclose()

    @Slot()
    def disconnectSimulation(self):
        if self._closed or self._pending:
            return
        self._generation += 1
        generation = self._generation
        self._pending = True
        self._telemetry = {}
        self.changed.emit()
        self._submit(self._disconnect(generation), generation)

    async def _disconnect(self, generation):
        await self._service.disconnect()
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
            self._monitor = None
        self._event.emit(generation, {"status": await self._service.get_status(),
                                     "pending": False, "telemetry": {},
                                     "log": "Simulation disconnected; monitoring stopped."})

    @Slot(str)
    def requestPosture(self, name):
        if self._closed or self._pending or not self._connected or self._busy:
            return
        self._busy = True
        self.changed.emit()
        self.logMessage.emit(f"Simulation: requesting {name}.")
        self._submit(self._posture_request(name, self._generation), self._generation)

    async def _posture_request(self, name, generation):
        try:
            result = await self._service.set_posture(name, 0.5)
            self._event.emit(generation, {"log": f"Simulation: {result.observed_posture} completed and verified (simulated state only)."})
        except asyncio.CancelledError:
            self._event.emit(generation, {"log": "Simulated posture request cancelled."})
            raise
        finally:
            self._event.emit(generation, {"status": await self._service.get_status()})

    @Slot()
    def close(self):
        """Cancel active work and stop/join the loop; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(self._disconnect(self._generation), self._loop).result(5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(5)
        if self._thread.is_alive():
            raise RuntimeError("Simulation worker did not stop")
