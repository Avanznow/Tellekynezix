"""Qt adapter for the simulation service; all async work runs on one worker loop."""

import asyncio
import math
import threading
from dataclasses import asdict
from concurrent.futures import CancelledError

from PySide6.QtCore import QObject, Property, Signal, Slot

from .simulation import SimulationRobotService
from .models import MotionState


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
        self._command_active = False
        self._motion_switching = False
        self._motion_request_id = 0
        self._stopping = False
        self._pending = False
        self._posture = "Unknown"
        self._telemetry = {}
        self._motion = asdict(MotionState())
        self._generation = 0
        self._closed = False
        self._monitor = None
        self._state_monitor = None
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
        return self._busy or self._command_active or self._stopping

    @Property(bool, notify=changed)
    def stopping(self):
        return self._stopping

    @Property(bool, notify=changed)
    def canSwitchMotion(self):
        """Allow motion selection when idle or during an automatic motion switch."""
        return (self._connected and not self._closed and not self._pending
                and not self._stopping
                and (not self.busy or self._motion_switching or self._motion["action"] in
                     ("forward", "backward", "turn_left", "turn_right")))

    @Property("QVariantMap", notify=changed)
    def motion(self):
        return self._motion

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
        if self.busy:
            labels = {"forward": "Walking forward", "backward": "Walking backward",
                      "turn_left": "Turning left", "turn_right": "Turning right",
                      "settling": "Settling into Stand", "posture": "Changing posture"}
            return labels.get(self._motion["action"], "Busy")
        return "Connected" if self._connected else "Disconnected"

    @Property("QVariantMap", notify=changed)
    def telemetry(self):
        return self._telemetry

    @Slot(int, object)
    def _receive(self, generation, event):
        if self._closed or generation != self._generation:
            return
        if "requestId" in event and event["requestId"] != self._motion_request_id:
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
        if "motion" in event:
            self._motion = event["motion"]
        if event.get("commandDone"):
            self._command_active = False
            self._motion_switching = False
        if "stopping" in event:
            self._stopping = event["stopping"]
        self.changed.emit()

    def _submit(self, coroutine, generation):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        def done(result):
            try:
                result.result()
            except CancelledError:
                pass
            except Exception as exc:
                self._event.emit(generation, {"pending": False, "commandDone": True,
                                             "log": f"Simulation error: {exc}"})
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
        self._state_monitor = asyncio.create_task(self._watch_state(generation))

    async def _watch_state(self, generation):
        while True:
            self._event.emit(generation, {
                "motion": asdict(await self._service.get_motion_state()),
                "status": await self._service.get_status(),
            })
            await asyncio.sleep(1 / 30)

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
        self._motion_request_id += 1
        generation = self._generation
        self._pending = True
        self._command_active = False
        self._motion_switching = False
        self._stopping = False
        self._telemetry = {}
        self.changed.emit()
        self._submit(self._disconnect(generation), generation)

    async def _disconnect(self, generation):
        await self._service.disconnect()
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
            self._monitor = None
        if self._state_monitor is not None:
            self._state_monitor.cancel()
            await asyncio.gather(self._state_monitor, return_exceptions=True)
            self._state_monitor = None
        self._event.emit(generation, {"status": await self._service.get_status(),
                                     "motion": asdict(await self._service.get_motion_state()),
                                     "pending": False, "telemetry": {},
                                     "log": "Simulation disconnected; monitoring stopped."})

    @Slot(str)
    def requestPosture(self, name):
        if self._closed or self._pending or not self._connected or self.busy:
            return
        self._busy = True
        self._command_active = True
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
            self._event.emit(generation, {"status": await self._service.get_status(),
                                         "motion": asdict(await self._service.get_motion_state()),
                                         "commandDone": True})

    @Slot(str)
    def requestMotion(self, command):
        if not self.canSwitchMotion:
            return
        self._motion_request_id += 1
        request_id = self._motion_request_id
        self._command_active = True
        self._motion_switching = True
        self.changed.emit()
        self._submit(self._start_motion(command, self._generation, request_id), self._generation)

    async def _start_motion(self, command, generation, request_id):
        """Latest selection wins; Stop/disconnect invalidate all pending selections."""
        def current():
            return (not self._closed and generation == self._generation
                    and request_id == self._motion_request_id)
        try:
            if not current():
                return
            if command not in await self._service.get_motion_capabilities():
                raise ValueError(f"Unsupported motion: {command}")
            state = await self._service.get_motion_state()
            if state.action == command:
                return  # Re-selecting an active direction does not restart it.
            if (await self._service.get_status()).busy:
                self._event.emit(generation, {"requestId": request_id,
                                             "log": f"Simulation: switching to {command}; stopping and settling first."})
                await self._service.stop_motion()
            if not current():
                return
            if await self._service.get_posture() != "Stand":
                self._event.emit(generation, {"requestId": request_id,
                                             "log": f"Simulation: preparing Stand before {command}."})
                await self._service.set_posture("Stand", 0.5)
            if not current():
                return
            motion = await self._service.start_motion(command)
            self._event.emit(generation, {"requestId": request_id,
                                         "log": f"Simulation: {command} started. Select another direction to switch, or Stop to finish.",
                                         "status": await self._service.get_status(),
                                         "motion": asdict(motion)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if current():
                self._event.emit(generation, {"requestId": request_id, "log": f"Simulation error: {exc}"})
        finally:
            if current():
                self._event.emit(generation, {"requestId": request_id,
                                             "status": await self._service.get_status(),
                                             "motion": asdict(await self._service.get_motion_state()),
                                             "commandDone": True})

    @Slot()
    def stopMotion(self):
        if self._closed or self._pending or not self._connected or self._stopping:
            return
        self._motion_request_id += 1
        self._motion_switching = False
        self._command_active = False
        self._stopping = True
        self.changed.emit()
        self.logMessage.emit("Simulation: stopping; travel halted, settling walking legs into Stand.")
        self._submit(self._stop_motion(self._generation), self._generation)

    async def _stop_motion(self, generation):
        try:
            await self._service.stop_motion()
            posture = await self._service.get_posture()
            message = ("Stopped; settled into Stand at the current position and heading."
                       if posture == "Stand" else "Stopped; posture transition frozen (posture Unknown)."
                       if posture is None else f"Stopped; posture remains {posture}.")
            self._event.emit(generation, {"log": "Simulation: " + message})
        finally:
            self._event.emit(generation, {"status": await self._service.get_status(),
                                         "motion": asdict(await self._service.get_motion_state()),
                                         "stopping": False})

    @Slot()
    def close(self):
        """Cancel active work and stop/join the loop; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        self._motion_request_id += 1
        try:
            asyncio.run_coroutine_threadsafe(self._disconnect(self._generation), self._loop).result(5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(5)
        if self._thread.is_alive():
            raise RuntimeError("Simulation worker did not stop")
