"""One-shot Windows microphone recognition, managed on the Qt GUI thread."""

import json
import math
import os
from pathlib import Path
import sys

from PySide6.QtCore import QObject, Property, QProcess, QTimer, Signal, Slot


MIN_CONFIDENCE = 0.60


class VoiceController(QObject):
    changed = Signal()

    def __init__(self, simulation, parent=None):
        super().__init__(parent)
        self._simulation = simulation
        self._process = None
        self._buffer = b""
        self._generation = None
        self._status = "Click Speak, then say one command."
        self._heard = ""
        self._confidence = ""
        self._result_received = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._expired)
        simulation.changed.connect(self._session_changed)

    @Property(bool, notify=changed)
    def listening(self):
        return self._process is not None

    @Property(str, notify=changed)
    def status(self):
        return self._status

    @Property(str, notify=changed)
    def heard(self):
        return self._heard

    @Property(str, notify=changed)
    def confidence(self):
        return self._confidence

    @Slot()
    def start(self):
        if self._process is not None:
            return
        if not self._simulation.connected or self._simulation.pending:
            self._status = "Connect simulation before speaking."
            self.changed.emit()
            return
        if sys.platform != "win32":
            self._status = "Microphone recognition currently requires Windows."
            self.changed.emit()
            return
        self._generation = self._simulation._generation
        self._buffer = b""
        self._result_received = False
        self._heard = ""
        self._confidence = ""
        self._status = "Preparing microphone..."
        process = QProcess(self)
        self._process = process
        process.readyReadStandardOutput.connect(lambda: self._read(process))
        process.finished.connect(lambda *_: self._finished(process))
        process.errorOccurred.connect(lambda *_: self._failed(process))
        program = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) /
                      "System32/WindowsPowerShell/v1.0/powershell.exe")
        process.start(program, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                "-File", str(Path(__file__).with_name("recognize_voice.ps1"))])
        self._timer.start(15000)
        self.changed.emit()

    def _session_changed(self):
        if self._process is not None and (
                not self._simulation.connected or self._generation != self._simulation._generation):
            self.cancel()

    def _read(self, process):
        if process is not self._process:
            return
        self._buffer += bytes(process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                result = json.loads(line)
                if not isinstance(result, dict):
                    raise ValueError("Expected recognition object")
                if result.get("ready") is True:
                    self._status = "Listening... say one command."
                elif "error" in result:
                    self._status = str(result["error"])
                else:
                    if self._result_received:
                        continue
                    self._result_received = True
                    confidence = result.get("confidence")
                    phrase = result.get("text")
                    if (not isinstance(phrase, str) or isinstance(confidence, bool)
                            or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
                            or not 0 <= confidence <= 1):
                        raise ValueError("Invalid recognition result")
                    self._heard = phrase
                    self._confidence = f"{confidence:.0%}" if phrase else ""
                    if not phrase:
                        self._status = "No command heard. Click Speak command and wait for Listening before speaking."
                        self._simulation.logMessage.emit("Voice: no command recognized.")
                    elif confidence < MIN_CONFIDENCE:
                        self._status = f"Confidence too low (requires {MIN_CONFIDENCE:.0%}). Click Speak command to retry."
                        self._simulation.logMessage.emit(
                            f"Voice rejected: {phrase!r}, confidence {confidence:.0%} (requires {MIN_CONFIDENCE:.0%}).")
                    elif self._generation != self._simulation._generation or not self._simulation.connected:
                        self._status = "Simulation session changed. Command discarded."
                    else:
                        # Release microphone ownership before invoking the same
                        # command functions as manual buttons.
                        self.cancel(keep_status=True)
                        accepted = self._simulation.executeVoiceCommand(phrase)
                        self._status = "Command submitted." if accepted else "Command unavailable in the current state."
            except (ValueError, UnicodeError):
                self._status = "Invalid response from speech recognizer."
            self.changed.emit()

    def _finished(self, process):
        if process is self._process:
            self._read(process)
            if self._status.startswith(("Preparing", "Listening")):
                self._status = "No command recognized. Try again."
            self._timer.stop()
            self._process = None
            self.changed.emit()
        process.deleteLater()

    def _failed(self, process):
        if process is self._process:
            self._status = "Speech recognizer could not start."
            self.cancel(keep_status=True)

    def _expired(self):
        self._status = "Listening timed out. Try again."
        self.cancel(keep_status=True)

    @Slot()
    def cancel(self, keep_status=False):
        process = self._process
        self._process = None
        self._timer.stop()
        if process is not None:
            process.kill()
            process.waitForFinished(1000)
            process.deleteLater()
        if not keep_status:
            self._status = "Listening cancelled."
        self.changed.emit()
