"""Exact vocabulary and GUI-thread recognition-result handling."""

import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import Mock

from PySide6.QtWidgets import QApplication

from robot_services.qt_simulation import SimulationController
from robot_services.voice_commands import COMMANDS, resolve_command


class VoiceCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = SimulationController()
        self.addCleanup(self.controller.close)

    def test_exact_vocabulary_rejects_unrelated_phrases(self):
        self.assertEqual(len(COMMANDS), 7)
        self.assertEqual(resolve_command("  WALK  FORWARD "), ("motion", "forward"))
        for text in ("don't stop", "please walk forward", "walk forward and sit down", "restart", None):
            self.assertIsNone(resolve_command(text))

    def test_voice_dispatch_uses_existing_controls_and_state_guards(self):
        self.controller.requestMotion = Mock()
        self.controller.requestPosture = Mock()
        self.controller.stopMotion = Mock()
        self.assertFalse(self.controller.executeVoiceCommand("walk forward"))
        self.controller._connected = True
        for phrase, action in (("walk forward", "forward"), ("walk backward", "backward"),
                              ("turn left", "turn_left"), ("turn right", "turn_right")):
            self.assertTrue(self.controller.executeVoiceCommand(phrase))
            self.controller.requestMotion.assert_called_with(action)
        self.assertTrue(self.controller.executeVoiceCommand("stand up"))
        self.controller.requestPosture.assert_called_with("Stand")
        self.assertTrue(self.controller.executeVoiceCommand("sit down"))
        self.controller.requestPosture.assert_called_with("Sit")
        self.controller._busy = True
        self.assertFalse(self.controller.executeVoiceCommand("sit down"))
        self.assertTrue(self.controller.executeVoiceCommand("stop"))
        self.controller.stopMotion.assert_called_once()

    def test_confidence_threshold_and_session_changes(self):
        self.controller._connected = True
        self.controller.executeVoiceCommand = Mock(return_value=True)
        voice = self.controller.voice
        voice._generation = self.controller._generation
        process = Mock()
        voice._process = process
        process.readAllStandardOutput.return_value = (
            json.dumps({"text": "walk forward", "confidence": .4}) + "\n"
        ).encode()
        voice._read(process)
        self.controller.executeVoiceCommand.assert_not_called()
        self.assertIn("Confidence too low", voice.status)
        self.assertEqual((voice.heard, voice.confidence), ("walk forward", "40%"))
        voice._result_received = False
        process.readAllStandardOutput.return_value = (
            json.dumps({"text": "walk forward", "confidence": .9}) + "\n"
        ).encode()
        voice._read(process)
        self.controller.executeVoiceCommand.assert_called_once_with("walk forward")
        self.assertEqual(voice.status, "Command submitted.")
        self.assertFalse(voice.listening)
        # Results arriving after a disconnect/cancel cannot act on a new session.
        voice._read(process)
        self.controller.executeVoiceCommand.assert_called_once()

    def test_silence_is_distinct_from_low_confidence(self):
        self.controller.executeVoiceCommand = Mock()
        voice = self.controller.voice
        process = Mock()
        voice._process = process
        process.readAllStandardOutput.return_value = b'{"text":"","confidence":0}\n'
        voice._read(process)
        self.assertIn("No command heard", voice.status)
        self.assertEqual((voice.heard, voice.confidence), ("", ""))
        self.controller.executeVoiceCommand.assert_not_called()

    def test_trial_threshold_accepts_60_and_64_percent_but_rejects_59(self):
        self.controller._connected = True
        self.controller.executeVoiceCommand = Mock(return_value=True)
        voice = self.controller.voice
        voice._generation = self.controller._generation
        for confidence, accepted in ((.59, False), (.60, True), (.64, True)):
            with self.subTest(confidence=confidence):
                self.controller.executeVoiceCommand.reset_mock()
                process = Mock()
                voice._process = process
                voice._result_received = False
                process.readAllStandardOutput.return_value = (
                    json.dumps({"text": "turn left", "confidence": confidence}) + "\n"
                ).encode()
                voice._read(process)
                if accepted:
                    self.controller.executeVoiceCommand.assert_called_once_with("turn left")
                    self.assertEqual(voice.status, "Command submitted.")
                else:
                    self.controller.executeVoiceCommand.assert_not_called()
                    self.assertIn("requires 60%", voice.status)

    def test_invalid_recognition_result_is_not_executed(self):
        self.controller.executeVoiceCommand = Mock()
        process = Mock()
        voice = self.controller.voice
        voice._process = process
        process.readAllStandardOutput.return_value = b'{"text":"stop","confidence":true}\n'
        voice._read(process)
        self.controller.executeVoiceCommand.assert_not_called()
        self.assertIn("Invalid response", voice.status)


if __name__ == "__main__":
    unittest.main()
