# Simulation voice commands

The Windows GUI uses a single recognition operation per click. In Manual NAO
Control, connect a local or remote simulation, click Speak command, wait for
Listening, and say one phrase. Cancel listening releases the microphone.
The status label shows the recognized phrase, confidence, and submission result.
Silence and low-confidence recognition have separate messages. The console
records rejected phrases and their confidence for evaluation.

| Phrase | Existing controller action |
| --- | --- |
| walk forward | requestMotion("forward") |
| walk backward | requestMotion("backward") |
| turn left | requestMotion("turn_left") |
| turn right | requestMotion("turn_right") |
| stand up | requestPosture("Stand") |
| sit down | requestPosture("Sit") |
| stop | stopMotion() |

Phrases match exactly after case and whitespace normalization. Recognition uses
a constrained English grammar and requires confidence of at least 0.60.
This simulation trial lowers the original 0.65 threshold after a correctly
recognized phrase scored 0.64. Repeat each phrase five times and record correct
actions, missed commands, and wrong actions before selecting the final threshold.
Recognition success submits a command subject to the same connected, busy,
and pending-state checks as manual control. The console records the recognized
phrase and subsequent simulation result.

## Windows setup

Recognition uses the default Windows microphone through System.Speech in a
separate Windows PowerShell process. An installed English desktop speech
recognizer is required. To check availability without recording audio:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File robot_services/recognize_voice.ps1 -Check
```

If no recognizer is available, install the applicable Windows English speech
language components, then repeat the check. The development subprocess initially
returned zero recognizers. A check from the user's Windows terminal confirmed
an en-US recognizer, and manual GUI trials exercised all seven commands in local
simulation. Some attempts needed repetition. Recognition accuracy and response
time still require measured trials.
The GUI shows a setup error if the recognizer or microphone is unavailable.

Each click waits up to five seconds for speech. The GUI ends the recognition
process after fifteen seconds overall. Speech recognition runs locally, with
no application audio upload or recording file. Switching simulation sessions,
disconnecting, manually issuing commands, leaving the Manual NAO Control tab,
and closing the application cancel pending recognition.

## Verification

```bash
python -B -m unittest discover -s tests -p test_voice_commands.py -v
```

Automated tests exercise phrase mapping, state guards, confidence handling,
and stale process results. The Qt/TCP integration test also routes recognized
command text to the remote simulation. A human microphone trial is still
required to measure recognition accuracy and end-to-end voice response.

For a manual trial, repeat each phrase several times and record the intended
phrase, displayed recognition, accepted/rejected outcome, and observed action.
Repeat with the remote simulation after confirming the local microphone.

API references:
[SpeechRecognitionEngine](https://learn.microsoft.com/en-us/dotnet/api/system.speech.recognition.speechrecognitionengine?view=netframework-4.8)
and [Recognize](https://learn.microsoft.com/en-us/dotnet/api/system.speech.recognition.speechrecognitionengine.recognize?view=netframework-4.8).
