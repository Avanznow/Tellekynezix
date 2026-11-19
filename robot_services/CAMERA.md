# Local camera preview

Manual NAO Control includes two local sources: a selected Windows webcam and a
video file. The source label identifies the preview. These sources work while
using either the local or remote simulation and run on the Windows GUI machine.
They provide a camera workflow for the proposal while NAO camera access is pending.

## Manual check

1. Start `GUI5.py` and open Manual NAO Control, then Camera.
2. Choose Local webcam or Test video. For Local webcam, select the laptop or USB
   camera from the device list before starting. For Test video, click Choose video and
   select an existing local video file.
3. Click Start Stream. Confirm the source label, frame dimensions, and
   approximate displayed frames per second.
4. Click Capture and choose a PNG destination. Open the saved frame to check it.
5. Click Stop Stream and confirm the preview clears. Restart to check recovery.
6. For a video file, let it reach the end and confirm the stopped message, then replay.

Camera access begins only after Start Stream. Switching source, leaving the
Camera panel, or closing the GUI stops capture and releases the video source.
Capture freezes one frame when clicked, then saves that frame after the file
dialog is accepted. Preview itself writes no recording and sends no video over
the simulation connection. Test videos play silently.

Stop Stream before choosing another webcam. The list refreshes when cameras
are plugged in or removed. Selection follows the device id if the list order
changes. Removing the selected webcam stops its preview and requests a new selection.

If a webcam is unavailable, busy, or blocked by Windows, the panel reports an
error. A source that delivers no frames within ten seconds also stops with
an error. Supported local video formats depend on the installed Qt media backend
and codecs. Manual Windows checks confirmed webcam start/stop and PNG capture,
including repeated use of a Logitech Webcam C930e and clearing the image on Stop.

## Verification

```powershell
python -B -m unittest discover -s tests -p test_camera_preview.py -v
```

Tests generate a short temporary video and decode it through Qt Multimedia.
They cover frame delivery, PNG capture, Stop, restart, end-of-video, invalid
source selection, and cleanup when the video output is destroyed.
No webcam activation is needed for these tests.

The displayed frame rate measures frames received by the GUI. It does not
measure capture-to-display latency. NAO camera acquisition and transport will
need a separate implementation and evaluation.
