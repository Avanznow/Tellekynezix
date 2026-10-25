# Articulated visual simulation model

This model reuses 27 original OBJ/MTL body parts from
`NA06_Manual_Control/Nao/nao6_body`. Converted Qt Quick 3D components and meshes
are kept in `parts/`; the hierarchy and approximate pivots are in `RigidNao.qml`.
No external model or texture is required. The original assets remain unchanged.

Rebuild with the project's activated Python environment:

```powershell
python tools/build_rigid_nao.py
```

The builder uses the installed PySide6 `balsam` importer. Rebuilding can change
generated files when the Qt version changes; review those changes before commit.

## Coordinates and state

- Source geometry is Y-up, with nominal forward along +Z.
- Position X/Z uses arbitrary model/scene units, not calibrated robot meters.
- Heading is in degrees; negative turns left, positive turns right.
- `sitAmount` is a 0..1 visual pose blend; it is not a measured joint angle.
- `walking` is a 0..1 gait blend; `gaitPhase` is in radians.
- Approximate hip, knee and ankle pivot heights are 1.55, 0.76 and 0.20 scene units.

The simulation backend owns animation state. QML only renders that state; it
does not independently advance a gait or send hardware commands. While connected,
choose a walking/turning command directly. The controller prepares Stand first
when posture is unknown or sitting, then starts the requested motion. Selecting another direction
automatically halts the current motion, settles the legs, and starts the new
selection. The latest direction selected during settling wins. Selecting the
already active direction leaves it running. Stop/disconnect cancel pending
switches, so no queued direction starts afterward. Stop halts translation and
heading changes immediately, then reduces the gait blend to zero at the same
position and heading. Posture controls are enabled again after settling.

Stopping a Sit/Stand transition freezes the pose and marks posture Unknown.
Disconnect freezes visual state, cancels active tasks, and clears posture/status;
reconnect does not resume old motion. Choose Stand to recover a known pose.

## Limits

This is procedural visualization, not a physics simulator, balance controller,
collision system, validated NAO trajectory or odometry source. The camera is fixed
by default so travel across the one-scene-unit floor grid is visible. The green
disc marks the original starting point. Enable Follow robot for longer walks;
turning it off holds the camera at its current location. IMU, range and battery readings remain
constant, explicitly labeled simulated fixtures; FSR/joint sensors remain
unavailable. Visual motion state is separate from sensor telemetry.
