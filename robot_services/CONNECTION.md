# Windows to Pop!_OS simulation connection

## Design

The Windows application acts as a TCP client and the Pop!_OS connector as
the server. Both a command-line probe and a GUI client use this protocol.
The server creates a SimulationRobotService for each TCP connection.

Protocol version 1 uses UTF-8 JSON Lines. The client sends one request and awaits
its response before sending the next. Each request contains a string id
representing a positive decimal integer that increases with every request,
a version, a method, and a params object. The reply echoes the id and
version and contains either ok=true with a result or ok=false with an error
code and message. Lines are limited to 65536 bytes.
Duplicate or out-of-order ids return an error. Clients must not automatically retry commands.

```json
{"version":1,"id":"2","method":"set_posture","params":{"name":"Stand","speed":1.0}}
{"version":1,"id":"2","ok":true,"result":{"requested_posture":"Stand","outcome":"completed_verified","observed_posture":"Stand","message":"Verified simulated state only"}}
```

| Method | Parameters | Result |
| --- | --- | --- |
| connect | Empty object | RobotStatus |
| get_status | Empty object | RobotStatus |
| get_motion_state | Empty object | Simulation MotionState for the viewer |
| set_posture | name (Stand or Sit), speed (greater than 0, at most 1) | PostureResult after simulated completion |
| read_telemetry | Empty object | Schema version 1 TelemetrySnapshot |
| disconnect | Empty object | Disconnected RobotStatus |

See [API.md](API.md) for field definitions. A connection initially has a
disconnected backend. Closing TCP releases that session. Reads, writes and
commands have timeouts (10 seconds by default). A command timeout cancels its
simulation operation and returns an error. EOF during an operation is handled
when the operation completes or times out. Server shutdown cancels active
client tasks and disconnects their services.

The server currently provides simulation only. It has no authentication or
encryption and is intended for local development through a loopback-bound
VirtualBox forwarding rule. Hardware control requires a separate design.
Remote walking, turning, and Stop commands are pending a later protocol increment.

## GUI use

Update the repository on both machines and restart the Pop!_OS server so it
includes `get_motion_state`. Start `GUI5.py` in the Windows environment.
In Manual NAO Control, select Remote simulation, enter the forwarded endpoint
(normally 127.0.0.1 and 8765), and click Connect simulation.

Stand Up and Sit Down now send requests to the Pop!_OS service. The GUI shows
the returned posture, telemetry, and visual pose. Remote preview refreshes
state every half second and shows completed posture poses. Local simulation
provides the existing smooth animations and motion controls.

Disconnect clears the session readings. If the server stops or a network request
fails, the remote client closes TCP and the GUI returns to Disconnected.
After the server is available again, click Connect simulation to create a fresh
session. Backend and endpoint selection are available while disconnected.

## Local round trip

From the project root, use the active project Python environment. In one terminal:

```bash
python -B -m robot_services.simulation_rpc serve
```

In another terminal:

```bash
python -B -m robot_services.simulation_rpc probe
```

Expected output includes:

```text
PASS: simulation connected, Stand completed and verified, then disconnected
```

## Windows and VirtualBox

Use a NAT adapter for the Pop!_OS VM and add a TCP port-forwarding rule in
VirtualBox's network settings. Set host IP to 127.0.0.1, host port to 8765,
guest port to 8765, and leave guest IP blank for the default NAT configuration.
If that host port is occupied, choose a different host port and pass it to the probe.

In the Pop!_OS project terminal:

```bash
source ~/nao-qi-test/bin/activate
python -B -m robot_services.simulation_rpc serve --host 0.0.0.0 --port 8765
```

In Windows PowerShell, from the project root:

```powershell
.\.venv\Scripts\python.exe -B -m robot_services.simulation_rpc probe --host 127.0.0.1 --port 8765
```

The probe connects the simulated backend, sends Stand once, verifies its result
and status, then disconnects. Keep the VM server terminal open and use Ctrl+C
to stop it when finished. This test requires the new files on both machines.

## Automated checks

```bash
python -B -m unittest discover -s tests -p test_simulation_rpc.py -v
```

These checks use real TCP sockets on loopback, including fragmented messages,
malformed requests, duplicate ids, command timeouts, isolated sessions, and the Stand round trip.
The Windows-to-VM route needs a separate run using the commands above.

VirtualBox forwarding reference:
[Oracle VirtualBox networking guide](https://docs.oracle.com/en/virtualization/virtualbox/7.2/user/networkingdetails.html).
