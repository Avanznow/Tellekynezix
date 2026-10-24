"""Rebuild rigid model assets with the active PySide6 environment.

Run from the repository root: python tools/build_rigid_nao.py
The original OBJ/MTL files remain the canonical source.
"""
from pathlib import Path
import sys
import subprocess

repo = Path(__file__).resolve().parents[1]
source = repo / "NA06_Manual_Control/Nao/nao6_body"
out = repo / "NA06_Manual_Control/rigid_model"
import PySide6
balsam = Path(PySide6.__file__).resolve().parent / ("balsam.exe" if sys.platform == "win32" else "balsam")
out.mkdir(parents=True, exist_ok=True)
parts = {"body": "Nao6_Body.obj", "head": "Nao6_Head.obj", "neck": "Nao6_Neck.obj"}
for side in ("Left", "Right"):
    for part in ("Bicep", "Elbow", "Forearm", "Shoulderpad", "Socket"):
        parts[f"{side.lower()}_{part.lower()}"] = f"{side}_Arm/Nao6_{side}_Arm_{part}.obj"
    for part in ("Calf", "Foot", "Joint", "Thigh"):
        parts[f"{side.lower()}_{part.lower()}"] = f"{side}_Leg/Nao6_{side}_Leg_{part}.obj"
    for finger in ("Bottom", "Left", "Right"):
        parts[f"{side.lower()}_finger_{finger.lower()}"] = f"{side}_Arm/{side}_Hand/Nao6_{side}_Hand_{finger}_Finger.obj"

components = {}
for name, relative in parts.items():
    asset = source / relative
    dest = out / "parts" / name
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(balsam), str(asset), "-o", str(dest)],
        capture_output=True, timeout=20,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    qmls = list(dest.glob("*.qml"))
    if len(qmls) != 1:
        raise RuntimeError(f"Missing converted component: {name}")
    components[name] = qmls[0].relative_to(out).as_posix()

def loader(name, pivot):
    x, y, z = pivot
    return f'Loader3D {{ source: "{components[name]}"; position: Qt.vector3d({-x}, {-y}, {-z}) }}'

qml = ['import QtQuick', 'import QtQuick3D', 'Node {', 'id: robot',
       'property real sitAmount: 0', 'property real gaitPhase: 0',
       'property real walking: 0', 'property real heading: 0',
       'eulerRotation.y: heading',
       'Node { id: pelvis; position.y: 1.55 - robot.sitAmount * 0.72 + robot.walking * 0.025 * Math.abs(Math.sin(robot.gaitPhase * 2))']
for name in ("body", "head", "neck"):
    qml.append(loader(name, (0, 1.55, 0)))

for side, sign in (("left", 1), ("right", -1)):
    hip = (sign * .47, 1.55, 0)
    knee = (sign * .47, .76, 0)
    ankle = (sign * .47, .20, 0)
    phase = 'robot.gaitPhase' if sign == 1 else '(robot.gaitPhase + Math.PI)'
    qml += [f'Node {{ position: Qt.vector3d({hip[0]}, 0, 0)',
            f'property real hipAngle: -85 * robot.sitAmount + robot.walking * 22 * Math.sin({phase})',
            'eulerRotation.x: hipAngle', loader(side + '_thigh', hip),
            loader(side + '_joint', hip),
            'Node { position: Qt.vector3d(0, -0.79, 0)',
            f'property real kneeAngle: 90 * robot.sitAmount + robot.walking * 30 * Math.max(0, Math.sin({phase}))',
            'eulerRotation.x: kneeAngle', loader(side + '_calf', knee),
            'Node { position: Qt.vector3d(0, -0.56, 0)',
            f'eulerRotation.x:  -5 * robot.sitAmount - robot.walking * (22 * Math.sin({phase}) + 30 * Math.max(0, Math.sin({phase})))',
            loader(side + '_foot', ankle), '}', '}', '}']
    shoulder = (sign * .60, 2.70, 0)
    qml += [f'Node {{ position: Qt.vector3d({shoulder[0]}, 1.15, 0)',
            f'eulerRotation.x: -robot.walking * 15 * Math.sin({phase}) - robot.sitAmount * 15']
    for part in ("bicep", "elbow", "forearm", "shoulderpad", "socket",
                 "finger_bottom", "finger_left", "finger_right"):
        qml.append(loader(side + '_' + part, shoulder))
    qml.append('}')
qml += ['}', '}']
(out / "RigidNao.qml").write_text('\n'.join(qml), encoding="utf-8")
print(f"Converted {len(parts)} rigid parts to {out}")
