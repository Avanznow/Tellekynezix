import QtQuick
import QtQuick3D
Node {
id: robot
property real sitAmount: 0
property real gaitPhase: 0
property real walking: 0
property real heading: 0
eulerRotation.y: heading
Node { id: pelvis; position.y: 1.55 - robot.sitAmount * 0.72 + robot.walking * 0.025 * Math.abs(Math.sin(robot.gaitPhase * 2))
Loader3D { source: "parts/body/Nao6_Body.qml"; position: Qt.vector3d(0, -1.55, 0) }
Loader3D { source: "parts/head/Nao6_Head.qml"; position: Qt.vector3d(0, -1.55, 0) }
Loader3D { source: "parts/neck/Nao6_Neck.qml"; position: Qt.vector3d(0, -1.55, 0) }
Node { position: Qt.vector3d(0.47, 0, 0)
property real hipAngle: -85 * robot.sitAmount + robot.walking * 22 * Math.sin(robot.gaitPhase)
eulerRotation.x: hipAngle
Loader3D { source: "parts/left_thigh/Nao6_Left_Leg_Thigh.qml"; position: Qt.vector3d(-0.47, -1.55, 0) }
Loader3D { source: "parts/left_joint/Nao6_Left_Leg_Joint.qml"; position: Qt.vector3d(-0.47, -1.55, 0) }
Node { position: Qt.vector3d(0, -0.79, 0)
property real kneeAngle: 90 * robot.sitAmount + robot.walking * 30 * Math.max(0, Math.sin(robot.gaitPhase))
eulerRotation.x: kneeAngle
Loader3D { source: "parts/left_calf/Nao6_Left_Leg_Calf.qml"; position: Qt.vector3d(-0.47, -0.76, 0) }
Node { position: Qt.vector3d(0, -0.56, 0)
eulerRotation.x:  -5 * robot.sitAmount - robot.walking * (22 * Math.sin(robot.gaitPhase) + 30 * Math.max(0, Math.sin(robot.gaitPhase)))
Loader3D { source: "parts/left_foot/Nao6_Left_Leg_Foot.qml"; position: Qt.vector3d(-0.47, -0.2, 0) }
}
}
}
Node { position: Qt.vector3d(0.6, 1.15, 0)
eulerRotation.x: -robot.walking * 15 * Math.sin(robot.gaitPhase) - robot.sitAmount * 15
Loader3D { source: "parts/left_bicep/Nao6_Left_Arm_Bicep.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_elbow/Nao6_Left_Arm_Elbow.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_forearm/Nao6_Left_Arm_Forearm.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_shoulderpad/Nao6_Left_Arm_Shoulderpad.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_socket/Nao6_Left_Arm_Socket.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_finger_bottom/Nao6_Left_Hand_Bottom_Finger.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_finger_left/Nao6_Left_Hand_Left_Finger.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
Loader3D { source: "parts/left_finger_right/Nao6_Left_Hand_Right_Finger.qml"; position: Qt.vector3d(-0.6, -2.7, 0) }
}
Node { position: Qt.vector3d(-0.47, 0, 0)
property real hipAngle: -85 * robot.sitAmount + robot.walking * 22 * Math.sin((robot.gaitPhase + Math.PI))
eulerRotation.x: hipAngle
Loader3D { source: "parts/right_thigh/Nao6_Right_Leg_Thigh.qml"; position: Qt.vector3d(0.47, -1.55, 0) }
Loader3D { source: "parts/right_joint/Nao6_Right_Leg_Joint.qml"; position: Qt.vector3d(0.47, -1.55, 0) }
Node { position: Qt.vector3d(0, -0.79, 0)
property real kneeAngle: 90 * robot.sitAmount + robot.walking * 30 * Math.max(0, Math.sin((robot.gaitPhase + Math.PI)))
eulerRotation.x: kneeAngle
Loader3D { source: "parts/right_calf/Nao6_Right_Leg_Calf.qml"; position: Qt.vector3d(0.47, -0.76, 0) }
Node { position: Qt.vector3d(0, -0.56, 0)
eulerRotation.x:  -5 * robot.sitAmount - robot.walking * (22 * Math.sin((robot.gaitPhase + Math.PI)) + 30 * Math.max(0, Math.sin((robot.gaitPhase + Math.PI))))
Loader3D { source: "parts/right_foot/Nao6_Right_Leg_Foot.qml"; position: Qt.vector3d(0.47, -0.2, 0) }
}
}
}
Node { position: Qt.vector3d(-0.6, 1.15, 0)
eulerRotation.x: -robot.walking * 15 * Math.sin((robot.gaitPhase + Math.PI)) - robot.sitAmount * 15
Loader3D { source: "parts/right_bicep/Nao6_Right_Arm_Bicep.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_elbow/Nao6_Right_Arm_Elbow.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_forearm/Nao6_Right_Arm_Forearm.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_shoulderpad/Nao6_Right_Arm_Shoulderpad.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_socket/Nao6_Right_Arm_Socket.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_finger_bottom/Nao6_Right_Hand_Bottom_Finger.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_finger_left/Nao6_Right_Hand_Left_Finger.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
Loader3D { source: "parts/right_finger_right/Nao6_Right_Hand_Right_Finger.qml"; position: Qt.vector3d(0.6, -2.7, 0) }
}
}
}