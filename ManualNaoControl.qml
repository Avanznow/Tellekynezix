/** File Name: ManualNaoControl.qml
** { Copyright: © 2025 Avatar BCI -- All rights reserved.
** ---------------------- [Cloud Computing Club;
** ---------------------- Brain Computer Interface Lab;
** ---------------------- CIDS / SCSU.] }
** Author: For the complete list of contributors, please refer to
** ---------------------- specific code file in the project's
** ---------------------- GitHub history and the UI Tab: [Developers] containing the 
** ---------------------- automatically updated complete list of authors.
** Description: 
** This QML file implements the Manual NAO Control tab, providing a user interface
** for previewing the NAO model and operating the simulation service.
** Posture and telemetry controls communicate only with the simulation backend.
** MIT License
 -----------------------------------------------------*/

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick3D
import "NA06_Manual_Control/rigid_model"
import "NA06_Manual_Control/camera_view" as NaoCamera

Rectangle {
    id: root
    objectName: "naoSimulationView"
    color: "#2f2f2f"
    property alias followRobot: followRobotCheck.checked
    property real cameraAnchorX: 0
    property real cameraAnchorZ: 0

    Connections {
        target: naoSimulation
        function onLogMessage(message) { root.appendLog(message) }
    }

    function appendLog(message) {
        var lines = consoleLog.text.split("\n")
        if (lines.length > 150) consoleLog.text = lines.slice(-100).join("\n")
        consoleLog.append(new Date().toLocaleTimeString() + "  " + message)
    }

    function reading(group, key) {
        var sample = naoSimulation.telemetry[group]
        if (!sample || !sample.values || !sample.values[key]) return "Unavailable"
        var item = sample.values[key]
        return item.value === null ? "Unavailable" : Number(item.value).toFixed(2) + " " + item.unit
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0
        Rectangle {
            Layout.preferredWidth: parent.width * 0.29
            Layout.minimumWidth: 245
            Layout.fillHeight: true
            color: "#718399"
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10
                Label {
                    text: "NAO Simulation"
                    color: "white"
                    font.pixelSize: 20
                    font.bold: true
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: "Motion controls\nChoose a direction; standing is automatic"
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                GridLayout {
                    columns: 2
                    Layout.fillWidth: true
                    Repeater {
                        model: [
                            {label: "Walking Backward", action: "backward", icon: "back.png"},
                            {label: "Walking Forward", action: "forward", icon: "forward.png"},
                            {label: "Turn Right", action: "turn_right", icon: "right.png"},
                            {label: "Turn Left", action: "turn_left", icon: "left.png"}
                        ]
                        delegate: Button {
                            objectName: "simulationMotion_" + modelData.action
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.min(110, root.height * 0.17)
                            enabled: naoSimulation.canSwitchMotion
                            contentItem: Column {
                                spacing: 4
                                Image {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    source: "GUI_Pics/" + modelData.icon
                                    width: 48; height: 48
                                    fillMode: Image.PreserveAspectFit
                                }
                                Label {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: modelData.label
                                }
                            }
                            onClicked: naoSimulation.requestMotion(modelData.action)
                        }
                    }
                }
                Label { text: "Simulated posture"; color: "white" }
                RowLayout {
                    Layout.fillWidth: true
                    Button {
                        objectName: "simulationStandButton"
                        text: "Stand Up"
                        Layout.fillWidth: true
                        enabled: naoSimulation.connected && !naoSimulation.busy && !naoSimulation.pending
                        onClicked: naoSimulation.requestPosture("Stand")
                    }
                    Button {
                        objectName: "simulationSitButton"
                        text: "Sit Down"
                        Layout.fillWidth: true
                        enabled: naoSimulation.connected && !naoSimulation.busy && !naoSimulation.pending
                        onClicked: naoSimulation.requestPosture("Sit")
                    }
                }
                Button {
                    objectName: "simulationStopButton"
                    text: naoSimulation.stopping ? "Stopping..." : "Stop"
                    Layout.fillWidth: true
                    enabled: naoSimulation.connected && naoSimulation.busy && !naoSimulation.pending && !naoSimulation.stopping
                    onClicked: naoSimulation.stopMotion()
                }
                Label { text: "Console"; color: "white" }
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 80
                    TextArea {
                        id: consoleLog
                        objectName: "simulationConsole"
                        readOnly: true
                        wrapMode: TextEdit.Wrap
                        font.pixelSize: 12
                        color: "white"
                        background: Rectangle { color: "#101820" }
                    }
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#2f2f2f"
            View3D {
                anchors.fill: parent
                environment: SceneEnvironment {
                    clearColor: "#2e2e2e"
                    backgroundMode: SceneEnvironment.Color
                }
                PerspectiveCamera {
                    objectName: "simulationViewCamera"
                    position: Qt.vector3d((root.followRobot ? naoSimulation.motion.x : root.cameraAnchorX) + 4.2,
                                         2.8, (root.followRobot ? naoSimulation.motion.z : root.cameraAnchorZ) + 7.2)
                    eulerRotation: Qt.vector3d(-9, 30, 0)
                    clipNear: 0.01
                    clipFar: 100
                }
                DirectionalLight { eulerRotation.x: -45; eulerRotation.y: 45; brightness: 1.5 }
                DirectionalLight { eulerRotation.x: 30; eulerRotation.y: -60; brightness: 1.2 }
                Model {
                    source: "#Cube"
                    position.y: -0.06
                    scale: Qt.vector3d(2, 0.001, 2)
                    materials: PrincipledMaterial { baseColor: "#344a59"; roughness: 1 }
                }
                Node {
                    objectName: "simulationGroundGrid"
                    // Fixed world reference: one square is one scene unit.
                    Repeater3D {
                        model: 41
                        delegate: Node {
                            required property int index
                            Model {
                                source: "#Cube"
                                position: Qt.vector3d(0, -0.005, index - 20)
                                scale: Qt.vector3d(0.4, 0.00003, 0.00008)
                                materials: PrincipledMaterial {
                                    baseColor: "#78919e"
                                    lighting: PrincipledMaterial.NoLighting
                                }
                            }
                            Model {
                                source: "#Cube"
                                position: Qt.vector3d(index - 20, -0.005, 0)
                                scale: Qt.vector3d(0.00008, 0.00003, 0.4)
                                materials: PrincipledMaterial {
                                    baseColor: "#78919e"
                                    lighting: PrincipledMaterial.NoLighting
                                }
                            }
                        }
                    }
                }
                Model {
                    objectName: "simulationStartMarker"
                    source: "#Cylinder"
                    position: Qt.vector3d(0, 0, 0)
                    scale: Qt.vector3d(0.012, 0.00005, 0.012)
                    materials: PrincipledMaterial {
                        baseColor: "#b1cb71"
                        lighting: PrincipledMaterial.NoLighting
                    }
                }
                RigidNao {
                    id: naoModel
                    objectName: "simulationNaoModel"
                    position: Qt.vector3d(naoSimulation.motion.x, 0, naoSimulation.motion.z)
                    heading: naoSimulation.motion.heading
                    sitAmount: naoSimulation.motion.sit_amount
                    walking: naoSimulation.motion.gait_weight
                    gaitPhase: naoSimulation.motion.gait_phase
                }
            }
            Label {
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 16
                width: parent.width - 24
                text: "SIMULATION\nNo hardware connection"
                color: "#d0ec8b"
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
            }
            Label {
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 58
                width: parent.width - 24
                text: "Procedural visual motion\nSensors remain constant test fixtures."
                color: "white"
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
            }
            CheckBox {
                id: followRobotCheck
                objectName: "simulationFollowCamera"
                anchors.top: parent.top
                anchors.topMargin: 96
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Follow robot"
                checked: false
                palette.windowText: "white"
                onCheckedChanged: {
                    if (!checked) {
                        root.cameraAnchorX = naoSimulation.motion.x
                        root.cameraAnchorZ = naoSimulation.motion.z
                    }
                }
            }
            Label {
                anchors.left: parent.left
                anchors.bottom: parent.bottom
                anchors.margins: 12
                text: "Grid: 1 scene unit\nGreen disc: starting point"
                color: "white"
                font.pixelSize: 11
            }
            Rectangle {
                objectName: "naoHardwareConnectionPanel"
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.margins: 12
                width: Math.min(205, parent.width - 24)
                height: hardwareFields.implicitHeight + 24
                color: "#242c4d"
                radius: 8
                border.color: "#3a4a6d"
                ColumnLayout {
                    id: hardwareFields
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 12
                    spacing: 8
                    Label { text: "NAO Connection"; font.bold: true; color: "white" }
                    GridLayout {
                        columns: 2
                        Layout.fillWidth: true
                        Label { text: "IP:"; color: "white" }
                        TextField {
                            objectName: "naoHardwareIp"
                            text: "192.168.23.53"
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            font.pixelSize: 12
                            selectByMouse: true
                        }
                        Label { text: "Port:"; color: "white" }
                        TextField {
                            objectName: "naoHardwarePort"
                            text: "9559"
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            font.pixelSize: 12
                            validator: IntValidator { bottom: 1; top: 65535 }
                            selectByMouse: true
                        }
                    }
                    Button {
                        objectName: "naoHardwareConnectButton"
                        text: "Connect"
                        Layout.fillWidth: true
                        enabled: false
                    }
                    Label {
                        text: "Hardware connection planned.\nSimulation uses its own Connect."
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        font.pixelSize: 11
                        color: "#d6deee"
                    }
                }
            }
        }
        Rectangle {
            Layout.preferredWidth: parent.width * 0.28
            Layout.minimumWidth: 250
            Layout.fillHeight: true
            color: "#233747"
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12
                Label { text: "Simulation session"; color: "white"; font.pixelSize: 18 }
                Label {
                    objectName: "simulationStatus"
                    text: "Status: " + naoSimulation.statusText
                    color: "white"
                }
                Label { text: "Posture: " + naoSimulation.posture; color: "white" }
                Button {
                    objectName: "simulationConnectButton"
                    Layout.fillWidth: true
                    text: naoSimulation.connected ? "Disconnect" : "Connect simulation"
                    enabled: !naoSimulation.pending
                    onClicked: {
                        if (naoSimulation.connected) naoSimulation.disconnectSimulation()
                        else naoSimulation.connectSimulation()
                    }
                }
                TabBar {
                    id: informationTabs
                    objectName: "naoInformationTabs"
                    Layout.fillWidth: true
                    TabButton { text: "Camera" }
                    TabButton { text: "Telemetry" }
                }
                StackLayout {
                    currentIndex: informationTabs.currentIndex
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Item {
                        NaoCamera.DroneCameraView {
                            objectName: "naoCameraPanel"
                            cameraController: null
                            placeholderText: "No camera connected"
                            availabilityText: "Live camera integration planned"
                            onLogToParent: function(message) { root.appendLog(message) }
                        }
                    }
                    ScrollView {
                        id: telemetryScroll
                        clip: true
                        contentWidth: availableWidth
                        ColumnLayout {
                            width: telemetryScroll.availableWidth
                            spacing: 12
                Label {
                    text: "Position (scene units):\nX: " + Number(naoSimulation.motion.x).toFixed(2) +
                          "  Z: " + Number(naoSimulation.motion.z).toFixed(2) +
                          "\nHeading: " + Number(naoSimulation.motion.heading).toFixed(1) + " deg"
                    color: "white"
                }
                Label { text: "Simulated telemetry"; color: "#d0ec8b"; font.bold: true }
                Label { text: "Battery: " + root.reading("battery", "charge"); color: "white" }
                Label {
                    text: !naoSimulation.telemetry.battery ? "Charging: Unavailable" :
                          "Charging: " + (naoSimulation.telemetry.battery.values.charging.value === 1 ? "Yes" : "No")
                    color: "white"
                }
                Label { text: "Left distance: " + root.reading("ultrasonic", "left"); color: "white" }
                Label { text: "Right distance: " + root.reading("ultrasonic", "right"); color: "white" }
                Label { text: "IMU acceleration (m/s²)"; color: "white" }
                Label {
                    text: "X: " + root.reading("imu", "acceleration_x") + "\nY: " +
                          root.reading("imu", "acceleration_y") + "\nZ: " + root.reading("imu", "acceleration_z")
                    color: "white"
                }
                Label {
                    text: "FSR / joints: Unavailable\nConstant test readings; not robot measurements."
                    color: "white"
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                }
                        }
                    }
                }

            }
        }
    }
}
