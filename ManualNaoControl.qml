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
    onVisibleChanged: {
        if (!visible) naoSimulation.voice.cancel()
    }

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

    component ConnectionBadge: Rectangle {
        implicitWidth: badgeRow.implicitWidth + 18
        implicitHeight: 28
        radius: 14
        color: naoSimulation.connected ? "#234d3c" : "#485767"
        border.color: naoSimulation.connected ? "#69ca8b" : "#a9b7c6"
        Row {
            id: badgeRow
            anchors.centerIn: parent
            spacing: 6
            Rectangle {
                width: 9; height: 9; radius: 5
                anchors.verticalCenter: parent.verticalCenter
                color: naoSimulation.connected ? "#6ce492" : "#bdc6d0"
            }
            Label {
                text: naoSimulation.connected ? "Connected" : "Disconnected"
                color: "white"
                font.pixelSize: 12
            }
        }
    }

    component ControlTile: Button {
        id: tile
        property string iconName: ""
        property bool selected: false
        property string fillColor: "#ffffff"
        property string labelColor: "#182b40"
        hoverEnabled: true
        padding: 7
        background: Rectangle {
            radius: 14
            color: tile.down ? Qt.darker(tile.fillColor, 1.12) :
                   tile.selected ? "#dff3e5" : tile.hovered && tile.enabled ? Qt.darker(tile.fillColor, 1.06) : tile.fillColor
            border.width: tile.activeFocus || tile.selected ? 2 : 1
            border.color: tile.activeFocus ? "#1676d2" : tile.selected ? "#198c32" : "#d0dce9"
            opacity: tile.enabled ? 1 : 0.65
        }
        contentItem: ColumnLayout {
            spacing: 3
            Image {
                visible: tile.iconName.length > 0
                source: tile.iconName ? "NA06_Manual_Control/control_icons/" + tile.iconName + ".svg" : ""
                Layout.alignment: Qt.AlignHCenter
                Layout.preferredWidth: Math.min(34, tile.height * 0.40)
                Layout.preferredHeight: Layout.preferredWidth
                sourceSize.width: 64
                sourceSize.height: 64
                opacity: tile.enabled ? 1 : 0.55
            }
            Label {
                text: tile.text
                color: tile.labelColor
                font.pixelSize: tile.objectName === "simulationStopButton" ? 18 : 13
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
            }
        }
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
                spacing: 7
                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "NAO Simulation"
                        color: "white"
                        font.pixelSize: 18
                        font.bold: true
                        Layout.fillWidth: true
                    }
                    ConnectionBadge { }
                }
                GridLayout {
                    columns: 3
                    rowSpacing: 8
                    columnSpacing: 8
                    Layout.fillWidth: true
                    Repeater {
                        model: [
                            {label: "Walk Forward", action: "forward", icon: "forward", row: 0, col: 1},
                            {label: "Turn Left", action: "turn_left", icon: "turn-left", row: 1, col: 0},
                            {label: "Turn Right", action: "turn_right", icon: "turn-right", row: 1, col: 2},
                            {label: "Walk Backward", action: "backward", icon: "backward", row: 2, col: 1}
                        ]
                        delegate: ControlTile {
                            objectName: "simulationMotion_" + modelData.action
                            text: modelData.label
                            iconName: modelData.icon
                            Layout.row: modelData.row
                            Layout.column: modelData.col
                            Layout.fillWidth: true
                            Layout.preferredWidth: 90
                            Layout.minimumWidth: 0
                            Layout.preferredHeight: Math.min(68, root.height * 0.09)
                            selected: naoSimulation.connected && naoSimulation.motion.action === modelData.action
                            enabled: naoSimulation.canSwitchMotion
                            onClicked: naoSimulation.requestMotion(modelData.action)
                        }
                    }
                    ControlTile {
                        objectName: "simulationStopButton"
                        text: naoSimulation.stopping ? "Stopping..." : "STOP"
                        fillColor: "#e52535"
                        labelColor: "white"
                        Layout.row: 1
                        Layout.column: 1
                        Layout.fillWidth: true
                        Layout.preferredWidth: 90
                        Layout.minimumWidth: 0
                        Layout.preferredHeight: Math.min(68, root.height * 0.09)
                        enabled: naoSimulation.motionSupported && naoSimulation.connected && naoSimulation.busy && !naoSimulation.pending && !naoSimulation.stopping
                        onClicked: naoSimulation.stopMotion()
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    ControlTile {
                        objectName: "simulationStandButton"
                        text: "Stand Up"
                        iconName: "stand"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.min(54, root.height * 0.075)
                        enabled: naoSimulation.connected && !naoSimulation.busy && !naoSimulation.pending
                        onClicked: naoSimulation.requestPosture("Stand")
                    }
                    ControlTile {
                        objectName: "simulationSitButton"
                        text: "Sit Down"
                        iconName: "sit"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.min(54, root.height * 0.075)
                        enabled: naoSimulation.connected && !naoSimulation.busy && !naoSimulation.pending
                        onClicked: naoSimulation.requestPosture("Sit")
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    ControlTile {
                        objectName: "simulationVoiceSpeak"
                        text: naoSimulation.voice.listening ? "Listening..." : "Speak Command"
                        iconName: "microphone"
                        fillColor: "#087bea"
                        labelColor: "white"
                        Layout.fillWidth: true
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 48
                        enabled: naoSimulation.connected && !naoSimulation.pending && !naoSimulation.voice.listening
                        onClicked: naoSimulation.voice.start()
                    }
                    ControlTile {
                        objectName: "simulationVoiceCancel"
                        text: "Cancel Listening"
                        fillColor: "#dce5f0"
                        Layout.fillWidth: true
                        Layout.preferredWidth: 120
                        Layout.preferredHeight: 48
                        enabled: naoSimulation.voice.listening
                        onClicked: naoSimulation.voice.cancel()
                    }
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    color: "white"
                    text: naoSimulation.voice.status +
                          (naoSimulation.voice.heard ? "\nHeard: " + naoSimulation.voice.heard +
                           " (confidence " + naoSimulation.voice.confidence + ")" : "")
                    ToolTip.visible: voiceHelp.containsMouse
                    ToolTip.text: "walk forward, walk backward, turn left, turn right, stand up, sit down, stop"
                    MouseArea { id: voiceHelp; anchors.fill: parent; hoverEnabled: true }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "Console"; color: "white"; font.bold: true; Layout.fillWidth: true }
                    Button {
                        objectName: "simulationConsoleClear"
                        text: "Clear"
                        implicitHeight: 28
                        enabled: consoleLog.length > 0
                        onClicked: consoleLog.clear()
                    }
                    Button {
                        objectName: "simulationConsoleCopy"
                        text: "Copy Log"
                        implicitHeight: 28
                        enabled: consoleLog.length > 0
                        onClicked: { consoleLog.selectAll(); consoleLog.copy(); consoleLog.deselect() }
                    }
                }
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 160
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
            CheckBox {
                id: followRobotCheck
                objectName: "simulationFollowCamera"
                anchors.left: parent.left
                anchors.leftMargin: 12
                anchors.bottom: departureLabel.top
                anchors.bottomMargin: 4
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
                id: departureLabel
                anchors.left: parent.left
                anchors.bottom: parent.bottom
                anchors.margins: 12
                text: "Initial departure: Green point."
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
            ScrollView {
                id: sidebarScroll
                anchors.fill: parent
                anchors.margins: 14
                clip: true
                contentWidth: availableWidth
            ColumnLayout {
                width: sidebarScroll.availableWidth
                spacing: 10
                NaoCamera.CameraPreview {
                    objectName: "naoCameraPanel"
                    Layout.fillWidth: true
                    Layout.preferredHeight: Math.max(330, root.height * 0.48)
                    cameraController: naoSimulation.cameraPreview
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label { text: "Simulation session"; color: "white"; font.pixelSize: 15 }
                    ComboBox {
                        id: simulationBackend
                        objectName: "simulationBackendSelector"
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        model: ["Local simulation", "Remote simulation"]
                        enabled: !naoSimulation.connected && !naoSimulation.pending
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Button {
                        objectName: "simulationConnectButton"
                        Layout.fillWidth: true
                        text: naoSimulation.connected ? "Disconnect" : "Connect simulation"
                        enabled: !naoSimulation.pending
                        onClicked: {
                            if (naoSimulation.connected) naoSimulation.disconnectSimulation()
                            else if (simulationBackend.currentIndex === 0 ||
                                     (simulationHost.text.trim().length > 0 && simulationPort.acceptableInput)) {
                                if (naoSimulation.configureSimulation(simulationBackend.currentIndex === 1,
                                                                      simulationHost.text, Number(simulationPort.text)))
                                    naoSimulation.connectSimulation()
                            }
                        }
                    }
                    ColumnLayout {
                        spacing: 3
                        Label {
                            text: "Status"
                            color: "white"
                            Layout.alignment: Qt.AlignHCenter
                        }
                        ConnectionBadge { }
                    }
                }
                RowLayout {
                    visible: simulationBackend.currentIndex === 1
                    Layout.fillWidth: true
                    TextField {
                        id: simulationHost
                        objectName: "simulationRemoteHost"
                        Layout.fillWidth: true
                        text: "127.0.0.1"
                        placeholderText: "Simulation host"
                        enabled: simulationBackend.enabled
                    }
                    TextField {
                        id: simulationPort
                        objectName: "simulationRemotePort"
                        Layout.preferredWidth: 75
                        text: "8765"
                        placeholderText: "Port"
                        validator: IntValidator { bottom: 1; top: 65535 }
                        enabled: simulationBackend.enabled
                    }
                }
                Label {
                    visible: simulationBackend.currentIndex === 1
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    color: "white"
                    text: "Motion, posture and telemetry from the remote simulation."
                }
                Label { text: "Posture: " + naoSimulation.posture; color: "white" }
                Button {
                    id: telemetryToggle
                    objectName: "naoTelemetryToggle"
                    Layout.fillWidth: true
                    checkable: true
                    checked: true
                    text: (checked ? "\u25be  " : "\u25b8  ") + "Telemetry"
                }
                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: telemetryToggle.checked ? root.height * 0.30 : 0
                    visible: telemetryToggle.checked
                    ScrollView {
                        anchors.fill: parent
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
}
