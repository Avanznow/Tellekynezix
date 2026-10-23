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
import "Nao.mesh"

Rectangle {
    id: root
    objectName: "naoSimulationView"
    color: "#2f2f2f"
    property bool animationInProgress: false
    property int heading: 0

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

    function preview(action) {
        if (animationInProgress) return
        var start = naoModel.position
        if (action === "left" || action === "right") {
            var step = action === "left" ? -90 : 90
            heading += step
            rotateAnim.from = naoModel.eulerRotation
            rotateAnim.to = Qt.vector3d(0, heading, 0)
            animationInProgress = true
            rotateAnim.start()
        } else {
            var distance = action === "forward" ? 50 : -50
            var angle = heading * Math.PI / 180
            moveAnim.from = start
            moveAnim.to = Qt.vector3d(start.x + distance * Math.sin(angle), start.y,
                                     start.z + distance * Math.cos(angle))
            animationInProgress = true
            moveAnim.start()
        }
        appendLog("Viewer preview: " + action + " (visual only; sensor values unchanged).")
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
                    text: "Viewer preview controls\nVisual movement only"
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                GridLayout {
                    columns: 2
                    Layout.fillWidth: true
                    Repeater {
                        model: [
                            {label: "Backward", action: "backward", icon: "back.png"},
                            {label: "Forward", action: "forward", icon: "forward.png"},
                            {label: "Right", action: "right", icon: "right.png"},
                            {label: "Left", action: "left", icon: "left.png"}
                        ]
                        delegate: Button {
                            objectName: "simulationPreview_" + modelData.action
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.min(110, root.height * 0.17)
                            enabled: !root.animationInProgress
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
                            onClicked: root.preview(modelData.action)
                        }
                    }
                }
                Label { text: "Simulated posture (status only)"; color: "white" }
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
                    position: Qt.vector3d(0, 250, 800)
                    eulerRotation.x: -10
                    clipFar: 5000
                }
                DirectionalLight { eulerRotation.x: -45; eulerRotation.y: 45; brightness: 1.5 }
                DirectionalLight { eulerRotation.x: 30; eulerRotation.y: -60; brightness: 1.2 }
                PointLight { position: Qt.vector3d(0, 400, 400); brightness: 800 }
                Nao {
                    id: naoModel
                    objectName: "simulationNaoModel"
                    scale: Qt.vector3d(100, 100, 100)
                    position: Qt.vector3d(0, -100, 0)
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
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottomMargin: 16
                width: parent.width - 24
                text: "Posture changes appear in status.\nThis model has no sitting animation yet."
                color: "white"
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
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
                Item { Layout.fillHeight: true }
            }
        }
    }
    PropertyAnimation {
        id: moveAnim
        target: naoModel
        property: "position"
        duration: 1000
        onStopped: root.animationInProgress = false
    }
    PropertyAnimation {
        id: rotateAnim
        target: naoModel
        property: "eulerRotation"
        duration: 800
        onStopped: root.animationInProgress = false
    }
}
