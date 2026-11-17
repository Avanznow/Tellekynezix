import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform
import "cameraview"

// Manual Drone Control Tab view
Rectangle {
    color: "#718399"

    readonly property color buttonBg: "#242c4d"
    readonly property color buttonBgInAction: '#191f35'
    readonly property color buttonBorder: "#3a4a62"
    readonly property int  buttonRadius: 10

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10

        // ====================================
        // ==== Left side - Drone Controls ====
        // ====================================
        Rectangle {
            id: controlsPanel
            Layout.fillWidth: false
            // Layout.preferredWidth: parent.width * 0.7
            Layout.preferredWidth: parent.width * 0.50
            Layout.fillHeight: true
            color: "transparent"

            // control row container
            ColumnLayout {
                anchors.fill: parent
                spacing: 5

                // ************************************
                // >> Top Row - Home, Up, Flight Log <<
                // ************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // Home Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/home.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Home"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: parent.color = buttonBgInAction
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                                onClicked: backend.doDroneTAction("go_home")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.095 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Up Button
                        Rectangle {
                            Layout.preferredWidth: 0.405 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/up.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Up"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("up") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.09 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Flight Log
                        Rectangle {
                            id: flightLogContainer
                            Layout.preferredWidth: 0.19 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: "#ffffff"
                            border.color: "#2E4053"
                            border.width: 1

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 3
                                anchors.topMargin: 5
                                spacing: 4

                                Text {
                                    text: "FLIGHT LOG"
                                    font.weight: 700
                                    color: "black"
                                    font.pixelSize: 12
                                    Layout.alignment: Qt.AlignHCenter
                                    font.letterSpacing: 1.1
                                }

                                ScrollView {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    clip: false

                                    TextArea {
                                        id: flightLog
                                        readOnly: true
                                        wrapMode: TextArea.Wrap
                                        font.pixelSize: Math.max(10, controlsPanel.height * 0.015)
                                        color: "black"

                                        background: Rectangle { color: "white" }

                                        Component.onCompleted: {
                                            backend.flightLogUpdated.connect(function (logList) {
                                                flightLog.text = logList.join("\n")
                                                flightLog.moveCursorSelection(TextArea.End)  // auto-scroll to bottom
                                            })
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // *******************************************
                // >>        2nd Row - Forward Button       <<
                // *******************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Flip Forward Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {visible: false}

                                Text {
                                    visible: true
                                    text: "↻"
                                    font.bold: true
                                    color: "#76ff03"
                                    font.pixelSize: parent.height * 0.8
                                    anchors.centerIn: parent
                                }
                            }

                            Text {
                                text: "Flip Forward"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("flip_forward") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // forward
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/forward.png"
                                }

                                Text { visible: false }

                            }

                            Text {
                                text: "Forward"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("forward") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // flip right
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: false
                                }

                                Text {
                                    visible: true
                                    text: "⟳"
                                    font.bold: true
                                    color: "#76ff03"
                                    anchors.centerIn: parent
                                    font.pixelSize: parent.height * 0.8
                                }
                            }

                            Text {
                                text: "Flip Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("flip_right") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }
                    }
                }

                // **************************************************************************************
                // >> 3rd Row - Directional Buttons (Turn Left, Left, Stream, Right, Turn Right)       <<
                // **************************************************************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // Turn Left Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/turnLeft.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Turn Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("turn_left") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // Left Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/left.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("left") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // Stream Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/stream.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Stream"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: parent.color = buttonBgInAction
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                                onClicked: backend.doDroneTAction("stream")
                            }
                        }

                        // Right Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/right.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("right") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // Turn Right Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/turnRight.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Turn Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("turn_right") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }
                    }
                }

                // *******************************************************
                // >> 4th Row - Back Button Row with Flip Buttons       << DONE
                // *******************************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Flip Back Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {visible: false}

                                Text {
                                    visible: true
                                    text: "↺"
                                    font.bold: true
                                    color: "#76ff03"
                                    font.pixelSize: parent.height * 0.8
                                    anchors.centerIn: parent
                                }
                            }

                            Text {
                                text: "Flip Back"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("flip_back") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // Back Button (Center)
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/back.png"
                                }

                                Text { visible: false }

                            }

                            Text {
                                text: "Back"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("backward") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // Flip Left Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {visible: false}

                                Text {
                                    visible: true
                                    text: "⟲"
                                    font.bold: true
                                    color: "#76ff03"
                                    font.pixelSize: parent.height * 0.8
                                    anchors.centerIn: parent
                                }
                            }

                            Text {
                                text: "Flip Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("flip_left") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }
                    }
                }
                // *******************************************************
                // >> 5th Row - Connect, Down, Takeoff, Land Buttons    <<
                // *******************************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // Connect Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/connect.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Connect"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: parent.color = buttonBgInAction
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                                onClicked: backend.doDroneTAction("connect")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.09 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Down Button
                        Rectangle {
                            Layout.preferredWidth: 0.405 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/down.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Down"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("down") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.085 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Takeoff Button
                        Rectangle {
                            Layout.preferredWidth: 0.0975 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/takeoff.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Takeoff"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("takeoff") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                                // onClicked: backend.takeoff()
                                cursorShape: Qt.PointingHandCursor
                            }
                        }

                        // Land Button
                        Rectangle {
                            Layout.preferredWidth: 0.0975 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/land.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Land"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onPressed: { parent.color = buttonBgInAction; backend.doDroneTAction("land") }
                                onReleased: parent.color = buttonBg
                                onEntered: parent.color = buttonBgInAction
                                onExited: parent.color = buttonBg
                            }
                        }
                    }
                }
            }
        }

        // ====================================
        // ==== 3D DRONE VIEW =================
        // ====================================
        // Qt Quick 3D axes: +X = right, +Y = up, -Z = drone nose (away from camera).
        // Primitive meshes (#Cube, #Cylinder, #Sphere) are 100 units, so
        // scale 0.40 = 40 units. Whole drone is ~170 units across.
        Rectangle {
            id: drone3DPanel

            Layout.preferredWidth: parent.width * 0.20
            Layout.minimumWidth: 250
            Layout.fillHeight: true
            color: "#718399"
            clip: true

            property string currentAction: "Hover"
            property bool landed: false

            // ---- Persistent pose (from backend.drone3DUpdated) ----
            // Scaled and clamped so the drone always stays inside the panel.
            property real poseScale: 0.3
            property real poseLimit: 45
            // Set to -1 if the model snaps back the wrong way after a turn.
            property real headingSign: 1
            property real posX: 0
            property real posY: 0
            property real posZ: 0
            property real heading: 0

            Behavior on posX { NumberAnimation { duration: 300; easing.type: Easing.OutCubic } }
            Behavior on posY { NumberAnimation { duration: 300; easing.type: Easing.OutCubic } }
            Behavior on posZ { NumberAnimation { duration: 300; easing.type: Easing.OutCubic } }
            Behavior on heading { NumberAnimation { duration: 380; easing.type: Easing.OutCubic } }

            // ---- Transient motion (button feedback) - always eases back to 0 ----
            property real pitch: 0      // negative = nose down
            property real roll: 0       // negative = right side down
            property real nudgeY: 0     // small hop during flips
            property real altitude: 0   // takeoff / land offset
            property real bob: 0        // idle hover bob

            // ---- Directional hold + auto return home ----
            // Each Up/Down/Forward/Back/Left/Right press moves one step and
            // the drone HOLDS there. After holdSeconds with no button press,
            // it flies back to home and hovers.
            property int  holdSeconds: 30
            property int  secondsLeft: 0
            property real stepSize: 22
            property real limitX: 34      // left / right
            property real limitZ: 60      // forward / back
            property real limitUp: 55
            property real limitDown: 40
            property int  moveDuration: 450
            property real offX: 0
            property real offY: 0
            property real offZ: 0
            readonly property bool awayFromHome: offX !== 0 || offY !== 0 || offZ !== 0
                                                 || posX !== 0 || posY !== 0 || posZ !== 0

            Behavior on offX { NumberAnimation { duration: drone3DPanel.moveDuration; easing.type: Easing.InOutCubic } }
            Behavior on offY { NumberAnimation { duration: drone3DPanel.moveDuration; easing.type: Easing.InOutCubic } }
            Behavior on offZ { NumberAnimation { duration: drone3DPanel.moveDuration; easing.type: Easing.InOutCubic } }

            function step(axis, dir) {
                moveDuration = 450
                if (axis === "x") offX = Math.max(-limitX, Math.min(limitX, offX + dir * stepSize))
                if (axis === "y") offY = Math.max(-limitDown, Math.min(limitUp, offY + dir * stepSize))
                if (axis === "z") offZ = Math.max(-limitZ, Math.min(limitZ, offZ + dir * stepSize))
            }

            // (Re)start the 30 s countdown. Called on every button press.
            function restartHoldTimer() {
                if (!awayFromHome || landed) { holdTimer.stop(); secondsLeft = 0; return }
                secondsLeft = holdSeconds
                holdTimer.restart()
            }

            function returnHome() {
                holdTimer.stop()
                secondsLeft = 0
                moveDuration = 1800                 // slow, smooth flight home
                offX = 0; offY = 0; offZ = 0
                posX = 0; posY = 0; posZ = 0        // also clear backend pose offset
                currentAction = "Returning home"
                actionLabelTimer.interval = 1900
                actionLabelTimer.restart()
            }

            Timer {
                id: holdTimer
                interval: 1000
                repeat: true
                onTriggered: {
                    drone3DPanel.secondsLeft -= 1
                    if (drone3DPanel.secondsLeft <= 0) drone3DPanel.returnHome()
                }
            }

            function clampPose(v) {
                return Math.max(-poseLimit, Math.min(poseLimit, v))
            }

            // ---------------- Animations ----------------

            // Idle hover: gentle bob whenever the drone is flying.
            SequentialAnimation {
                running: !drone3DPanel.landed
                loops: Animation.Infinite
                NumberAnimation { target: drone3DPanel; property: "bob"; to: 4;  duration: 1100; easing.type: Easing.InOutSine }
                NumberAnimation { target: drone3DPanel; property: "bob"; to: -4; duration: 1100; easing.type: Easing.InOutSine }
            }

            // Tilt (pitch or roll) then settle back to level.
            SequentialAnimation {
                id: pitchTilt
                property real peak: 0
                NumberAnimation { target: drone3DPanel; property: "pitch"; to: pitchTilt.peak; duration: 160; easing.type: Easing.OutCubic }
                NumberAnimation { target: drone3DPanel; property: "pitch"; to: 0; duration: 600; easing.type: Easing.OutBack }
            }
            SequentialAnimation {
                id: rollTilt
                property real peak: 0
                NumberAnimation { target: drone3DPanel; property: "roll"; to: rollTilt.peak; duration: 160; easing.type: Easing.OutCubic }
                NumberAnimation { target: drone3DPanel; property: "roll"; to: 0; duration: 600; easing.type: Easing.OutBack }
            }

            // 360-degree flips with a small hop.
            ParallelAnimation {
                id: pitchFlip
                property real endAngle: -360
                SequentialAnimation {
                    NumberAnimation { target: drone3DPanel; property: "pitch"; from: 0; to: pitchFlip.endAngle; duration: 700; easing.type: Easing.InOutQuad }
                    ScriptAction { script: drone3DPanel.pitch = 0 }
                }
                SequentialAnimation {
                    NumberAnimation { target: drone3DPanel; property: "nudgeY"; to: 22; duration: 350; easing.type: Easing.OutQuad }
                    NumberAnimation { target: drone3DPanel; property: "nudgeY"; to: 0;  duration: 450; easing.type: Easing.InOutSine }
                }
            }
            ParallelAnimation {
                id: rollFlip
                property real endAngle: 360
                SequentialAnimation {
                    NumberAnimation { target: drone3DPanel; property: "roll"; from: 0; to: rollFlip.endAngle; duration: 700; easing.type: Easing.InOutQuad }
                    ScriptAction { script: drone3DPanel.roll = 0 }
                }
                SequentialAnimation {
                    NumberAnimation { target: drone3DPanel; property: "nudgeY"; to: 22; duration: 350; easing.type: Easing.OutQuad }
                    NumberAnimation { target: drone3DPanel; property: "nudgeY"; to: 0;  duration: 450; easing.type: Easing.InOutSine }
                }
            }

            // Takeoff / land.
            SequentialAnimation {
                id: takeoffAnim
                ScriptAction { script: drone3DPanel.landed = false }
                NumberAnimation { target: drone3DPanel; property: "altitude"; to: 0; duration: 1200; easing.type: Easing.OutCubic }
            }
            SequentialAnimation {
                id: landAnim
                NumberAnimation { target: drone3DPanel; property: "altitude"; to: -55; duration: 1400; easing.type: Easing.InOutQuad }
                ScriptAction { script: { drone3DPanel.landed = true; drone3DPanel.bob = 0; drone3DPanel.currentAction = "Landed" } }
            }

            function stopTransient() {
                pitchTilt.stop(); rollTilt.stop()
                pitchFlip.stop(); rollFlip.stop()
                pitch = 0; roll = 0
                nudgeY = 0
            }

            function tilt(tiltAnim, peak) {
                tiltAnim.peak = peak
                tiltAnim.restart()
            }

            function animateDroneAction(action) {
                // Stop previous motion so rapid presses don't stack.
                stopTransient()

                currentAction = action.replace(/_/g, " ")
                actionLabelTimer.interval = 900
                actionLabelTimer.restart()

                if (action === "go_home") {
                    if (!landed) returnHome()
                    return
                }
                if (action === "takeoff") {
                    landAnim.stop(); takeoffAnim.restart()
                    restartHoldTimer()
                    return
                }
                if (action === "land") {
                    // Lands where it is; no auto-return while on the ground.
                    takeoffAnim.stop(); landAnim.restart()
                    holdTimer.stop(); secondsLeft = 0
                    return
                }
                // A landed drone doesn't move.
                if (landed) return

                switch (action) {
                case "forward":      tilt(pitchTilt, -15); step("z", -1); break
                case "backward":     tilt(pitchTilt,  15); step("z",  1); break
                case "left":         tilt(rollTilt,   15); step("x", -1); break
                case "right":        tilt(rollTilt,  -15); step("x",  1); break
                case "up":           step("y",  1); break
                case "down":         step("y", -1); break
                case "turn_left":    heading += 45; break
                case "turn_right":   heading -= 45; break
                case "flip_forward": pitchFlip.endAngle = -360; pitchFlip.restart(); break
                case "flip_back":    pitchFlip.endAngle =  360; pitchFlip.restart(); break
                case "flip_left":    rollFlip.endAngle  =  360; rollFlip.restart();  break
                case "flip_right":   rollFlip.endAngle  = -360; rollFlip.restart();  break
                }

                // Any button press while away from home resets the 30 s hold.
                restartHoldTimer()
            }

            Connections {
                target: backend

                // Immediate visual response on button press.
                function onDrone3DPreview(action) { drone3DPanel.animateDroneAction(action) }

                // Commands from BCI/AI where there's no QML button press.
                function onDrone3DAction(action) { drone3DPanel.animateDroneAction(action) }

                // Python X/Y/Z = lateral/forward/altitude.
                // Qt Quick 3D: X = lateral, Y = altitude, -Z = forward.
                function onDrone3DUpdated(x, y, z, rotation) {
                    drone3DPanel.posX = drone3DPanel.clampPose( x * drone3DPanel.poseScale)
                    drone3DPanel.posY = drone3DPanel.clampPose( z * drone3DPanel.poseScale)
                    drone3DPanel.posZ = drone3DPanel.clampPose(-y * drone3DPanel.poseScale)
                    drone3DPanel.heading = rotation * drone3DPanel.headingSign
                    drone3DPanel.restartHoldTimer()
                }
            }

            Timer {
                id: actionLabelTimer
                interval: 900
                repeat: false
                onTriggered: {
                    // Keep the label while takeoff/land is still playing.
                    if (landAnim.running || takeoffAnim.running) { restart(); return }
                    drone3DPanel.currentAction = drone3DPanel.landed ? "Landed" : "Hover"
                }
            }

            // ---------------- 3D scene ----------------
            View3D {
                id: droneView
                anchors.fill: parent

                environment: SceneEnvironment {
                    backgroundMode: SceneEnvironment.Transparent
                    clearColor: "transparent"
                    antialiasingMode: SceneEnvironment.MSAA
                    antialiasingQuality: SceneEnvironment.High
                    // Soft ambient occlusion for contact shading
                    aoStrength: 35
                    aoDistance: 6
                    aoSoftness: 50
                }

                // Chase camera: behind and above, looking at the drone.
                PerspectiveCamera {
                    position: Qt.vector3d(0, 120, 250)
                    eulerRotation.x: -25
                    fieldOfViewOrientation: PerspectiveCamera.Horizontal
                    fieldOfView: 60
                    clipNear: 1
                    clipFar: 2000
                }

                // Key, rim and fill lights
                DirectionalLight { eulerRotation.x: -50; eulerRotation.y: -25; brightness: 1.6 }
                DirectionalLight { eulerRotation.x: -15; eulerRotation.y: 170; brightness: 0.8 }
                DirectionalLight { eulerRotation.x: 40;  eulerRotation.y: 20;  brightness: 0.35 }

                // Soft ground shadow - shrinks/fades as the drone climbs.
                Model {
                    id: shadowModel
                    property real lift: Math.max(0, Math.min(1, (droneRoot.position.y + 55) / 140))
                    source: "#Cylinder"
                    position: Qt.vector3d(droneRoot.position.x, -72, droneRoot.position.z)
                    scale: Qt.vector3d(1.5 - lift * 0.5, 0.001, 1.4 - lift * 0.5)
                    materials: PrincipledMaterial {
                        baseColor: "black"
                        lighting: PrincipledMaterial.NoLighting
                        alphaMode: PrincipledMaterial.Blend
                        opacity: 0.28 - shadowModel.lift * 0.14
                    }
                }

                Node {
                    id: droneRoot
                    position: Qt.vector3d(
                        drone3DPanel.posX + drone3DPanel.offX,
                        drone3DPanel.posY + drone3DPanel.offY + drone3DPanel.altitude + drone3DPanel.bob + drone3DPanel.nudgeY,
                        drone3DPanel.posZ + drone3DPanel.offZ)
                    eulerRotation.y: drone3DPanel.heading

                    Node {
                        id: droneVisual
                        eulerRotation.x: drone3DPanel.pitch
                        eulerRotation.z: drone3DPanel.roll

                        // ---- Materials (matte gray, realistic) ----
                        PrincipledMaterial { id: bodyMat;   baseColor: "#5b6169"; roughness: 0.45; metalness: 0.05; clearcoatAmount: 0.35; clearcoatRoughnessAmount: 0.3 }
                        PrincipledMaterial { id: shellMat;  baseColor: "#757c85"; roughness: 0.35; metalness: 0.05; clearcoatAmount: 0.5;  clearcoatRoughnessAmount: 0.2 }
                        PrincipledMaterial { id: darkMat;   baseColor: "#2c3035"; roughness: 0.6 }
                        PrincipledMaterial { id: armMat;    baseColor: "#4e545b"; roughness: 0.5; metalness: 0.05 }
                        PrincipledMaterial { id: motorMat;  baseColor: "#3a3f46"; roughness: 0.35; metalness: 0.6 }
                        PrincipledMaterial { id: bellMat;   baseColor: "#a7adb4"; roughness: 0.25; metalness: 0.9 }
                        PrincipledMaterial { id: bladeMat;  baseColor: "#17191d"; roughness: 0.55 }
                        PrincipledMaterial { id: glassMat;  baseColor: "#0a0d12"; roughness: 0.05; metalness: 0.3 }
                        PrincipledMaterial {
                            id: discMat
                            baseColor: "#c9d3df"
                            lighting: PrincipledMaterial.NoLighting
                            alphaMode: PrincipledMaterial.Blend
                            opacity: 0.10
                        }

                        // ---- Body (smooth ellipsoids instead of boxes) ----
                        Model { source: "#Sphere"; position: Qt.vector3d(0, 0, 2);   scale: Qt.vector3d(0.44, 0.15, 0.94); materials: [bodyMat] }
                        Model { source: "#Sphere"; position: Qt.vector3d(0, 5, 6);   scale: Qt.vector3d(0.38, 0.10, 0.76); materials: [shellMat] }
                        Model { source: "#Sphere"; position: Qt.vector3d(0, -5, 4);  scale: Qt.vector3d(0.36, 0.08, 0.76); materials: [darkMat] }
                        // Rear battery grip
                        Model { source: "#Sphere"; position: Qt.vector3d(0, 1, 40);  scale: Qt.vector3d(0.34, 0.11, 0.12); materials: [darkMat] }
                        // Top cooling vent
                        Model { source: "#Cube";   position: Qt.vector3d(0, 10.1, 14); scale: Qt.vector3d(0.16, 0.004, 0.22); materials: [darkMat] }
                        // Front obstacle sensors
                        Model { source: "#Sphere"; position: Qt.vector3d(-8, 3, -43); scale: Qt.vector3d(0.06, 0.035, 0.02); materials: [glassMat] }
                        Model { source: "#Sphere"; position: Qt.vector3d( 8, 3, -43); scale: Qt.vector3d(0.06, 0.035, 0.02); materials: [glassMat] }

                        // ---- Gimbal camera under the nose ----
                        Model { source: "#Sphere";   position: Qt.vector3d(0, -7, -38);   scale: Qt.vector3d(0.08, 0.06, 0.08); materials: [darkMat] }
                        Model { source: "#Cube";     position: Qt.vector3d(0, -11, -41);  scale: Qt.vector3d(0.14, 0.025, 0.03); materials: [motorMat] }
                        Model { source: "#Sphere";   position: Qt.vector3d(0, -13, -44);  scale: Qt.vector3d(0.12, 0.10, 0.10); materials: [darkMat] }
                        Model { source: "#Cylinder"; position: Qt.vector3d(0, -13, -49);  eulerRotation.x: 90; scale: Qt.vector3d(0.065, 0.02, 0.065); materials: [motorMat] }
                        Model { source: "#Sphere";   position: Qt.vector3d(0, -13, -50.2); scale: Qt.vector3d(0.05, 0.05, 0.012)
                                materials: PrincipledMaterial { baseColor: "#0b1a2a"; roughness: 0.05; metalness: 0.9 } }

                        // ---- Arms, motors, legs, nav lights and rotors ----
                        // Front arms sit higher than rear arms (folding-arm layout).
                        // ax/az = where the arm meets the body. dir = spin direction.
                        Repeater3D {
                            model: [
                                { x: -58, y:  2, z: -46, ax: -16, az: -26, dir:  1, front: true  },
                                { x:  58, y:  2, z: -46, ax:  16, az: -26, dir: -1, front: true  },
                                { x: -56, y: -3, z:  48, ax: -16, az:  28, dir: -1, front: false },
                                { x:  56, y: -3, z:  48, ax:  16, az:  28, dir:  1, front: false }
                            ]
                            delegate: Node {
                                id: motorNode
                                required property var modelData
                                readonly property real dx: modelData.ax - modelData.x
                                readonly property real dz: modelData.az - modelData.z
                                position: Qt.vector3d(modelData.x, modelData.y, modelData.z)

                                // Tapered arm from motor to body
                                Model {
                                    source: "#Sphere"
                                    position: Qt.vector3d(motorNode.dx / 2, 0, motorNode.dz / 2)
                                    eulerRotation.y: Math.atan2(motorNode.dx, motorNode.dz) * 180 / Math.PI
                                    scale: Qt.vector3d(0.075, 0.055, Math.sqrt(motorNode.dx * motorNode.dx + motorNode.dz * motorNode.dz) / 100 + 0.06)
                                    materials: [armMat]
                                }
                                // Motor base + silver bell
                                Model { source: "#Cylinder"; position: Qt.vector3d(0, 1, 0); scale: Qt.vector3d(0.13, 0.09, 0.13); materials: [motorMat] }
                                Model { source: "#Cylinder"; position: Qt.vector3d(0, 6, 0); scale: Qt.vector3d(0.11, 0.04, 0.11); materials: [bellMat] }
                                // Landing leg
                                Model {
                                    source: "#Cylinder"
                                    position: Qt.vector3d(0, modelData.front ? -9 : -7, 0)
                                    scale: Qt.vector3d(0.028, modelData.front ? 0.16 : 0.12, 0.028)
                                    materials: [darkMat]
                                }
                                // Nav light: red front, green rear
                                Model {
                                    source: "#Sphere"
                                    position: Qt.vector3d(0, -4, 0)
                                    scale: Qt.vector3d(0.04, 0.025, 0.04)
                                    materials: PrincipledMaterial {
                                        baseColor: motorNode.modelData.front ? "#ff3b30" : "#34c759"
                                        emissiveFactor: motorNode.modelData.front ? Qt.vector3d(1.0, 0.15, 0.1) : Qt.vector3d(0.15, 0.9, 0.3)
                                    }
                                }
                                // Rotor
                                Node {
                                    position: Qt.vector3d(0, 10, 0)

                                    NumberAnimation on eulerRotation.y {
                                        from: 0
                                        to: 360 * motorNode.modelData.dir
                                        duration: 260
                                        loops: Animation.Infinite
                                        running: !drone3DPanel.landed
                                    }

                                    // Two tapered, slightly pitched blades
                                    Model { source: "#Sphere"; position: Qt.vector3d( 13.5, 0, 0); eulerRotation.x:  6 * motorNode.modelData.dir; scale: Qt.vector3d(0.28, 0.008, 0.05); materials: [bladeMat] }
                                    Model { source: "#Sphere"; position: Qt.vector3d(-13.5, 0, 0); eulerRotation.x: -6 * motorNode.modelData.dir; scale: Qt.vector3d(0.28, 0.008, 0.05); materials: [bladeMat] }
                                    // Hub
                                    Model { source: "#Cylinder"; position: Qt.vector3d(0, 1.5, 0); scale: Qt.vector3d(0.06, 0.04, 0.06); materials: [bellMat] }
                                    // Faint motion-blur disc while spinning
                                    Model {
                                        source: "#Cylinder"
                                        visible: !drone3DPanel.landed
                                        scale: Qt.vector3d(0.58, 0.003, 0.58)
                                        materials: [discMat]
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Text {
                text: drone3DPanel.currentAction.toUpperCase()
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 7
                color: "#1f2933"
                font.bold: true
                font.pixelSize: 11
                z: 10
            }

            // Countdown until auto return home.
            Text {
                visible: holdTimer.running
                text: "RETURNING HOME IN " + drone3DPanel.secondsLeft + "s"
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottomMargin: 8
                color: "#1f2933"
                font.bold: true
                font.pixelSize: 10
                z: 10
            }
        }

        // ====================================
        // ==== Right side of Camera View =====
        // ====================================
            
        CameraView {
            id: cameraPanel
            Layout.preferredWidth: parent.width * 0.3
            Layout.minimumWidth: 300
            Layout.fillHeight: true
            // width: parent.width * 0.3
            // height: parent.height
            cameraController: cameraController
        }
    }
}
