import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    color: "#718399"
    width: 1100
    height: 700

    onVisibleChanged: {
        if (visible) {
            developersBackend.devChart()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 20
        anchors.margins: 10

        // ===================== TOP SECTION =====================
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height / 2   // Top half of the window
            spacing: 15

            // ---------- GOLD PANEL ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Gold"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 26
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        radius: 8
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: developersBackend.goldPath

                            onStatusChanged: {
                                if (status === Image.Error) {
                                    console.log("Gold chart not available.")
                                }
                            }
                        }
                    }
                }
            }

            // ---------- SILVER PANEL ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Silver"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 26
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        radius: 8
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: developersBackend.silverPath

                            onStatusChanged: {
                                if (status === Image.Error) {
                                    console.log("Silver chart not available.")
                                }
                            }
                        }
                    }
                }
            }

            // ---------- BRONZE PANEL ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Bronze"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 26
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        radius: 8
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: developersBackend.bronzePath

                            onStatusChanged: {
                                if (status === Image.Error) {
                                    console.log("Bronze chart not available.")
                                }
                            }
                        }
                    }
                }
            }
        }

        // ===================== BOTTOM SECTION =====================
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height / 2    // Bottom half of the window
            spacing: 10

            // ---------- DEVELOPERS LOG ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "List Of Developers"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 22
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                        TextArea {
                            id: devText
                            text: developersBackend.devListText
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: 12
                            color: "black"
                            background: Rectangle { color: "white"; radius: 6 }
                        }
                    }
                }

                Component.onCompleted: {
                    ticketText.text = developersBackend.getTicketsByDev()
                }
            }

            // ---------- MEDAL + REFRESH ----------
            Rectangle {
                color: "#718399"
                Layout.preferredWidth: 230
                Layout.fillHeight: true
                radius: 10
                border.color: "transparent"

                Column {
                    anchors.centerIn: parent
                    width: parent.width
                    spacing: 0   // minimal space between image and button

                    // Medal Image
                    Image {
                        source: developersBackend.medalPath
                        width: 225
                        height: 285
                        fillMode: Image.PreserveAspectFit
                        smooth: true
                        antialiasing: true
                        anchors.horizontalCenter: parent.horizontalCenter

                        onStatusChanged: {
                            if (status === Image.Error) {
                                console.log("Medal image not available.")
                            }
                        }
                    }

                    Button {
                        text: "Refresh"
                        font.bold: true
                        width: 200
                        height: 50
                        anchors.horizontalCenter: parent.horizontalCenter

                        background: Rectangle {
                            color: "#003366"
                            radius: 5
                            border.color: "#001F3F"
                        }

                        contentItem: Text {
                            text: qsTr("Refresh")
                            font.bold: true
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        onClicked: {
                            developersBackend.devChartForce()
                            ticketText.text = developersBackend.getTicketsByDev()
                        }
                    }
                }
            }

            // ---------- TICKET BY DEVELOPER LOG ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Ticket by Developer Log"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 22
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                        TextArea {
                            id: ticketText
                            text: "Console output here..."
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: 12
                            color: "black"
                            background: Rectangle { color: "white"; radius: 6 }
                        }
                    }
                }
            }
        }
    }
}
