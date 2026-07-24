import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Transfer Data is renamed to Cloud Computing
Rectangle {
    id: cloudTabRoot
    color: "#718399"
    objectName: "cloudTabRoot"

    // Original baseline signal interfaces
    signal saveConfig(string host, string username, string privateKeyDir, string targetDir, bool ignoreHostKey, string sourceDir, string configPath)
    signal loadConfig(string configPath)
    signal clearConfig()
    signal upload(string host, string username, string privateKeyDir, string password, bool ignoreHostKey, string sourceDir, string targetDir)
    
    // NEW SIGNAL: Dispatched when user changes Substrate endpoint connection parameter
    signal updateSubstrateEndpoint(string rpcUrl)

    // NEW STATE PROPERTIES: Bound directly to the backend Python engine thread for real-time status updates
    property bool isP2pSecure: false
    property string p2pStatusMessage: "Awaiting State Engine Connection..."
    property string latestBlockHash: "Initializing Node Validation..."

    RowLayout {
        anchors.fill: parent
        anchors.margins: 15
        spacing: 15

        // =========================================================================
        // LEFT VIEWPORT: Dedicated Configuration Credentials Entry Form
        // =========================================================================
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width * 0.45
            clip: true

            Rectangle {
                width: parent.width
                implicitHeight: contentLayout.implicitHeight + 20
                color: "#5a6b7d"
                border.color: "#CCCCCC"
                border.width: 1
                radius: 4

                ColumnLayout {
                    id: contentLayout
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 10

                    Label {
                        text: "Target IP"
                        color: "white"
                        font.bold: true
                    }
                    TextField {
                        id: hostInput
                        objectName: "hostInput"
                        Layout.fillWidth: true
                        text: ""
                    }

                    Label {
                        text: "Target Username"
                        color: "white"
                        font.bold: true
                    }
                    TextField {
                        id: usernameInput
                        objectName: "usernameInput"
                        Layout.fillWidth: true
                        text: ""
                    }

                    Label {
                        text: "Target Password"
                        color: "white"
                        font.bold: true
                    }
                    TextField {
                        id: passwordInput
                        objectName: "passwordInput"
                        Layout.fillWidth: true
                        echoMode: TextInput.Password
                        text: ""
                    }

                    // NEW LAB DEPLOYMENT EXTENSION: Allows runtime mutation between your local sandbox and lab server
                    Label {
                        text: "Substrate Blockchain RPC URL"
                        color: "orange"
                        font.bold: true
                    }
                    TextField {
                        id: rpcUrlInput
                        objectName: "rpcUrlInput"
                        Layout.fillWidth: true
                        text: "ws://127.0.0.1:9944"
                        placeholderText: "ws://127.0.0.1:9944"
                        onEditingFinished: cloudTabRoot.updateSubstrateEndpoint(text)
                    }

                    Label {
                        text: "Private Key Directory:"
                        color: "white"
                        font.bold: true
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.topMargin: -5
                        height: 40
                        color: "transparent"
                        border.color: "#CCCCCC"
                        border.width: 1
                        radius: 4
                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 4
                            spacing: 8
                            TextField {
                                id: privateKeyDirInput
                                objectName: "privateKeyDirInput"
                                Layout.fillWidth: true
                                text: ""
                            }
                            Button {
                                id: privateKeyDirButton
                                objectName: "privateKeyDirButton"
                                text: "Browse"
                                font.bold: true
                                onClicked: privateKeyFileDialog.open()
                                contentItem: Text {
                                    text: parent.text
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle { color: "#2C3E50"; radius: 4 }
                            }
                        }
                    }

                    CheckBox {
                        id: ignoreHostKeyCheckbox
                        objectName: "ignoreHostKeyCheckbox"
                        text: "Ignore Host Key"
                        font.bold: true
                        checked: true
                        contentItem: Text {
                            text: parent.text
                            font.bold: true
                            color: "white"
                            leftPadding: parent.indicator.width + parent.spacing
                        }
                    }

                    Label {
                        text: "Source Directory:"
                        color: "white"
                        font.bold: true
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.topMargin: -5
                        height: 40
                        color: "transparent"
                        border.width: 1
                        border.color: "#CCCCCC"
                        radius: 4
                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 4
                            spacing: 8
                            TextField {
                                id: sourceDirInput
                                objectName: "sourceDirInput"
                                text: ""
                                Layout.fillWidth: true
                            }
                            Button {
                                id: sourceDirButton
                                objectName: "sourceDirButton"
                                text: "Browse"
                                font.bold: true
                                onClicked: sourceDirFileDialog.open()
                                contentItem: Text {
                                    text: parent.text
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle { color: "#2C3E50"; radius: 4 }
                            }
                        }
                    }

                    Label {
                        text: "Target Directory:"
                        color: "white"
                        font.bold: true
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.topMargin: -5
                        height: 40
                        color: "transparent"
                        border.width: 1
                        border.color: "#CCCCCC"
                        radius: 4
                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 4
                            spacing: 8
                            TextField {
                                id: targetDirInput
                                placeholderText: "/home/"
                                objectName: "targetDirInput"
                                Layout.fillWidth: true
                                text: "/home/"
                            }
                            Button {
                                id: targetDirButton
                                objectName: "targetDirButton"
                                text: "Browse"
                                font.bold: true
                                onClicked: targetDirFileDialog.open()
                                contentItem: Text {
                                    text: parent.text
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle { color: "#2C3E50"; radius: 4 }
                            }
                        }
                    }
                    
                    // Bottom operation control buttons Row
                    RowLayout {
                        Layout.alignment: Qt.AlignHCenter
                        spacing: 8
                        Button {
                            id: saveConfigButton
                            objectName: "saveConfigButton"
                            text: "Save Config"
                            font.bold: true
                            onClicked: configFileDialog.open()
                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle { color: "#2C3E50"; radius: 4 }
                        }
                        Button {
                            id: loadConfigButton
                            objectName: "loadConfigButton"
                            text: "Load Config"
                            font.bold: true
                            onClicked: configFileDialog.open()
                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle { color: "#2C3E50"; radius: 4 }
                        }
                        Button {
                            id: clearConfigButton
                            objectName: "clearConfigButton"
                            text: "Clear Config"
                            font.bold: true
                            onClicked: {
                                console.log("Clear Config clicked");
                                cloudTabRoot.clearConfig();
                            }
                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle { color: "#2C3E50"; radius: 4 }
                        }
                        Button {
                            id: uploadButton
                            objectName: "uploadButton"
                            text: "Upload"
                            font.bold: true
                            onClicked: {
                                console.log("Upload clicked");
                                cloudTabRoot.upload(hostInput.text, usernameInput.text, privateKeyDirInput.text, passwordInput.text, ignoreHostKeyCheckbox.checked, sourceDirInput.text, targetDirInput.text);
                            }
                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle { color: "#2C3E50"; radius: 4 }
                        }
                    }
                }
            }
        }
        
        // =========================================================================
        // RIGHT VIEWPORT: New P2P Cybersecurity Sub-Panel Frame
        // =========================================================================
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width * 0.55
            color: "#4a5866"
            border.color: "#95A5A6"
            border.width: 2
            radius: 6
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 15
                Label {
                    text: "P2P Quantum Cybersecurity"
                    color: "#ECF0F1"
                    font.pixelSize: 18
                    font.bold: true
                }
                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: "#7F8C8D"
                }
                Label {
                    text: "View Event Mode"
                    color: "orange"
                    font.pixelSize: 14
                    font.bold: true
                }
                // Dynamic Status Visual Indicator Highlight Box (Handles Graceful Degradation colors)
                Rectangle {
                    id: statusHighlightBox
                    Layout.fillWidth: true
                    Layout.preferredHeight: 80
                    color: cloudTabRoot.isP2pSecure ? "#2ECC71" : "#E74C3C"
                    radius: 4
                    border.color: "white"
                    border.width: 1
                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: 5
                        Text {
                            text: cloudTabRoot.isP2pSecure ? "BLOCKCHAIN ENCRYPTED CHANNEL ACTIVE" : "SECURING CONNECTION..."
                            color: "white"
                            font.bold: true
                            font.pixelSize: 14
                            renderType: Text.NativeRendering
                            horizontalAlignment: Text.AlignHCenter
                        }
                    }
                }
                Label {
                    text: "Live Network Conditions Log Stream:"
                    color: "#BDC3C7"
                    font.bold: true
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#2C3E50"
                    radius: 4
                    border.color: "#7F8C8D"
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 8
                        Text {
                            text: "Status: " + cloudTabRoot.p2pStatusMessage
                            color: "#ECF0F1"
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            font.family: "Courier"
                            font.pixelSize: 12
                        }
                        Text {
                            text: "Latest Handshake Block Finality Hash:\n" + cloudTabRoot.latestBlockHash
                            color: cloudTabRoot.isP2pSecure ? "#2ECC71" : "#F1C40F"
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            font.family: "Courier"
                            font.pixelSize: 11
                        }
                        Item { Layout.fillHeight: true } // Bottom expansion spacer
                    }
                }
            }
        }
    }

    // Standard Functional File Dialog Triggers
    FileDialog { id: privateKeyFileDialog; title: "Select Private Key Directory"; onAccepted: { privateKeyDirInput.text = fileUrl.toLocalFile() } }
    FileDialog { id: sourceDirFileDialog; title: "Select Source Directory"; onAccepted: { sourceDirInput.text = fileUrl.toLocalFile() } }
    FileDialog { id: targetDirFileDialog; title: "Select Target Directory"; onAccepted: { targetDirInput.text = fileUrl.toLocalFile() } }
    FileDialog {
        id: configFileDialog
        title: "Select Configuration File"
        onAccepted: {
            if (saveConfigButton.down) {
                saveConfig(hostInput.text, usernameInput.text, privateKeyDirInput.text, targetDirInput.text, ignoreHostKeyCheckbox.checked, sourceDirInput.text, fileUrl.toLocalFile());
            } else {
                loadConfig(fileUrl.toLocalFile());
            }
        }
    }
}