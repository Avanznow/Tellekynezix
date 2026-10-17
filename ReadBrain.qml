import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Read Brain view
Rectangle {
    id: root
    property string selectedModel: "Random Forest"
    property string currentFramework: "PyTorch"
    property string selectedHeadset: "OpenBCI"
    property string neurosityStatus: "Offline"
    property string signalQualityStatus: "EEG quality: no assessed input"
    // Structured quality view from backend.qualityViewChanged (signal_quality.gui_view)
    property string qualityState: "idle"
    property string qualityHeadline: ""
    property string qualitySummary: ""
    property var qualityReasons: []
    property var qualityChannels: []
    // Improved GaussianNB chain: "<participant>: calibrated | training participant | global fallback"
    property string gnbStatus: ""
    // Subject-specific GaussianNB: "global" = one model for everyone, "subject" = this user's own model trained in
    // the guided interleaved cue session (backend.startGaussianNBTraining); progress arrives in gnbCue.
    property string gnbMode: "global"
    property bool gnbTraining: false
    property var gnbCue: ({})
    property string gnbExpectation: ""
    property string gnbTrainingMessage: ""
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    Row {
        anchors.fill: parent
        spacing: width * 0.02

        // Left Column
        Column {
            width: parent.width * 0.5
            height: parent.height
            spacing: height * 0.02
            anchors.left: parent.left

            // Control Mode
            Row {
                width: parent.width * 0.4
                spacing: parent.width * 0.02
                anchors.horizontalCenter: parent.horizontalCenter

                RadioButton {
                    id: manualControl
                    text: "Manual Control"
                    checked: true
                    font.pixelSize: parent.width * 0.05

                    contentItem: Text {
                        text: manualControl.text
                        color: "white"
                        font.pixelSize: manualControl.font.pixelSize
                        font.bold: true
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: manualControl.indicator.width + manualControl.spacing
                    }
                }

                RadioButton {
                    id: autopilot
                    text: "Autopilot"
                    font.pixelSize: parent.width * 0.05
                    contentItem: Text {
                        text: autopilot.text
                        color: "white"
                        font.pixelSize: autopilot.font.pixelSize
                        font.bold: true
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: autopilot.indicator.width + autopilot.spacing
                    }
                }
            }

            // Brainwave Image with Transparent Button
            Rectangle {
                width: parent.width * 0.25
                height: parent.height * 0.2
                color: "#242c4d"
                anchors.horizontalCenter: parent.horizontalCenter

                Image {
                    source: "GUI_Pics/brain.png"
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                }

                Button {
                    anchors.fill: parent
                    background: Item {
                    }
                    contentItem: Text {
                        text: "Read my mind..."
                        font.pixelSize: parent.width * 0.1
                        color: "white"
                        anchors.centerIn: parent
                    }
                    onClicked: backend.readMyMind()
                }

                // EEG quality tools, beside the button that produces quality decisions
                Column {
                    anchors.left: parent.right
                    anchors.leftMargin: 12
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 6
                    Button {
                        text: "Quality dashboard..."
                        onClicked: backend.openQualityDashboard()
                    }
                    Button {
                        text: "Check recording..."
                        onClicked: backend.checkRecordingQuality()
                    }
                    // GaussianNB (improved chain): the participant id picks the per-participant feature
                    // statistics; a new user records ~30 s of EEG once (or picks one of their recordings).
                    Row {
                        visible: selectedModel === "GaussianNB"
                        spacing: 6
                        TextField {
                            id: gnbParticipantField
                            width: 130
                            placeholderText: "participant id"
                            font.pixelSize: 12
                            Component.onCompleted: text = backend.gaussianNBParticipant()
                            onEditingFinished: backend.setGaussianNBParticipant(text)
                        }
                        Button {
                            text: "Calibrate (30 s)"
                            onClicked: {
                                backend.setGaussianNBParticipant(gnbParticipantField.text)
                                backend.calibrateGaussianNB()
                            }
                        }
                    }
                    Row {
                        visible: selectedModel === "GaussianNB"
                        spacing: 6
                        Button {
                            text: "Calibrate from recording..."
                            onClicked: {
                                backend.setGaussianNBParticipant(gnbParticipantField.text)
                                backend.calibrateGaussianNBFromRecording()
                            }
                        }
                        Text {
                            text: root.gnbStatus
                            color: "white"
                            font.pixelSize: 12
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }
                    // Subject-specific GaussianNB: "My model" runs a model fitted on this user's own labelled
                    // windows (84 % per participant on Data_clean vs 49 % for the global chain). It needs one
                    // guided session first: 6 rounds x 6 commands x 20 s cues in random order with rest gaps.
                    Row {
                        visible: selectedModel === "GaussianNB"
                        spacing: 6
                        Component.onCompleted: {
                            root.gnbMode = backend.gaussianNBMode()
                            var plan = backend.gaussianNBTrainingPlan()
                            if (plan && plan.expectation)
                                root.gnbExpectation = plan.expectation
                        }
                        RadioButton {
                            id: gnbGlobalMode
                            text: "Global model"
                            checked: root.gnbMode === "global"
                            onClicked: {
                                root.gnbMode = "global"
                                backend.setGaussianNBMode("global")
                            }
                            contentItem: Text {
                                text: gnbGlobalMode.text
                                color: "white"
                                font.pixelSize: 12
                                verticalAlignment: Text.AlignVCenter
                                leftPadding: gnbGlobalMode.indicator.width + gnbGlobalMode.spacing
                            }
                        }
                        RadioButton {
                            id: gnbSubjectMode
                            text: "My model"
                            checked: root.gnbMode === "subject"
                            onClicked: {
                                root.gnbMode = "subject"
                                backend.setGaussianNBMode("subject")
                            }
                            contentItem: Text {
                                text: gnbSubjectMode.text
                                color: "white"
                                font.pixelSize: 12
                                verticalAlignment: Text.AlignVCenter
                                leftPadding: gnbSubjectMode.indicator.width + gnbSubjectMode.spacing
                            }
                        }
                    }
                    Row {
                        visible: selectedModel === "GaussianNB" && root.gnbMode === "subject"
                        spacing: 6
                        Button {
                            text: root.gnbTraining ? "Stop training" : "Train my model (15 min)"
                            onClicked: {
                                if (root.gnbTraining) {
                                    backend.stopGaussianNBTraining()
                                } else {
                                    backend.setGaussianNBParticipant(gnbParticipantField.text)
                                    root.gnbTrainingMessage = ""
                                    backend.startGaussianNBTraining()
                                }
                            }
                        }
                        Button {
                            text: "Train from folder..."
                            enabled: !root.gnbTraining
                            onClicked: {
                                backend.setGaussianNBParticipant(gnbParticipantField.text)
                                backend.trainGaussianNBFromFolder()
                            }
                        }
                    }
                    Text {
                        visible: selectedModel === "GaussianNB" && root.gnbMode === "subject"
                        width: 300
                        wrapMode: Text.WordWrap
                        color: "white"
                        font.pixelSize: 11
                        text: (root.gnbTrainingMessage.length > 0 ? root.gnbTrainingMessage + "\n" : "")
                              + "Learning curve (Data_clean): about 4 recordings per command for ~75 %, 6 for ~80 % window accuracy. "
                              + root.gnbExpectation
                    }
                }
            }

            // EEG quality: withholding indicator, channel grid and quality tools (compact for 1200x800)
            Rectangle {
                id: qualityPanel
                width: parent.width * 0.9
                height: qualityColumn.implicitHeight + 10
                radius: 6
                color: "#242c4d"
                border.width: 2
                border.color: root.qualityState === "accepted" ? "#66bb6a"
                            : root.qualityState === "withheld" ? "#ff8a65" : "#8a94a6"
                anchors.horizontalCenter: parent.horizontalCenter

                Column {
                    id: qualityColumn
                    x: 6
                    y: 5
                    width: parent.width - 12
                    spacing: 3

                    Label {
                        // Before the first structured view (or while assessing) show the plain status line.
                        text: (root.qualityHeadline !== "" && root.signalQualityStatus.indexOf("Assessing") < 0
                               ? root.qualityHeadline : root.signalQualityStatus)
                              + (root.qualitySummary !== "" ? "  ·  " + root.qualitySummary : "")
                        color: root.qualityState === "withheld" ? "#ffcf70"
                             : root.qualityState === "accepted" ? "#a5d6a7" : "white"
                        font.bold: true
                        font.pixelSize: 13
                        width: parent.width
                        elide: Text.ElideRight
                    }
                    Label {
                        id: qualityReasonsLabel
                        visible: root.qualityReasons.length > 0
                        text: root.qualityReasons.join("  ·  ")
                        color: "white"
                        font.pixelSize: 11
                        width: parent.width
                        elide: Text.ElideRight
                        HoverHandler { id: reasonsHover }
                        ToolTip.visible: reasonsHover.hovered
                        ToolTip.text: root.qualityReasons.join("\n")
                    }
                    Flow {
                        width: parent.width
                        spacing: 2
                        Repeater {
                            model: root.qualityChannels
                            Rectangle {
                                width: 26
                                height: 18
                                radius: 3
                                color: modelData.status === "pass" ? "#2e7d4f"
                                     : modelData.status === "fail" ? "#c62828"
                                     : modelData.status === "warn" ? "#b7791f" : "#5a6275"
                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.label.replace("Ch ", "")
                                    color: "white"
                                    font.pixelSize: 10
                                }
                                HoverHandler { id: channelHover }
                                ToolTip.visible: channelHover.hovered
                                ToolTip.text: modelData.name + ": " + modelData.status + " - " + modelData.detail
                            }
                        }
                    }
                }
            }
            Label {
                text: "The model says ..."
                color: "white"
                font.pixelSize: parent.width * 0.03
                anchors.horizontalCenter: parent.horizontalCenter
            }

            GroupBox {
                width: parent.width * 0.4
                height: parent.height * 0.15
                anchors.horizontalCenter: parent.horizontalCenter

                Row {
                    width: parent.width
                    height: parent.height * 0.28
                    spacing: parent.width * 0.01
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.5
                        height: parent.height
                        Text {
                            text: "Count"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.5
                        height: parent.height
                        Text {
                            text: "Label"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                }

                ListView {
                    id: predictionListView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: ListModel {
                    }
                    delegate: RowLayout {
                        spacing: 150
                        Text { 
                            text: model.count; font.bold: true; color: "white"; width: 80 
                        }
                        Text { 
                            text: model.label; font.bold: true; color: "white"; width: 80 
                        }
                    }
                }
            }

            // Action Buttons
            Row {
                width: parent.width * 0.6
                height: parent.height * 0.08
                spacing: parent.width * 0.02
                anchors.horizontalCenter: parent.horizontalCenter
                Button {
                    id: notThinking
                    text: "Not what I was thinking..."
                    font.pixelSize: parent.width * 0.03
                    width: (parent.width - parent.spacing) * 0.5
                    height: parent.height
                    background: Rectangle { 
                        color: "#242c4d" 
                    }
                    contentItem: Text {
                        text: notThinking.text
                        color: "white"
                        font.pixelSize: notThinking.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }
                    onClicked: backend.notWhatIWasThinking(manualInput.text)
                }

                Button {
                    id: executeBtn
                    text: "Action"
                    font.pixelSize: parent.width * 0.03
                    width: (parent.width - parent.spacing) * 0.5
                    height: parent.height
                    background: Rectangle { 
                        color: "#242c4d" 
                    }
                    contentItem: Text {
                        text: executeBtn.text
                        color: "white"
                        font.pixelSize: executeBtn.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }
                    onClicked: backend.executeAction()
                }
            }

            // Manual Input and Keep Alive
            Row {
                width: parent.width * .8
                height: parent.height * 0.03
                spacing: parent.width * 0.01
                anchors.horizontalCenter: parent.horizontalCenter
                TextField {
                    id: manualInput
                    placeholderText: "Manual Command"
                    font.pixelSize: parent.width * 0.03
                    width: parent.width * 0.6
                    height: parent.height
                }
                Button {
                    width: parent.width * 0.3
                    height: parent.height
                    background: Rectangle { 
                        color: "#242c4d" 
                    }

                    contentItem: Text {
                        text: qsTr("Run")
                        color: "white"
                        font.pixelSize: parent.height * 0.5
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        anchors.fill: parent
                    }
                    onClicked: {
                        backend.keepDroneAlive(text=manualInput.text)
                        manualInput.text = ""
                    }
                }
            }

            Text { 
              anchors.horizontalCenter: parent.horizontalCenter
              text: "Flight Log"
              color: "white"
              font.bold: true
              font.pixelSize: 15
              horizontalAlignment: Text.AlignHCenter
              Layout.alignment: Qt.AlignHCenter
            }
            // Flight Log
            GroupBox { 
              width: parent.width * 0.7
              height: parent.height * 0.2
              anchors.horizontalCenter: parent.horizontalCenter              
              background: Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumHeight: root.minControlSize
                Layout.minimumWidth: root.minControlSize
                implicitWidth: Math.max(root.minControlSize, Math.min(root.width * 0.35, 420))
                implicitHeight: Math.max(root.minControlSize,Math.min(root.height * 0.25, 220))
                color:"#5f6b7a"
                radius: 6 
                border.color:"#d0d6df"
                border.width: 1 
              }
              ScrollView {
                anchors.fill: parent
                clip: true

                TextArea {
                  id: flightLog
                  wrapMode: Text.WrapAnywhere
                  readOnly: true
                  font.pixelSize: parent.width * 0.025
                  color: "black"
                  background: Rectangle { color: "white" }
                }
              }
            }

            Connections {
                target: backend
                function onImagesReady(imageData) {
                    imageModel.clear();
                    for (let item of imageData) {
                        imageModel.append(item);
                    }
                }
                function onLogMessage(message) {
                    var timestamp = new Date().toLocaleString()
                    consoleLog.append(message + " at " + timestamp)
                }
                function onNeurosityStatusChanged(status) {
                    neurosityStatus = status
                }
                function onSignalQualityChanged(status) {
                    root.signalQualityStatus = status
                }
                function onQualityViewChanged(view) {
                    root.qualityState = view.state
                    root.qualityHeadline = view.headline
                    root.qualitySummary = view.summary
                    root.qualityReasons = view.reasons
                    root.qualityChannels = view.channels
                }
                function onGnbParticipantStatusChanged(status) {
                    root.gnbStatus = status
                }
                function onGnbTrainingCueChanged(cue) {
                    root.gnbCue = cue
                    root.gnbTraining = cue.phase !== "finished"
                }
                function onGnbTrainingFinished(summary) {
                    root.gnbTraining = false
                    root.gnbTrainingMessage = summary.trained
                        ? ("Trained: " + summary.message + " (" + summary.status + ")")
                        : ("Not trained: " + summary.message)
                }
            }
        }

        // Right Column (Prediction Table and Console Log)
        Column {
            width: parent.width * 0.45
            height: parent.height
            spacing: parent.height * 0.02
            anchors.right: parent.right

            // Predictions Table
            GroupBox {
                title: "Predictions Table"
                width: parent.width * 0.9
                height: parent.height * 0.35

                label: Text {
                    text: qsTr("Predictions Table")
                    color: "white"
                    font.pixelSize: parent.width * 0.03
                    font.bold: true
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.top: parent.top
                    anchors.topMargin: 5
                }

                Row {
                    width: parent.width
                    height: parent.height * 0.1
                    spacing: parent.width * 0.001
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.33
                        height: parent.height
                        Text {
                            text: "Predictions Count"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.33
                        height: parent.height
                        Text {
                            text: "Server Predictions"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.33
                        height: parent.height
                        Text {
                            text: "Prediction Label"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                }

                ListView {
                    Layout.preferredWidth: 700
                    Layout.preferredHeight: 550
                    model: ListModel {
                        ListElement { 
                            count: "1"; server: "Prediction A"; label: "Label A" 
                        }
                        ListElement { 
                            count: "2"; server: "Prediction B"; label: "Label B" 
                        }
                    }

                    delegate: Rectangle {
                        width: parent.width
                        height: 40
                        color: "white"

                        RowLayout {
                            anchors.fill: parent
                            spacing: 50
                            Text { 
                                text: model.count; font.bold: true; color: "black"; width: 120 
                            }
                            Text { 
                                text: model.server; font.bold: true; color: "black"; width: 200 
                            }
                            Text { 
                                text: model.label; font.bold: true; color: "black"; width: 120 
                            }
                        }
                    }
                }
            }

        Text {
            text: "Console Log"
            color: "white"
            font.bold: true
            font.pixelSize: 15
            horizontalAlignment: Text.AlignHCenter
            Layout.alignment: Qt.AlignHCenter
        }
            // Console Log
            GroupBox {
              width: parent.width * 0.9
              height: parent.height * 0.25 
              background: Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumHeight: root.minControlSize
                Layout.minimumWidth: root.minControlSize
                implicitWidth: Math.max(root.minControlSize, Math.min(root.width * 0.35, 420))
                implicitHeight: Math.max(root.minControlSize,Math.min(root.height * 0.25, 220))
                color:"#5f6b7a"
                radius: 6 
                border.color:"#d0d6df"
                border.width: 1 
              }


                ScrollView {
                    anchors.fill: parent
                    clip: true

                    TextArea {
                        id: consoleLog
                        wrapMode: Text.WrapAnywhere
                        readOnly: true
                        font.pixelSize: 14
                        color: "black"
                        background: Rectangle { color: "white" }
                    }
                }
            }

            // Spacer to push Model Selection buttons down
            Item {
                width: parent.width
                height: parent.height * 0.02
            }

            // Model Selection Buttons in dark box
            Rectangle {
                width: parent.width * 0.95
                height: parent.height * 0.25
                color: "#2c3e50"
                
                Column {
                    anchors.fill: parent
                    anchors.margins: parent.height * 0.05
                    spacing: parent.height * 0.1
                    
                    // Model Selection Row
                    Row {
                        width: parent.width
                        height: (parent.height - parent.spacing * 2) / 3
                        spacing: parent.width * 0.02
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "Random Forest" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "Random Forest" ? 3 : 1
                            
                            Text {
                                text: "Random Forest"
                                font.pixelSize: parent.height * 0.25
                                font.bold: true
                                color: selectedModel === "Random Forest" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "Random Forest"
                                    backend.selectModel("Random Forest")
                                }
                            }
                        }
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "GaussianNB" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "GaussianNB" ? 3 : 1
                            
                            Text {
                                text: "GaussianNB"
                                font.pixelSize: parent.height * 0.25
                                font.bold: true
                                color: selectedModel === "GaussianNB" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "GaussianNB"
                                    backend.selectModel("GaussianNB")
                                }
                            }
                        }
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "Deep Learning" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "Deep Learning" ? 3 : 1
                            
                            Text {
                                text: "Deep Learning"
                                font.pixelSize: parent.height * 0.25
                                font.bold: true
                                color: selectedModel === "Deep Learning" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "Deep Learning"
                                    backend.selectModel("Deep Learning")
                                }
                            }
                        }
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "Riemannian + LR" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "Riemannian + LR" ? 3 : 1

                            Text {
                                text: "Riemannian + LR"
                                font.pixelSize: parent.height * 0.19
                                font.bold: true
                                color: selectedModel === "Riemannian + LR" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "Riemannian + LR"
                                    backend.selectModel("Riemannian + LR")
                                }
                            }
                        }
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "Band Power + SVM" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "Band Power + SVM" ? 3 : 1

                            Text {
                                text: "Band Power + SVM"
                                font.pixelSize: parent.height * 0.16
                                font.bold: true
                                color: selectedModel === "Band Power + SVM" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "Band Power + SVM"
                                    backend.selectModel("Band Power + SVM")
                                }
                            }
                        }
                    }

                    // Framework Selection Row
                    Row {
                        width: parent.width
                        height: (parent.height - parent.spacing * 2) / 3
                        spacing: parent.width * 0.02
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: currentFramework === "PyTorch" ? "yellow" : "#4a9d6f"
                            border.width: currentFramework === "PyTorch" ? 3 : 1
                            
                            Text {
                                text: "PyTorch"
                                font.pixelSize: parent.height * 0.3
                                font.bold: true
                                color: currentFramework === "PyTorch" ? "yellow" : "white"
                                anchors.centerIn: parent
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "PyTorch"
                                    backend.selectFramework("PyTorch")
                                }
                            }
                        }

                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: currentFramework === "TensorFlow" ? "yellow" : "#4a9d6f"
                            border.width: currentFramework === "TensorFlow" ? 3 : 1
                            
                            Text {
                                text: "TensorFlow"
                                font.pixelSize: parent.height * 0.3
                                font.bold: true
                                color: currentFramework === "TensorFlow" ? "yellow" : "white"
                                anchors.centerIn: parent
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "TensorFlow"
                                    backend.selectFramework("TensorFlow")
                                }
                            }
                        }
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: currentFramework === "JAX" ? "yellow" : "#4a9d6f"
                            border.width: currentFramework === "JAX" ? 3 : 1
                            
                            Text {
                                text: "JAX"
                                font.pixelSize: parent.height * 0.3
                                font.bold: true
                                color: currentFramework === "JAX" ? "yellow" : "white"
                                anchors.centerIn: parent
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "JAX"
                                    backend.selectFramework("JAX")
                                }
                            }
                        }
                        
                    }

            //Headset select row
            Row{
                width: parent.width
                height: (parent.height - parent.spacing * 2) / 3
                spacing: parent.width * 0.02

                Rectangle {
                    width: (parent.width - parent.spacing ) / 2
                    height: parent.height
                    color: "#2d7a4a"
                    radius: 5
                    border.color: selectedHeadset === "OpenBCI" ? "yellow" : "#4a9d6f"
                    border.width: selectedHeadset === "OpenBCI" ? 3 : 1

                    Text {
                        text: "OpenBCI"
                        font.pixelSize: parent.height * 0.3
                        font.bold: true
                        color: selectedHeadset === "OpenBCI" ? "yellow" : "white"
                        anchors.centerIn: parent
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            selectedHeadset = "OpenBCI"
                            backend.setBCISource("openbci")
                            
                        }
                    }
                }

                Rectangle {
                    width: (parent.width - parent.spacing ) / 2 
                    height: parent.height
                    color: "#2d7a4a"
                    radius: 5
                    border.color: selectedHeadset === "Neurosity" ? "yellow" : "#4a9d6f"
                    border.width: selectedHeadset === "Neurosity" ? 3 : 1

                    Text {
                        text: "Neurosity(" + neurosityStatus + ")"
                        font.pixelSize: parent.height * 0.3
                        font.bold: true
                        color: selectedHeadset === "Neurosity" ? "yellow" : "white"
                        anchors.centerIn: parent
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                        selectedHeadset = "Neurosity"
                        backend.setBCISource("neurosity")
                        
                        }
                    }
                }

                    }



                }
            }          
        }
    }

    // Connect Button - Fixed to bottom left
    Rectangle {
        width: parent.width * 0.06
        height: parent.height * 0.10
        color: "#242c4d"
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: parent.width * 0.01
        anchors.bottomMargin: parent.height * 0.01

        Image {
            source: "GUI_Pics/connect.png"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
        }

        Button {
            anchors.fill: parent
            background: Item {
            }
            contentItem: Text {
                text: "Connect"
                font.pixelSize: parent.width * 0.15
                color: "white"
                anchors.centerIn: parent
            }
            onClicked: backend.connectDrone()
        }
    }

    // Radio Buttons - Fixed to bottom left, next to connect
   Row {
        width: parent.width * 0.25
        height: parent.height * 0.06
        spacing: width * 0.02
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: parent.width * 0.14
        anchors.bottomMargin: parent.height * 0.03      
        RadioButton {
            id: liveRadio
            text: "Live Data"
            font.pixelSize: parent.width * 0.04
            font.bold: true
            checked: true
            contentItem: Text {
                text: liveRadio.text
                color: "white"
                font.pixelSize: liveRadio.font.pixelSize
                font.bold: liveRadio.font.bold
                verticalAlignment: Text.AlignVCenter
                leftPadding: liveRadio.indicator.width + liveRadio.spacing
            }
            onClicked: backend.setDataMode("live")
        }

        RadioButton {
            id: syntheticRadio
            text: "Synthetic Data"
            font.pixelSize: parent.width * 0.04
            font.bold: true
            checked: false
            contentItem: Text {
                text: syntheticRadio.text
                color: "white"
                font.pixelSize: syntheticRadio.font.pixelSize
                font.bold: syntheticRadio.font.bold
                verticalAlignment: Text.AlignVCenter
                leftPadding: syntheticRadio.indicator.width + syntheticRadio.spacing
            }
            onClicked: backend.setDataMode("synthetic")
        }
         
    }

    // Guided labelled session (subject-specific GaussianNB): the cue the user should think about, the countdown,
    // and how many quality-checked windows were kept so far. Shown over the whole view while the session runs.
    Rectangle {
        id: gnbTrainingOverlay
        anchors.fill: parent
        z: 50
        visible: root.gnbTraining
        color: "#e6213040"
        property string phase: root.gnbCue && root.gnbCue.phase ? root.gnbCue.phase : ""
        property string command: root.gnbCue && root.gnbCue.command ? String(root.gnbCue.command).toUpperCase() : ""
        property int remaining: root.gnbCue && root.gnbCue.remaining !== undefined ? root.gnbCue.remaining : 0
        property int cueIndex: root.gnbCue && root.gnbCue.index !== undefined ? root.gnbCue.index : -1
        property int nCues: root.gnbCue && root.gnbCue.n_cues ? root.gnbCue.n_cues : 0
        property int windowsTotal: root.gnbCue && root.gnbCue.windows_total !== undefined ? root.gnbCue.windows_total : 0

        MouseArea { anchors.fill: parent }   // swallow clicks on the controls behind the overlay

        Column {
            anchors.centerIn: parent
            spacing: 18
            width: parent.width * 0.8

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                color: "#cfd8e3"
                font.pixelSize: 22
                text: gnbTrainingOverlay.phase === "settle" ? "Get ready - sit still, headset settling"
                    : gnbTrainingOverlay.phase === "rest" ? "Rest - next command:"
                    : gnbTrainingOverlay.phase === "cue" ? "Think the command now:"
                    : gnbTrainingOverlay.phase === "cue_done" ? "Saved" : ""
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                color: gnbTrainingOverlay.phase === "cue" ? "yellow" : "white"
                font.pixelSize: gnbTrainingOverlay.phase === "cue" ? 72 : 44
                font.bold: true
                text: gnbTrainingOverlay.command
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                color: "white"
                font.pixelSize: 40
                text: gnbTrainingOverlay.remaining > 0 ? gnbTrainingOverlay.remaining + " s" : ""
            }
            ProgressBar {
                anchors.horizontalCenter: parent.horizontalCenter
                width: parent.width
                from: 0
                to: Math.max(1, gnbTrainingOverlay.nCues)
                value: Math.max(0, gnbTrainingOverlay.cueIndex + (gnbTrainingOverlay.phase === "cue_done" ? 1 : 0))
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                color: "#cfd8e3"
                font.pixelSize: 16
                text: (gnbTrainingOverlay.cueIndex >= 0
                       ? "Cue " + (gnbTrainingOverlay.cueIndex + 1) + " of " + gnbTrainingOverlay.nCues
                         + (root.gnbCue.round ? "  -  round " + root.gnbCue.round + " of " + root.gnbCue.rounds : "")
                       : "")
                      + "  -  windows kept: " + gnbTrainingOverlay.windowsTotal
                      + (root.gnbCue.phase === "cue_done" && root.gnbCue.n_windows !== undefined
                         ? " (+" + root.gnbCue.n_windows + ", " + root.gnbCue.n_blocks_kept + "/" + root.gnbCue.n_blocks + " blocks)" : "")
            }
            Button {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Stop and keep what was recorded"
                onClicked: backend.stopGaussianNBTraining()
            }
        }
    }
}
