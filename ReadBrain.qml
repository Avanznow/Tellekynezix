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
    property string signalQualityStatus: "EEG quality: no assessed input"   // one-line status of the capture check
    // Replay Recording source: "Replay: <file> at <position> of <length> s" while a recording stands in for the headset
    property string replayRecording: ""
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

                // Model-specific controls, beside the button that produces decisions
                Column {
                    anchors.left: parent.right
                    anchors.leftMargin: 12
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 6
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
                        Button {
                            text: "Forget my model"
                            enabled: !root.gnbTraining
                            // removes this participant's stored model, statistics and windows for the selected framework
                            onClicked: {
                                backend.setGaussianNBParticipant(gnbParticipantField.text)
                                backend.forgetGaussianNBUser()
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

            // EEG quality: one-line status of the capture check (a rejected capture produces no command)
            Label {
                text: root.signalQualityStatus
                color: root.signalQualityStatus.indexOf("withheld") >= 0 ? "#ffcf70" : "white"
                font.pixelSize: 13
                width: parent.width * 0.9
                elide: Text.ElideRight
                horizontalAlignment: Text.AlignHCenter
                anchors.horizontalCenter: parent.horizontalCenter
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
                    id: modelSaysHeader
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
                    anchors.top: modelSaysHeader.bottom
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    clip: true
                    model: ListModel {
                        id: modelSaysModel          // filled from backend.predictionsTableUpdated (count, label)
                    }
                    delegate: Item {
                        width: ListView.view.width
                        height: 22
                        Text {
                            x: 0; width: parent.width * 0.5; horizontalAlignment: Text.AlignHCenter
                            text: model.count; font.bold: true; color: "white"
                        }
                        Text {
                            x: parent.width * 0.5; width: parent.width * 0.5; horizontalAlignment: Text.AlignHCenter
                            text: model.label; font.bold: true; color: "white"
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
                // One row per Read my mind (or manual command) in both tables; the backend sends the whole list.
                function onPredictionsTableUpdated(rows) {
                    predictionsModel.clear()
                    modelSaysModel.clear()
                    for (let row of rows) {
                        predictionsModel.append({ count: String(row.count), server: String(row.server), label: String(row.label) })
                        modelSaysModel.append({ count: String(row.count), label: String(row.label) })
                    }
                    predictionsTableView.positionViewAtEnd()
                    predictionListView.positionViewAtEnd()
                }
                // The flight log arrives newest first.
                function onFlightLogUpdated(entries) {
                    flightLog.text = entries.join("\n")
                }
                function onNeurosityStatusChanged(status) {
                    neurosityStatus = status
                }
                function onSignalQualityChanged(status) {
                    root.signalQualityStatus = status
                }
                function onReplayRecordingChanged(status) {
                    root.replayRecording = status
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
                    id: predictionsHeader
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
                    id: predictionsTableView
                    anchors.top: predictionsHeader.bottom
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    clip: true
                    model: ListModel {
                        id: predictionsModel        // filled from backend.predictionsTableUpdated (count, server, label)
                    }

                    delegate: Rectangle {
                        width: ListView.view.width
                        height: 28
                        color: index % 2 ? "#f2f4f7" : "white"

                        Text {
                            x: 0; width: parent.width * 0.33; anchors.verticalCenter: parent.verticalCenter; horizontalAlignment: Text.AlignHCenter
                            text: model.count; font.bold: true; color: "black"
                        }
                        Text {
                            x: parent.width * 0.33; width: parent.width * 0.33; anchors.verticalCenter: parent.verticalCenter; horizontalAlignment: Text.AlignHCenter
                            text: model.server; font.bold: true; color: "black"
                        }
                        Text {
                            x: parent.width * 0.66; width: parent.width * 0.34; anchors.verticalCenter: parent.verticalCenter; horizontalAlignment: Text.AlignHCenter
                            text: model.label; font.bold: true; color: model.label === "WITHHELD" ? "#b25c00" : "black"
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
        width: parent.width * 0.38
        height: parent.height * 0.06
        spacing: width * 0.02
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: parent.width * 0.14
        anchors.bottomMargin: parent.height * 0.03      
        RadioButton {
            id: liveRadio
            text: "Live Data"
            font.pixelSize: parent.width * 0.028
            font.bold: true
            checked: false      // the backend starts in Synthetic Data mode (BrainwavesBackend.current_data_mode)
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
            font.pixelSize: parent.width * 0.028
            font.bold: true
            checked: true
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

        RadioButton {
            id: replayRadio
            text: "Replay Recording"
            font.pixelSize: parent.width * 0.028
            font.bold: true
            checked: false
            contentItem: Text {
                text: replayRadio.text
                color: "white"
                font.pixelSize: replayRadio.font.pixelSize
                font.bold: replayRadio.font.bold
                verticalAlignment: Text.AlignVCenter
                leftPadding: replayRadio.indicator.width + replayRadio.spacing
            }
            // opens the file dialog; the chosen OpenBCI export then stands in for the headset
            onClicked: backend.chooseReplayRecording()
        }
         
    }

    // Replay Recording: which file stands in for the headset and how far the replay has advanced; clicking it
    // opens the file dialog again. Empty (hidden) in Live and Synthetic mode.
    Label {
        visible: root.replayRecording !== ""
        text: root.replayRecording
        color: "#cfe3ff"
        font.pixelSize: 13
        padding: 4
        background: Rectangle { color: "#e0213040"; radius: 3 }    // readable over the flight log's white box
        width: parent.width * 0.38
        elide: Text.ElideRight
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: parent.width * 0.14
        anchors.bottomMargin: parent.height * 0.095
        MouseArea {
            anchors.fill: parent
            onClicked: backend.chooseReplayRecording()
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
