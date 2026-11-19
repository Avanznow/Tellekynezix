import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtMultimedia

Rectangle {
    id: panel
    anchors.fill: parent
    color: "#2c3e50"
    property var cameraController
    onVisibleChanged: { if (!visible && cameraController) cameraController.stop() }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8
        Label { text: "CAMERA VIEW"; color: "white"; font.bold: true; Layout.alignment: Qt.AlignHCenter }
        ComboBox {
            id: sourceSelector
            objectName: "naoPreviewSource"
            Layout.fillWidth: true
            model: ["Local webcam", "Test video"]
            enabled: !panel.cameraController.active
            onActivated: panel.cameraController.chooseSource(currentIndex === 0 ? "webcam" : "video")
        }
        ComboBox {
            objectName: "naoPreviewCameraSelector"
            Layout.fillWidth: true
            visible: sourceSelector.currentIndex === 0
            model: panel.cameraController.cameras
            currentIndex: panel.cameraController.selectedCameraIndex
            displayText: count > 0 && currentIndex >= 0 ? currentText : "No webcam selected"
            enabled: !panel.cameraController.active && count > 0
            onActivated: panel.cameraController.chooseCamera(currentIndex)
        }
        Button {
            text: "Choose video..."
            visible: sourceSelector.currentIndex === 1
            enabled: !panel.cameraController.active
            Layout.fillWidth: true
            onClicked: videoFile.open()
        }
        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: panel.cameraController.sourceLabel
            color: "white"
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#11161a"
            clip: true
            VideoOutput {
                id: preview
                objectName: "naoPreviewVideo"
                anchors.fill: parent
                fillMode: VideoOutput.PreserveAspectFit
                visible: panel.cameraController.active && panel.cameraController.hasFrame
                Component.onCompleted: panel.cameraController.attachOutput(preview)
            }
            Label {
                anchors.centerIn: parent
                text: "Preview stopped"
                color: "#c0cadd"
                visible: !panel.cameraController.hasFrame
            }
        }
        Label { text: panel.cameraController.frameInfo; color: "white"; Layout.fillWidth: true }
        Label {
            text: panel.cameraController.status
            color: "white"
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
        RowLayout {
            Layout.fillWidth: true
            Button {
                objectName: "naoCameraStartButton"
                text: "Start Stream"
                Layout.fillWidth: true
                enabled: !panel.cameraController.active
                onClicked: panel.cameraController.start()
            }
            Button {
                objectName: "naoCameraStopButton"
                text: "Stop Stream"
                Layout.fillWidth: true
                enabled: panel.cameraController.active
                onClicked: panel.cameraController.stop()
            }
            Button {
                objectName: "naoCameraCaptureButton"
                text: "Capture"
                Layout.fillWidth: true
                enabled: panel.cameraController.active && panel.cameraController.hasFrame
                onClicked: { if (panel.cameraController.prepareCapture()) captureFile.open() }
            }
        }
    }
    FileDialog {
        id: videoFile
        title: "Choose a local test video"
        nameFilters: ["Video files (*.mp4 *.avi *.mov *.mkv *.webm)", "All files (*)"]
        onAccepted: panel.cameraController.chooseVideo(selectedFile)
    }
    FileDialog {
        id: captureFile
        title: "Save preview frame"
        fileMode: FileDialog.SaveFile
        nameFilters: ["PNG image (*.png)"]
        defaultSuffix: "png"
        onAccepted: panel.cameraController.capture(selectedFile)
        onRejected: panel.cameraController.cancelCapture()
    }
}
