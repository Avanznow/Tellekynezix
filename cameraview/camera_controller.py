"""
Camera Controller for Tello Drone
Handles camera streaming and video capture functionality
"""
import cv2
import base64
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot, QTimer


class CameraController(QObject):
    # Signals for QML communication
    frameReady = Signal(str)  # Send base64 encoded frame to QML
    streamStatusChanged = Signal(bool)  # Stream on/off status
    logMessage = Signal(str)  # Log messages
    
    def __init__(self, tello_instance=None):
        super().__init__()
        self.tello = tello_instance
        self.is_streaming = False
        self.stream_thread = None
        self.current_frame = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        capture_dir = Path(__file__).resolve().parent.parent / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = capture_dir
        
    def set_tello_instance(self, tello_instance):
        """Set the Tello drone instance"""
        self.tello = tello_instance
        
    @Slot()
    def start_camera_stream(self):
        """Start the camera stream from Tello drone"""
        if self.tello is None:
            self.logMessage.emit("Error: Tello drone not connected")
            return
            
        if self.is_streaming:
            self.logMessage.emit("Camera stream already running")
            return
            
        try:
            # Start Tello video stream
            self.tello.streamon()
            self.current_frame = None
            self.is_streaming = True
            self.streamStatusChanged.emit(True)
            
            # Start frame processing timer
            self.frame_timer.start(33)  # ~30 FPS
            
            self.logMessage.emit("Camera stream started")
            
        except Exception as e:
            self.logMessage.emit(f"Error starting camera stream: {e}")
            self.is_streaming = False
            self.streamStatusChanged.emit(False)
    
    @Slot()
    def stop_camera_stream(self):
        """Stop the camera stream"""
        if not self.is_streaming:
            return
            
        try:
            self.frame_timer.stop()
            
            if self.tello:
                self.tello.streamoff()
                
            self.is_streaming = False
            self.current_frame = None
            self.streamStatusChanged.emit(False)
            self.logMessage.emit("Camera stream stopped")
            
        except Exception as e:
            self.logMessage.emit(f"Error stopping camera stream: {e}")
    
    def process_frame(self):
        """Process and emit video frames"""
        if not self.is_streaming or not self.tello:
            return
            
        try:
            # Get frame from Tello
            frame = self.tello.get_frame_read().frame

            if frame is not None:
                self.current_frame = frame.copy()
                encoded, buffer = cv2.imencode(".jpg", frame)
                if encoded:
                    jpg_as_text = base64.b64encode(buffer).decode("ascii")
                    self.frameReady.emit(f"data:image/jpeg;base64,{jpg_as_text}")
                
        except Exception as e:
            self.logMessage.emit(f"Error processing frame: {e}")
    
    @Slot()
    def capture_photo(self):
        """Capture a single photo from the stream"""
        if not self.is_streaming or not self.tello:
            self.logMessage.emit("Camera stream not active")
            return
            
        try:
            frame = self.current_frame
            if frame is None:
                frame = self.tello.get_frame_read().frame

            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = self.capture_dir / f"drone_photo_{timestamp}.jpg"
                if cv2.imwrite(str(filename), frame):
                    self.logMessage.emit(f"Photo captured: {filename}")
                else:
                    self.logMessage.emit(f"Error capturing photo: could not write {filename}")
            else:
                self.logMessage.emit("No frame available for capture")
                
        except Exception as e:
            self.logMessage.emit(f"Error capturing photo: {e}")