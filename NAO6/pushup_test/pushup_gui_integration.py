#!/usr/bin/env python3

"""
NAO Push-Up Test GUI Integration
Integrates push-up testing with the main Avatar GUI
"""

import sys
import os
import threading
import time
from datetime import datetime
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from pushup_test_automation import PushupTestAutomation

class PushupTestGUI(QObject):
    # Signals for GUI communication
    testStarted = Signal(str)
    testProgress = Signal(str, int)  # message, progress percentage
    testCompleted = Signal(str, dict)  # status, results
    testError = Signal(str)
    temperatureUpdate = Signal(dict)  # temperature data
    
    def __init__(self, nao_ip="192.168.23.53", port=9559):
        super().__init__()
        self.nao_ip = nao_ip
        self.port = port
        
        self.automation = PushupTestAutomation(nao_ip, port)
        self.test_running = False
        self.current_test_thread = None
        
        # Temperature monitoring timer
        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self._emit_temperature_update)
        self.temp_timer.setInterval(2000)  # Update every 2 seconds
    
    @Slot()
    def startPushupTest(self):
        """Start the push-up test from GUI"""
        if self.test_running:
            self.testError.emit("Test already running")
            return
        
        self.test_running = True
        self.testStarted.emit("Starting NAO Push-Up Endurance Test...")
        
        # Run test in separate thread
        self.current_test_thread = threading.Thread(target=self._run_test_thread)
        self.current_test_thread.daemon = True
        self.current_test_thread.start()
    
    def _run_test_thread(self):
        """Run test in background thread"""
        try:
            # Update progress
            self.testProgress.emit("Checking robot temperature...", 10)
            time.sleep(1)
            
            if not self.automation.check_robot_temperature():
                self.testProgress.emit("Waiting for robot cooldown...", 20)
                self.automation.wait_for_cooldown()
            
            self.testProgress.emit("Starting video recording...", 30)
            video_file = self.automation.start_video_recording()
            
            self.testProgress.emit("Initializing NAO controller...", 40)
            from nao_pushup_controller import NAOPushupController
            controller = NAOPushupController(self.nao_ip, self.port)
            
            self.testProgress.emit("Moving to initial position...", 50)
            if not controller.connect_to_nao():
                self.testError.emit("Failed to connect to NAO robot")
                return
            
            if not controller.get_initial_position():
                self.testError.emit("Failed to move to initial position")
                return
            
            # Start temperature monitoring
            controller.start_temperature_monitoring()
            self.temp_timer.start()
            
            self.testProgress.emit("Starting push-up repetitions...", 60)
            
            # Perform push-ups with progress updates
            target_reps = 15
            successful_reps = 0
            
            for rep in range(1, target_reps + 1):
                progress = 60 + int((rep / target_reps) * 30)
                self.testProgress.emit(f"Performing push-up {rep}/{target_reps}...", progress)
                
                if controller.perform_pushup(rep):
                    successful_reps += 1
                else:
                    break
            
            self.testProgress.emit("Analyzing performance...", 90)
            time.sleep(1)
            
            # Analyze results
            analysis = controller.analyze_performance()
            results = controller.save_test_results()
            
            self.testProgress.emit("Finalizing test...", 100)
            
            # Stop monitoring
            self.temp_timer.stop()
            controller.stop_temperature_monitoring()
            self.automation.stop_video_recording()
            
            # Generate report
            report_file = self.automation.generate_test_report(
                {'status': 'completed', 'video_file': video_file},
                controller.test_results,
                analysis
            )
            
            # Prepare results for GUI
            gui_results = {
                'total_reps': successful_reps,
                'target_reps': target_reps,
                'fatigue_passed': analysis.get('fatigue_passed', False),
                'thermal_passed': analysis.get('thermal_passed', False),
                'overall_passed': analysis.get('overall_passed', False),
                'max_temperature': analysis.get('max_temperature', 0),
                'fatigue_ratio': analysis.get('fatigue_ratio', 0),
                'video_file': video_file,
                'report_file': report_file,
                'analysis': analysis,
                'files': results
            }
            
            self.testCompleted.emit("completed", gui_results)
            
        except Exception as e:
            self.testError.emit(f"Test failed: {str(e)}")
        
        finally:
            self.test_running = False
            self.temp_timer.stop()
    
    def _emit_temperature_update(self):
        """Emit temperature updates to GUI"""
        if self.automation and hasattr(self.automation, 'controller') and self.automation.controller:
            try:
                # Get latest temperature data
                if self.automation.controller.test_results['temperature_data']:
                    latest_temp = self.automation.controller.test_results['temperature_data'][-1]
                    self.temperatureUpdate.emit(latest_temp['temperatures'])
            except:
                pass
    
    @Slot()
    def stopPushupTest(self):
        """Stop the currently running test"""
        if self.test_running:
            self.test_running = False
            self.temp_timer.stop()
            
            if self.automation:
                self.automation.stop_video_recording()
            
            self.testCompleted.emit("stopped", {"message": "Test stopped by user"})
    
    @Slot()
    def getRobotStatus(self):
        """Get current robot status"""
        try:
            from naoqi import ALProxy
            temp_proxy = ALProxy("ALTemperature", self.nao_ip, self.port)
            
            # Get key joint temperatures
            critical_joints = ["RShoulderPitch", "LShoulderPitch", "RHipPitch", "LHipPitch"]
            temps = {}
            
            for joint in critical_joints:
                try:
                    temps[joint] = temp_proxy.getTemperature(joint)
                except:
                    temps[joint] = None
            
            return {
                'connected': True,
                'temperatures': temps,
                'ready': all(temp < 50 for temp in temps.values() if temp is not None)
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'ready': False
            }

# Integration with main GUI
class PushupTestIntegration:
    """Integration class for adding push-up test to main GUI"""
    
    @staticmethod
    def add_to_gui(gui_app):
        """Add push-up test functionality to existing GUI"""
        
        # Create push-up test instance
        pushup_test = PushupTestGUI()
        
        # Expose to QML
        if hasattr(gui_app, 'engine'):
            gui_app.engine.rootContext().setContextProperty("pushupTest", pushup_test)
        
        return pushup_test

# Example usage in main GUI
def example_integration():
    """Example of how to integrate with main GUI"""
    
    # In your main GUI initialization (GUI5.py):
    
    # Import the integration
    from NAO6.pushup_test.pushup_gui_integration import PushupTestIntegration
    
    # After creating the engine:
    # pushup_test = PushupTestIntegration.add_to_gui(self)
    
    # In QML, you can then call:
    # pushupTest.startPushupTest()
    # pushupTest.stopPushupTest()
    # pushupTest.getRobotStatus()
    
    pass

if __name__ == "__main__":
    # Test the GUI integration
    app = PushupTestGUI()
    
    print("NAO Push-Up Test GUI Integration Ready")
    print("Available methods:")
    print("- startPushupTest()")
    print("- stopPushupTest()") 
    print("- getRobotStatus()")
    print("\nSignals:")
    print("- testStarted(message)")
    print("- testProgress(message, percentage)")
    print("- testCompleted(status, results)")
    print("- testError(error)")
    print("- temperatureUpdate(temperatures)")
