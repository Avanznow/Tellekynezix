#!/usr/bin/env python3

"""
NAO Walking Test GUI Integration
Integrates walking and navigation testing with the main Avatar GUI
"""

import sys
import os
import threading
import time
from datetime import datetime
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from walking_test_automation import WalkingTestAutomation

class WalkingTestGUI(QObject):
    # Signals for GUI communication
    testStarted = Signal(str)
    testProgress = Signal(str, int)  # message, progress percentage
    testCompleted = Signal(str, dict)  # status, results
    testError = Signal(str)
    stabilityUpdate = Signal(dict)  # stability data
    gaitUpdate = Signal(dict)  # gait data
    obstacleUpdate = Signal(str)  # obstacle warning
    
    def __init__(self, nao_ip="192.168.23.53", port=9559):
        super().__init__()
        self.nao_ip = nao_ip
        self.port = port
        
        self.automation = WalkingTestAutomation(nao_ip, port)
        self.test_running = False
        self.current_test_thread = None
        
        # Monitoring timers
        self.stability_timer = QTimer()
        self.stability_timer.timeout.connect(self._emit_stability_update)
        self.stability_timer.setInterval(2000)  # Update every 2 seconds
        
        self.gait_timer = QTimer()
        self.gait_timer.timeout.connect(self._emit_gait_update)
        self.gait_timer.setInterval(1000)  # Update every second
        
        # Test configuration
        self.test_config = {
            'route_distance': 5.0,
            'obstacles': [
                {'position': 2.0, 'type': 'door_sill', 'height': 0.02},
                {'position': 3.5, 'type': 'cord', 'height': 0.01}
            ]
        }
    
    @Slot()
    def startWalkingTest(self):
        """Start the walking test from GUI"""
        if self.test_running:
            self.testError.emit("Test already running")
            return
        
        self.test_running = True
        self.testStarted.emit("Starting NAO Walking and Navigation Test...")
        
        # Run test in separate thread
        self.current_test_thread = threading.Thread(target=self._run_test_thread)
        self.current_test_thread.daemon = True
        self.current_test_thread.start()
    
    def _run_test_thread(self):
        """Run test in background thread"""
        try:
            # Update progress
            self.testProgress.emit("Preparing test area...", 5)
            time.sleep(1)
            
            if not self.automation.assess_environmental_conditions():
                self.testProgress.emit("Environmental conditions not suitable...", 0)
                return
            
            self.testProgress.emit("Setting up video recording...", 10)
            video_files = self.automation.setup_video_recording()
            
            if not video_files:
                self.testError.emit("Failed to setup video recording")
                return
            
            self.testProgress.emit("Initializing NAO controller...", 20)
            from nao_walking_controller import NAOWalkingController
            controller = NAOWalkingController(self.nao_ip, self.port)
            controller.route_config = self.test_config
            self.automation.controller = controller
            
            self.testProgress.emit("Starting video recording...", 30)
            if not self.automation.start_video_recording():
                self.testError.emit("Failed to start video recording")
                return
            
            self.testProgress.emit("Connecting to NAO robot...", 40)
            if not controller.connect_to_nao():
                self.testError.emit("Failed to connect to NAO robot")
                return
            
            self.testProgress.emit("Starting monitoring systems...", 50)
            controller.start_monitoring()
            self.stability_timer.start()
            self.gait_timer.start()
            
            self.testProgress.emit("Moving to starting position...", 60)
            controller.posture_proxy.goToPosture("StandInit", 1.0)
            time.sleep(2)
            
            self.testProgress.emit("Starting outbound journey...", 70)
            
            # Execute walking test with progress updates
            target_distance = self.test_config['route_distance']
            
            # Outbound journey
            while controller.current_position < target_distance and controller.walking_direction == 1:
                progress = 70 + int((controller.current_position / target_distance) * 10)
                self.testProgress.emit(f"Walking outbound: {controller.current_position:.1f}m / {target_distance:.1f}m", progress)
                
                # Check for obstacles
                for obstacle in self.test_config['obstacles']:
                    if abs(controller.current_position - obstacle['position']) < 0.1:
                        self.obstacleUpdate.emit(f"Obstacle ahead: {obstacle['type']}")
                
                if not controller.execute_walk_step(0.1):
                    break
                
                time.sleep(0.5)  # Allow GUI updates
            
            # Turnaround
            if controller.current_position >= target_distance - 0.1:
                self.testProgress.emit("Executing turnaround maneuver...", 80)
                controller.execute_turnaround()
            
            # Return journey
            self.testProgress.emit("Starting return journey...", 85)
            while controller.current_position > 0 and controller.walking_direction == -1:
                progress = 85 + int(((target_distance - controller.current_position) / target_distance) * 10)
                self.testProgress.emit(f"Walking return: {controller.current_position:.1f}m to go", progress)
                
                if not controller.execute_walk_step(-0.1):
                    break
                
                time.sleep(0.5)
            
            self.testProgress.emit("Analyzing performance...", 95)
            time.sleep(1)
            
            # Analyze results
            analysis = controller.analyze_performance()
            results = controller.save_test_results()
            
            self.testProgress.emit("Finalizing test...", 100)
            
            # Stop monitoring
            self.stability_timer.stop()
            self.gait_timer.stop()
            controller.stop_monitoring()
            self.automation.stop_video_recording()
            
            # Generate report
            report_file = self.automation._generate_comprehensive_report(results, analysis)
            
            # Prepare results for GUI
            gui_results = {
                'route_completed': analysis['route_completed'],
                'total_distance': analysis['total_distance'],
                'target_distance': analysis['target_distance'],
                'avg_stability': analysis['avg_stability'],
                'stability_passed': analysis['stability_passed'],
                'gait_consistency': analysis['gait_consistency'],
                'gait_passed': analysis['gait_passed'],
                'critical_errors': analysis['critical_errors'],
                'overall_passed': analysis['overall_passed'],
                'video_files': video_files,
                'report_file': report_file,
                'analysis': analysis,
                'files': results
            }
            
            self.testCompleted.emit("completed", gui_results)
            
        except Exception as e:
            self.testError.emit(f"Walking test failed: {str(e)}")
        
        finally:
            self.test_running = False
            self.stability_timer.stop()
            self.gait_timer.stop()
    
    def _emit_stability_update(self):
        """Emit stability updates to GUI"""
        if self.automation and hasattr(self.automation, 'controller') and self.automation.controller:
            try:
                if self.automation.controller.stability_monitor.stability_history:
                    latest_stability = self.automation.controller.stability_monitor.stability_history[-1]
                    stability_data = {
                        'score': latest_stability['stability_score'],
                        'acc_magnitude': latest_stability['acc_magnitude'],
                        'gyr_magnitude': latest_stability['gyr_magnitude'],
                        'timestamp': latest_stability['timestamp']
                    }
                    self.stabilityUpdate.emit(stability_data)
            except:
                pass
    
    def _emit_gait_update(self):
        """Emit gait updates to GUI"""
        if self.automation and hasattr(self.automation, 'controller') and self.automation.controller:
            try:
                if self.automation.controller.gait_analyzer.step_history:
                    latest_gait = self.automation.controller.gait_analyzer.step_history[-1]
                    gait_data = {
                        'foot_contact_quality': latest_gait['foot_contact_quality'],
                        'joint_consistency': latest_gait['joint_consistency'],
                        'overall_quality': latest_gait['overall_quality'],
                        'timestamp': latest_gait['timestamp']
                    }
                    self.gaitUpdate.emit(gait_data)
            except:
                pass
    
    @Slot()
    def stopWalkingTest(self):
        """Stop the currently running test"""
        if self.test_running:
            self.test_running = False
            self.stability_timer.stop()
            self.gait_timer.stop()
            
            if self.automation:
                self.automation.stop_video_recording()
            
            self.testCompleted.emit("stopped", {"message": "Test stopped by user"})
    
    @Slot()
    def getRobotStatus(self):
        """Get current robot status"""
        try:
            from naoqi import ALProxy
            motion_proxy = ALProxy("ALMotion", self.nao_ip, self.port)
            memory_proxy = ALProxy("ALMemory", self.nao_ip, self.port)
            
            # Get basic status
            battery_level = memory_proxy.getData("Device/SubDeviceList/Battery/Charge/Sensor/Value")
            
            # Get position estimate
            position = memory_proxy.getData("Robot/Position")
            
            # Get stability estimate
            acc_x = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
            acc_y = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
            acc_z = memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
            acc_magnitude = (acc_x**2 + acc_y**2 + acc_z**2)**0.5
            
            return {
                'connected': True,
                'battery_level': battery_level,
                'position': position,
                'stability_estimate': acc_magnitude,
                'ready': battery_level > 20 and acc_magnitude < 15
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'ready': False
            }
    
    @Slot(str)
    def configureTest(self, config_json):
        """Configure test parameters from GUI"""
        try:
            import json
            config = json.loads(config_json)
            
            # Update test configuration
            self.test_config.update(config)
            
            # Update automation configuration
            self.automation.test_config.update(config)
            
            print(f"Test configuration updated: {config}")
            return True
            
        except Exception as e:
            print(f"Failed to configure test: {e}")
            return False
    
    @Slot(str)
    def getTestConfiguration(self):
        """Get current test configuration"""
        import json
        return json.dumps(self.test_config, indent=2)

# Integration with main GUI
class WalkingTestIntegration:
    """Integration class for adding walking test to main GUI"""
    
    @staticmethod
    def add_to_gui(gui_app):
        """Add walking test functionality to existing GUI"""
        
        # Create walking test instance
        walking_test = WalkingTestGUI()
        
        # Expose to QML
        if hasattr(gui_app, 'engine'):
            gui_app.engine.rootContext().setContextProperty("walkingTest", walking_test)
        
        return walking_test

# Example usage in main GUI
def example_integration():
    """Example of how to integrate with main GUI"""
    
    # In your main GUI initialization (GUI5.py):
    
    # Import the integration
    from NAO6.walking_test.walking_gui_integration import WalkingTestIntegration
    
    # After creating the engine:
    # walking_test = WalkingTestIntegration.add_to_gui(self)
    
    # In QML, you can then call:
    # walkingTest.startWalkingTest()
    # walkingTest.stopWalkingTest()
    # walkingTest.getRobotStatus()
    
    pass

if __name__ == "__main__":
    # Test the GUI integration
    app = WalkingTestGUI()
    
    print("NAO Walking Test GUI Integration Ready")
    print("Available methods:")
    print("- startWalkingTest()")
    print("- stopWalkingTest()") 
    print("- getRobotStatus()")
    print("- configureTest(config_json)")
    print("- getTestConfiguration()")
    
    print("\nSignals:")
    print("- testStarted(message)")
    print("- testProgress(message, percentage)")
    print("- testCompleted(status, results)")
    print("- testError(error)")
    print("- stabilityUpdate(stability_data)")
    print("- gaitUpdate(gait_data)")
    print("- obstacleUpdate(warning)")
