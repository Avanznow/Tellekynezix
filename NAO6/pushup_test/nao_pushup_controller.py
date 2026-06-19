#!/usr/bin/env python2.7

"""
NAO Robot Push-Up Endurance Test
Implements comprehensive push-up testing with temperature monitoring and performance analysis
"""

from naoqi import ALProxy
import time
import json
import csv
import os
from datetime import datetime
import threading
import math

class NAOPushupController:
    def __init__(self, nao_ip="192.168.23.53", port=9559):
        self.nao_ip = nao_ip
        self.port = port
        
        # Initialize NAO proxies
        self.motion_proxy = None
        self.posture_proxy = None
        self.memory_proxy = None
        self.temperature_proxy = None
        
        # Test data storage
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'repetitions': [],
            'temperature_data': [],
            'errors': [],
            'status': 'not_started'
        }
        
        # Performance tracking
        self.rep_times = []
        self.current_rep = 0
        self.target_reps = 15  # Aim for 15-20 reps
        
        # Temperature monitoring
        self.monitoring_active = False
        self.temperature_thread = None
        
        # Motor joints to monitor
        self.motor_joints = [
            "HeadYaw", "HeadPitch",
            "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
            "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
            "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
            "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"
        ]
        
    def connect_to_nao(self):
        """Establish connection to NAO robot"""
        try:
            self.motion_proxy = ALProxy("ALMotion", self.nao_ip, self.port)
            self.posture_proxy = ALProxy("ALRobotPosture", self.nao_ip, self.port)
            self.memory_proxy = ALProxy("ALMemory", self.nao_ip, self.port)
            self.temperature_proxy = ALProxy("ALTemperature", self.nao_ip, self.port)
            
            # Wake up robot
            self.motion_proxy.wakeUp()
            time.sleep(1)
            
            print("Successfully connected to NAO robot")
            return True
            
        except Exception as e:
            print(f"Failed to connect to NAO: {e}")
            return False
    
    def start_temperature_monitoring(self):
        """Start monitoring motor temperatures in background thread"""
        self.monitoring_active = True
        self.temperature_thread = threading.Thread(target=self._monitor_temperatures)
        self.temperature_thread.daemon = True
        self.temperature_thread.start()
        print("Temperature monitoring started")
    
    def _monitor_temperatures(self):
        """Background thread to monitor motor temperatures"""
        while self.monitoring_active:
            try:
                timestamp = time.time()
                temp_reading = {'timestamp': timestamp, 'temperatures': {}}
                
                for joint in self.motor_joints:
                    try:
                        temp = self.temperature_proxy.getTemperature(joint)
                        temp_reading['temperatures'][joint] = temp
                    except:
                        temp_reading['temperatures'][joint] = None
                
                self.test_results['temperature_data'].append(temp_reading)
                
                # Check for overheating (threshold: 70°C)
                max_temp = max([t for t in temp_reading['temperatures'].values() if t is not None])
                if max_temp > 70:
                    self.test_results['errors'].append({
                        'timestamp': timestamp,
                        'type': 'OVERHEATING',
                        'message': f'Motor temperature exceeded 70°C: {max_temp}°C',
                        'joint': joint
                    })
                    print(f"WARNING: Motor overheating detected! Max temp: {max_temp}°C")
                
                time.sleep(1)  # Sample every second
                
            except Exception as e:
                print(f"Error monitoring temperature: {e}")
                time.sleep(1)
    
    def stop_temperature_monitoring(self):
        """Stop temperature monitoring"""
        self.monitoring_active = False
        if self.temperature_thread:
            self.temperature_thread.join(timeout=2)
        print("Temperature monitoring stopped")
    
    def get_initial_position(self):
        """Move NAO to initial push-up position (on all fours)"""
        try:
            print("Moving to initial push-up position...")
            
            # Stiffness on
            self.motion_proxy.stiffnessInterpolation("Body", 1.0, 1.0)
            
            # Move to push-up ready position
            angles = [
                # Head
                ["HeadYaw", 0.0],
                ["HeadPitch", 0.0],
                
                # Left arm (support position)
                ["LShoulderPitch", 90.0 * math.pi / 180],  # Forward
                ["LShoulderRoll", 45.0 * math.pi / 180],   # Outward
                ["LElbowYaw", 0.0],
                ["LElbowRoll", -90.0 * math.pi / 180],   # Bent
                ["LWristYaw", 0.0],
                ["LHand", 0.0],
                
                # Right arm (support position)
                ["RShoulderPitch", 90.0 * math.pi / 180],  # Forward
                ["RShoulderRoll", -45.0 * math.pi / 180],  # Outward
                ["RElbowYaw", 0.0],
                ["RElbowRoll", -90.0 * math.pi / 180],    # Bent
                ["RWristYaw", 0.0],
                ["RHand", 0.0],
                
                # Legs (support position)
                ["LHipYawPitch", 0.0],
                ["LHipRoll", 0.0],
                ["LHipPitch", -45.0 * math.pi / 180],     # Bent
                ["LKneePitch", 90.0 * math.pi / 180],     # Bent
                ["LAnklePitch", -45.0 * math.pi / 180],
                ["LAnkleRoll", 0.0],
                
                ["RHipYawPitch", 0.0],
                ["RHipRoll", 0.0],
                ["RHipPitch", -45.0 * math.pi / 180],     # Bent
                ["RKneePitch", 90.0 * math.pi / 180],     # Bent
                ["RAnklePitch", -45.0 * math.pi / 180],
                ["RAnkleRoll", 0.0]
            ]
            
            # Apply angles with smooth interpolation
            names = [angle[0] for angle in angles]
            angle_lists = [angle[1] for angle in angles]
            times = [1.5] * len(angles)  # 1.5 seconds for each joint
            
            self.motion_proxy.angleInterpolation(names, angle_lists, times, True)
            time.sleep(2)  # Wait for movement to complete
            
            print("Initial position achieved")
            return True
            
        except Exception as e:
            print(f"Error moving to initial position: {e}")
            return False
    
    def perform_pushup(self, rep_number):
        """Execute a single push-up repetition"""
        try:
            rep_start_time = time.time()
            
            print(f"Starting push-up repetition #{rep_number}")
            
            # Phase 1: Lower body (down position)
            down_angles = [
                ["LShoulderPitch", 120.0 * math.pi / 180],   # More forward
                ["RShoulderPitch", 120.0 * math.pi / 180],   # More forward
                ["LElbowRoll", -120.0 * math.pi / 180],      # More bent
                ["RElbowRoll", -120.0 * math.pi / 180],      # More bent
                ["LHipPitch", -60.0 * math.pi / 180],        # More bent
                ["RHipPitch", -60.0 * math.pi / 180],        # More bent
            ]
            
            names = [angle[0] for angle in down_angles]
            angle_lists = [angle[1] for angle in down_angles]
            times = [1.0] * len(down_angles)
            
            self.motion_proxy.angleInterpolation(names, angle_lists, times, True)
            time.sleep(1.5)  # Hold down position
            
            # Phase 2: Raise body (up position)
            up_angles = [
                ["LShoulderPitch", 90.0 * math.pi / 180],    # Back to start
                ["RShoulderPitch", 90.0 * math.pi / 180],    # Back to start
                ["LElbowRoll", -90.0 * math.pi / 180],       # Back to start
                ["RElbowRoll", -90.0 * math.pi / 180],       # Back to start
                ["LHipPitch", -45.0 * math.pi / 180],        # Back to start
                ["RHipPitch", -45.0 * math.pi / 180],        # Back to start
            ]
            
            names = [angle[0] for angle in up_angles]
            angle_lists = [angle[1] for angle in up_angles]
            times = [1.0] * len(up_angles)
            
            self.motion_proxy.angleInterpolation(names, angle_lists, times, True)
            time.sleep(1.0)  # Hold up position
            
            rep_end_time = time.time()
            rep_duration = rep_end_time - rep_start_time
            
            # Store repetition data
            rep_data = {
                'rep_number': rep_number,
                'start_time': rep_start_time,
                'end_time': rep_end_time,
                'duration': rep_duration,
                'status': 'completed'
            }
            
            self.test_results['repetitions'].append(rep_data)
            self.rep_times.append(rep_duration)
            
            print(f"Push-up #{rep_number} completed in {rep_duration:.2f} seconds")
            return True
            
        except Exception as e:
            error_msg = f"Error in push-up #{rep_number}: {e}"
            print(error_msg)
            self.test_results['errors'].append({
                'timestamp': time.time(),
                'type': 'MOVEMENT_ERROR',
                'message': error_msg,
                'rep_number': rep_number
            })
            return False
    
    def run_endurance_test(self, target_reps=15):
        """Run the complete push-up endurance test"""
        print(f"Starting NAO Push-Up Endurance Test - Target: {target_reps} reps")
        
        # Initialize test
        self.test_results['start_time'] = time.time()
        self.test_results['status'] = 'running'
        self.target_reps = target_reps
        
        try:
            # Connect to NAO
            if not self.connect_to_nao():
                return False
            
            # Start temperature monitoring
            self.start_temperature_monitoring()
            
            # Get initial position
            if not self.get_initial_position():
                return False
            
            # Perform push-ups
            successful_reps = 0
            for rep in range(1, target_reps + 1):
                self.current_rep = rep
                
                if self.perform_pushup(rep):
                    successful_reps += 1
                else:
                    print(f"Failed to complete push-up #{rep}")
                    break
                
                # Check for critical errors
                critical_errors = [e for e in self.test_results['errors'] if e['type'] in ['OVERHEATING', 'FALL_DETECTED']]
                if critical_errors:
                    print("Critical error detected - stopping test")
                    break
            
            # Test completion
            self.test_results['end_time'] = time.time()
            self.test_results['status'] = 'completed'
            self.test_results['successful_reps'] = successful_reps
            
            print(f"Test completed: {successful_reps}/{target_reps} push-ups successful")
            
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            self.test_results['status'] = 'failed'
            return False
        
        finally:
            # Cleanup
            self.stop_temperature_monitoring()
            self._cleanup()
    
    def analyze_performance(self):
        """Analyze test performance against acceptance criteria"""
        if len(self.rep_times) < 10:
            return {"status": "insufficient_data", "message": "Need at least 10 repetitions"}
        
        # Calculate average times for first 3 and last 3 reps
        first_three_avg = sum(self.rep_times[:3]) / 3
        last_three_avg = sum(self.rep_times[-3:]) / 3
        
        # Check 15% fatigue threshold
        fatigue_threshold = 1.15  # 15% slower
        fatigue_ratio = last_three_avg / first_three_avg
        
        # Check thermal performance
        max_temp = 0
        for temp_data in self.test_results['temperature_data']:
            for joint, temp in temp_data['temperatures'].items():
                if temp and temp > max_temp:
                    max_temp = temp
        
        # Check for errors
        critical_errors = [e for e in self.test_results['errors'] if e['type'] in ['OVERHEATING', 'FALL_DETECTED']]
        
        analysis = {
            'total_reps': len(self.rep_times),
            'first_three_avg': first_three_avg,
            'last_three_avg': last_three_avg,
            'fatigue_ratio': fatigue_ratio,
            'fatigue_passed': fatigue_ratio <= fatigue_threshold,
            'max_temperature': max_temp,
            'thermal_passed': max_temp < 70 and len(critical_errors) == 0,
            'critical_errors': critical_errors,
            'overall_passed': (
                len(self.rep_times) >= 10 and
                fatigue_ratio <= fatigue_threshold and
                max_temp < 70 and
                len(critical_errors) == 0
            )
        }
        
        return analysis
    
    def save_test_results(self, filename_prefix="nao_pushup_test"):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save temperature data as CSV
        csv_filename = f"{filename_prefix}_temperature_{timestamp}.csv"
        with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + self.motor_joints)
            
            for temp_data in self.test_results['temperature_data']:
                row = [temp_data['timestamp']]
                for joint in self.motor_joints:
                    row.append(temp_data['temperatures'].get(joint, ''))
                writer.writerow(row)
        
        # Save performance analysis
        analysis = self.analyze_performance()
        analysis_filename = f"{filename_prefix}_analysis_{timestamp}.json"
        with open(analysis_filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Test results saved:")
        print(f"  - Detailed results: {json_filename}")
        print(f"  - Temperature data: {csv_filename}")
        print(f"  - Performance analysis: {analysis_filename}")
        
        return {
            'json_file': json_filename,
            'csv_file': csv_filename,
            'analysis_file': analysis_filename
        }
    
    def _cleanup(self):
        """Clean up and reset robot"""
        try:
            if self.motion_proxy:
                # Move to safe position
                self.posture_proxy.goToPosture("StandInit", 0.5)
                time.sleep(1)
                
                # Rest robot
                self.motion_proxy.rest()
                print("Robot cleaned up and rested")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    """Main function to run the push-up test"""
    controller = NAOPushupController()
    
    try:
        # Run the test
        success = controller.run_endurance_test(target_reps=15)
        
        if success:
            # Analyze performance
            analysis = controller.analyze_performance()
            print("\n=== PERFORMANCE ANALYSIS ===")
            print(f"Total Repetitions: {analysis['total_reps']}")
            print(f"First 3 Avg Time: {analysis['first_three_avg']:.2f}s")
            print(f"Last 3 Avg Time: {analysis['last_three_avg']:.2f}s")
            print(f"Fatigue Ratio: {analysis['fatigue_ratio']:.2f}")
            print(f"Fatigue Test Passed: {'YES' if analysis['fatigue_passed'] else 'NO'}")
            print(f"Max Temperature: {analysis['max_temperature']:.1f}°C")
            print(f"Thermal Test Passed: {'YES' if analysis['thermal_passed'] else 'NO'}")
            print(f"Overall Test Passed: {'YES' if analysis['overall_passed'] else 'NO'}")
            
            # Save results
            controller.save_test_results()
        else:
            print("Test failed to complete")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        controller.stop_temperature_monitoring()
        controller._cleanup()
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        controller._cleanup()

if __name__ == "__main__":
    main()
