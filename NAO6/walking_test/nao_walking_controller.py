#!/usr/bin/env python2.7

"""
NAO Robot Walking and Navigation Test Controller
Implements comprehensive walking test with obstacle avoidance, gait analysis, and stability monitoring
"""

from naoqi import ALProxy
import time
import json
import math
import threading
from datetime import datetime
from collections import deque

class NAOWalkingController:
    def __init__(self, nao_ip="192.168.23.53", port=9559):
        self.nao_ip = nao_ip
        self.port = port
        
        # Initialize NAO proxies
        self.motion_proxy = None
        self.posture_proxy = None
        self.memory_proxy = None
        self.navigation_proxy = None
        self.vision_proxy = None
        
        # Walking parameters
        self.walking_config = {
            'step_length': 0.04,      # 4cm steps
            'step_height': 0.020,     # 2cm step height
            'torso_height': 0.385,    # 38.5cm torso height
            'max_step_duration': 0.4, # Max time per step
            'stability_threshold': 0.7 # Stability confidence threshold
        }
        
        # Test data storage
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'route_completed': False,
            'gait_data': [],
            'stability_data': [],
            'obstacle_data': [],
            'errors': [],
            'status': 'not_started'
        }
        
        # Route configuration
        self.route_config = {
            'total_distance': 5.0,    # 5 meters
            'obstacles': [
                {'position': 2.0, 'type': 'door_sill', 'height': 0.02},
                {'position': 3.5, 'type': 'cord', 'height': 0.01}
            ],
            'turnaround_point': 5.0,
            'return_distance': 5.0
        }
        
        # Gait analysis
        self.gait_analyzer = GaitAnalyzer()
        self.stability_monitor = StabilityMonitor()
        self.obstacle_detector = ObstacleDetector()
        
        # Current state
        self.current_position = 0.0
        self.walking_direction = 1  # 1 for forward, -1 for backward
        self.phase = 'outbound'  # 'outbound' or 'return'
        
        # Monitoring threads
        self.monitoring_active = False
        self.gait_thread = None
        self.stability_thread = None
        
    def connect_to_nao(self):
        """Establish connection to NAO robot"""
        try:
            self.motion_proxy = ALProxy("ALMotion", self.nao_ip, self.port)
            self.posture_proxy = ALProxy("ALRobotPosture", self.nao_ip, self.port)
            self.memory_proxy = ALProxy("ALMemory", self.nao_ip, self.port)
            self.navigation_proxy = ALProxy("ALNavigation", self.nao_ip, self.port)
            self.vision_proxy = ALProxy("ALVideoDevice", self.nao_ip, self.port)
            
            # Wake up robot
            self.motion_proxy.wakeUp()
            time.sleep(1)
            
            # Initialize walking
            self._configure_walking()
            
            print("Successfully connected to NAO robot")
            return True
            
        except Exception as e:
            print(f"Failed to connect to NAO: {e}")
            return False
    
    def _configure_walking(self):
        """Configure walking parameters for stability"""
        try:
            # Set walking configuration
            self.motion_proxy.setMotionConfig([
                ["ENABLE_STIFFNESS", True],
                ["ENABLE_FOOT_CONTACT_PROTECTION", True],
                ["ENABLE_BALANCING", True],
                ["ENABLE_ARMS_BALANCING", True]
            ])
            
            # Set walking parameters
            self.motion_proxy.setWalkConfig([
                ["MAX_STEP_X", self.walking_config['step_length']],
                ["MAX_STEP_Y", 0.0],
                ["MAX_STEP_THETA", 0.1],
                ["STEP_HEIGHT", self.walking_config['step_height']],
                ["TORSO_HEIGHT", self.walking_config['torso_height']]
            ])
            
            print("Walking configuration completed")
            
        except Exception as e:
            print(f"Error configuring walking: {e}")
    
    def start_monitoring(self):
        """Start gait and stability monitoring"""
        self.monitoring_active = True
        
        # Start gait analysis thread
        self.gait_thread = threading.Thread(target=self._monitor_gait)
        self.gait_thread.daemon = True
        self.gait_thread.start()
        
        # Start stability monitoring thread
        self.stability_thread = threading.Thread(target=self._monitor_stability)
        self.stability_thread.daemon = True
        self.stability_thread.start()
        
        print("Monitoring systems started")
    
    def _monitor_gait(self):
        """Monitor gait patterns and foot placement"""
        while self.monitoring_active:
            try:
                # Get foot contact and position data
                left_foot_contact = self.memory_proxy.getData("Device/SubDeviceList/LFoot/Contact/FrontLeft")
                right_foot_contact = self.memory_proxy.getData("Device/SubDeviceList/RFoot/Contact/FrontLeft")
                
                # Get joint angles for gait analysis
                left_hip_pitch = self.memory_proxy.getData("Device/SubDeviceList/LHipPitch/Position/Sensor/Value")
                right_hip_pitch = self.memory_proxy.getData("Device/SubDeviceList/RHipPitch/Position/Sensor/Value")
                left_knee_pitch = self.memory_proxy.getData("Device/SubDeviceList/LKneePitch/Position/Sensor/Value")
                right_knee_pitch = self.memory_proxy.getData("Device/SubDeviceList/RKneePitch/Position/Sensor/Value")
                
                # Get position estimate
                position = self.memory_proxy.getData("Robot/Position")
                
                gait_data = {
                    'timestamp': time.time(),
                    'left_foot_contact': left_foot_contact,
                    'right_foot_contact': right_foot_contact,
                    'left_hip_pitch': left_hip_pitch,
                    'right_hip_pitch': right_hip_pitch,
                    'left_knee_pitch': left_knee_pitch,
                    'right_knee_pitch': right_knee_pitch,
                    'position': position
                }
                
                self.test_results['gait_data'].append(gait_data)
                self.gait_analyzer.analyze_step(gait_data)
                
                time.sleep(0.1)  # 10 Hz sampling
                
            except Exception as e:
                print(f"Error monitoring gait: {e}")
                time.sleep(0.1)
    
    def _monitor_stability(self):
        """Monitor robot stability and balance"""
        while self.monitoring_active:
            try:
                # Get accelerometer and gyroscope data
                acc_x = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
                acc_y = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
                acc_z = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
                
                gyr_x = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value")
                gyr_y = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value")
                gyr_z = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeZ/Sensor/Value")
                
                # Get center of mass estimate
                com_x = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
                com_y = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
                
                stability_data = {
                    'timestamp': time.time(),
                    'accelerometer': {'x': acc_x, 'y': acc_y, 'z': acc_z},
                    'gyroscope': {'x': gyr_x, 'y': gyr_y, 'z': gyr_z},
                    'center_of_mass': {'x': com_x, 'y': com_y}
                }
                
                self.test_results['stability_data'].append(stability_data)
                stability_score = self.stability_monitor.analyze_stability(stability_data)
                
                # Check for instability
                if stability_score < self.walking_config['stability_threshold']:
                    self.test_results['errors'].append({
                        'timestamp': time.time(),
                        'type': 'INSTABILITY',
                        'message': f'Low stability detected: {stability_score:.2f}',
                        'score': stability_score
                    })
                
                time.sleep(0.05)  # 20 Hz sampling
                
            except Exception as e:
                print(f"Error monitoring stability: {e}")
                time.sleep(0.05)
    
    def detect_obstacles(self):
        """Detect obstacles in the walking path"""
        try:
            # Get camera image for obstacle detection
            image_client = self.vision_proxy.subscribe("python_client", 0, 0, 10)
            image = self.vision_proxy.getImageRemote(image_client)
            self.vision_proxy.unsubscribe(image_client)
            
            # Process image for obstacle detection
            obstacles = self.obstacle_detector.detect_obstacles(image)
            
            for obstacle in obstacles:
                self.test_results['obstacle_data'].append({
                    'timestamp': time.time(),
                    'obstacle': obstacle
                })
            
            return obstacles
            
        except Exception as e:
            print(f"Error detecting obstacles: {e}")
            return []
    
    def execute_walk_step(self, distance):
        """Execute a single walking step"""
        try:
            # Check for obstacles
            obstacles = self.detect_obstacles()
            
            if obstacles:
                # Handle obstacle avoidance
                for obstacle in obstacles:
                    if obstacle['distance'] < 0.5:  # Obstacle within 50cm
                        return self._handle_obstacle(obstacle)
            
            # Execute normal walking step
            self.motion_proxy.post.moveTo(distance, 0, 0)
            
            # Wait for step completion
            step_duration = abs(distance) / self.walking_config['step_length'] * 0.4
            time.sleep(step_duration)
            
            # Update position
            self.current_position += distance
            
            return True
            
        except Exception as e:
            print(f"Error executing walk step: {e}")
            self.test_results['errors'].append({
                'timestamp': time.time(),
                'type': 'WALKING_ERROR',
                'message': str(e)
            })
            return False
    
    def _handle_obstacle(self, obstacle):
        """Handle obstacle avoidance"""
        try:
            print(f"Handling obstacle: {obstacle['type']} at {obstacle['distance']:.2f}m")
            
            # Step over small obstacles
            if obstacle['height'] < 0.03:  # Less than 3cm
                # Lift foot higher
                self.motion_proxy.setWalkConfig([
                    ["STEP_HEIGHT", 0.04]  # Increase step height
                ])
                
                # Take careful step
                self.motion_proxy.post.moveTo(0.1, 0, 0)
                time.sleep(0.5)
                
                # Restore normal step height
                self.motion_proxy.setWalkConfig([
                    ["STEP_HEIGHT", self.walking_config['step_height']]
                ])
                
                self.current_position += 0.1
                return True
            
            else:
                # Go around larger obstacles
                # Side step to avoid
                self.motion_proxy.post.moveTo(0, 0.2, 0)  # Step right
                time.sleep(1)
                
                self.motion_proxy.post.moveTo(0.3, 0, 0)  # Move forward
                time.sleep(1)
                
                self.motion_proxy.post.moveTo(0, -0.2, 0)  # Step left
                time.sleep(1)
                
                self.current_position += 0.3
                return True
                
        except Exception as e:
            print(f"Error handling obstacle: {e}")
            return False
    
    def execute_turnaround(self):
        """Execute turnaround at end of route"""
        try:
            print("Executing turnaround maneuver")
            
            # Stop walking
            self.motion_proxy.stopMove()
            time.sleep(0.5)
            
            # Turn 180 degrees
            self.motion_proxy.post.moveTo(0, 0, math.pi)
            time.sleep(2)
            
            # Update direction
            self.walking_direction = -1
            self.phase = 'return'
            
            return True
            
        except Exception as e:
            print(f"Error executing turnaround: {e}")
            return False
    
    def run_walking_test(self):
        """Run the complete walking test"""
        print(f"Starting NAO Walking Test - Route: {self.route_config['total_distance']}m out and back")
        
        # Initialize test
        self.test_results['start_time'] = time.time()
        self.test_results['status'] = 'running'
        
        try:
            # Connect to NAO
            if not self.connect_to_nao():
                return False
            
            # Start monitoring
            self.start_monitoring()
            
            # Move to standing position
            self.posture_proxy.goToPosture("StandInit", 1.0)
            time.sleep(2)
            
            # Execute outbound journey
            print("Starting outbound journey...")
            while self.current_position < self.route_config['turnaround_point'] and self.walking_direction == 1:
                if not self.execute_walk_step(0.1):  # 10cm steps
                    break
                
                # Check for predefined obstacles
                for obstacle in self.route_config['obstacles']:
                    if abs(self.current_position - obstacle['position']) < 0.1:
                        print(f"Approaching predefined obstacle: {obstacle['type']}")
                        self._handle_predefined_obstacle(obstacle)
            
            # Execute turnaround
            if self.current_position >= self.route_config['turnaround_point'] - 0.1:
                if not self.execute_turnaround():
                    return False
            
            # Execute return journey
            print("Starting return journey...")
            while self.current_position > 0 and self.walking_direction == -1:
                if not self.execute_walk_step(-0.1):  # 10cm steps backward
                    break
            
            # Test completion
            self.test_results['end_time'] = time.time()
            self.test_results['route_completed'] = abs(self.current_position) < 0.1
            self.test_results['status'] = 'completed'
            
            print(f"Walking test completed: Route {'completed' if self.test_results['route_completed'] else 'not completed'}")
            print(f"Final position: {self.current_position:.2f}m")
            
            return True
            
        except Exception as e:
            print(f"Walking test failed: {e}")
            self.test_results['status'] = 'failed'
            return False
        
        finally:
            # Cleanup
            self.stop_monitoring()
            self._cleanup()
    
    def _handle_predefined_obstacle(self, obstacle):
        """Handle predefined obstacles in the route"""
        try:
            print(f"Handling predefined obstacle: {obstacle['type']}")
            
            if obstacle['type'] == 'door_sill':
                # Step over door sill
                self.motion_proxy.setWalkConfig([
                    ["STEP_HEIGHT", 0.03]  # Higher step for sill
                ])
                self.motion_proxy.post.moveTo(0.15, 0, 0)
                time.sleep(0.8)
                self.motion_proxy.setWalkConfig([
                    ["STEP_HEIGHT", self.walking_config['step_height']]
                ])
                
            elif obstacle['type'] == 'cord':
                # Careful step over cord
                self.motion_proxy.post.moveTo(0.05, 0, 0)  # Short step
                time.sleep(0.3)
                self.motion_proxy.post.moveTo(0.05, 0, 0)  # Another short step
                time.sleep(0.3)
            
            self.current_position += 0.1
            
        except Exception as e:
            print(f"Error handling predefined obstacle: {e}")
    
    def analyze_performance(self):
        """Analyze walking performance against acceptance criteria"""
        # Route completion analysis
        route_completed = self.test_results['route_completed']
        
        # Stability analysis
        stability_scores = [self.stability_monitor.get_stability_score() for data in self.test_results['stability_data']]
        avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0
        min_stability = min(stability_scores) if stability_scores else 0
        
        # Gait consistency analysis
        gait_consistency = self.gait_analyzer.get_consistency_score()
        
        # Error analysis
        critical_errors = [e for e in self.test_results['errors'] if e['type'] in ['FALL_DETECTED', 'COLLISION', 'INSTABILITY']]
        
        # Obstacle handling analysis
        obstacles_handled = len([o for o in self.test_results['obstacle_data'] if 'handled' in o.get('obstacle', {})])
        
        analysis = {
            'route_completed': route_completed,
            'total_distance': abs(self.current_position),
            'target_distance': self.route_config['total_distance'] * 2,  # Out and back
            'avg_stability': avg_stability,
            'min_stability': min_stability,
            'stability_passed': avg_stability > self.walking_config['stability_threshold'],
            'gait_consistency': gait_consistency,
            'gait_passed': gait_consistency > 0.8,  # 80% consistency threshold
            'critical_errors': critical_errors,
            'obstacles_handled': obstacles_handled,
            'overall_passed': (
                route_completed and
                avg_stability > self.walking_config['stability_threshold'] and
                gait_consistency > 0.8 and
                len(critical_errors) == 0
            )
        }
        
        return analysis
    
    def save_test_results(self, filename_prefix="nao_walking_test"):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save gait data as CSV
        gait_filename = f"{filename_prefix}_gait_{timestamp}.csv"
        self._save_gait_data(gait_filename)
        
        # Save stability data as CSV
        stability_filename = f"{filename_prefix}_stability_{timestamp}.csv"
        self._save_stability_data(stability_filename)
        
        # Save performance analysis
        analysis = self.analyze_performance()
        analysis_filename = f"{filename_prefix}_analysis_{timestamp}.json"
        with open(analysis_filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Walking test results saved:")
        print(f"  - Detailed results: {json_filename}")
        print(f"  - Gait data: {gait_filename}")
        print(f"  - Stability data: {stability_filename}")
        print(f"  - Performance analysis: {analysis_filename}")
        
        return {
            'json_file': json_filename,
            'gait_file': gait_filename,
            'stability_file': stability_filename,
            'analysis_file': analysis_filename
        }
    
    def _save_gait_data(self, filename):
        """Save gait data to CSV"""
        import csv
        with open(filename, 'w') as f:
            if self.test_results['gait_data']:
                writer = csv.writer(f)
                headers = ['timestamp', 'left_foot_contact', 'right_foot_contact', 
                          'left_hip_pitch', 'right_hip_pitch', 'left_knee_pitch', 'right_knee_pitch']
                writer.writerow(headers)
                
                for data in self.test_results['gait_data']:
                    row = [data['timestamp'], data['left_foot_contact'], data['right_foot_contact'],
                           data['left_hip_pitch'], data['right_hip_pitch'], 
                           data['left_knee_pitch'], data['right_knee_pitch']]
                    writer.writerow(row)
    
    def _save_stability_data(self, filename):
        """Save stability data to CSV"""
        import csv
        with open(filename, 'w') as f:
            if self.test_results['stability_data']:
                writer = csv.writer(f)
                headers = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'com_x', 'com_y']
                writer.writerow(headers)
                
                for data in self.test_results['stability_data']:
                    row = [data['timestamp'],
                           data['accelerometer']['x'], data['accelerometer']['y'], data['accelerometer']['z'],
                           data['gyroscope']['x'], data['gyroscope']['y'], data['gyroscope']['z'],
                           data['center_of_mass']['x'], data['center_of_mass']['y']]
                    writer.writerow(row)
    
    def stop_monitoring(self):
        """Stop monitoring threads"""
        self.monitoring_active = False
        
        if self.gait_thread:
            self.gait_thread.join(timeout=2)
        
        if self.stability_thread:
            self.stability_thread.join(timeout=2)
        
        print("Monitoring stopped")
    
    def _cleanup(self):
        """Clean up and reset robot"""
        try:
            if self.motion_proxy:
                # Stop walking
                self.motion_proxy.stopMove()
                time.sleep(0.5)
                
                # Move to safe position
                self.posture_proxy.goToPosture("StandInit", 0.5)
                time.sleep(1)
                
                # Rest robot
                self.motion_proxy.rest()
                print("Robot cleaned up and rested")
        except Exception as e:
            print(f"Error during cleanup: {e}")


class GaitAnalyzer:
    """Analyzes gait patterns and consistency"""
    
    def __init__(self):
        self.step_history = deque(maxlen=100)
        self.consistency_threshold = 0.8
    
    def analyze_step(self, gait_data):
        """Analyze individual step data"""
        # Calculate step quality metrics
        foot_contact_quality = self._analyze_foot_contact(gait_data)
        joint_angle_consistency = self._analyze_joint_angles(gait_data)
        
        step_quality = {
            'timestamp': gait_data['timestamp'],
            'foot_contact_quality': foot_contact_quality,
            'joint_consistency': joint_angle_consistency,
            'overall_quality': (foot_contact_quality + joint_angle_consistency) / 2
        }
        
        self.step_history.append(step_quality)
        return step_quality
    
    def _analyze_foot_contact(self, gait_data):
        """Analyze foot contact patterns"""
        left_contact = gait_data['left_foot_contact']
        right_contact = gait_data['right_foot_contact']
        
        # Ideal pattern: alternating foot contact
        if (left_contact > 0.5 and right_contact < 0.5) or (right_contact > 0.5 and left_contact < 0.5):
            return 1.0
        elif left_contact > 0.5 and right_contact > 0.5:
            return 0.7  # Both feet on ground (stable but inefficient)
        else:
            return 0.3  # Poor foot contact
    
    def _analyze_joint_angles(self, gait_data):
        """Analyze joint angle consistency"""
        # Check for reasonable joint angles
        left_hip = gait_data['left_hip_pitch']
        right_hip = gait_data['right_hip_pitch']
        left_knee = gait_data['left_knee_pitch']
        right_knee = gait_data['right_knee_pitch']
        
        # Basic consistency check
        if all(-1.5 < angle < 1.5 for angle in [left_hip, right_hip, left_knee, right_knee]):
            return 0.9
        else:
            return 0.5
    
    def get_consistency_score(self):
        """Calculate overall gait consistency"""
        if not self.step_history:
            return 0.0
        
        recent_steps = list(self.step_history)[-20:]  # Last 20 steps
        avg_quality = sum(step['overall_quality'] for step in recent_steps) / len(recent_steps)
        
        return avg_quality


class StabilityMonitor:
    """Monitors robot stability and balance"""
    
    def __init__(self):
        self.stability_history = deque(maxlen=200)
        self.instability_threshold = 0.7
    
    def analyze_stability(self, stability_data):
        """Analyze stability from sensor data"""
        # Calculate stability metrics
        acc_magnitude = math.sqrt(
            stability_data['accelerometer']['x']**2 +
            stability_data['accelerometer']['y']**2 +
            stability_data['accelerometer']['z']**2
        )
        
        gyr_magnitude = math.sqrt(
            stability_data['gyroscope']['x']**2 +
            stability_data['gyroscope']['y']**2 +
            stability_data['gyroscope']['z']**2
        )
        
        # Calculate stability score (0-1, higher is better)
        stability_score = self._calculate_stability_score(acc_magnitude, gyr_magnitude, stability_data['center_of_mass'])
        
        self.stability_history.append({
            'timestamp': stability_data['timestamp'],
            'stability_score': stability_score,
            'acc_magnitude': acc_magnitude,
            'gyr_magnitude': gyr_magnitude
        })
        
        return stability_score
    
    def _calculate_stability_score(self, acc_mag, gyr_mag, com):
        """Calculate overall stability score"""
        # Accelerometer stability (should be close to 9.8 m/s² when upright)
        acc_stability = 1.0 - min(abs(acc_mag - 9.8) / 5.0, 1.0)
        
        # Gyroscope stability (should be low when stable)
        gyr_stability = 1.0 - min(gyr_mag / 2.0, 1.0)
        
        # Center of mass stability (should be near zero)
        com_stability = 1.0 - min((abs(com['x']) + abs(com['y'])) / 0.5, 1.0)
        
        # Weighted average
        overall_stability = (acc_stability * 0.4 + gyr_stability * 0.3 + com_stability * 0.3)
        
        return overall_stability
    
    def get_stability_score(self):
        """Get current stability score"""
        if not self.stability_history:
            return 0.0
        
        return self.stability_history[-1]['stability_score']


class ObstacleDetector:
    """Detects and classifies obstacles"""
    
    def __init__(self):
        self.obstacle_types = {
            'door_sill': {'height_range': (0.015, 0.025), 'color_range': (100, 200)},
            'cord': {'height_range': (0.005, 0.015), 'color_range': (50, 150)},
            'unknown': {'height_range': (0, 0.1), 'color_range': (0, 255)}
        }
    
    def detect_obstacles(self, image_data):
        """Detect obstacles from camera image"""
        # Simplified obstacle detection
        # In a real implementation, this would use computer vision
        
        obstacles = []
        
        # Mock obstacle detection for demonstration
        # In reality, this would process the image_data
        mock_obstacles = [
            {'type': 'door_sill', 'distance': 2.1, 'height': 0.02, 'handled': False},
            {'type': 'cord', 'distance': 3.6, 'height': 0.01, 'handled': False}
        ]
        
        for obstacle in mock_obstacles:
            obstacles.append(obstacle)
        
        return obstacles


def main():
    """Main function to run the walking test"""
    controller = NAOWalkingController()
    
    try:
        # Run the test
        success = controller.run_walking_test()
        
        if success:
            # Analyze performance
            analysis = controller.analyze_performance()
            print("\n=== WALKING TEST ANALYSIS ===")
            print(f"Route Completed: {'YES' if analysis['route_completed'] else 'NO'}")
            print(f"Total Distance: {analysis['total_distance']:.2f}m")
            print(f"Target Distance: {analysis['target_distance']:.2f}m")
            print(f"Average Stability: {analysis['avg_stability']:.2f}")
            print(f"Stability Test Passed: {'YES' if analysis['stability_passed'] else 'NO'}")
            print(f"Gait Consistency: {analysis['gait_consistency']:.2f}")
            print(f"Gait Test Passed: {'YES' if analysis['gait_passed'] else 'NO'}")
            print(f"Critical Errors: {len(analysis['critical_errors'])}")
            print(f"Overall Test Passed: {'YES' if analysis['overall_passed'] else 'NO'}")
            
            # Save results
            controller.save_test_results()
        else:
            print("Walking test failed to complete")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        controller.stop_monitoring()
        controller._cleanup()
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        controller._cleanup()

if __name__ == "__main__":
    main()
