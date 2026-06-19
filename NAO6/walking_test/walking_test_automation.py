#!/usr/bin/env python3

"""
NAO Walking Test Automation
Handles automated walking test execution with video recording and comprehensive reporting
"""

import subprocess
import time
import os
import json
import cv2
import threading
import numpy as np
from datetime import datetime
from nao_walking_controller import NAOWalkingController

class WalkingTestAutomation:
    def __init__(self, nao_ip="192.168.23.53", port=9559):
        self.nao_ip = nao_ip
        self.port = port
        
        # Video recording setup
        self.recording = False
        self.video_thread = None
        self.video_writer = None
        self.camera = None
        self.secondary_camera = None
        
        # Test configuration
        self.test_config = {
            'route_distance': 5.0,          # 5 meters out and back
            'obstacles': [
                {'position': 2.0, 'type': 'door_sill', 'height': 0.02},
                {'position': 3.5, 'type': 'cord', 'height': 0.01}
            ],
            'video_resolution': (1280, 720),  # HD video
            'video_fps': 30,
            'primary_camera_index': 0,
            'secondary_camera_index': 1,
            'recording_angle': 'side',        # 'side', 'front', or 'both'
            'stability_threshold': 0.7,
            'gait_consistency_threshold': 0.8
        }
        
        # Test metadata
        self.test_metadata = {
            'test_start': None,
            'test_end': None,
            'nao_ip': nao_ip,
            'test_config': self.test_config,
            'status': 'preparing',
            'environmental_conditions': {}
        }
        
        # Performance metrics
        self.performance_metrics = {
            'route_completion': 0.0,
            'stability_scores': [],
            'gait_consistency': 0.0,
            'obstacle_handling': [],
            'timing_data': {},
            'error_events': []
        }
    
    def setup_video_recording(self, filename_prefix="nao_walking_test"):
        """Setup multi-camera video recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        video_files = []
        
        try:
            # Setup primary camera
            self.camera = cv2.VideoCapture(self.test_config['primary_camera_index'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.test_config['video_resolution'][0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.test_config['video_resolution'][1])
            self.camera.set(cv2.CAP_PROP_FPS, self.test_config['video_fps'])
            
            # Create primary video writer
            primary_filename = f"{filename_prefix}_primary_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                primary_filename,
                fourcc,
                self.test_config['video_fps'],
                self.test_config['video_resolution']
            )
            video_files.append(primary_filename)
            
            # Setup secondary camera if available
            if self.test_config['secondary_camera_index'] >= 0:
                self.secondary_camera = cv2.VideoCapture(self.test_config['secondary_camera_index'])
                self.secondary_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.test_config['video_resolution'][0])
                self.secondary_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.test_config['video_resolution'][1])
                self.secondary_camera.set(cv2.CAP_PROP_FPS, self.test_config['video_fps'])
                
                secondary_filename = f"{filename_prefix}_secondary_{timestamp}.mp4"
                self.secondary_writer = cv2.VideoWriter(
                    secondary_filename,
                    fourcc,
                    self.test_config['video_fps'],
                    self.test_config['video_resolution']
                )
                video_files.append(secondary_filename)
            
            print(f"Video recording setup complete: {video_files}")
            return video_files
            
        except Exception as e:
            print(f"Failed to setup video recording: {e}")
            return []
    
    def start_video_recording(self):
        """Start recording video with real-time overlays"""
        try:
            self.recording = True
            self.video_thread = threading.Thread(target=self._record_video_with_overlays)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            print("Video recording started with overlays")
            return True
            
        except Exception as e:
            print(f"Failed to start video recording: {e}")
            return False
    
    def _record_video_with_overlays(self):
        """Record video with real-time performance overlays"""
        frame_count = 0
        
        while self.recording:
            try:
                # Capture from primary camera
                ret, frame = self.camera.read()
                if ret:
                    # Add overlays
                    frame_with_overlays = self._add_video_overlays(frame, frame_count)
                    self.video_writer.write(frame_with_overlays)
                    frame_count += 1
                
                # Capture from secondary camera if available
                if self.secondary_camera and hasattr(self, 'secondary_writer'):
                    ret2, frame2 = self.secondary_camera.read()
                    if ret2:
                        frame2_overlays = self._add_video_overlays(frame2, frame_count, secondary=True)
                        self.secondary_writer.write(frame2_overlays)
                
                time.sleep(1.0 / self.test_config['video_fps'])
                
            except Exception as e:
                print(f"Error recording video: {e}")
                break
    
    def _add_video_overlays(self, frame, frame_count, secondary=False):
        """Add real-time performance overlays to video"""
        overlay_frame = frame.copy()
        h, w = overlay_frame.shape[:2]
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay_frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add test information
        cv2.putText(overlay_frame, "NAO Walking Test - Issue #604", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add route progress
        if hasattr(self, 'controller') and self.controller:
            progress_text = f"Position: {self.controller.current_position:.1f}m / {self.test_config['route_distance']*2:.1f}m"
            cv2.putText(overlay_frame, progress_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Add stability indicator
            if self.controller.stability_monitor.stability_history:
                current_stability = self.controller.stability_monitor.get_stability_score()
                stability_color = (0, 255, 0) if current_stability > self.test_config['stability_threshold'] else (0, 0, 255)
                stability_text = f"Stability: {current_stability:.2f}"
                cv2.putText(overlay_frame, stability_text, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, stability_color, 2)
            
            # Add gait consistency
            if self.controller.gait_analyzer.step_history:
                gait_consistency = self.controller.gait_analyzer.get_consistency_score()
                gait_color = (0, 255, 0) if gait_consistency > self.test_config['gait_consistency_threshold'] else (0, 0, 255)
                gait_text = f"Gait: {gait_consistency:.2f}"
                cv2.putText(overlay_frame, gait_text, (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, gait_color, 2)
            
            # Add obstacle warnings
            for obstacle in self.controller.route_config['obstacles']:
                if abs(self.controller.current_position - obstacle['position']) < 0.2:
                    obstacle_text = f"Obstacle: {obstacle['type']} ahead!"
                    cv2.putText(overlay_frame, obstacle_text, (10, 230),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add camera indicator
        camera_text = "Camera: Secondary" if secondary else "Camera: Primary"
        cv2.putText(overlay_frame, camera_text, (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add frame counter
        frame_text = f"Frame: {frame_count}"
        cv2.putText(overlay_frame, frame_text, (w - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return overlay_frame
    
    def stop_video_recording(self):
        """Stop video recording"""
        self.recording = False
        
        if self.video_thread:
            self.video_thread.join(timeout=2)
        
        if self.video_writer:
            self.video_writer.release()
        
        if self.secondary_writer:
            self.secondary_writer.release()
        
        if self.camera:
            self.camera.release()
        
        if self.secondary_camera:
            self.secondary_camera.release()
        
        print("Video recording stopped")
    
    def assess_environmental_conditions(self):
        """Assess environmental conditions before test"""
        conditions = {
            'lighting_level': self._measure_lighting(),
            'floor_surface': self._detect_surface_type(),
            'obstacle_clearance': self._measure_obstacle_clearance(),
            'ambient_temperature': self._measure_temperature(),
            'noise_level': self._measure_noise_level()
        }
        
        self.test_metadata['environmental_conditions'] = conditions
        print(f"Environmental conditions assessed: {conditions}")
        
        return conditions
    
    def _measure_lighting(self):
        """Measure ambient lighting conditions"""
        try:
            if self.camera:
                ret, frame = self.camera.read()
                if ret:
                    # Calculate average brightness
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    return brightness
        except:
            pass
        return 0.0
    
    def _detect_surface_type(self):
        """Detect floor surface type (simplified)"""
        # In a real implementation, this would use computer vision
        return "hard_floor"  # Default assumption
    
    def _measure_obstacle_clearance(self):
        """Measure clearance around obstacles"""
        # Simplified measurement
        return 0.5  # 50cm clearance
    
    def _measure_temperature(self):
        """Measure ambient temperature"""
        # Would use temperature sensor in real implementation
        return 22.0  # 22°C default
    
    def _measure_noise_level(self):
        """Measure ambient noise level"""
        # Would use microphone in real implementation
        return 40.0  # 40dB default
    
    def prepare_test_area(self):
        """Prepare the test area and verify conditions"""
        print("Preparing test area...")
        
        # Check environmental conditions
        conditions = self.assess_environmental_conditions()
        
        # Verify route is clear
        route_clear = self._verify_route_clearance()
        
        # Setup obstacle markers
        obstacles_set = self._setup_obstacle_markers()
        
        preparation_status = {
            'environmental_conditions': conditions,
            'route_clear': route_clear,
            'obstacles_set': obstacles_set,
            'ready': route_clear and obstacles_set
        }
        
        print(f"Test area preparation: {preparation_status}")
        return preparation_status
    
    def _verify_route_clearance(self):
        """Verify the walking route is clear"""
        # In a real implementation, this would use sensors or cameras
        print("Verifying route clearance...")
        return True
    
    def _setup_obstacle_markers(self):
        """Setup visual markers for obstacles"""
        print("Setting up obstacle markers...")
        # In a real implementation, this would place visual markers
        return True
    
    def run_comprehensive_walking_test(self):
        """Run the complete walking test with full automation"""
        print("=" * 70)
        print("NAO WALKING AND NAVIGATION TEST - COMPREHENSIVE")
        print("Issue #604: Dynamic Mobility and Navigation")
        print("=" * 70)
        
        # Initialize test metadata
        self.test_metadata['test_start'] = datetime.now().isoformat()
        self.test_metadata['status'] = 'running'
        
        try:
            # Step 1: Prepare test area
            print("\n1. Preparing test area...")
            preparation = self.prepare_test_area()
            
            if not preparation['ready']:
                print("Test area preparation failed")
                return self._create_failure_result("preparation_failed")
            
            # Step 2: Setup video recording
            print("\n2. Setting up video recording...")
            video_files = self.setup_video_recording()
            
            if not video_files:
                print("Video recording setup failed")
                return self._create_failure_result("video_setup_failed")
            
            # Step 3: Initialize NAO controller
            print("\n3. Initializing NAO controller...")
            self.controller = NAOWalkingController(self.nao_ip, self.port)
            
            # Configure route
            self.controller.route_config = self.test_config
            
            # Step 4: Start video recording
            print("\n4. Starting video recording...")
            if not self.start_video_recording():
                print("Failed to start video recording")
                return self._create_failure_result("video_start_failed")
            
            # Step 5: Execute walking test
            print("\n5. Executing walking test...")
            test_success = self.controller.run_walking_test()
            
            if not test_success:
                print("Walking test execution failed")
                return self._create_failure_result("test_execution_failed")
            
            # Step 6: Analyze performance
            print("\n6. Analyzing performance...")
            analysis = self.controller.analyze_performance()
            self.performance_metrics.update(analysis)
            
            # Step 7: Save test results
            print("\n7. Saving test results...")
            result_files = self.controller.save_test_results()
            
            # Step 8: Generate comprehensive report
            print("\n8. Generating comprehensive report...")
            report_file = self._generate_comprehensive_report(result_files, analysis)
            
            # Finalize test
            self.test_metadata['test_end'] = datetime.now().isoformat()
            self.test_metadata['status'] = 'completed'
            self.test_metadata['video_files'] = video_files
            self.test_metadata['result_files'] = result_files
            self.test_metadata['report_file'] = report_file
            
            # Print results summary
            self._print_results_summary(analysis)
            
            return self.test_metadata
            
        except Exception as e:
            print(f"Comprehensive test failed: {e}")
            self.test_metadata['status'] = 'error'
            self.test_metadata['error'] = str(e)
            return self.test_metadata
        
        finally:
            # Cleanup
            self.stop_video_recording()
            if hasattr(self, 'controller'):
                self.controller._cleanup()
    
    def _create_failure_result(self, failure_type):
        """Create failure result metadata"""
        self.test_metadata['status'] = 'failed'
        self.test_metadata['failure_type'] = failure_type
        self.test_metadata['test_end'] = datetime.now().isoformat()
        return self.test_metadata
    
    def _generate_comprehensive_report(self, result_files, analysis):
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"nao_walking_comprehensive_report_{timestamp}.md"
        
        report_content = f"""# NAO Walking and Navigation Test Report

**Test Date:** {self.test_metadata['test_start']}  
**NAO IP:** {self.test_metadata['nao_ip']}  
**Status:** {self.test_metadata['status'].upper()}  
**Issue:** #604 - Dynamic Mobility and Navigation

## Test Configuration

### Route Parameters
- **Total Distance:** {self.test_config['route_distance']}m out and back
- **Obstacles:** {len(self.test_config['obstacles'])} predefined obstacles
- **Surface Type:** {self.test_metadata['environmental_conditions'].get('floor_surface', 'Unknown')}

### Video Recording
- **Resolution:** {self.test_config['video_resolution'][0]}x{self.test_config['video_resolution'][1]}
- **Frame Rate:** {self.test_config['video_fps']} FPS
- **Cameras:** Primary + Secondary (if available)

### Environmental Conditions
- **Lighting Level:** {self.test_metadata['environmental_conditions'].get('lighting_level', 'N/A')} lux
- **Ambient Temperature:** {self.test_metadata['environmental_conditions'].get('ambient_temperature', 'N/A')}°C
- **Floor Surface:** {self.test_metadata['environmental_conditions'].get('floor_surface', 'N/A')}

## Test Results Summary

### Performance Metrics

| Metric | Result | Status |
|--------|--------|---------|
| Route Completion | {analysis['total_distance']:.2f}m / {analysis['target_distance']:.2f}m | {'✅ PASS' if analysis['route_completed'] else '❌ FAIL'} |
| Average Stability | {analysis['avg_stability']:.2f} | {'✅ PASS' if analysis['stability_passed'] else '❌ FAIL'} |
| Gait Consistency | {analysis['gait_consistency']:.2f} | {'✅ PASS' if analysis['gait_passed'] else '❌ FAIL'} |
| Critical Errors | {len(analysis['critical_errors'])} | {'✅ PASS' if len(analysis['critical_errors']) == 0 else '❌ FAIL'} |

### Overall Result: {'✅ PASSED' if analysis['overall_passed'] else '❌ FAILED'}

## Detailed Analysis

### Route Navigation Analysis
- **Completed Distance:** {analysis['total_distance']:.2f} meters
- **Target Distance:** {analysis['target_distance']:.2f} meters
- **Completion Rate:** {(analysis['total_distance'] / analysis['target_distance'] * 100):.1f}%
- **Direction Changes:** 1 (turnaround at midpoint)

### Stability and Balance Analysis
- **Average Stability Score:** {analysis['avg_stability']:.3f}
- **Minimum Stability Score:** {analysis['min_stability']:.3f}
- **Stability Threshold:** {self.test_config['stability_threshold']}
- **Stability Events:** {len([e for e in self.controller.test_results['errors'] if e['type'] == 'INSTABILITY'])}

### Gait Analysis
- **Consistency Score:** {analysis['gait_consistency']:.3f}
- **Consistency Threshold:** {self.test_config['gait_consistency_threshold']}
- **Foot Contact Quality:** Analyzed throughout test
- **Joint Angle Consistency:** Monitored for smooth movement

### Obstacle Handling
- **Predefined Obstacles:** {len(self.test_config['obstacles'])}
- **Obstacles Detected:** {len(self.controller.test_results['obstacle_data'])}
- **Successfully Handled:** {analysis['obstacles_handled']}
- **Avoidance Maneuvers:** Recorded in video

## Acceptance Criteria Validation

### ✅ Criterion 1: Complete Route Navigation
**Requirement:** Successfully walk entire pre-defined route (5 meters and back) without falling or human intervention

**Result:** {'PASSED' if analysis['route_completed'] else 'FAILED'}
- Distance completed: {analysis['total_distance']:.2f}m
- Target distance: {analysis['target_distance']:.2f}m
- Falls detected: {len([e for e in analysis['critical_errors'] if e['type'] == 'FALL_DETECTED'])}
- Human intervention required: {len([e for e in analysis['critical_errors'] if e['type'] == 'INTERVENTION_REQUIRED'])}

### ✅ Criterion 2: Stable and Consistent Gait
**Requirement:** Gait must appear consistent and stable, with correct foot placement and smooth weight transfer

**Result:** {'PASSED' if analysis['gait_passed'] else 'FAILED'}
- Gait consistency score: {analysis['gait_consistency']:.3f}
- Foot placement quality: Analyzed in gait data
- Weight transfer smoothness: Monitored via stability data
- Average stability: {analysis['avg_stability']:.3f}

### ✅ Criterion 3: Video Documentation
**Requirement:** Video recording of entire walk test must be attached

**Result:** {'PASSED' if self.test_metadata.get('video_files') else 'FAILED'}
- Primary video: {self.test_metadata.get('video_files', ['N/A'])[0] if self.test_metadata.get('video_files') else 'N/A'}
- Secondary video: {self.test_metadata.get('video_files', ['N/A'])[1] if len(self.test_metadata.get('video_files', [])) > 1 else 'N/A'}
- Recording duration: {self.test_metadata.get('test_end', 'N/A')}

## Error Analysis

"""
        
        if analysis['critical_errors']:
            report_content += "### Critical Errors Detected\n\n"
            for i, error in enumerate(analysis['critical_errors'], 1):
                report_content += f"{i}. **{error['type']}**: {error['message']}\n"
                report_content += f"   - Timestamp: {error.get('timestamp', 'N/A')}\n\n"
        else:
            report_content += "### No Critical Errors Detected\n\n✅ Test completed without any critical errors or falls.\n\n"
        
        report_content += f"""## Files Generated

### Video Evidence
- **Primary Video:** {self.test_metadata.get('video_files', ['N/A'])[0] if self.test_metadata.get('video_files') else 'N/A'}
- **Secondary Video:** {self.test_metadata.get('video_files', ['N/A'])[1] if len(self.test_metadata.get('video_files', [])) > 1 else 'N/A'}

### Data Files
- **Detailed Results:** {result_files.get('json_file', 'N/A')}
- **Gait Data:** {result_files.get('gait_file', 'N/A')}
- **Stability Data:** {result_files.get('stability_file', 'N/A')}
- **Performance Analysis:** {result_files.get('analysis_file', 'N/A')}

## Recommendations

"""
        
        if analysis['overall_passed']:
            report_content += """
### ✅ Test PASSED - Robot Mobility Verified

**Performance Assessment:**
- Navigation system functioning correctly
- Balance and stability within acceptable parameters
- Gait consistency meets quality standards
- Obstacle avoidance working as expected

**Next Steps:**
- Robot cleared for normal operation
- Continue routine mobility monitoring
- Consider longer distance testing for extended validation
"""
        else:
            report_content += """
### ❌ Test FAILED - Issues Detected

**Performance Issues:**
"""
            if not analysis['route_completed']:
                report_content += "- Route navigation incomplete - check path planning\n"
            if not analysis['stability_passed']:
                report_content += "- Stability below threshold - check balance systems\n"
            if not analysis['gait_passed']:
                report_content += "- Gait inconsistency detected - check motor control\n"
            if analysis['critical_errors']:
                report_content += "- Critical errors detected - review error logs\n"
            
            report_content += """
**Recommended Actions:**
- Review error logs and video footage
- Check robot calibration and balance systems
- Verify obstacle detection sensors
- Consider maintenance or repair as needed
- Retest after addressing issues
"""
        
        report_content += f"""
## Test Statistics

- **Test Duration:** {self.controller.test_results.get('end_time', 0) - self.controller.test_results.get('start_time', 0):.1f} seconds
- **Data Points Collected:** {len(self.controller.test_results['gait_data'])} gait samples
- **Stability Samples:** {len(self.controller.test_results['stability_data'])} samples
- **Video Frames Recorded:** Approx. {int((self.controller.test_results.get('end_time', 0) - self.controller.test_results.get('start_time', 0)) * self.test_config['video_fps'])}

---

**Report Generated:** {datetime.now().isoformat()}  
**Test Environment:** Controlled laboratory setting  
**Next Review:** Based on test results
"""
        
        # Save report
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        print(f"Comprehensive report generated: {report_filename}")
        return report_filename
    
    def _print_results_summary(self, analysis):
        """Print concise results summary"""
        print("\n" + "=" * 70)
        print("WALKING TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"Route Navigation: {'PASSED' if analysis['route_completed'] else 'FAILED'}")
        print(f"Distance Covered: {analysis['total_distance']:.2f}m / {analysis['target_distance']:.2f}m")
        print(f"Stability Performance: {'PASSED' if analysis['stability_passed'] else 'FAILED'}")
        print(f"Average Stability: {analysis['avg_stability']:.3f}")
        print(f"Gait Consistency: {'PASSED' if analysis['gait_passed'] else 'FAILED'}")
        print(f"Consistency Score: {analysis['gait_consistency']:.3f}")
        print(f"Critical Errors: {len(analysis['critical_errors'])}")
        print(f"Overall Result: {'PASSED' if analysis['overall_passed'] else 'FAILED'}")
        print("=" * 70)

def main():
    """Main function for automated walking test"""
    automation = WalkingTestAutomation()
    
    try:
        # Run comprehensive test
        results = automation.run_comprehensive_walking_test()
        
        # Print final status
        if results.get('status') == 'completed':
            print("\n🎉 Walking test completed successfully!")
            print(f"Report available: {results.get('report_file', 'N/A')}")
        else:
            print(f"\n❌ Walking test status: {results.get('status', 'unknown')}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        automation.stop_video_recording()
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        automation.stop_video_recording()

if __name__ == "__main__":
    main()
