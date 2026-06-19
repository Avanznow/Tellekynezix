#!/usr/bin/env python3

"""
NAO Push-Up Test Automation
Handles test scheduling, video recording, and report generation
"""

import subprocess
import time
import os
import json
import cv2
import threading
from datetime import datetime
from nao_pushup_controller import NAOPushupController

class PushupTestAutomation:
    def __init__(self, nao_ip="192.168.23.53", port=9559):
        self.nao_ip = nao_ip
        self.port = port
        
        # Video recording
        self.recording = False
        self.video_thread = None
        self.video_writer = None
        self.camera = None
        
        # Test configuration
        self.test_config = {
            'target_reps': 15,
            'warmup_time': 300,  # 5 minutes warmup
            'cooldown_time': 180,  # 3 minutes cooldown
            'video_resolution': (640, 480),
            'video_fps': 30,
            'camera_index': 0  # Default webcam
        }
        
    def start_video_recording(self, filename_prefix="nao_pushup_test"):
        """Start video recording of the test"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{filename_prefix}_{timestamp}.mp4"
        
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(self.test_config['camera_index'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.test_config['video_resolution'][0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.test_config['video_resolution'][1])
            self.camera.set(cv2.CAP_PROP_FPS, self.test_config['video_fps'])
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                self.test_config['video_fps'],
                self.test_config['video_resolution']
            )
            
            self.recording = True
            self.video_thread = threading.Thread(target=self._record_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            print(f"Video recording started: {video_filename}")
            return video_filename
            
        except Exception as e:
            print(f"Failed to start video recording: {e}")
            return None
    
    def _record_video(self):
        """Video recording thread"""
        while self.recording:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Add timestamp overlay
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add test status overlay
                    cv2.putText(frame, "NAO Push-Up Test", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    self.video_writer.write(frame)
                else:
                    print("Failed to capture frame")
                    break
                    
                time.sleep(1.0 / self.test_config['video_fps'])
                
            except Exception as e:
                print(f"Error recording video: {e}")
                break
    
    def stop_video_recording(self):
        """Stop video recording"""
        self.recording = False
        if self.video_thread:
            self.video_thread.join(timeout=2)
        
        if self.video_writer:
            self.video_writer.release()
        
        if self.camera:
            self.camera.release()
        
        print("Video recording stopped")
    
    def check_robot_temperature(self):
        """Check if robot is cooled down enough to start test"""
        print("Checking robot temperature...")
        
        try:
            from naoqi import ALProxy
            temp_proxy = ALProxy("ALTemperature", self.nao_ip, self.port)
            
            # Check critical joint temperatures
            critical_joints = ["RShoulderPitch", "LShoulderPitch", "RHipPitch", "LHipPitch"]
            max_temp = 0
            
            for joint in critical_joints:
                try:
                    temp = temp_proxy.getTemperature(joint)
                    max_temp = max(max_temp, temp)
                    print(f"{joint}: {temp}°C")
                except:
                    print(f"Could not read temperature for {joint}")
            
            if max_temp > 50:  # 50°C threshold for starting test
                print(f"Robot too warm: {max_temp}°C. Waiting for cooldown...")
                return False
            else:
                print(f"Robot temperature acceptable: {max_temp}°C")
                return True
                
        except Exception as e:
            print(f"Error checking temperature: {e}")
            return False
    
    def wait_for_cooldown(self):
        """Wait for robot to cool down before test"""
        print("Waiting for robot cooldown...")
        
        cooldown_start = time.time()
        while time.time() - cooldown_start < self.test_config['cooldown_time']:
            if self.check_robot_temperature():
                print("Robot cooled down sufficiently")
                return True
            
            remaining = self.test_config['cooldown_time'] - (time.time() - cooldown_start)
            print(f"Cooldown remaining: {remaining:.0f}s")
            time.sleep(30)  # Check every 30 seconds
        
        print("Cooldown period completed")
        return True
    
    def run_comprehensive_test(self):
        """Run the complete push-up test with video recording and reporting"""
        print("=" * 60)
        print("NAO PUSH-UP ENDURANCE TEST - COMPREHENSIVE")
        print("=" * 60)
        
        # Test metadata
        test_metadata = {
            'test_start': datetime.now().isoformat(),
            'nao_ip': self.nao_ip,
            'test_config': self.test_config,
            'status': 'preparing'
        }
        
        # Step 1: Check robot temperature
        if not self.check_robot_temperature():
            print("Robot too warm - waiting for cooldown")
            self.wait_for_cooldown()
        
        # Step 2: Start video recording
        video_filename = self.start_video_recording()
        test_metadata['video_file'] = video_filename
        
        # Step 3: Initialize controller and run test
        controller = NAOPushupController(self.nao_ip, self.port)
        
        try:
            # Run the push-up test
            success = controller.run_endurance_test(self.test_config['target_reps'])
            
            if success:
                test_metadata['status'] = 'completed'
                
                # Analyze performance
                analysis = controller.analyze_performance()
                test_metadata['analysis'] = analysis
                
                # Save test results
                result_files = controller.save_test_results()
                test_metadata['result_files'] = result_files
                
                # Generate comprehensive report
                report_file = self.generate_test_report(test_metadata, controller.test_results, analysis)
                test_metadata['report_file'] = report_file
                
                print("\n" + "=" * 60)
                print("TEST RESULTS SUMMARY")
                print("=" * 60)
                print(f"Total Repetitions: {analysis['total_reps']}")
                print(f"Fatigue Performance: {'PASSED' if analysis['fatigue_passed'] else 'FAILED'}")
                print(f"Thermal Performance: {'PASSED' if analysis['thermal_passed'] else 'FAILED'}")
                print(f"Overall Result: {'PASSED' if analysis['overall_passed'] else 'FAILED'}")
                print(f"Max Temperature: {analysis['max_temperature']:.1f}°C")
                print(f"Fatigue Ratio: {analysis['fatigue_ratio']:.2f}")
                print(f"Video File: {video_filename}")
                print(f"Report File: {report_file}")
                print("=" * 60)
                
                return test_metadata
                
            else:
                test_metadata['status'] = 'failed'
                print("Test failed to complete")
                return test_metadata
        
        except Exception as e:
            test_metadata['status'] = 'error'
            test_metadata['error'] = str(e)
            print(f"Test error: {e}")
            return test_metadata
        
        finally:
            # Step 4: Stop video recording
            self.stop_video_recording()
            
            # Step 5: Cleanup
            controller._cleanup()
    
    def generate_test_report(self, metadata, test_results, analysis):
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"nao_pushup_report_{timestamp}.md"
        
        report_content = f"""# NAO Push-Up Endurance Test Report

**Test Date:** {metadata['test_start']}  
**NAO IP:** {metadata['nao_ip']}  
**Status:** {metadata['status'].upper()}

## Test Configuration

- **Target Repetitions:** {metadata['test_config']['target_reps']}
- **Warmup Time:** {metadata['test_config']['warmup_time']} seconds
- **Cooldown Time:** {metadata['test_config']['cooldown_time']} seconds
- **Video Resolution:** {metadata['test_config']['video_resolution']}
- **Video FPS:** {metadata['test_config']['video_fps']}

## Test Results

### Performance Summary

- **Total Repetitions Completed:** {analysis['total_reps']}
- **First 3 Reps Average Time:** {analysis['first_three_avg']:.2f} seconds
- **Last 3 Reps Average Time:** {analysis['last_three_avg']:.2f} seconds
- **Fatigue Ratio:** {analysis['fatigue_ratio']:.2f}
- **Maximum Temperature:** {analysis['max_temperature']:.1f}°C

### Acceptance Criteria Results

| Criteria | Requirement | Result | Status |
|----------|-------------|--------|---------|
| Minimum Repetitions | ≥ 10 push-ups | {analysis['total_reps']} push-ups | {'✅ PASS' if analysis['total_reps'] >= 10 else '❌ FAIL'} |
| Thermal Performance | < 70°C, no overheating | {analysis['max_temperature']:.1f}°C | {'✅ PASS' if analysis['thermal_passed'] else '❌ FAIL'} |
| Fatigue Performance | ≤ 15% slower | {analysis['fatigue_ratio']:.1%} | {'✅ PASS' if analysis['fatigue_passed'] else '❌ FAIL'} |

### Overall Result: {'✅ PASSED' if analysis['overall_passed'] else '❌ FAILED'}

## Detailed Analysis

### Fatigue Analysis
- **Acceptable Fatigue Ratio:** ≤ 1.15 (15% slower)
- **Actual Fatigue Ratio:** {analysis['fatigue_ratio']:.2f}
- **Performance Change:** {(analysis['fatigue_ratio'] - 1) * 100:.1f}% {'slower' if analysis['fatigue_ratio'] > 1 else 'faster'}

### Thermal Analysis
- **Temperature Threshold:** 70°C
- **Maximum Recorded Temperature:** {analysis['max_temperature']:.1f}°C
- **Temperature Margin:** {70 - analysis['max_temperature']:.1f}°C

### Errors and Issues
"""
        
        if metadata.get('analysis', {}).get('critical_errors'):
            report_content += "\n**Critical Errors:**\n"
            for error in analysis['critical_errors']:
                report_content += f"- {error['type']}: {error['message']}\n"
        else:
            report_content += "\n**No critical errors detected.**\n"
        
        report_content += f"""
## Files Generated

- **Video Recording:** {metadata.get('video_file', 'N/A')}
- **Detailed Results:** {metadata.get('result_files', {}).get('json_file', 'N/A')}
- **Temperature Data:** {metadata.get('result_files', {}).get('csv_file', 'N/A')}
- **Performance Analysis:** {metadata.get('result_files', {}).get('analysis_file', 'N/A')}

## Recommendations

"""
        
        if analysis['overall_passed']:
            report_content += """
✅ **Test PASSED** - NAO robot meets all acceptance criteria

- Robot demonstrates excellent endurance and thermal performance
- No significant fatigue detected over test duration
- Temperature levels remained within safe operating range
- Ready for production deployment

"""
        else:
            report_content += """
❌ **Test FAILED** - Issues detected that need attention

"""
            if not analysis['fatigue_passed']:
                report_content += "- **Fatigue Issue:** Performance degradation exceeded 15% threshold\n"
            if not analysis['thermal_passed']:
                report_content += "- **Thermal Issue:** Temperature exceeded safe operating limits\n"
            if analysis['total_reps'] < 10:
                report_content += "- **Endurance Issue:** Failed to complete minimum repetitions\n"
            
            report_content += """
Recommendations:
- Allow additional cooldown time before retesting
- Check for mechanical issues in joint movements
- Verify proper robot calibration
- Consider reducing test intensity if issues persist

"""
        
        report_content += f"""
---
**Report Generated:** {datetime.now().isoformat()}  
**Test Duration:** {test_results.get('end_time', 0) - test_results.get('start_time', 0):.1f} seconds
"""
        
        # Save report
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        print(f"Test report generated: {report_filename}")
        return report_filename

def main():
    """Main function for automated testing"""
    automation = PushupTestAutomation()
    
    try:
        # Run comprehensive test
        results = automation.run_comprehensive_test()
        
        # Print final status
        if results.get('status') == 'completed':
            print("\n🎉 Test completed successfully!")
        else:
            print(f"\n❌ Test status: {results.get('status', 'unknown')}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        automation.stop_video_recording()
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        automation.stop_video_recording()

if __name__ == "__main__":
    main()
