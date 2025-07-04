"""
Advanced testing module with multiple runs and fullscreen display
"""

import cv2
import time
import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Optional
import statistics

class AdvancedVehicleDetectionTester:
    """Advanced testing class with multiple runs per video"""
    
    def __init__(self, detector, config):
        """Initialize advanced tester"""
        self.detector = detector
        self.config = config
        self.test_results = []
        
    def test_video_multiple_runs(self, video_path: str, area_points: List, 
                                runs: int = 10, fullscreen: bool = True) -> Dict:
        """Test single video multiple times for better accuracy"""
        print(f"\n🎬 Testing {video_path} with {runs} runs")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print(f"📊 Video Info:")
        print(f"   - Frames: {total_frames}")
        print(f"   - Duration: {duration:.1f}s")  
        print(f"   - FPS: {fps:.1f}")
        print(f"   - Size: {width}x{height}")
        
        cap.release()
        
        # Multiple runs
        all_run_results = []
        
        for run in range(runs):
            print(f"\n🔄 Run {run + 1}/{runs}")
            run_result = self._single_test_run(video_path, area_points, fullscreen, run + 1)
            if run_result:
                all_run_results.append(run_result)
                print(f"   ✅ Run {run + 1}: {run_result['avg_detection']:.1f} vehicles avg")
        
        if not all_run_results:
            return None
        
        # Calculate statistics across all runs
        final_result = self._calculate_run_statistics(
            video_path, all_run_results, width, height, fps, total_frames, duration)
        
        self.test_results.append(final_result)
        self._print_comprehensive_result(final_result)
        
        return final_result
    
    def _single_test_run(self, video_path: str, area_points: List, 
                        fullscreen: bool, run_number: int) -> Dict:
        """Single test run with optional fullscreen display"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Setup display
        window_name = f"Vehicle Detection - Run {run_number}"
        if fullscreen:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Initialize metrics
        detection_counts = []
        inference_times = []
        frame_count = 0
        processed_frames = 0
        
        start_time = time.time()
        
        print(f"      🎥 Processing frames... (Press 'q' to skip run, 's' to speed up)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % self.config.FRAME_SKIP != 0:
                continue
            
            processed_frames += 1
            
            # Resize frame
            if fullscreen:
                # Use original resolution for better accuracy in fullscreen
                frame_resized = frame
                scale_x = 1.0
                scale_y = 1.0
                scaled_points = area_points
            else:
                frame_resized = cv2.resize(frame, 
                                         (self.config.RESIZE_WIDTH, 
                                          self.config.RESIZE_HEIGHT))
                scale_x = self.config.RESIZE_WIDTH / frame.shape[1]
                scale_y = self.config.RESIZE_HEIGHT / frame.shape[0]
                scaled_points = [(int(x * scale_x), int(y * scale_y)) 
                               for x, y in area_points]
            
            # Measure inference time
            inference_start = time.time()
            vehicle_count, detections = self.detector.detect_vehicles(
                frame_resized, scaled_points)
            inference_time = time.time() - inference_start
            
            detection_counts.append(vehicle_count)
            inference_times.append(inference_time)
            
            # Draw results
            result_frame = self.detector.draw_detections(
                frame_resized, detections, scaled_points)
            
            # Add run info
            cv2.putText(result_frame, f"Run {run_number} - Frame {processed_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow(window_name, result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"      ⏭️ Run {run_number} skipped by user")
                break
            elif key == ord('s'):
                # Speed up by skipping more frames
                frame_count += self.config.FRAME_SKIP * 2
        
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        if not detection_counts:
            return None
        
        # Calculate metrics for this run
        return {
            'avg_detection': np.mean(detection_counts),
            'max_detection': max(detection_counts),
            'min_detection': min(detection_counts),
            'std_detection': np.std(detection_counts),
            'avg_inference_ms': np.mean(inference_times) * 1000,
            'total_time': total_time,
            'processed_frames': processed_frames,
            'processing_fps': processed_frames / total_time
        }
    
    def _calculate_run_statistics(self, video_path: str, run_results: List[Dict],
                                 width: int, height: int, fps: float, 
                                 total_frames: int, duration: float) -> Dict:
        """Calculate comprehensive statistics across multiple runs"""
        
        # Extract metrics from all runs
        avg_detections = [r['avg_detection'] for r in run_results]
        max_detections = [r['max_detection'] for r in run_results]
        min_detections = [r['min_detection'] for r in run_results]
        inference_times = [r['avg_inference_ms'] for r in run_results]
        
        # Calculate statistics
        final_result = {
            'video_name': video_path,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            },
            'runs_completed': len(run_results),
            'detection_stats': {
                'mean': statistics.mean(avg_detections),
                'median': statistics.median(avg_detections),
                'std_dev': statistics.stdev(avg_detections) if len(avg_detections) > 1 else 0,
                'min': min(avg_detections),
                'max': max(avg_detections),
            },
            'peak_detection': {
                'mean': statistics.mean(max_detections),
                'max': max(max_detections)
            },
            'performance': {
                'avg_inference_ms': statistics.mean(inference_times),
                'std_inference_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0
            },
            'accuracy_assessment': self._assess_accuracy(avg_detections),
            'timestamp': datetime.now().isoformat()
        }
        
        return final_result
    
    def _assess_accuracy(self, detections: List[float]) -> Dict:
        """Assess detection accuracy and consistency"""
        if len(detections) < 2:
            return {"consistency": "insufficient_data", "reliability": "unknown"}
        
        std_dev = statistics.stdev(detections)
        mean_val = statistics.mean(detections)
        cv = (std_dev / mean_val * 100) if mean_val > 0 else 0  # Coefficient of variation
        
        # Assess consistency
        if cv < 10:
            consistency = "excellent"
        elif cv < 20:
            consistency = "good"
        elif cv < 30:
            consistency = "fair"
        else:
            consistency = "poor"
        
        # Assess reliability based on standard deviation
        if std_dev < 0.5:
            reliability = "very_high"
        elif std_dev < 1.0:
            reliability = "high"
        elif std_dev < 2.0:
            reliability = "moderate"
        else:
            reliability = "low"
        
        return {
            "consistency": consistency,
            "reliability": reliability,
            "coefficient_of_variation": cv,
            "std_deviation": std_dev
        }
    
    def _print_comprehensive_result(self, result: Dict):
        """Print detailed test results"""
        print(f"\n📊 COMPREHENSIVE RESULTS - {os.path.basename(result['video_name'])}")
        print("=" * 70)
        
        # Video info
        info = result['video_info']
        print(f"📹 Video: {info['width']}x{info['height']}, {info['fps']:.1f}fps, {info['duration']:.1f}s")
        
        # Detection statistics
        stats = result['detection_stats']
        print(f"\n🎯 Detection Statistics ({result['runs_completed']} runs):")
        print(f"   Mean:      {stats['mean']:.2f} vehicles")
        print(f"   Median:    {stats['median']:.2f} vehicles")
        print(f"   Std Dev:   {stats['std_dev']:.2f}")
        print(f"   Range:     {stats['min']:.1f} - {stats['max']:.1f}")
        
        # Accuracy assessment
        accuracy = result['accuracy_assessment']
        print(f"\n🎯 Accuracy Assessment:")
        print(f"   Consistency: {accuracy['consistency'].upper()}")
        print(f"   Reliability: {accuracy['reliability'].upper()}")
        print(f"   CV:          {accuracy['coefficient_of_variation']:.1f}%")
        
        # Performance
        perf = result['performance']
        print(f"\n⚡ Performance:")
        print(f"   Inference:   {perf['avg_inference_ms']:.1f} ± {perf['std_inference_ms']:.1f} ms")
    
    def run_comprehensive_testing(self, runs_per_video: int = 10, fullscreen: bool = True):
        """Run comprehensive testing on all videos"""
        print(f"\n🚀 COMPREHENSIVE TESTING - {runs_per_video} RUNS PER VIDEO")
        print("=" * 80)
        
        if fullscreen:
            print("🖥️ Using FULLSCREEN mode for better accuracy")
        
        print("\n📋 Controls during testing:")
        print("   'q' = Skip current run")
        print("   's' = Speed up current run")
        print("   ESC = Use default area selection")
        
        for i, video_path in enumerate(self.config.VIDEO_PATHS, 1):
            if not os.path.exists(video_path):
                print(f"\n⚠️ Video {i}: {video_path} not found - SKIPPED")
                continue
            
            print(f"\n📹 VIDEO {i}/{len(self.config.VIDEO_PATHS)}: {os.path.basename(video_path)}")
            
            # Get area points (interactive or default)
            area_points = self._get_area_points_interactive(video_path)
            
            # Run multiple tests
            self.test_video_multiple_runs(video_path, area_points, runs_per_video, fullscreen)
        
        # Generate final report
        self.generate_final_report()
        self.save_comprehensive_results()
    
    def _get_area_points_interactive(self, video_path: str) -> List:
        """Get area points interactively or use defaults"""
        print(f"\n🖱️ Select detection area for {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot read frame, using default points")
            return self._get_default_points(video_path)
        
        points = []
        window_name = f"Area Selection - {os.path.basename(video_path)}"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                print(f"   Point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("🖱️ Click 4 points for detection area (ESC for default)")
        
        while True:
            display_frame = frame.copy()
            
            # Draw points and polygon
            for i, pt in enumerate(points):
                cv2.circle(display_frame, pt, 8, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(points) >= 3:
                cv2.polylines(display_frame, [np.array(points)], 
                             len(points) == 4, (255, 0, 0), 3)
            
            cv2.putText(display_frame, f"Points: {len(points)}/4", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "ESC=Default, ENTER=Done", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("⚠️ Using default area points")
                points = self._get_default_points(video_path)
                break
            elif key == 13 or len(points) == 4:  # ENTER or 4 points
                if len(points) == 4:
                    break
        
        cv2.destroyAllWindows()
        print(f"✅ Area selected: {points}")
        return points
    
    def _get_default_points(self, video_path: str) -> List:
        """Get default points for video"""
        video_name = os.path.basename(video_path)
        return self.config.DEFAULT_TEST_POINTS.get(
            video_name, 
            [(100, 200), (800, 200), (800, 400), (100, 400)]
        )
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        if not self.test_results:
            print("❌ No test results available")
            return
        
        print(f"\n📋 FINAL COMPREHENSIVE REPORT")
        print("=" * 80)
        print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Videos tested: {len(self.test_results)}")
        
        # Summary table
        print(f"\n📊 SUMMARY TABLE:")
        print("| Video | Runs | Mean ± Std | Range | Consistency | Reliability |")
        print("|-------|------|------------|-------|-------------|-------------|")
        
        for result in self.test_results:
            video_name = os.path.basename(result['video_name'])[:15]
            runs = result['runs_completed']
            stats = result['detection_stats']
            accuracy = result['accuracy_assessment']
            
            print(f"| {video_name:<15} | {runs:4d} | "
                  f"{stats['mean']:4.1f}±{stats['std_dev']:4.1f} | "
                  f"{stats['min']:4.1f}-{stats['max']:4.1f} | "
                  f"{accuracy['consistency']:<11} | {accuracy['reliability']:<11} |")
        
        # Overall statistics
        all_means = [r['detection_stats']['mean'] for r in self.test_results]
        overall_mean = statistics.mean(all_means)
        overall_std = statistics.stdev(all_means) if len(all_means) > 1 else 0
        
        print(f"\n🎯 OVERALL STATISTICS:")
        print(f"   System Mean: {overall_mean:.2f} ± {overall_std:.2f} vehicles")
        print(f"   Total Runs:  {sum(r['runs_completed'] for r in self.test_results)}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        high_reliability = sum(1 for r in self.test_results 
                              if r['accuracy_assessment']['reliability'] in ['very_high', 'high'])
        
        if high_reliability >= len(self.test_results) * 0.8:
            print("   ✅ System shows high reliability across videos")
        else:
            print("   ⚠️ Consider adjusting detection parameters for better consistency")
    
    def save_comprehensive_results(self, filename: str = "comprehensive_test_results.json"):
        """Save comprehensive results to JSON"""
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Comprehensive results saved to: {filepath}")
