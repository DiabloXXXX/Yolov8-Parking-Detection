"""
Comprehensive accuracy testing with full screen mode and multiple runs
"""

import cv2
import numpy as np
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd

class ComprehensiveAccuracyTester:
    """Advanced testing class with multiple runs and full screen mode"""
    
    def __init__(self, detector, config):
        """Initialize comprehensive tester"""
        self.detector = detector
        self.config = config
        self.test_results = []
        self.full_screen_mode = True
        self.runs_per_video = 10
        
    def test_video_multiple_runs(self, video_path: str, area_points: List[Tuple[int, int]], 
                                ground_truth: int = None) -> Dict:
        """Test single video with multiple runs for better accuracy"""
        print(f"\n🎬 COMPREHENSIVE TESTING: {video_path}")
        print(f"🔄 Running {self.runs_per_video} tests per video for accuracy")
        print("=" * 70)
        
        all_run_results = []
        
        for run_num in range(1, self.runs_per_video + 1):
            print(f"\n🏃 Run {run_num}/{self.runs_per_video}")
            run_result = self._single_test_run(video_path, area_points, run_num)
            
            if run_result:
                all_run_results.append(run_result)
                print(f"   ✅ Run {run_num}: {run_result['avg_detection']:.1f} vehicles detected")
            else:
                print(f"   ❌ Run {run_num}: Failed")
        
        if not all_run_results:
            print("❌ All test runs failed!")
            return None
        
        # Calculate comprehensive statistics
        combined_result = self._calculate_comprehensive_stats(
            video_path, all_run_results, ground_truth)
        
        self.test_results.append(combined_result)
        self._print_comprehensive_result(combined_result)
        
        return combined_result
    
    def _single_test_run(self, video_path: str, area_points: List[Tuple[int, int]], 
                        run_num: int) -> Dict:
        """Single test run with full screen processing"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Get original video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use full resolution for better accuracy
        detection_counts = []
        inference_times = []
        frame_count = 0
        processed_frames = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame for full screen mode (no skipping)
            processed_frames += 1
            
            # Use original resolution for maximum accuracy
            inference_start = time.time()
            vehicle_count, detections = self.detector.detect_vehicles(frame, area_points)
            inference_time = time.time() - inference_start
            
            detection_counts.append(vehicle_count)
            inference_times.append(inference_time)
            
            # Progress indicator every 100 frames
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"      Progress: {progress:.1f}% ({processed_frames}/{total_frames})")
        
        cap.release()
        total_time = time.time() - start_time
        
        # Calculate run statistics
        avg_detection = statistics.mean(detection_counts) if detection_counts else 0
        median_detection = statistics.median(detection_counts) if detection_counts else 0
        mode_detection = statistics.mode(detection_counts) if detection_counts else 0
        std_dev = statistics.stdev(detection_counts) if len(detection_counts) > 1 else 0
        
        return {
            'run_number': run_num,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'avg_detection': avg_detection,
            'median_detection': median_detection,
            'mode_detection': mode_detection,
            'std_deviation': std_dev,
            'max_detection': max(detection_counts) if detection_counts else 0,
            'min_detection': min(detection_counts) if detection_counts else 0,
            'avg_inference_time': statistics.mean(inference_times) if inference_times else 0,
            'total_processing_time': total_time,
            'processing_fps': processed_frames / total_time if total_time > 0 else 0,
            'resolution': f"{original_width}x{original_height}",
            'detection_counts': detection_counts[:100]  # Store first 100 for analysis
        }
    
    def _calculate_comprehensive_stats(self, video_path: str, run_results: List[Dict], 
                                     ground_truth: int = None) -> Dict:
        """Calculate comprehensive statistics from multiple runs"""
        
        # Extract metrics from all runs
        avg_detections = [r['avg_detection'] for r in run_results]
        median_detections = [r['median_detection'] for r in run_results]
        inference_times = [r['avg_inference_time'] for r in run_results]
        processing_times = [r['total_processing_time'] for r in run_results]
        
        # Calculate overall statistics
        overall_avg = statistics.mean(avg_detections)
        overall_median = statistics.median(avg_detections)
        overall_std = statistics.stdev(avg_detections) if len(avg_detections) > 1 else 0
        consistency_score = (1 - (overall_std / overall_avg)) * 100 if overall_avg > 0 else 0
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(overall_avg, ground_truth)
        
        # Determine reliability level
        reliability = self._determine_reliability(overall_std, consistency_score)
        
        return {
            'video_name': video_path,
            'total_runs': len(run_results),
            'resolution': run_results[0]['resolution'],
            'total_frames': run_results[0]['total_frames'],
            
            # Detection Statistics
            'overall_avg_detection': round(overall_avg, 2),
            'overall_median_detection': round(overall_median, 2),
            'detection_std_dev': round(overall_std, 2),
            'consistency_score': round(consistency_score, 1),
            'min_avg_detection': round(min(avg_detections), 2),
            'max_avg_detection': round(max(avg_detections), 2),
            
            # Performance Statistics
            'avg_inference_time_ms': round(statistics.mean(inference_times) * 1000, 2),
            'avg_processing_time': round(statistics.mean(processing_times), 2),
            'avg_processing_fps': round(statistics.mean([r['processing_fps'] for r in run_results]), 2),
            
            # Accuracy Assessment
            'ground_truth': ground_truth,
            'accuracy_percentage': accuracy_metrics['percentage'],
            'accuracy_level': accuracy_metrics['level'],
            'reliability_level': reliability,
            
            # Detailed Results
            'run_details': run_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_accuracy_metrics(self, detected_avg: float, ground_truth: int) -> Dict:
        """Calculate accuracy metrics compared to ground truth"""
        if ground_truth is None:
            return {'percentage': 'N/A', 'level': 'No Ground Truth'}
        
        accuracy_pct = max(0, (1 - abs(detected_avg - ground_truth) / max(ground_truth, 1)) * 100)
        
        if accuracy_pct >= 95:
            level = "Excellent (>95%)"
        elif accuracy_pct >= 90:
            level = "Very Good (90-95%)"
        elif accuracy_pct >= 80:
            level = "Good (80-90%)"
        elif accuracy_pct >= 70:
            level = "Fair (70-80%)"
        elif accuracy_pct >= 60:
            level = "Poor (60-70%)"
        else:
            level = "Very Poor (<60%)"
        
        return {'percentage': round(accuracy_pct, 1), 'level': level}
    
    def _determine_reliability(self, std_dev: float, consistency: float) -> str:
        """Determine reliability level based on consistency"""
        if consistency >= 95:
            return "Very Reliable (>95% consistent)"
        elif consistency >= 90:
            return "Reliable (90-95% consistent)"
        elif consistency >= 80:
            return "Moderately Reliable (80-90% consistent)"
        elif consistency >= 70:
            return "Somewhat Reliable (70-80% consistent)"
        else:
            return "Unreliable (<70% consistent)"
    
    def _print_comprehensive_result(self, result: Dict):
        """Print detailed comprehensive test results"""
        print(f"\n📊 COMPREHENSIVE RESULTS - {result['video_name']}")
        print("=" * 80)
        
        print(f"🎥 Video Info:")
        print(f"   Resolution: {result['resolution']}")
        print(f"   Total Frames: {result['total_frames']}")
        print(f"   Test Runs: {result['total_runs']}")
        
        print(f"\n🎯 Detection Results:")
        print(f"   Average Detection: {result['overall_avg_detection']} vehicles")
        print(f"   Median Detection: {result['overall_median_detection']} vehicles")
        print(f"   Range: {result['min_avg_detection']} - {result['max_avg_detection']} vehicles")
        print(f"   Standard Deviation: {result['detection_std_dev']}")
        print(f"   Consistency Score: {result['consistency_score']}%")
        
        print(f"\n⚡ Performance:")
        print(f"   Avg Inference Time: {result['avg_inference_time_ms']} ms/frame")
        print(f"   Avg Processing FPS: {result['avg_processing_fps']} fps")
        print(f"   Avg Processing Time: {result['avg_processing_time']} seconds")
        
        print(f"\n🎯 Accuracy Assessment:")
        if result['ground_truth']:
            print(f"   Ground Truth: {result['ground_truth']} vehicles")
            print(f"   Accuracy: {result['accuracy_percentage']}%")
        print(f"   Accuracy Level: {result['accuracy_level']}")
        print(f"   Reliability: {result['reliability_level']}")
    
    def run_comprehensive_testing_all_videos(self):
        """Run comprehensive testing on all videos"""
        print("\n🚀 COMPREHENSIVE ACCURACY TESTING - FULL SCREEN MODE")
        print("=" * 80)
        print(f"🔄 {self.runs_per_video} runs per video for maximum accuracy")
        print(f"📺 Full resolution processing (no frame skipping)")
        print("=" * 80)
        
        for video_path in self.config.VIDEO_PATHS:
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping missing video: {video_path}")
                continue
            
            video_name = os.path.basename(video_path)
            ground_truth = self.config.GROUND_TRUTH_DATA.get(video_name)
            
            # Use default area points or get from config
            area_points = self.config.DEFAULT_TEST_POINTS.get(
                video_name, [(100, 200), (800, 200), (800, 400), (100, 400)])
            
            # Run comprehensive testing
            result = self.test_video_multiple_runs(video_path, area_points, ground_truth)
            
            if result:
                print(f"✅ Completed testing: {video_name}")
            else:
                print(f"❌ Failed testing: {video_name}")
        
        # Generate final comprehensive report
        self.generate_comprehensive_report()
        self.save_comprehensive_results()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report for all videos"""
        if not self.test_results:
            print("❌ No test results available for report")
            return
        
        print(f"\n📋 FINAL COMPREHENSIVE ACCURACY REPORT")
        print("=" * 100)
        print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Videos Tested: {len(self.test_results)}")
        print(f"🔄 Runs per Video: {self.runs_per_video}")
        print(f"📺 Mode: Full Screen Resolution")
        
        # Summary table
        print(f"\n📊 SUMMARY TABLE:")
        print("+" + "-" * 98 + "+")
        print(f"| {'Video':<20} | {'Avg Detect':<10} | {'Accuracy':<10} | {'Consistency':<12} | {'Reliability':<15} | {'Ground Truth':<10} |")
        print("+" + "-" * 98 + "+")
        
        for result in self.test_results:
            video_name = os.path.basename(result['video_name'])[:18]
            accuracy = f"{result['accuracy_percentage']}%" if result['accuracy_percentage'] != 'N/A' else 'N/A'
            consistency = f"{result['consistency_score']}%"
            reliability = result['reliability_level'][:13]
            ground_truth = result['ground_truth'] if result['ground_truth'] else 'N/A'
            
            print(f"| {video_name:<20} | {result['overall_avg_detection']:<10} | {accuracy:<10} | {consistency:<12} | {reliability:<15} | {ground_truth:<10} |")
        
        print("+" + "-" * 98 + "+")
        
        # Overall statistics
        all_accuracies = [r['accuracy_percentage'] for r in self.test_results if r['accuracy_percentage'] != 'N/A']
        all_consistency = [r['consistency_score'] for r in self.test_results]
        all_inference = [r['avg_inference_time_ms'] for r in self.test_results]
        
        print(f"\n📈 OVERALL SYSTEM PERFORMANCE:")
        if all_accuracies:
            print(f"   Average Accuracy: {statistics.mean(all_accuracies):.1f}%")
            print(f"   Best Accuracy: {max(all_accuracies):.1f}%")
            print(f"   Worst Accuracy: {min(all_accuracies):.1f}%")
        
        print(f"   Average Consistency: {statistics.mean(all_consistency):.1f}%")
        print(f"   Average Inference Time: {statistics.mean(all_inference):.1f} ms/frame")
        print(f"   Total Test Runs: {len(self.test_results) * self.runs_per_video}")
    
    def save_comprehensive_results(self, filename: str = "comprehensive_accuracy_results.json"):
        """Save comprehensive results to JSON file"""
        import os
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        # Prepare data for JSON export
        export_data = {
            'test_metadata': {
                'runs_per_video': self.runs_per_video,
                'full_screen_mode': self.full_screen_mode,
                'total_videos': len(self.test_results),
                'total_test_runs': len(self.test_results) * self.runs_per_video,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive results saved to: {filepath}")

def main():
    """Main function to run comprehensive testing"""
    import sys
    import os
    sys.path.append('src')
    
    from src.config import Config
    from src.vehicle_detector import VehicleDetector
    
    # Load configuration
    config = Config('config/config.yaml')
    
    # Initialize detector
    detector = VehicleDetector(config)
    
    # Initialize comprehensive tester
    tester = ComprehensiveAccuracyTester(detector, config)
    
    # Run comprehensive testing
    tester.run_comprehensive_testing_all_videos()

if __name__ == "__main__":
    main()
