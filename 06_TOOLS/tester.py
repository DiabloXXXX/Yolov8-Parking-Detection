"""
Testing module for vehicle detection system
Provides comprehensive testing and evaluation capabilities
"""

import cv2
import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class VehicleDetectionTester:
    """Comprehensive testing class for vehicle detection system"""
    
    def __init__(self, detector, config):
        """Initialize tester"""
        self.detector = detector
        self.config = config
        self.test_results = []
        
    def test_single_video(self, video_path: str, area_points: List, 
                         ground_truth: Optional[int] = None) -> Dict:
        """Test single video with comprehensive metrics"""
        print(f"🎬 Testing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize metrics
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
            if frame_count % self.config.FRAME_SKIP != 0:
                continue
            
            processed_frames += 1
            
            # Resize frame
            frame_resized = cv2.resize(frame, 
                                     (self.config.RESIZE_WIDTH, 
                                      self.config.RESIZE_HEIGHT))
            
            # Scale points
            scale_x = self.config.RESIZE_WIDTH / width
            scale_y = self.config.RESIZE_HEIGHT / height
            scaled_points = [(int(x * scale_x), int(y * scale_y)) 
                           for x, y in area_points]
            
            # Measure inference time
            inference_start = time.time()
            vehicle_count, detections = self.detector.detect_vehicles(
                frame_resized, scaled_points)
            inference_time = time.time() - inference_start
            
            detection_counts.append(vehicle_count)
            inference_times.append(inference_time)
            
            if processed_frames % 30 == 0:
                print(f"   Progress: {processed_frames}/{total_frames//self.config.FRAME_SKIP}")
        
        cap.release()
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_detection = np.mean(detection_counts) if detection_counts else 0
        max_detection = max(detection_counts) if detection_counts else 0
        min_detection = min(detection_counts) if detection_counts else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(avg_detection, ground_truth)
        
        result = {
            'video_name': video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'avg_detection': round(avg_detection, 2),
            'max_detection': max_detection,
            'min_detection': min_detection,
            'avg_inference_ms': round(avg_inference * 1000, 2),
            'total_time': round(total_time, 2),
            'processing_fps': round(processed_frames / total_time, 2),
            'ground_truth': ground_truth,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        self._print_result(result)
        
        return result
    
    def _calculate_accuracy(self, detected: float, ground_truth: Optional[int]) -> str:
        """Calculate accuracy based on ground truth"""
        if ground_truth is None:
            return "No ground truth"
        
        accuracy_pct = (1 - abs(detected - ground_truth) / max(ground_truth, 1)) * 100
        
        if accuracy_pct >= 90:
            return f"Excellent ({accuracy_pct:.1f}%)"
        elif accuracy_pct >= 75:
            return f"Good ({accuracy_pct:.1f}%)"
        elif accuracy_pct >= 60:
            return f"Fair ({accuracy_pct:.1f}%)"
        else:
            return f"Poor ({accuracy_pct:.1f}%)"
    
    def _print_result(self, result: Dict):
        """Print test result"""
        print(f"✅ Results for {result['video_name']}:")
        print(f"   📊 Average detection: {result['avg_detection']} vehicles")
        print(f"   ⚡ Inference time: {result['avg_inference_ms']} ms/frame")
        print(f"   🎯 Accuracy: {result['accuracy']}")
        print(f"   ⏱️ Processing time: {result['total_time']} seconds")
    
    def run_comprehensive_testing(self):
        """Run testing on all available videos"""
        print("🚀 COMPREHENSIVE TESTING")
        print("=" * 60)
        
        for video_path in self.config.VIDEO_PATHS:
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping: {video_path} (not found)")
                continue
            
            video_name = os.path.basename(video_path)
            ground_truth = self.config.GROUND_TRUTH_DATA.get(video_name)
            area_points = self.config.DEFAULT_TEST_POINTS.get(
                video_name, [(100, 200), (800, 200), (800, 400), (100, 400)])
            
            self.test_single_video(video_path, area_points, ground_truth)
        
        self.generate_report()
        self.save_results()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        if not self.test_results:
            print("❌ No test results available")
            return
        
        print(f"📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Videos tested: {len(self.test_results)}")
        print()
        
        # Table header
        print("| Video | Avg Detection | Inference (ms) | Accuracy | Processing Time |")
        print("|-------|---------------|----------------|----------|-----------------|")
        
        for result in self.test_results:
            video_name = os.path.basename(result['video_name'])[:20]
            print(f"| {video_name:<20} | {result['avg_detection']:<13} | "
                  f"{result['avg_inference_ms']:<14} | {result['accuracy']:<8} | "
                  f"{result['total_time']:<15} |")
        
        # Summary
        avg_inference = np.mean([r['avg_inference_ms'] for r in self.test_results])
        avg_fps = np.mean([r['processing_fps'] for r in self.test_results])
        total_time = sum(r['total_time'] for r in self.test_results)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Average inference time: {avg_inference:.2f} ms/frame")
        print(f"   Average processing FPS: {avg_fps:.2f}")
        print(f"   Total testing time: {total_time:.2f} seconds")
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file"""
        import os
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def plot_results(self):
        """Plot test results visualization"""
        if not self.test_results:
            print("❌ No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        videos = [os.path.basename(r['video_name']) for r in self.test_results]
        detections = [r['avg_detection'] for r in self.test_results]
        inference_times = [r['avg_inference_ms'] for r in self.test_results]
        processing_fps = [r['processing_fps'] for r in self.test_results]
        
        # Detection counts
        ax1.bar(videos, detections)
        ax1.set_title('Average Vehicle Detection by Video')
        ax1.set_ylabel('Vehicle Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Inference times
        ax2.bar(videos, inference_times)
        ax2.set_title('Inference Time by Video')
        ax2.set_ylabel('Time (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Processing FPS
        ax3.bar(videos, processing_fps)
        ax3.set_title('Processing FPS by Video')
        ax3.set_ylabel('FPS')
        ax3.tick_params(axis='x', rotation=45)
        
        # Ground truth comparison
        ground_truths = [self.config.GROUND_TRUTH_DATA.get(
            os.path.basename(r['video_name']), 0) for r in self.test_results]
        
        x = np.arange(len(videos))
        width = 0.35
        
        ax4.bar(x - width/2, detections, width, label='Detected')
        ax4.bar(x + width/2, ground_truths, width, label='Ground Truth')
        ax4.set_title('Detection vs Ground Truth')
        ax4.set_ylabel('Vehicle Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(videos, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('output/test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Results visualization saved to: output/test_results.png")
