#!/usr/bin/env python3
"""
Performance Benchmark Test for YOLOv8 Vehicle Detection
Tests processing speed, memory usage, and system performance
"""

import sys
import os
import time
import psutil
import cv2
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "01_CORE"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_DIR))

from src.config import Config
from src.vehicle_detector import VehicleDetector

class PerformanceBenchmark:
    """Performance benchmark testing"""
    
    def __init__(self):
        self.config = Config("02_CONFIG/config.yaml")
        self.detector = VehicleDetector(self.config)
        self.results = {
            'system_info': self.get_system_info(),
            'model_tests': [],
            'resolution_tests': [],
            'tracking_performance': [],
            'memory_usage': []
        }
    
    def get_system_info(self):
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'python_version': sys.version,
            'opencv_version': cv2.__version__
        }
    
    def test_model_performance(self):
        """Test different model sizes performance"""
        print("🤖 TESTING MODEL PERFORMANCE")
        print("-" * 40)
        
        # Model paths to test
        models_to_test = [
            ("03_MODELS/yolov8s.pt", "YOLOv8s"),
            ("03_MODELS/yolov8m.pt", "YOLOv8m"),
            ("03_MODELS/yolov8l.pt", "YOLOv8l")
        ]
        
        # Test video
        video_path = self.config.VIDEO_PATHS[0] if self.config.VIDEO_PATHS else None
        if not video_path or not os.path.exists(video_path):
            print("❌ No test video available")
            return
        
        for model_path, model_name in models_to_test:
            if not os.path.exists(model_path):
                print(f"⚠️ Skipping {model_name} - file not found")
                continue
            
            print(f"\n🧪 Testing {model_name}")
            
            # Load model
            start_load_time = time.time()
            try:
                test_config = Config("02_CONFIG/config.yaml")
                test_config.MODEL_PATH = model_path
                test_detector = VehicleDetector(test_config)
                load_time = time.time() - start_load_time
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
            
            # Test processing speed
            cap = cv2.VideoCapture(video_path)
            processing_times = []
            memory_usage = []
            
            for i in range(30):  # Test 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                # Measure memory before
                memory_before = psutil.virtual_memory().used
                
                # Measure processing time
                start_time = time.time()
                total_detected, _ = test_detector.detect_vehicles(frame_resized, area_points)
                processing_time = time.time() - start_time
                
                # Measure memory after
                memory_after = psutil.virtual_memory().used
                memory_diff = (memory_after - memory_before) / (1024**2)  # MB
                
                processing_times.append(processing_time)
                memory_usage.append(memory_diff)
            
            cap.release()
            
            # Calculate metrics
            avg_processing_time = np.mean(processing_times)
            fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
            avg_memory = np.mean(memory_usage)
            
            result = {
                'model_name': model_name,
                'load_time': load_time,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
                'avg_memory_usage': avg_memory,
                'frames_tested': len(processing_times)
            }
            
            self.results['model_tests'].append(result)
            
            print(f"   Load time: {load_time:.2f}s")
            print(f"   Processing: {avg_processing_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
            print(f"   Memory: {avg_memory:.1f}MB per frame")
    
    def test_resolution_performance(self):
        """Test different resolution performance"""
        print(f"\n📐 TESTING RESOLUTION PERFORMANCE")
        print("-" * 40)
        
        # Resolution sizes to test
        resolutions = [
            (320, 240, "QVGA"),
            (640, 480, "VGA"),
            (1280, 720, "HD"),
            (1920, 1080, "FHD")
        ]
        
        video_path = self.config.VIDEO_PATHS[0] if self.config.VIDEO_PATHS else None
        if not video_path or not os.path.exists(video_path):
            return
        
        for width, height, name in resolutions:
            print(f"\n🧪 Testing {name} ({width}x{height})")
            
            cap = cv2.VideoCapture(video_path)
            processing_times = []
            detection_counts = []
            
            for i in range(20):  # Test 20 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (width, height))
                area_points = [(0, 0), (width, 0), (width, height), (0, height)]
                
                start_time = time.time()
                total_detected, _ = self.detector.detect_vehicles(frame_resized, area_points)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                detection_counts.append(total_detected)
            
            cap.release()
            
            avg_processing_time = np.mean(processing_times)
            fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
            avg_detected = np.mean(detection_counts)
            
            result = {
                'resolution_name': name,
                'width': width,
                'height': height,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
                'avg_detected': avg_detected,
                'pixel_count': width * height
            }
            
            self.results['resolution_tests'].append(result)
            
            print(f"   Processing: {avg_processing_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
            print(f"   Avg detected: {avg_detected:.1f}")
    
    def test_tracking_performance(self):
        """Test tracking system performance"""
        print(f"\n🎯 TESTING TRACKING PERFORMANCE")
        print("-" * 40)
        
        video_path = self.config.VIDEO_PATHS[0] if self.config.VIDEO_PATHS else None
        if not video_path or not os.path.exists(video_path):
            return
        
        # Test with and without tracking
        tracking_configs = [
            (False, "Without Tracking"),
            (True, "With Tracking")
        ]
        
        for tracking_enabled, config_name in tracking_configs:
            print(f"\n🧪 Testing {config_name}")
            
            # Configure tracking
            test_config = Config("02_CONFIG/config.yaml")
            test_config.TRACKING_ENABLED = tracking_enabled
            test_detector = VehicleDetector(test_config)
            
            cap = cv2.VideoCapture(video_path)
            processing_times = []
            detection_counts = []
            
            for i in range(50):  # Test 50 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                start_time = time.time()
                
                if tracking_enabled:
                    parked_count, moving_count, detections = test_detector.detect_vehicles_with_tracking(
                        frame_resized, area_points)
                    total_detected = len(detections)
                else:
                    total_detected, _ = test_detector.detect_vehicles(frame_resized, area_points)
                
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                detection_counts.append(total_detected)
            
            cap.release()
            
            avg_processing_time = np.mean(processing_times)
            fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
            avg_detected = np.mean(detection_counts)
            
            result = {
                'tracking_enabled': tracking_enabled,
                'config_name': config_name,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
                'avg_detected': avg_detected
            }
            
            self.results['tracking_performance'].append(result)
            
            print(f"   Processing: {avg_processing_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
            print(f"   Avg detected: {avg_detected:.1f}")
    
    def run_full_benchmark(self):
        """Run complete performance benchmark"""
        print("🚀 PERFORMANCE BENCHMARK TEST")
        print("=" * 50)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System info
        sys_info = self.results['system_info']
        print(f"\n💻 SYSTEM INFO:")
        print(f"   CPU: {sys_info['cpu_count']} cores @ {sys_info['cpu_freq']:.0f}MHz")
        print(f"   Memory: {sys_info['memory_total']}GB")
        print(f"   OpenCV: {sys_info['opencv_version']}")
        
        # Change to project root
        os.chdir(PROJECT_ROOT)
        
        try:
            # Run all tests
            self.test_model_performance()
            self.test_resolution_performance()
            self.test_tracking_performance()
            
            # Save results
            self.save_benchmark_results()
            
            # Show summary
            self.show_benchmark_summary()
            
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            import traceback
            traceback.print_exc()
    
    def show_benchmark_summary(self):
        """Show benchmark summary"""
        print(f"\n📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Best model performance
        if self.results['model_tests']:
            best_model = max(self.results['model_tests'], key=lambda x: x['fps'])
            print(f"🏆 Best Model: {best_model['model_name']} ({best_model['fps']:.1f} FPS)")
        
        # Best resolution
        if self.results['resolution_tests']:
            best_res = max(self.results['resolution_tests'], key=lambda x: x['fps'])
            print(f"📐 Best Resolution: {best_res['resolution_name']} ({best_res['fps']:.1f} FPS)")
        
        # Tracking overhead
        if len(self.results['tracking_performance']) >= 2:
            without_tracking = next(x for x in self.results['tracking_performance'] if not x['tracking_enabled'])
            with_tracking = next(x for x in self.results['tracking_performance'] if x['tracking_enabled'])
            overhead = ((with_tracking['avg_processing_time'] - without_tracking['avg_processing_time']) 
                       / without_tracking['avg_processing_time']) * 100
            print(f"🎯 Tracking Overhead: {overhead:.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if self.results['model_tests']:
            fastest_model = min(self.results['model_tests'], key=lambda x: x['avg_processing_time'])
            if fastest_model['fps'] < 15:
                print("   - Consider using faster hardware or smaller model")
            print(f"   - Use {fastest_model['model_name']} for best speed")
        
        if self.results['resolution_tests']:
            good_res = [r for r in self.results['resolution_tests'] if r['fps'] >= 20]
            if good_res:
                best_balance = max(good_res, key=lambda x: x['pixel_count'])
                print(f"   - Recommended resolution: {best_balance['resolution_name']}")
    
    def save_benchmark_results(self):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        output_dir = Path(self.config.OUTPUT_DIR).parent / "benchmark_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"benchmark_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Benchmark results saved: {results_file}")

def main():
    """Main function"""
    try:
        benchmark = PerformanceBenchmark()
        benchmark.run_full_benchmark()
        print(f"\n✅ Benchmark completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
