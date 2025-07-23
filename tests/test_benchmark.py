import pytest
import os
import time
import psutil
import cv2
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# Import modules from src
from src.config import Config
from src.vehicle_detector import VehicleDetector

# Define PROJECT_ROOT for consistent pathing in tests
PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="module")
def config_and_detector():
    """Fixture to provide a Config and VehicleDetector instance"""
    config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
    detector = VehicleDetector(config)
    return config, detector

class TestPerformanceBenchmark:
    """Performance benchmark testing using pytest"""
    
    def get_system_info(self):
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'python_version': sys.version,
            'opencv_version': cv2.__version__
        }
    
    def test_model_performance(self, config_and_detector):
        """Test different model sizes performance"""
        config, _ = config_and_detector # We'll create new detector instances
        
        print("\n🤖 TESTING MODEL PERFORMANCE")
        print("-" * 40)
        
        models_to_test = [
            (str(PROJECT_ROOT / "models" / "yolov8s.pt"), "YOLOv8s"),
            # Add other models if available and desired for testing
            # (str(PROJECT_ROOT / "models" / "yolov8m.pt"), "YOLOv8m"),
            # (str(PROJECT_ROOT / "models" / "yolov8l.pt"), "YOLOv8l")
        ]
        
        video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
        assert video_path is not None, "❌ No test video available for model performance test!"
        
        results = []
        for model_path, model_name in models_to_test:
            if not os.path.exists(model_path):
                print(f"⚠️ Skipping {model_name} - file not found")
                continue
            
            print(f"\n🧪 Testing {model_name}")
            
            start_load_time = time.time()
            try:
                test_config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
                test_config.MODEL_PATH = model_path
                test_detector = VehicleDetector(test_config)
                load_time = time.time() - start_load_time
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
            
            cap = cv2.VideoCapture(video_path)
            processing_times = []
            
            for i in range(30):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                start_time = time.time()
                total_detected, _ = test_detector.detect_vehicles(frame_resized, area_points)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
            
            cap.release()
            
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
            
            result = {
                'model_name': model_name,
                'load_time': load_time,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
                'frames_tested': len(processing_times)
            }
            results.append(result)
            
            print(f"   Load time: {load_time:.2f}s")
            print(f"   Processing: {avg_processing_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
        
        assert len(results) > 0, "No models were tested for performance."

    def test_resolution_performance(self, config_and_detector):
        """Test different resolution performance"""
        config, detector = config_and_detector
        
        print(f"\n📐 TESTING RESOLUTION PERFORMANCE")
        print("-" * 40)
        
        resolutions = [
            (320, 240, "QVGA"),
            (640, 480, "VGA"),
            (1280, 720, "HD"),
            # (1920, 1080, "FHD") # May be too slow for quick tests
        ]
        
        video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
        assert video_path is not None, "❌ No test video available for resolution test!"
        
        results = []
        for width, height, name in resolutions:
            print(f"\n🧪 Testing {name} ({width}x{height})")
            
            cap = cv2.VideoCapture(video_path)
            processing_times = []
            
            for i in range(20):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (width, height))
                area_points = [(0, 0), (width, 0), (width, height), (0, height)]
                
                start_time = time.time()
                total_detected, _ = detector.detect_vehicles(frame_resized, area_points)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
            
            cap.release()
            
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
            
            result = {
                'resolution_name': name,
                'width': width,
                'height': height,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
            }
            results.append(result)
            
            print(f"   Processing: {avg_processing_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
        
        assert len(results) > 0, "No resolutions were tested for performance."

    def test_tracking_performance(self, config_and_detector):
        """Test tracking system performance"""
        config, _ = config_and_detector # We'll create new detector instances
        
        print(f"\n🎯 TESTING TRACKING PERFORMANCE")
        print("-" * 40)
        
        video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
        assert video_path is not None, "❌ No test video available for tracking performance test!"
        
        tracking_configs = [
            (False, "Without Tracking"),
            (True, "With Tracking")
        ]
        
        results = []
        for tracking_enabled, config_name in tracking_configs:
            print(f"\n🧪 Testing {config_name}")
            
            test_config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
            test_config.TRACKING_ENABLED = tracking_enabled
            test_detector = VehicleDetector(test_config)
            
            cap = cv2.VideoCapture(video_path)
            processing_times = []
            
            for i in range(50):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                start_time = time.time()
                
                if tracking_enabled:
                    parked_count, moving_count, detections = test_detector.detect_vehicles_with_tracking(
                        frame_resized, area_points)
                else:
                    total_detected, _ = test_detector.detect_vehicles(frame_resized, area_points)
                
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
            
            cap.release()
            
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
            
            result = {
                'tracking_enabled': tracking_enabled,
                'config_name': config_name,
                'avg_processing_time': avg_processing_time,
                'fps': fps,
            }
            results.append(result)
            
            print(f"   Processing: {avg_processing_time:.3f}s")
            print(f"   FPS: {fps:.1f}")
        
        assert len(results) > 0, "No tracking configurations were tested."