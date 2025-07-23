import pytest
import os
import time
import cv2
from pathlib import Path
import numpy as np

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

def test_quick_accuracy(config_and_detector):
    """Quick accuracy test on one video using pytest"""
    config, detector = config_and_detector
    
    print("\n⚡ QUICK ACCURACY TEST (Pytest)")
    print("=" * 40)
    
    video_path = None
    for path in config.VIDEO_PATHS:
        if os.path.exists(path):
            video_path = path
            break
    
    assert video_path is not None, "❌ No video files found for testing!"
    
    video_name = os.path.basename(video_path)
    print(f"🎬 Testing video: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"❌ Cannot open video: {video_path}"
    
    test_frames = 50
    frame_skip = 20
    
    detection_counts = []
    processing_times = []
    
    detector.tracker.reset_tracker()
    
    print("🔄 Running detection test...")
    
    tested_frames = 0
    frame_count = 0
    while tested_frames < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % frame_skip != 0:
            continue
        
        tested_frames += 1
        
        frame_resized = cv2.resize(frame, (640, 480))
        area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
        
        start_time = time.time()
        
        if hasattr(detector, 'tracker') and config.TRACKING_ENABLED:
            parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                frame_resized, area_points)
            total_detected = len(detections)
        else:
            total_detected, detections = detector.detect_vehicles(frame_resized, area_points)
            parked_count = total_detected
            moving_count = 0
        
        processing_time = time.time() - start_time
        
        detection_counts.append(total_detected)
        processing_times.append(processing_time)
        
        if tested_frames % 10 == 0:
            print(f"   Frame {tested_frames}/{test_frames}: {total_detected} vehicles, {processing_time:.3f}s")
    
    cap.release()
    
    assert len(detection_counts) > 0, "No detections recorded during quick test."
    
    avg_detected = np.mean(detection_counts)
    std_detected = np.std(detection_counts)
    avg_processing_time = np.mean(processing_times)
    
    fps_performance = 1 / avg_processing_time if avg_processing_time > 0 else 0
    detection_stability = 1 - (std_detected / avg_detected) if avg_detected > 0 else 0
    
    print("\n📊 QUICK TEST RESULTS:")
    print("-" * 50)
    print(f"📹 Video: {video_name}")
    print(f"🎯 Frames tested: {tested_frames}")
    print(f"🚗 Average vehicles detected per frame: {avg_detected:.1f}")
    print(f"📊 Detection stability: {detection_stability:.2%}")
    print(f"⚡ Average processing time: {avg_processing_time:.3f}s")
    print(f"🚀 Performance FPS: {fps_performance:.1f}")
    print("\n✅ Quick testing completed!")

def test_config_comparison(config_and_detector):
    """Compare different configuration settings using pytest"""
    config, detector = config_and_detector # Use the fixture, though 'detector' is re-initialized in loop
    
    print("\n🔄 CONFIGURATION COMPARISON TEST (Pytest)")
    print("=" * 45)
    
    conf_thresholds = [0.2, 0.3, 0.4, 0.5]
    results = []
    
    video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
    assert video_path is not None, "❌ No video files found for config comparison!"
    
    for conf in conf_thresholds:
        print(f"\n🎯 Testing confidence threshold: {conf}")
        
        temp_config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
        temp_config.CONF_THRESHOLD = conf
        temp_detector = VehicleDetector(temp_config)
        
        cap = cv2.VideoCapture(video_path)
        detection_counts = []
        
        for i in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            
            if i % 5 != 0:
                continue
            
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            total_detected, _ = temp_detector.detect_vehicles(frame_resized, area_points)
            detection_counts.append(total_detected)
        
        cap.release()
        
        avg_detected = np.mean(detection_counts) if detection_counts else 0
        std_detected = np.std(detection_counts) if detection_counts else 0
        
        stability = 1 - (std_detected / avg_detected) if avg_detected > 0 else 0
        
        results.append({
            'conf_threshold': conf,
            'avg_detected': avg_detected,
            'std_detected': std_detected,
            'stability': stability
        })
        
        print(f"   Average detected: {avg_detected:.1f}")
        print(f"   Stability: {stability:.2%}")
    
    print("\n📊 CONFIGURATION COMPARISON RESULTS:")
    print("-" * 40)
    print("Conf.  | Avg Det. | Stability")
    print("-" * 40)
    for result in results:
        print(f"{result['conf_threshold']:.1f}    | {result['avg_detected']:6.1f}   | {result['stability']:.2%}")
    
    best_result = max(results, key=lambda x: x['stability'] * (1 if 3 <= x['avg_detected'] <= 10 else 0.5))
    print(f"\n💡 RECOMMENDED: conf_threshold = {best_result['conf_threshold']}")
    
    # Assert that at least one result was generated
    assert len(results) > 0, "No configuration comparison results were generated."