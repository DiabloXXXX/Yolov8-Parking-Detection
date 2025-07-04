#!/usr/bin/env python3
"""
Quick Accuracy Test for YOLOv8 Vehicle Detection
Simple and fast accuracy testing script
"""

import sys
import os
import time
import cv2
from pathlib import Path
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "01_CORE"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_DIR))

from src.config import Config
from src.vehicle_detector import VehicleDetector

def quick_accuracy_test():
    """Quick accuracy test on one video"""
    print("⚡ QUICK ACCURACY TEST")
    print("=" * 40)
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    try:
        # Load config and detector
        config = Config("02_CONFIG/config.yaml")
        detector = VehicleDetector(config)
        
        # Test on first available video
        video_path = None
        for path in config.VIDEO_PATHS:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            print("❌ No video files found!")
            return
        
        video_name = os.path.basename(video_path)
        print(f"🎬 Testing video: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Test parameters
        test_frames = 50  # Test only 50 frames for speed
        frame_skip = 20   # Skip frames
        
        detection_counts = []
        processing_times = []
        tracking_consistency = []
        
        frame_count = 0
        tested_frames = 0
        
        # Reset tracker
        detector.tracker.reset_tracker()
        
        print("🔄 Running detection test...")
        
        while tested_frames < test_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % frame_skip != 0:
                continue
            
            tested_frames += 1
            
            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            # Measure processing time
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
            
            # Store results
            detection_counts.append(total_detected)
            processing_times.append(processing_time)
            tracking_consistency.append(parked_count + moving_count)
            
            # Progress indicator
            if tested_frames % 10 == 0:
                print(f"   Frame {tested_frames}/{test_frames}: {total_detected} vehicles, {processing_time:.3f}s")
        
        cap.release()
        
        # Get final tracking state - vehicles still being tracked
        if hasattr(detector, 'tracker') and detector.tracker.tracked_vehicles:
            final_parked_vehicles = detector.tracker.get_parked_vehicles()
            final_moving_vehicles = detector.tracker.get_moving_vehicles()
            total_tracked = len(detector.tracker.tracked_vehicles)
            final_parked_count = len(final_parked_vehicles)
            final_moving_count = len(final_moving_vehicles)
        else:
            final_parked_count = int(avg_detected)  # Fallback
            final_moving_count = 0
            total_tracked = final_parked_count
        
        # Calculate results
        avg_detected = np.mean(detection_counts)
        std_detected = np.std(detection_counts)
        avg_processing_time = np.mean(processing_times)
        max_detected = np.max(detection_counts)
        min_detected = np.min(detection_counts)
        
        # Performance metrics
        fps_performance = 1 / avg_processing_time if avg_processing_time > 0 else 0
        detection_stability = 1 - (std_detected / avg_detected) if avg_detected > 0 else 0
        
        print("\n📊 QUICK TEST RESULTS")
        print("-" * 50)
        print(f"📹 Video: {video_name}")
        print(f"🎯 Frames tested: {tested_frames}")
        print(f"🚗 Average vehicles detected per frame: {avg_detected:.1f}")
        print(f"📈 Detection range: {min_detected} - {max_detected}")
        print(f"📊 Detection stability: {detection_stability:.2%}")
        print(f"⚡ Average processing time: {avg_processing_time:.3f}s")
        print(f"🚀 Performance FPS: {fps_performance:.1f}")
        print()
        print("🅿️ FINAL PARKING DETECTION RESULTS:")
        print(f"   🟢 Total kendaraan PARKIR terdeteksi: {final_parked_count}")
        print(f"   🔴 Total kendaraan BERGERAK terdeteksi: {final_moving_count}")
        print(f"   📊 Total kendaraan dalam tracking: {total_tracked}")
        
        # Show individual vehicle details if available
        if hasattr(detector, 'tracker') and detector.tracker.tracked_vehicles:
            print("\n🚗 DETAIL KENDARAAN YANG TERDETEKSI:")
            for vehicle_id, vehicle in detector.tracker.tracked_vehicles.items():
                status = "PARKIR" if vehicle.is_parked else "BERGERAK" 
                duration = vehicle.parking_duration if vehicle.is_parked else vehicle.moving_duration
                print(f"   Vehicle {vehicle_id}: {status} ({duration:.1f}s)")
        
        # Simple rating
        if detection_stability >= 0.8 and fps_performance >= 15:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif detection_stability >= 0.7 and fps_performance >= 10:
            rating = "GOOD ⭐⭐⭐⭐"
        elif detection_stability >= 0.6 and fps_performance >= 5:
            rating = "FAIR ⭐⭐⭐"
        else:
            rating = "NEEDS IMPROVEMENT ⭐⭐"
        
        print(f"\n🏆 Quick Rating: {rating}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if fps_performance < 10:
            print("   - Consider using smaller model (yolov8s.pt)")
            print("   - Increase frame_skip in config")
        if detection_stability < 0.7:
            print("   - Adjust conf_threshold in config")
            print("   - Check min_area and max_area settings")
        if avg_detected < 2:
            print("   - Lower conf_threshold for more detections")
        if avg_detected > 15:
            print("   - Raise conf_threshold to reduce false positives")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def compare_configurations():
    """Compare different configuration settings"""
    print("\n🔄 CONFIGURATION COMPARISON TEST")
    print("=" * 45)
    
    # Test different confidence thresholds
    conf_thresholds = [0.2, 0.3, 0.4, 0.5]
    results = []
    
    for conf in conf_thresholds:
        print(f"\n🎯 Testing confidence threshold: {conf}")
        
        # Temporarily modify config
        config = Config("02_CONFIG/config.yaml")
        config.CONF_THRESHOLD = conf
        
        detector = VehicleDetector(config)
        
        # Quick test on first video
        video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
        if not video_path or not os.path.exists(video_path):
            continue
        
        cap = cv2.VideoCapture(video_path)
        detection_counts = []
        
        for i in range(20):  # Test 20 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            if i % 5 != 0:  # Skip frames
                continue
            
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            total_detected, _ = detector.detect_vehicles(frame_resized, area_points)
            detection_counts.append(total_detected)
        
        cap.release()
        
        avg_detected = np.mean(detection_counts) if detection_counts else 0
        std_detected = np.std(detection_counts) if detection_counts else 0
        
        results.append({
            'conf_threshold': conf,
            'avg_detected': avg_detected,
            'std_detected': std_detected,
            'stability': 1 - (std_detected / avg_detected) if avg_detected > 0 else 0
        })
        
        print(f"   Average detected: {avg_detected:.1f}")
        print(f"   Stability: {results[-1]['stability']:.2%}")
    
    # Show comparison
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("-" * 40)
    print("Conf.  | Avg Det. | Stability")
    print("-" * 40)
    for result in results:
        print(f"{result['conf_threshold']:.1f}    | {result['avg_detected']:6.1f}   | {result['stability']:7.2%}")
    
    # Recommend best configuration
    best_result = max(results, key=lambda x: x['stability'] * (1 if 3 <= x['avg_detected'] <= 10 else 0.5))
    print(f"\n💡 RECOMMENDED: conf_threshold = {best_result['conf_threshold']}")

def main():
    """Main function"""
    try:
        quick_accuracy_test()
        
        # Ask if user wants configuration comparison
        try:
            response = input("\n❓ Run configuration comparison test? (y/n): ").lower()
            if response == 'y':
                compare_configurations()
        except:
            pass  # Skip if input not available
        
        print(f"\n✅ Quick testing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
