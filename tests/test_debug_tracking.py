import pytest
import os
import time
import cv2
from pathlib import Path

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

def test_debug_tracking_logic(config_and_detector):
    """Debug tracking logic with detailed output using pytest"""
    config, detector = config_and_detector
    
    print("\n🐛 DEBUG MODE: Tracking Logic Verification (Pytest)")
    print("=" * 60)
    
    video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
    assert video_path is not None, "❌ No video files found for debugging!"
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"❌ Cannot open video: {video_path}"
    
    detector.tracker.reset_tracker()
    
    frame_count = 0
    start_time = time.time()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area_points = [(0, 0), (width, 0), (width, height), (0, height)]
    
    print(f"📹 Processing: {video_path}")
    print(f"🔲 Area: Full screen ({width}x{height})")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = time.time()
            
            if frame_count % 10 == 0:
                frame_resized = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
                
                scale_x = config.RESIZE_WIDTH / width
                scale_y = config.RESIZE_HEIGHT / height
                scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in area_points]
                
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, scaled_points)
                
                print(f"\n📊 Frame {frame_count} | Time: {current_time - start_time:.1f}s")
                print(f"   Total detections: {len(detections)}")
                print(f"   📗 Parked/Diam: {parked_count}")
                print(f"   📙 Moving: {moving_count}")
                
                for detection in detections:
                    vehicle_id = detection.get('vehicle_id', 0)
                    is_parked = detection.get('is_parked', True)
                    is_moving = detection.get('is_moving', False)
                    parking_duration = detection.get('parking_duration', 0.0)
                    moving_duration = detection.get('moving_duration', 0.0)
                    class_name = detection.get('class_name', 'unknown')
                    
                    status_symbol = "🟢" if is_parked and not is_moving else "🟠" if is_moving else "🟡"
                    
                    if is_moving:
                        print(f"   {status_symbol} ID:{vehicle_id} {class_name} | BERGERAK {moving_duration:.1f}s")
                    elif is_parked:
                        if parking_duration >= 2.0:
                            print(f"   {status_symbol} ID:{vehicle_id} {class_name} | PARKIR {parking_duration:.1f}s")
                        else:
                            print(f"   {status_symbol} ID:{vehicle_id} {class_name} | DIAM {parking_duration:.1f}s")
                    else:
                        print(f"   {status_symbol} ID:{vehicle_id} {class_name} | TRANSISI")
            
            if frame_count >= 100:
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        cap.release()
        print("✅ Debug tracking completed!")