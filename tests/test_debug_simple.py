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

def test_debug_tracking_status(config_and_detector):
    """Debug tracking logic with real-time status output using pytest"""
    config, detector = config_and_detector
    
    print("\n🔧 DEBUGGING TRACKING STATUS (Pytest)")
    print("=" * 40)
    
    video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
    assert video_path is not None, "❌ No video files found for debugging!"
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"❌ Cannot open {video_path}"
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area_points = [(0, 0), (width, 0), (width, height), (0, height)]
    
    frame_count = 0
    
    print("🎬 Processing frames... (Press Ctrl+C to stop)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            frame_count += 1
            
            if frame_count % 10 != 0:
                continue
            
            frame_resized = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
            
            scale_x = config.RESIZE_WIDTH / width
            scale_y = config.RESIZE_HEIGHT / height
            scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in area_points]
            
            if hasattr(detector, 'tracker') and config.TRACKING_ENABLED:
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, scaled_points)
                
                print(f"\n--- Frame {frame_count} ---")
                print(f"Total: {len(detections)} | Parked: {parked_count} | Moving: {moving_count}")
                
                for detection in detections:
                    vid = detection.get('vehicle_id', 0)
                    is_parked = detection.get('is_parked', True)
                    is_moving = detection.get('is_moving', False)
                    park_dur = detection.get('parking_duration', 0.0)
                    move_dur = detection.get('moving_duration', 0.0)
                    
                    if is_moving and not is_parked:
                        status = f"BERGERAK ({move_dur:.1f}s)"
                        symbol = "🟠"
                    elif is_parked and not is_moving:
                        if park_dur >= 2.0:
                            status = f"PARKIR ({park_dur:.1f}s)"
                        else:
                            status = f"DIAM ({park_dur:.1f}s)"
                        symbol = "🟢"
                    else:
                        status = f"TRANSISI (p:{is_parked}, m:{is_moving})"
                        symbol = "🟡"
                    
                    print(f"  {symbol} Vehicle {vid}: {status}")
                
                time.sleep(0.1)  # Reduced pause for faster testing
            
            if frame_count > 50:  # Stop after a few frames for quick debug
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        cap.release()
        print("✅ Debug completed")