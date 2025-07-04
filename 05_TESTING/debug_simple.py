#!/usr/bin/env python3
"""
Script debug sederhana untuk melihat status tracking secara real-time
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import time
from src.config import Config
from src.vehicle_detector import VehicleDetector

def main():
    print("🔧 DEBUGGING TRACKING STATUS")
    print("=" * 40)
    
    # Load config
    config = Config()
    detector = VehicleDetector(config)
    
    # Test video
    video_path = "parking_area/video/park2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Full screen area
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
            
            # Process every 10th frame
            if frame_count % 10 != 0:
                continue
            
            # Resize frame
            frame_resized = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
            
            # Scale area points
            scale_x = config.RESIZE_WIDTH / width
            scale_y = config.RESIZE_HEIGHT / height
            scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in area_points]
            
            # Detect with tracking
            if hasattr(detector, 'tracker') and config.TRACKING_ENABLED:
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, scaled_points)
                
                print(f"\n--- Frame {frame_count} ---")
                print(f"Total: {len(detections)} | Parked: {parked_count} | Moving: {moving_count}")
                
                # Show status of each vehicle
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
                
                time.sleep(0.5)  # Pause untuk readability
            
            if frame_count > 500:  # Stop after ~50 seconds
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    cap.release()
    print("✅ Debug completed")

if __name__ == "__main__":
    main()
