#!/usr/bin/env python3
"""
Test movement detection dengan threshold yang sangat rendah
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import time
from src.config import Config
from src.vehicle_detector import VehicleDetector

def main():
    print("🔧 TESTING MOVEMENT DETECTION")
    print("=" * 40)
    
    config = Config()
    print(f"Movement threshold: {config.MAX_MOVEMENT_THRESHOLD}px")
    
    detector = VehicleDetector(config)
    
    # Test dengan video yang berbeda
    video_path = "parking_area/video/park1.mp4"  # Video berbeda
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area_points = [(0, 0), (width, 0), (width, height), (0, height)]
    
    frame_count = 0
    
    print(f"🎬 Processing {video_path} with very low threshold...")
    
    try:
        for i in range(50):  # Process 50 frames only
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame for more data
            if frame_count % 3 != 0:
                continue
            
            # Resize frame
            frame_resized = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
            
            # Scale area points
            scale_x = config.RESIZE_WIDTH / width
            scale_y = config.RESIZE_HEIGHT / height
            scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in area_points]
            
            # Detect with tracking
            parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                frame_resized, scaled_points)
            
            print(f"\nFrame {frame_count}: P:{parked_count} M:{moving_count} T:{len(detections)}")
            
            # Show detailed info
            for detection in detections:
                vid = detection.get('vehicle_id', 0)
                is_parked = detection.get('is_parked', True)
                is_moving = detection.get('is_moving', False)
                move_dur = detection.get('moving_duration', 0.0)
                
                print(f"  V{vid}: parked={is_parked}, moving={is_moving}, move_dur={move_dur:.1f}s")
            
            # Stop early if we find moving vehicles
            if moving_count > 0:
                print("🎯 Found moving vehicles!")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    cap.release()
    print("✅ Test completed")

if __name__ == "__main__":
    main()
