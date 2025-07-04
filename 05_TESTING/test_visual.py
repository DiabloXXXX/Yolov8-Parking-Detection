#!/usr/bin/env python3
"""
Visual test for movement detection validation
Shows real-time status transitions with visual feedback
"""

import cv2
import time
from src.config import Config
from src.vehicle_detector import VehicleDetector

def main():
    print("🎯 VISUAL MOVEMENT DETECTION TEST")
    print("========================================")
    
    # Load configuration
    config = Config()
    
    # Initialize detector
    detector = VehicleDetector(config)
    
    # Test on one video
    video_path = "parking_area/video/park1.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    print(f"🎬 Testing movement detection on: {video_path}")
    print("📋 Watch for status changes:")
    print("   🟢 Green: DIAM/PARKIR (stationary)")
    print("   🟠 Orange: BERGERAK (moving)")
    print("   🟡 Yellow: TRANSISI (transition)")
    print("   Press 'q' to quit, 'space' to pause")
    
    frame_count = 0
    paused = False
    
    # Get video dimensions for full screen detection
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Full screen detection area
    area_points = [
        (0, 0),
        (video_width, 0),
        (video_width, video_height),
        (0, video_height)
    ]
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video reached")
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % 2 != 0:  # Process every 2nd frame
                continue
        
        # Detect vehicles with tracking
        parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(frame, area_points)
        
        # Draw detections with tracking info
        result_frame = detector.draw_detections(frame, detections, area_points)
        
        # Add debug info overlay
        debug_text = f"Frame: {frame_count} | Parked: {parked_count} | Moving: {moving_count}"
        cv2.putText(result_frame, debug_text, (50, video_height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Print status changes to console
        if detections:
            for detection in detections:
                is_moving = detection.get('is_moving', False)
                is_parked = detection.get('is_parked', True)
                vehicle_id = detection.get('vehicle_id', 0)
                
                if is_moving and not is_parked:
                    status_emoji = "🟠"
                    status_text = "BERGERAK"
                elif is_parked and not is_moving:
                    status_emoji = "🟢"
                    if detection.get('parking_duration', 0) >= 2.0:
                        status_text = "PARKIR"
                    else:
                        status_text = "DIAM"
                else:
                    status_emoji = "🟡"
                    status_text = "TRANSISI"
                
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"{status_emoji} Vehicle {vehicle_id}: {status_text}")
        
        # Display frame
        cv2.imshow('Movement Detection Test', result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to pause/unpause
            paused = not paused
            print("⏸️ Paused" if paused else "▶️ Resumed")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Visual test completed")

if __name__ == "__main__":
    main()
