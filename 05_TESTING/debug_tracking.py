"""
Debug script untuk memverifikasi logic tracking
Menampilkan detail status kendaraan dalam real-time
"""

import sys
import os
sys.path.append('src')

from config import Config
from vehicle_detector import VehicleDetector
import cv2
import time

def debug_tracking():
    """Debug tracking logic dengan output detail"""
    print("🐛 DEBUG MODE: Tracking Logic Verification")
    print("=" * 60)
    
    # Load config dan detector
    config = Config()
    detector = VehicleDetector(config)
    
    # Test dengan video pertama
    video_path = "parking_area/video/park2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Reset tracker
    detector.tracker.reset_tracker()
    
    frame_count = 0
    start_time = time.time()
    
    # Area full screen
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area_points = [(0, 0), (width, 0), (width, height), (0, height)]
    
    print(f"📹 Processing: {video_path}")
    print(f"🔲 Area: Full screen ({width}x{height})")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = time.time()
        
        # Process setiap 10 frame untuk debug yang lebih jelas
        if frame_count % 10 == 0:
            # Resize frame
            frame_resized = cv2.resize(frame, (1920, 1080))
            
            # Scale area points
            scale_x = 1920 / width
            scale_y = 1080 / height
            scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in area_points]
            
            # Detect dengan tracking
            parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                frame_resized, scaled_points)
            
            # Debug output
            print(f"\n📊 Frame {frame_count} | Time: {current_time - start_time:.1f}s")
            print(f"   Total detections: {len(detections)}")
            print(f"   📗 Parked/Diam: {parked_count}")
            print(f"   📙 Moving: {moving_count}")
            
            # Detail per kendaraan
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
        
        # Break setelah 100 frame untuk debug
        if frame_count >= 100:
            break
    
    cap.release()
    print("\n✅ Debug tracking completed!")
    print(f"📊 Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    debug_tracking()
