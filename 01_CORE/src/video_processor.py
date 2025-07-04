"""
Video processing module for vehicle detection
Handles video input, area selection, and processing modes
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path

class VideoProcessor:
    """Video processing class for vehicle detection"""
    
    def __init__(self, detector, config):
        """Initialize video processor"""
        self.detector = detector
        self.config = config
        self.area_points = []
        self.current_video_path = None
        self.video_info = {}
        
    def select_video(self, video_path: Optional[str] = None) -> str:
        """Select video file for processing"""
        if video_path and os.path.exists(video_path):
            self.current_video_path = video_path
            print(f"✅ Video selected: {video_path}")
            return video_path
        
        # Show video selection menu
        print("📁 Available videos:")
        valid_videos = []
        
        for i, video in enumerate(self.config.VIDEO_PATHS, 1):
            if os.path.exists(video):
                valid_videos.append(video)
                print(f"{i}. {video}")
            else:
                print(f"{i}. {video} (❌ not found)")
        
        if not valid_videos:
            raise Exception("❌ No valid video files found!")
        
        # Get user selection
        while True:
            try:
                choice = input(f"🎬 Select video (1-{len(self.config.VIDEO_PATHS)}): ")
                video_index = int(choice) - 1
                
                if 0 <= video_index < len(self.config.VIDEO_PATHS):
                    selected_video = self.config.VIDEO_PATHS[video_index]
                    if os.path.exists(selected_video):
                        self.current_video_path = selected_video
                        print(f"✅ Video selected: {selected_video}")
                        return selected_video
                    else:
                        print("❌ Selected video file not found!")
                else:
                    print("❌ Invalid selection!")
            except (ValueError, KeyboardInterrupt):
                # Use default video
                default_video = valid_videos[0]
                self.current_video_path = default_video
                print(f"⚠️ Using default video: {default_video}")
                return default_video
    
    def validate_video(self, video_path: str) -> bool:
        """Validate video file and get information"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        # Validate resolution
        if width < self.config.MIN_WIDTH or height < self.config.MIN_HEIGHT:
            print(f"⚠️ Video resolution ({width}x{height}) below minimum ({self.config.MIN_WIDTH}x{self.config.MIN_HEIGHT})")
            return False
        
        # Store video info
        self.video_info = {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'path': video_path
        }
        
        print(f"✅ Video validated: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s")
        return True
    
    def select_detection_area(self, video_path: str) -> List[Tuple[int, int]]:
        """Interactive area selection for vehicle detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Cannot read first frame")
        
        print("🖱️ Click 4 points to define parking area (ESC to use default)")
        
        points = []
        window_name = f"Select Parking Area - {os.path.basename(video_path)}"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                print(f"   Point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set fullscreen if configured
        if hasattr(self.config, 'FULLSCREEN_MODE') and self.config.FULLSCREEN_MODE:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            display_frame = frame.copy()
            
            # Draw existing points
            for i, pt in enumerate(points):
                cv2.circle(display_frame, pt, 8, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw polygon if we have enough points
            if len(points) >= 3:
                cv2.polylines(display_frame, [np.array(points)], 
                             len(points) == 4, (255, 0, 0), 3)
            
            # Add instructions
            cv2.putText(display_frame, f"Points: {len(points)}/4", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "ESC=Default, ENTER=Done", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("⚠️ Using default area points")
                points = self._get_default_points(video_path)
                break
            elif key == 13 or len(points) == 4:  # ENTER or 4 points
                if len(points) == 4:
                    break
        
        cv2.destroyAllWindows()
        
        if len(points) != 4:
            print("⚠️ Using default area points")
            points = self._get_default_points(video_path)
        
        self.area_points = points
        print(f"🟢 Detection area selected: {points}")
        return points
    
    def _get_default_points(self, video_path: str) -> List[Tuple[int, int]]:
        """Get default detection points for video (Full screen by default)"""
        video_name = os.path.basename(video_path)
        
        # Get actual video dimensions for full screen detection
        if hasattr(self, 'video_info') and self.video_info:
            width = self.video_info['width']
            height = self.video_info['height']
        else:
            # Fallback: get video dimensions
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        
        # Create full screen area using actual video dimensions
        default_fullscreen = [(0, 0), (width, 0), (width, height), (0, height)]
        
        # Use config points if available, otherwise use full screen
        configured_points = self.config.DEFAULT_TEST_POINTS.get(video_name, default_fullscreen)
        
        # If configured points are the generic 1920x1080, use actual video dimensions
        if configured_points == [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]:
            configured_points = default_fullscreen
            
        print(f"🔲 Using detection area: Full screen ({width}x{height})")
        return configured_points
    
    def process_video_realtime(self, video_path: str, area_points: List[Tuple[int, int]]) -> None:
        """Process video in real-time with detection visualization"""
        # Reset tracker untuk video baru (mencegah delay detection)
        if hasattr(self.detector, 'tracker'):
            self.detector.tracker.reset_tracker()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        frame_count = 0
        window_name = f"Vehicle Detection - {os.path.basename(video_path)}"
        
        print(f"🎬 Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for efficiency
                if frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                # Resize frame
                frame_resized = cv2.resize(frame, 
                                         (self.config.RESIZE_WIDTH, 
                                          self.config.RESIZE_HEIGHT))
                
                # Scale area points
                scale_x = self.config.RESIZE_WIDTH / self.video_info['width']
                scale_y = self.config.RESIZE_HEIGHT / self.video_info['height']
                
                scaled_points = [(int(x * scale_x), int(y * scale_y)) 
                               for x, y in area_points]
                
                # Detect vehicles with tracking
                if hasattr(self.detector, 'tracker') and self.config.TRACKING_ENABLED:
                    parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
                        frame_resized, scaled_points)
                else:
                    # Fallback ke deteksi tanpa tracking
                    vehicle_count, detections = self.detector.detect_vehicles(
                        frame_resized, scaled_points)
                    parked_count = vehicle_count
                    moving_count = 0
                
                # Draw results
                result_frame = self.detector.draw_detections(
                    frame_resized, detections, scaled_points)
                
                # Setup window for fullscreen if configured
                if hasattr(self.config, 'FULLSCREEN_MODE') and self.config.FULLSCREEN_MODE:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                cv2.imshow(window_name, result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Video processing completed")
    
    def run_interactive_mode(self, video_path: Optional[str] = None) -> None:
        """Run interactive detection mode"""
        print("🚀 INTERACTIVE VEHICLE DETECTION MODE")
        print("=" * 50)
        
        # Select video
        selected_video = self.select_video(video_path)
        
        # Validate video
        if not self.validate_video(selected_video):
            return
        
        # Select detection area
        area_points = self.select_detection_area(selected_video)
        
        # Process video in real-time
        self.process_video_realtime(selected_video, area_points)
    
    def run_auto_mode(self) -> None:
        """Run automatic detection on all videos with default settings"""
        print("🤖 AUTOMATIC DETECTION MODE")
        print("=" * 50)
        
        for video_path in self.config.VIDEO_PATHS:
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping missing video: {video_path}")
                continue
            
            print(f"📹 Processing: {os.path.basename(video_path)}")
            
            if not self.validate_video(video_path):
                continue
            
            # Use default points
            area_points = self._get_default_points(video_path)
            
            # Process video
            self.process_video_realtime(video_path, area_points)
    
    def run_test_mode(self) -> None:
        """Run comprehensive testing mode"""
        print("🧪 TESTING MODE")
        print("=" * 50)
        
        # Import tester from tools directory
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent / "06_TOOLS"))
        from tester import VehicleDetectionTester
        
        tester = VehicleDetectionTester(self.detector, self.config)
        tester.run_comprehensive_testing()
