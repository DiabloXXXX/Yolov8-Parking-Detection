"""
Real-time monitoring dashboard
"""

import cv2
import numpy as np
import time
from datetime import datetime
import json
import threading
from queue import Queue

class RealTimeMonitor:
    """Real-time monitoring with live dashboard"""
    
    def __init__(self, detector, config):
        self.detector = detector
        self.config = config
        self.stats = {
            'total_vehicles': 0,
            'peak_count': 0,
            'avg_count': 0,
            'detection_history': [],
            'start_time': time.time(),
            'frame_count': 0
        }
        self.running = False
        self.data_queue = Queue()
    
    def start_monitoring(self, video_path: str, area_points: list):
        """Start real-time monitoring with dashboard"""
        self.running = True
        
        # Start dashboard thread
        dashboard_thread = threading.Thread(
            target=self._dashboard_worker, 
            daemon=True
        )
        dashboard_thread.start()
        
        # Start video processing
        self._process_video_realtime(video_path, area_points)
    
    def _process_video_realtime(self, video_path: str, area_points: list):
        """Process video with real-time statistics"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Create dashboard window
        cv2.namedWindow("Real-time Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-time Dashboard", 1200, 800)
        
        frame_count = 0
        detection_history = []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % self.config.FRAME_SKIP != 0:
                continue
            
            # Resize frame
            frame_resized = cv2.resize(frame, 
                                     (self.config.RESIZE_WIDTH, 
                                      self.config.RESIZE_HEIGHT))
            
            # Scale area points
            video_info = self._get_video_info(cap)
            scale_x = self.config.RESIZE_WIDTH / video_info['width']
            scale_y = self.config.RESIZE_HEIGHT / video_info['height']
            
            scaled_points = [(int(x * scale_x), int(y * scale_y)) 
                           for x, y in area_points]
            
            # Detect vehicles
            start_time = time.time()
            vehicle_count, detections = self.detector.detect_vehicles(
                frame_resized, scaled_points)
            inference_time = (time.time() - start_time) * 1000
            
            # Update statistics
            detection_history.append(vehicle_count)
            if len(detection_history) > 100:  # Keep last 100 readings
                detection_history.pop(0)
            
            self.stats.update({
                'current_count': vehicle_count,
                'peak_count': max(self.stats['peak_count'], vehicle_count),
                'avg_count': np.mean(detection_history),
                'detection_history': detection_history.copy(),
                'frame_count': frame_count,
                'inference_time': inference_time
            })
            
            # Create dashboard
            dashboard_frame = self._create_dashboard(frame_resized, detections, scaled_points)
            
            # Display
            cv2.imshow("Real-time Dashboard", dashboard_frame)
            
            # Send data to queue for logging
            self.data_queue.put({
                'timestamp': datetime.now().isoformat(),
                'vehicle_count': vehicle_count,
                'inference_time': inference_time
            })
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Save current statistics
                self._save_realtime_stats()
        
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
    
    def _create_dashboard(self, frame, detections, area_points):
        """Create comprehensive dashboard display"""
        # Create larger canvas
        dashboard_height = 800
        dashboard_width = 1200
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        # Video section (left side)
        video_height = 400
        video_width = int(video_height * frame.shape[1] / frame.shape[0])
        frame_resized = cv2.resize(frame, (video_width, video_height))
        
        # Draw detections on frame
        result_frame = self.detector.draw_detections(
            frame_resized.copy(), detections, 
            self._scale_points(area_points, video_width/frame.shape[1], video_height/frame.shape[0])
        )
        
        dashboard[50:50+video_height, 50:50+video_width] = result_frame
        
        # Statistics panel (right side)
        stats_x = video_width + 100
        stats_y = 50
        
        # Current statistics
        stats_text = [
            f"REAL-TIME VEHICLE MONITORING",
            f"Current Count: {self.stats.get('current_count', 0)}",
            f"Peak Count: {self.stats.get('peak_count', 0)}",
            f"Average Count: {self.stats.get('avg_count', 0):.1f}",
            f"Total Frames: {self.stats.get('frame_count', 0)}",
            f"Inference Time: {self.stats.get('inference_time', 0):.1f}ms",
            f"Runtime: {time.time() - self.stats['start_time']:.0f}s"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = stats_y + i * 40
            font_scale = 1.2 if i == 0 else 0.8
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            
            cv2.putText(dashboard, text, (stats_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Detection history graph
        history = self.stats.get('detection_history', [])
        if len(history) > 1:
            graph_x = stats_x
            graph_y = stats_y + 300
            graph_width = 400
            graph_height = 200
            
            # Draw graph background
            cv2.rectangle(dashboard, (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height), 
                         (50, 50, 50), -1)
            
            # Draw graph
            max_val = max(history) if history else 1
            points = []
            for i, val in enumerate(history):
                x = graph_x + int(i * graph_width / len(history))
                y = graph_y + graph_height - int(val * graph_height / max_val)
                points.append((x, y))
            
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(dashboard, points[i-1], points[i], (0, 255, 0), 2)
            
            # Graph labels
            cv2.putText(dashboard, "Detection History", (graph_x, graph_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "S - Save Stats",
            "P - Pause/Resume"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(dashboard, instruction, (50, dashboard_height - 100 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return dashboard
    
    def _scale_points(self, points, scale_x, scale_y):
        """Scale points for display"""
        return [(int(x * scale_x), int(y * scale_y)) for x, y in points]
    
    def _get_video_info(self, cap):
        """Get video information"""
        return {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS)
        }
    
    def _dashboard_worker(self):
        """Background worker for data logging"""
        log_data = []
        
        while self.running:
            try:
                data = self.data_queue.get(timeout=1)
                log_data.append(data)
                
                # Save every 100 data points
                if len(log_data) >= 100:
                    self._save_log_data(log_data)
                    log_data = []
                    
            except:
                continue
        
        # Save remaining data
        if log_data:
            self._save_log_data(log_data)
    
    def _save_log_data(self, data):
        """Save log data to file"""
        filename = f"output/realtime_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_realtime_stats(self):
        """Save current statistics"""
        filename = f"output/realtime_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"💾 Statistics saved to: {filename}")

print("✅ Real-time monitoring dashboard ready!")
