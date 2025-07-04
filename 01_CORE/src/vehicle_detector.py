"""
Vehicle Detection module using YOLOv8
Handles vehicle detection and filtering for parking areas
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import time
from typing import List, Tuple, Dict, Optional
from vehicle_tracker import VehicleTracker

class VehicleDetector:
    """Main vehicle detector class with parking tracking"""
    
    def __init__(self, config):
        """Initialize detector with configuration"""
        self.config = config
        self.model = None
        self.class_list = []
        self.detection_history = []
        
        # Initialize tracker
        self.tracker = VehicleTracker(config)
        
        self._load_model()
        self._load_class_list()
        
        print("✅ Vehicle detector with tracking initialized successfully!")
        print(f"🚗 Target vehicles: {list(config.TARGET_VEHICLE_CLASSES.values())}")
        print(f"❌ Excluded: motorcycle, bicycle")
        print(f"⏱️ Parking detection: {self.tracker.min_parking_time}s minimum stationary time")
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.config.MODEL_PATH)
            print(f"✅ Model loaded: {self.config.MODEL_PATH}")
        except Exception as e:
            raise Exception(f"❌ Failed to load model: {e}")
    
    def _load_class_list(self):
        """Load class list from file"""
        try:
            with open(self.config.CLASS_LIST_PATH, 'r', encoding='utf-8') as f:
                self.class_list = f.read().splitlines()
            print(f"✅ Class list loaded: {len(self.class_list)} classes")
        except Exception as e:
            raise Exception(f"❌ Failed to load class list: {e}")
    
    def detect_vehicles(self, frame: np.ndarray, area_points: List[Tuple[int, int]]) -> Tuple[int, List[Dict]]:
        """
        Detect vehicles in the specified area with parking tracking
        
        Args:
            frame: Input frame
            area_points: List of polygon points defining the detection area
            
        Returns:
            Tuple of (parked_vehicle_count, detection_details)
        """
        current_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, 
                           conf=self.config.CONF_THRESHOLD,
                           iou=self.config.IOU_THRESHOLD,
                           verbose=False)
        
        if len(results[0].boxes) == 0:
            # Update tracker with empty detections
            self.tracker.update_tracks([], current_time)
            return 0, []
        
        # Process detections
        detections_data = results[0].boxes.data.cpu().numpy()
        df = pd.DataFrame(detections_data).astype('float')
        
        raw_detections = []
        
        for index, row in df.iterrows():
            x1, y1, x2, y2, conf, cls_id = row[:6]
            x1, y1, x2, y2, cls_id = map(int, [x1, y1, x2, y2, cls_id])
            
            # Get class name
            if cls_id < len(self.class_list):
                class_name = self.class_list[cls_id]
            else:
                continue
            
            # Check if vehicle is target type
            if self._is_target_vehicle(cls_id, class_name):
                # Calculate center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check if vehicle is in detection area
                if self._point_in_polygon(cx, cy, area_points):
                    # Validate detection area
                    area = (x2 - x1) * (y2 - y1)
                    if self.config.MIN_AREA <= area <= self.config.MAX_AREA:
                        raw_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (cx, cy),
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': class_name,
                            'area': area
                        })
        
        # Update tracking with raw detections
        tracked_vehicles = self.tracker.update_tracks(raw_detections, current_time)
        
        # Only count parked vehicles (stationary for ≥2 seconds)
        parked_vehicles = self.tracker.get_parked_vehicles()
        parked_count = len(parked_vehicles)
        
        # Convert tracked vehicles to detection format for compatibility
        detection_details = []
        for vehicle in tracked_vehicles:
            detection = {
                'bbox': vehicle.bbox,
                'center': vehicle.center,
                'confidence': vehicle.confidence,
                'class_id': vehicle.id % 10,  # For class display
                'class_name': vehicle.class_name,
                'area': (vehicle.bbox[2] - vehicle.bbox[0]) * (vehicle.bbox[3] - vehicle.bbox[1]),
                'vehicle_id': vehicle.id,
                'is_parked': vehicle.is_parked,
                'parking_duration': vehicle.parking_duration,
                'positions_history': len(vehicle.positions)
            }
            detection_details.append(detection)
        
        return parked_count, detection_details
    
    def _is_target_vehicle(self, class_id: int, class_name: str) -> bool:
        """
        Check if detected object is a target vehicle type
        
        Args:
            class_id: COCO class ID
            class_name: Class name string
            
        Returns:
            True if target vehicle, False otherwise
        """
        # Method 1: Check by COCO class ID (more accurate)
        if class_id in self.config.TARGET_VEHICLE_CLASSES:
            return True
        
        # Method 2: Fallback keyword check
        class_lower = class_name.lower()
        for keyword in self.config.PARKING_VEHICLE_KEYWORDS:
            if keyword in class_lower:
                # Double check not motorcycle/bike
                if 'motor' not in class_lower and 'bike' not in class_lower:
                    return True
        
        return False
    
    def _point_in_polygon(self, x: int, y: int, polygon_points: List[Tuple[int, int]]) -> bool:
        """
        Check if point is inside polygon using OpenCV
        
        Args:
            x, y: Point coordinates
            polygon_points: List of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        polygon_array = np.array(polygon_points, np.int32)
        result = cv2.pointPolygonTest(polygon_array, (x, y), False)
        return result >= 0
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement techniques for better detection
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Additional sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       area_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            detections: List of detection details
            area_points: Detection area points
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Draw detection area
        if len(area_points) >= 3:
            area_array = np.array(area_points, np.int32)
            cv2.polylines(result_frame, [area_array], True, (255, 255, 0), 2)
            # Skip fillPoly for better performance and avoid errors
        
        # Count parked and moving vehicles
        parked_count = 0
        moving_count = 0
        
        # Draw detections with tracking info
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get tracking info with new logic
            is_parked = detection.get('is_parked', True)  # Default state: DIAM
            is_moving = detection.get('is_moving', False)
            parking_duration = detection.get('parking_duration', 0.0)
            moving_duration = detection.get('moving_duration', 0.0)
            vehicle_id = detection.get('vehicle_id', 0)
            
            # FIXED: Logic untuk menentukan status dan counting
            if is_moving and not is_parked:
                # Status: BERGERAK (is_moving=True, is_parked=False)
                moving_count += 1
                color = (0, 165, 255)  # Orange untuk bergerak
                status = f"BERGERAK {moving_duration:.1f}s"
            elif is_parked and not is_moving:
                # Status: DIAM/PARKIR (is_parked=True, is_moving=False)
                parked_count += 1
                color = (0, 255, 0)  # Green untuk diam/parkir
                if parking_duration >= 2.0:
                    status = f"PARKIR {parking_duration:.1f}s"
                else:
                    status = f"DIAM {parking_duration:.1f}s"
            else:
                # State tidak valid atau transisi - default DIAM
                parked_count += 1
                color = (255, 255, 0)  # Yellow untuk transisi/error
                status = "TRANSISI"
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw vehicle info
            label = f"ID:{vehicle_id} {class_name.upper()} {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw parking status
            cv2.putText(result_frame, status, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw area info
        cv2.putText(result_frame, "Area Deteksi: LAYAR PENUH", (50, 30), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw vehicle counts with parking info
        parked_text = f"Kendaraan Parkir: {parked_count} (>=2 detik)"
        moving_text = f"Kendaraan Bergerak: {moving_count}"
        total_text = f"Total Terdeteksi: {len(detections)}"
        
        cv2.putText(result_frame, parked_text, (50, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(result_frame, moving_text, (50, 100), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)
        cv2.putText(result_frame, total_text, (50, 130), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        
        return result_frame
    
    def detect_vehicles_with_tracking(self, frame: np.ndarray, area_points: List[Tuple[int, int]]) -> Tuple[int, int, List[Dict]]:
        """
        Detect vehicles with tracking menggunakan logic baru:
        - State awal: DIAM 
        - Butuh bergerak ≥1 detik untuk masuk kategori BERGERAK
        - Kembali ke DIAM jika tidak bergerak ≥2 detik
        
        Args:
            frame: Input frame
            area_points: List of polygon points defining the detection area
            
        Returns:
            Tuple of (parked_count, moving_count, all_detections)
        """
        import time
        current_time = time.time()
        
        # Get basic detections (tanpa tracking dulu)
        results = self.model(frame, 
                           conf=self.config.CONF_THRESHOLD,
                           iou=self.config.IOU_THRESHOLD,
                           verbose=False)
        
        if len(results[0].boxes) == 0:
            return 0, 0, []
        
        # Process raw detections
        detections_data = results[0].boxes.data.cpu().numpy()
        df = pd.DataFrame(detections_data).astype('float')
        
        raw_detections = []
        for index, row in df.iterrows():
            x1, y1, x2, y2, conf, cls_id = row[:6]
            x1, y1, x2, y2, cls_id = map(int, [x1, y1, x2, y2, cls_id])
            
            # Get class name
            if cls_id < len(self.class_list):
                class_name = self.class_list[cls_id]
            else:
                continue
            
            # Check if vehicle is target type
            if self._is_target_vehicle(cls_id, class_name):
                # Calculate center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check if vehicle is in detection area
                if self._point_in_polygon(cx, cy, area_points):
                    # Validate detection area
                    area = (x2 - x1) * (y2 - y1)
                    if self.config.MIN_AREA <= area <= self.config.MAX_AREA:
                        raw_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (cx, cy),
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': class_name,
                            'area': area
                        })
        
        # Update tracking dengan raw detections
        tracked_vehicles = self.tracker.update_tracks(raw_detections, current_time)
        
        # Separate parked and moving vehicles berdasarkan logic baru
        parked_vehicles = []
        moving_vehicles = []
        
        for vehicle in tracked_vehicles:
            if vehicle.is_parked and not vehicle.is_moving:
                parked_vehicles.append(vehicle)
            elif vehicle.is_moving:
                moving_vehicles.append(vehicle)
            else:
                # State transisi - defaultkan ke DIAM (state awal)
                parked_vehicles.append(vehicle)
        
        parked_count = len(parked_vehicles)
        moving_count = len(moving_vehicles)
        
        # Convert tracked vehicles ke detection format untuk compatibility
        all_detections = []
        for vehicle in tracked_vehicles:
            detection = {
                'bbox': vehicle.bbox,
                'center': vehicle.center,
                'confidence': vehicle.confidence,
                'class_id': 2,  # Default to car class
                'class_name': vehicle.class_name,
                'area': (vehicle.bbox[2] - vehicle.bbox[0]) * (vehicle.bbox[3] - vehicle.bbox[1]),
                'vehicle_id': vehicle.id,
                'is_parked': vehicle.is_parked,
                'is_moving': vehicle.is_moving,
                'parking_duration': vehicle.parking_duration,
                'moving_duration': vehicle.moving_duration
            }
            all_detections.append(detection)
        
        return parked_count, moving_count, all_detections
    
    def draw_detections_with_tracking(self, frame: np.ndarray, area_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw detection results with tracking information
        
        Args:
            frame: Input frame
            area_points: Detection area points
            
        Returns:
            Frame with drawn detections and tracking info
        """
        return self.tracker.draw_tracks(frame, area_points)
