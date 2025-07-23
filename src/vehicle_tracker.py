"""
Vehicle tracking module for parking detection
Tracks vehicles to differentiate between parked and moving vehicles
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TrackedVehicle:
    """Data class for tracked vehicle information"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    class_name: str
    confidence: float
    first_seen: float
    last_seen: float
    positions: List[Tuple[int, int]]  # History of center positions
    is_parked: bool = True  # Default state: DIAM/PARKIR (state awal)
    is_moving: bool = False  # Status bergerak
    parking_duration: float = 0.0
    moving_duration: float = 0.0  # Durasi bergerak
    stationary_start: float = 0.0  # Waktu mulai diam
    movement_start: float = 0.0  # Waktu mulai bergerak
    last_movement: float = 0.0  # Terakhir kali bergerak

class VehicleTracker:
    """Vehicle tracking class for parking detection"""
    
    def __init__(self, config):
        """Initialize tracker"""
        self.config = config
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.next_id = 0
        self.last_reset_time = time.time()
        
        # Tracking parameters from config
        self.max_distance_threshold = getattr(config, 'MAX_DISTANCE_THRESHOLD', 100)
        self.min_parking_time = getattr(config, 'MIN_PARKING_TIME', 2.0)
        self.min_moving_time = getattr(config, 'MIN_MOVING_TIME', 0.8)  # Increased to reduce sensitivity
        self.max_movement_threshold = getattr(config, 'MAX_MOVEMENT_THRESHOLD', 10)  # Increased threshold
        self.cleanup_timeout = getattr(config, 'CLEANUP_TIMEOUT', 5.0)
        
    def reset_tracker(self):
        """Reset tracker for new video to avoid delay detection issues"""
        self.tracked_vehicles.clear()
        self.next_id = 0
        print("🔄 Tracker reset for new video - clearing all buffers and states")
        
        # Juga reset timer dan buffer jika ada
        self.last_reset_time = time.time()
        
    def update_tracks(self, detections: List[Dict], current_time: float) -> List[TrackedVehicle]:
        """
        Update vehicle tracks with new detections
        
        Args:
            detections: List of current frame detections
            current_time: Current timestamp
            
        Returns:
            List of currently tracked vehicles
        """
        # Convert detections to centers for matching
        detection_centers = [(det['center'], det) for det in detections]
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for vehicle_id, tracked_vehicle in list(self.tracked_vehicles.items()):
            best_match_idx = None
            best_distance = float('inf')
            
            # Find closest detection to this tracked vehicle
            for i, (center, detection) in enumerate(detection_centers):
                if i not in unmatched_detections:
                    continue
                    
                distance = self._calculate_distance(tracked_vehicle.center, center)
                
                if distance < self.max_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Update existing track
                detection = detections[best_match_idx]
                self._update_vehicle(tracked_vehicle, detection, current_time)
                matched_tracks.append(vehicle_id)
                unmatched_detections.remove(best_match_idx)
        
        # Create new tracks for unmatched detections
        for i in unmatched_detections:
            detection = detections[i]
            new_vehicle = TrackedVehicle(
                id=self.next_id,
                bbox=detection['bbox'],
                center=detection['center'],
                class_name=detection['class_name'],
                confidence=detection['confidence'],
                first_seen=current_time,
                last_seen=current_time,
                positions=[detection['center']],
                is_parked=True,  # State awal: DIAM
                is_moving=False,
                stationary_start=current_time,  # Mulai diam dari awal
                movement_start=0.0,
                last_movement=0.0
            )
            self.tracked_vehicles[self.next_id] = new_vehicle
            self.next_id += 1
        
        # Remove old tracks that haven't been seen recently
        self._cleanup_old_tracks(current_time)
        
        # Update parking status for all vehicles
        self._update_parking_status(current_time)
        
        return list(self.tracked_vehicles.values())
    
    def _update_vehicle(self, vehicle: TrackedVehicle, detection: Dict, current_time: float):
        """Update existing vehicle with new detection"""
        vehicle.bbox = detection['bbox']
        vehicle.center = detection['center']
        vehicle.confidence = detection['confidence']
        vehicle.last_seen = current_time
        
        # Add position to history (keep only recent positions)
        vehicle.positions.append(detection['center'])
        if len(vehicle.positions) > 30:  # Keep last 30 positions (1 second at 30fps)
            vehicle.positions.pop(0)
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _cleanup_old_tracks(self, current_time: float):
        """Remove tracks that haven't been updated recently"""
        to_remove = []
        for vehicle_id, vehicle in self.tracked_vehicles.items():
            if current_time - vehicle.last_seen > self.cleanup_timeout:
                to_remove.append(vehicle_id)
        
        for vehicle_id in to_remove:
            del self.tracked_vehicles[vehicle_id]
    
    def _update_parking_status(self, current_time: float):
        """
        Update parking status dengan logic yang BENAR:
        - State awal: DIAM (is_parked=True, is_moving=False)
        - Jika bergerak ≥0.5 detik: BERGERAK (is_moving=True, is_parked=False)
        - Jika diam ≥2 detik: PARKIR (is_parked=True, is_moving=False)
        """
        for vehicle in self.tracked_vehicles.values():
            # Pastikan timer sudah diinisialisasi
            if not hasattr(vehicle, 'stationary_start') or vehicle.stationary_start == 0:
                vehicle.stationary_start = vehicle.first_seen
            if not hasattr(vehicle, 'movement_start') or vehicle.movement_start == 0:
                vehicle.movement_start = 0.0
            
            if len(vehicle.positions) >= 2:
                # Calculate movement over recent positions
                recent_positions = vehicle.positions[-10:]  # Diperluas dari 5 ke 10 posisi untuk lebih akurat
                if len(recent_positions) >= 2:
                    max_movement = self._calculate_max_movement(recent_positions)
                    
                    # Filter untuk goyangan kamera - gunakan threshold yang lebih ketat
                    if len(recent_positions) >= 3:
                        direction_consistency = self._check_movement_consistency(recent_positions)
                    else:
                        direction_consistency = 1.0
                        
                    # Pergerakan harus melewati threshold DAN konsisten untuk dianggap bergerak
                    is_currently_moving = (max_movement > self.max_movement_threshold) and (direction_consistency > 0.6)
                    
                    print(f"Vehicle {vehicle.id}: movement={max_movement:.1f}px, consistency={direction_consistency:.2f}, threshold={self.max_movement_threshold}px, is_moving={is_currently_moving}")
                    
                    if is_currently_moving:
                        # KENDARAAN BERGERAK SAAT INI
                        
                        # PERBAIKAN: Jika belum pernah bergerak atau sudah lama tidak bergerak
                        if vehicle.movement_start == 0.0 or (current_time - vehicle.last_movement) > 1.0:
                            vehicle.movement_start = current_time
                            print(f"Vehicle {vehicle.id}: START moving timer at {current_time}")
                        
                        # Update last movement time
                        vehicle.last_movement = current_time
                        
                        # Hitung durasi bergerak
                        moving_duration = current_time - vehicle.movement_start
                        vehicle.moving_duration = moving_duration
                        
                        # PERBAIKAN: Perlu bergerak ≥0.8 detik secara konsisten untuk dianggap BERGERAK
                        # Ini mengurangi sensitivitas terhadap goyangan kamera
                        if moving_duration >= self.min_moving_time:
                            # Hanya update status jika pergerakan konsisten (tidak random/goyang)
                            vehicle.is_moving = True
                            vehicle.is_parked = False
                            vehicle.parking_duration = 0.0
                            # Reset stationary timer saat bergerak
                            vehicle.stationary_start = current_time
                            print(f"Vehicle {vehicle.id}: STATUS = BERGERAK ({moving_duration:.1f}s) ✅")
                        else:
                            # Transisi: bergerak tapi belum cukup lama
                            vehicle.is_moving = False
                            vehicle.is_parked = True
                            print(f"Vehicle {vehicle.id}: STATUS = TRANSISI (moving {moving_duration:.1f}s, need {self.min_moving_time}s) ⏳")
                    
                    else:
                        # KENDARAAN TIDAK BERGERAK SAAT INI
                        
                        # PERBAIKAN: Jika sebelumnya bergerak, mulai timer diam
                        if vehicle.is_moving or vehicle.movement_start > 0:
                            vehicle.stationary_start = current_time
                            vehicle.is_moving = False
                            vehicle.movement_start = 0.0  # Reset movement timer
                            print(f"Vehicle {vehicle.id}: STOP moving, start stationary timer at {current_time}")
                        
                        # Hitung durasi diam
                        stationary_duration = current_time - vehicle.stationary_start
                        vehicle.parking_duration = stationary_duration
                        vehicle.moving_duration = 0.0
                        
                        # Set status DIAM/PARKIR
                        vehicle.is_parked = True
                        vehicle.is_moving = False
                        
                        if stationary_duration >= self.min_parking_time:
                            print(f"Vehicle {vehicle.id}: STATUS = PARKIR ({stationary_duration:.1f}s) 🅿️")
                        else:
                            print(f"Vehicle {vehicle.id}: STATUS = DIAM ({stationary_duration:.1f}s) ⏸️")
            
            else:
                # Kendaraan baru atau data tidak cukup - default DIAM
                vehicle.is_parked = True
                vehicle.is_moving = False
                vehicle.parking_duration = current_time - vehicle.first_seen
                vehicle.moving_duration = 0.0
                # print(f"Vehicle {vehicle.id}: STATUS = DIAM (new/insufficient data)")
    
    def _calculate_max_movement(self, positions: List[Tuple[int, int]]) -> float:
        """
        Calculate maximum movement distance from recent positions
        Uses multiple methods to detect real movement vs camera shake
        """
        if len(positions) < 2:
            return 0.0
        
        # Method 1: Distance between first and last position (more practical)
        first_pos = positions[0]
        last_pos = positions[-1]
        direct_distance = self._calculate_distance(first_pos, last_pos)
        
        # Method 2: Maximum distance between any two consecutive positions
        max_consecutive_distance = 0.0
        avg_consecutive_distance = 0.0
        total_distance = 0.0
        
        for i in range(len(positions) - 1):
            distance = self._calculate_distance(positions[i], positions[i + 1])
            max_consecutive_distance = max(max_consecutive_distance, distance)
            total_distance += distance
        
        # Calculate average movement (helps filter out camera shake)
        avg_consecutive_distance = total_distance / (len(positions) - 1) if len(positions) > 1 else 0
        
        # Check for consistent direction (untuk membedakan gerakan asli vs goyangan)
        direction_consistency = self._check_movement_consistency(positions)
        
        # Filter jitters/goyangan:
        # 1. Goyangan cenderung memiliki jarak konsekutif yang tinggi tapi jarak langsung yang rendah
        # 2. Goyangan memiliki arah yang tidak konsisten
        is_likely_jitter = (direct_distance < 3.0 and max_consecutive_distance > 5.0 and direction_consistency < 0.7)
        
        # Use weighted score to determine real movement
        # Prioritize consistent directional movement over random jittering
        if is_likely_jitter:
            # Ini kemungkinan besar goyangan kamera, kurangi skor pergerakan secara signifikan
            movement = direct_distance * 0.5
        elif direction_consistency > 0.7:  # Consistent direction
            movement = max(direct_distance, max_consecutive_distance)
        else:
            # Jika arah tidak konsisten (kemungkinan goyangan), kurangi skor pergerakan
            movement = max(direct_distance, max_consecutive_distance) * direction_consistency
        
        return movement
        
    def _check_movement_consistency(self, positions: List[Tuple[int, int]]) -> float:
        """
        Check how consistent the movement direction is
        Returns 0.0-1.0 score (0 = random movement, 1 = perfectly consistent direction)
        """
        if len(positions) < 3:
            return 1.0  # Not enough data, assume consistent
            
        # Calculate primary direction based on first and last position
        first_pos = positions[0]
        last_pos = positions[-1]
        primary_dx = last_pos[0] - first_pos[0]
        primary_dy = last_pos[1] - first_pos[1]
        
        # Normalize primary direction vector
        primary_magnitude = math.sqrt(primary_dx**2 + primary_dy**2)
        if primary_magnitude == 0:
            return 0.0  # No movement
            
        primary_dx /= primary_magnitude
        primary_dy /= primary_magnitude
        
        # Check how consistent each step is with the primary direction
        consistency_scores = []
        
        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]
            
            step_dx = pos2[0] - pos1[0]
            step_dy = pos2[1] - pos1[1]
            
            step_magnitude = math.sqrt(step_dx**2 + step_dy**2)
            if step_magnitude == 0:
                continue  # Skip zero movement
                
            # Normalize step vector
            step_dx /= step_magnitude
            step_dy /= step_magnitude
            
            # Dot product gives cosine of angle between vectors (-1 to 1)
            dot_product = (primary_dx * step_dx) + (primary_dy * step_dy)
            
            # Convert to 0-1 score (1 = same direction, 0 = opposite direction)
            consistency = (dot_product + 1) / 2
            consistency_scores.append(consistency)
        
        # Return average consistency score
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        return 1.0  # No movement segments, assume consistent
    
    def get_parked_vehicles(self) -> List[TrackedVehicle]:
        """Get list of vehicles that are currently parked"""
        return [vehicle for vehicle in self.tracked_vehicles.values() if vehicle.is_parked]
    
    def get_moving_vehicles(self) -> List[TrackedVehicle]:
        """Get list of vehicles that are currently moving"""
        return [vehicle for vehicle in self.tracked_vehicles.values() if not vehicle.is_parked]
    
    def draw_tracks(self, frame: np.ndarray, area_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            area_points: Detection area points
            
        Returns:
            Frame with tracking visualization
        """
        result_frame = frame.copy()
        
        # Draw detection area
        if len(area_points) >= 3:
            area_array = np.array(area_points, np.int32)
            cv2.polylines(result_frame, [area_array], True, (255, 0, 2), 2)
        
        parked_count = 0
        moving_count = 0
        
        # Draw tracked vehicles
        for vehicle in self.tracked_vehicles.values():
            x1, y1, x2, y2 = vehicle.bbox
            cx, cy = vehicle.center
            
            # Choose color based on parking status
            if vehicle.is_parked:
                if vehicle.parking_duration >= self.min_parking_time:
                    color = (0, 255, 0)  # Green for parked
                    status = "PARKIR"
                else:
                    color = (0, 255, 255)  # Yellow for temporary stop
                    status = "DIAM"
                parked_count += 1
            else:
                color = (0, 165, 255)  # Orange for moving
                status = "BERGERAK"
                moving_count += 1
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(result_frame, (cx, cy), 4, color, -1)
            
            # Draw tracking trail (recent positions)
            if len(vehicle.positions) > 1:
                for i in range(1, min(len(vehicle.positions), 10)):
                    pt1 = vehicle.positions[i-1]
                    pt2 = vehicle.positions[i]
                    alpha = i / 10.0  # Fade trail
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(result_frame, pt1, pt2, trail_color, 2)
            
            # Draw vehicle info
            info_text = f"ID:{vehicle.id} {status}"
            if vehicle.is_parked:
                info_text += f" {vehicle.parking_duration:.1f}s"
            else:
                info_text += f" {vehicle.moving_duration:.1f}s"
            
            # Background for text
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_frame, (x1, y1-25), (x1 + text_size[0] + 5, y1), color, -1)
            cv2.putText(result_frame, info_text, (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw summary
        summary_text = f"PARKIR: {parked_count} | BERGERAK: {moving_count}"
        cv2.putText(result_frame, summary_text, (50, 60), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.putText(result_frame, summary_text, (50, 60), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        return result_frame
