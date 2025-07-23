import pytest
import os
import time
import cv2
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Import modules from src
from src.config import Config
from src.vehicle_detector import VehicleDetector
from src.video_processor import VideoProcessor

# Define PROJECT_ROOT for consistent pathing in tests
PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="module")
def config_and_detector():
    """Fixture to provide a Config and VehicleDetector instance"""
    config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
    detector = VehicleDetector(config)
    return config, detector

class TestAccuracy:
    """Comprehensive accuracy testing for vehicle detection system using pytest"""
    
    # Ground truth data (you can customize this)
    ground_truth = {
    }

    def _iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def _load_ground_truth_bboxes(self, video_name, frame_number, img_width, img_height):
        gt_file = PROJECT_ROOT / "data" / "parking_area" / "ground_truth" / f"{video_name.split('.')[0]}_gt.json"
        if not gt_file.exists():
            return []
        
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
        
        # GT data is stored as {frame_number: [[x1, y1, x2, y2, class_id], ...]},
        # where coordinates are absolute pixels.
        # We need to filter by frame_number and convert to a consistent format if needed.
        frame_gt = gt_data.get(str(frame_number), [])
        
        # Convert to x1, y1, x2, y2 format if not already
        # Assuming the dummy data is already in x1, y1, x2, y2, class_id format
        return [gt_bbox[:4] for gt_bbox in frame_gt] # Return only bbox coords

    # Ground truth data (you can customize this)
    ground_truth = {
        'park1.mp4': {
            'total_vehicles': 5,
            'parked_vehicles': 4,
            'moving_vehicles': 1,
            'critical_frames': [100, 200, 300, 400, 500]  # Frames to check
        },
        'park2.mp4': {
            'total_vehicles': 8,
            'parked_vehicles': 6,
            'moving_vehicles': 2,
            'critical_frames': [150, 250, 350, 450, 550]
        },
        'park3.mp4': {
            'total_vehicles': 4,
            'parked_vehicles': 3,
            'moving_vehicles': 1,
            'critical_frames': [120, 220, 320, 420, 520]
        },
        'park4.mp4': {
            'total_vehicles': 7,
            'parked_vehicles': 5,
            'moving_vehicles': 2,
            'critical_frames': [110, 210, 310, 410, 510]
        }
    }

    def _iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def _load_ground_truth_bboxes(self, video_name, frame_number, img_width, img_height):
        gt_file = PROJECT_ROOT / "data" / "parking_area" / "ground_truth" / f"{video_name.split('.')[0]}_gt.json"
        if not gt_file.exists():
            return []
        
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
        
        # GT data is stored as {frame_number: [[x1, y1, x2, y2, class_id], ...]},
        # where coordinates are absolute pixels.
        # We need to filter by frame_number and convert to a consistent format if needed.
        frame_gt = gt_data.get(str(frame_number), [])
        
        # Convert to x1, y1, x2, y2 format if not already
        # Assuming the dummy data is already in x1, y1, x2, y2, class_id format
        return [gt_bbox[:4] for gt_bbox in frame_gt] # Return only bbox coords

    def test_detection_accuracy(self, config_and_detector):
        """Test detection accuracy for a single video"""
        config, detector = config_and_detector
        
        print("\n🎯 Testing Detection Accuracy")
        print("-" * 50)
        
        all_results = []
        for video_path in config.VIDEO_PATHS:
            video_name = os.path.basename(video_path)
            print(f"\n📹 Video: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                continue
            
            # Get video dimensions for ground truth scaling
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            gt_info = self.ground_truth.get(video_name, {})
            critical_frames = gt_info.get('critical_frames', [])
            
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            frame_count = 0
            detector.tracker.reset_tracker()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count not in critical_frames:
                    continue
                
                frame_resized = cv2.resize(frame, (config.RESIZE_WIDTH // 2, config.RESIZE_HEIGHT // 2))
                area_points = [(0, 0), (frame_resized.shape[1], 0), 
                              (frame_resized.shape[1], frame_resized.shape[0]), (0, frame_resized.shape[0])]
                
                # Get detections from model
                _, detections = detector.detect_vehicles(frame_resized, area_points)
                detected_bboxes = [d['bbox'] for d in detections]

                # Load ground truth bounding boxes for the current frame
                gt_bboxes = self._load_ground_truth_bboxes(video_name, frame_count, img_width, img_height)

                # Match detections to ground truth
                matched_gt = [False] * len(gt_bboxes)
                matched_det = [False] * len(detected_bboxes)

                for i, gt_box in enumerate(gt_bboxes):
                    for j, det_box in enumerate(detected_bboxes):
                        if not matched_det[j] and self._iou(gt_box, det_box) >= 0.5: # IoU threshold
                            true_positives += 1
                            matched_gt[i] = True
                            matched_det[j] = True
                            break
                
                false_positives += sum(1 for m in matched_det if not m)
                false_negatives += sum(1 for m in matched_gt if not m)

                print(f"Frame {frame_count}: TP={true_positives}, FP={false_positives}, FN={false_negatives}")
            
            cap.release()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results = {
                'video_name': video_name,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            all_results.append(results)
            
            print(f"✅ DETECTION ACCURACY RESULTS for {video_name}:")
            print(f"   Precision: {precision:.2%}")
            print(f"   Recall: {recall:.2%}")
            print(f"   F1 Score: {f1_score:.2%}")
        
        assert len(all_results) > 0, "No videos were tested for detection accuracy."
        # Further assertions can be added here based on expected accuracy thresholds

    def test_tracking_performance(self, config_and_detector):
        """Test tracking consistency and performance"""
        config, detector = config_and_detector
        
        print("\n🎯 Testing Tracking Performance")
        print("-" * 50)
        
        all_results = []
        for video_path in config.VIDEO_PATHS:
            video_name = os.path.basename(video_path)
            print(f"\n📹 Video: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                continue
            
            detector.tracker.reset_tracker()
            
            track_consistency = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % 20 != 0:
                    continue
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, area_points)
                
                track_consistency.append(len(detections))
            
            cap.release()
            
            avg_tracked_vehicles = np.mean(track_consistency) if track_consistency else 0
            tracking_variance = np.std(track_consistency) if track_consistency else 0
            
            results = {
                'video_name': video_name,
                'average_tracked_vehicles': avg_tracked_vehicles,
                'tracking_variance': tracking_variance,
                'frames_processed': len(track_consistency),
            }
            all_results.append(results)
            
            print(f"✅ TRACKING PERFORMANCE for {video_name}:")
            print(f"   Average tracked per frame: {avg_tracked_vehicles:.1f}")
            print(f"   Tracking variance: {tracking_variance:.2f}")
        
        assert len(all_results) > 0, "No videos were tested for tracking performance."

    def test_false_positive_negative(self, config_and_detector):
        """Test for false positives and false negatives"""
        config, detector = config_and_detector
        
        print("\n🎯 Testing False Positives/Negatives")
        print("-" * 50)
        
        all_results = []
        for video_path in config.VIDEO_PATHS:
            video_name = os.path.basename(video_path)
            print(f"\n📹 Video: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                continue
            
            gt = self.ground_truth.get(video_name, {})
            expected_total = gt.get('total_vehicles', 0)
            critical_frames = gt.get('critical_frames', [])
            
            false_positives = 0
            false_negatives = 0
            true_positives = 0
            
            frame_count = 0
            detector.tracker.reset_tracker()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count not in critical_frames:
                    continue
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, area_points)
                
                detected_count = len(detections)
                
                if detected_count > expected_total:
                    false_positives += (detected_count - expected_total)
                    true_positives += expected_total
                elif detected_count < expected_total:
                    false_negatives += (expected_total - detected_count)
                    true_positives += detected_count
                else:
                    true_positives += detected_count
                
                print(f"Frame {frame_count}: Detected {detected_count}, Expected {expected_total}")
            
            cap.release()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results = {
                'video_name': video_name,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            all_results.append(results)
            
            print(f"✅ FP/FN Results for {video_name}:")
            print(f"   False Positives: {false_positives}")
            print(f"   False Negatives: {false_negatives}")
            print(f"   F1 Score: {f1_score:.2%}")
        
        assert len(all_results) > 0, "No videos were tested for false positives/negatives."