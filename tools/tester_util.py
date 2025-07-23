"""
Testing module for vehicle detection system
Provides comprehensive testing and evaluation capabilities
"""

import sys
import cv2
import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Define PROJECT_ROOT for consistent pathing
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

class VehicleDetectionTester:
    """Comprehensive testing class for vehicle detection system"""

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
        
        frame_gt = gt_data.get(str(frame_number), [])
        return [gt_bbox[:4] for gt_bbox in frame_gt] # Return only bbox coords

    
    def __init__(self, detector, config):
        """Initialize tester"""
        self.detector = detector
        self.config = config
        self.test_results = []
        
    def test_single_video(self, video_path: str, area_points: List, 
                         ground_truth_bboxes: Optional[List[List[int]]] = None) -> Dict:
        """Test single video with comprehensive metrics"""
        print(f"🎬 Testing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize metrics
        detection_counts = []
        inference_times = []
        frame_count = 0
        processed_frames = 0

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % self.config.FRAME_SKIP != 0:
                continue
            
            processed_frames += 1
            
            # Resize frame
            frame_resized = cv2.resize(frame, 
                                     (self.config.RESIZE_WIDTH, 
                                      self.config.RESIZE_HEIGHT))
            
            # Scale points
            scale_x = self.config.RESIZE_WIDTH / width
            scale_y = self.config.RESIZE_HEIGHT / height
            scaled_points = [(int(x * scale_x), int(y * scale_y)) 
                           for x, y in area_points]
            
            # Measure inference time
            inference_start = time.time()
            vehicle_count, detections = self.detector.detect_vehicles(
                frame_resized, scaled_points)
            inference_time = time.time() - inference_start
            
            detection_counts.append(vehicle_count)
            inference_times.append(inference_time)
            
            if processed_frames % 30 == 0:
                print(f"   Progress: {processed_frames}/{total_frames//self.config.FRAME_SKIP}")

            # Evaluate accuracy if ground truth is provided for this frame
            if ground_truth_bboxes:
                detected_bboxes = [d['bbox'] for d in detections]

                matched_gt = [False] * len(ground_truth_bboxes)
                matched_det = [False] * len(detected_bboxes)

                for i, gt_box in enumerate(ground_truth_bboxes):
                    for j, det_box in enumerate(detected_bboxes):
                        if not matched_det[j] and self._iou(gt_box, det_box) >= 0.5: # IoU threshold
                            total_true_positives += 1
                            matched_gt[i] = True
                            matched_det[j] = True
                            break
                
                total_false_positives += sum(1 for m in matched_det if not m)
                total_false_negatives += sum(1 for m in matched_gt if not m)
        
        cap.release()
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_detection = np.mean(detection_counts) if detection_counts else 0
        max_detection = max(detection_counts) if detection_counts else 0
        min_detection = min(detection_counts) if detection_counts else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        # Calculate accuracy
        precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            'video_name': video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'avg_detection': round(avg_detection, 2),
            'max_detection': max_detection,
            'min_detection': min_detection,
            'avg_inference_ms': round(avg_inference * 1000, 2),
            'total_time': round(total_time, 2),
            'processing_fps': round(processed_frames / total_time, 2),
            'true_positives': total_true_positives,
            'false_positives': total_false_positives,
            'false_negatives': total_false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: Dict):
        """Print test result"""
        print(f"✅ Results for {result['video_name']}:")
        print(f"   📊 Average detection: {result['avg_detection']} vehicles")
        print(f"   ⚡ Inference time: {result['avg_inference_ms']} ms/frame")
        print(f"   🎯 Precision: {result['precision']:.2f}")
        print(f"   🎯 Recall: {result['recall']:.2f}")
        print(f"   🎯 F1 Score: {result['f1_score']:.2f}")
        print(f"   ⏱️ Processing time: {result['total_time']} seconds")
    
    def run_comprehensive_testing(self):
        """Run testing on all available videos"""
        print("🚀 COMPREHENSIVE TESTING")
        print("=" * 60)
        
        for video_path in self.config.VIDEO_PATHS:
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping: {video_path} (not found)")
                continue
            
            video_name = os.path.basename(video_path)
            
            # Get video dimensions for ground truth loading
            cap = cv2.VideoCapture(video_path)
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Load ground truth bounding boxes for all critical frames in this video
            gt_bboxes_for_video = self._load_ground_truth_bboxes(video_name, -1, img_width, img_height) # -1 to load all frames

            # Use full frame as area_points for comprehensive testing
            area_points = [(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)]
            
            self.test_single_video(video_path, area_points, gt_bboxes_for_video)
        
        self.generate_report()
        self.save_results()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        if not self.test_results:
            print("❌ No test results available")
            return
        
        print(f"📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Videos tested: {len(self.test_results)}")
        print()
        
        # Table header
        print("| Video | Avg Det | Precision | Recall | F1 Score | Proc Time |")
        print("|-------|---------|-----------|--------|----------|-----------|")
        
        for result in self.test_results:
            video_name = os.path.basename(result['video_name'])[:15]
            print(f"| {video_name:<15} | {result['avg_detection']:<9.2f} | {result['precision']:<9.2f} | {result['recall']:<6.2f} | {result['f1_score']:<8.2f} | {result['total_time']:<9.2f} |")
        
        # Summary
        avg_precision = np.mean([r['precision'] for r in self.test_results])
        avg_recall = np.mean([r['recall'] for r in self.test_results])
        avg_f1_score = np.mean([r['f1_score'] for r in self.test_results])
        avg_inference = np.mean([r['avg_inference_ms'] for r in self.test_results])
        avg_fps = np.mean([r['processing_fps'] for r in self.test_results])
        total_time = sum(r['total_time'] for r in self.test_results)
        
        print(f"\nSUMMARY:")
        print(f"   Average Precision: {avg_precision:.2f}")
        print(f"   Average Recall: {avg_recall:.2f}")
        print(f"   Average F1 Score: {avg_f1_score:.2f}")
        print(f"   Average inference time: {avg_inference:.2f} ms/frame")
        print(f"   Average processing FPS: {avg_fps:.2f}")
        print(f"   Total testing time: {total_time:.2f} seconds")
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file"""
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def plot_results(self):
        """Plot test results visualization"""
        if not self.test_results:
            print("❌ No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        videos = [os.path.basename(r['video_name']) for r in self.test_results]
        detections = [r['avg_detection'] for r in self.test_results]
        inference_times = [r['avg_inference_ms'] for r in self.test_results]
        processing_fps = [r['processing_fps'] for r in self.test_results]
        
        # Detection counts
        ax1.bar(videos, detections)
        ax1.set_title('Average Vehicle Detection by Video')
        ax1.set_ylabel('Vehicle Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Inference times
        ax2.bar(videos, inference_times)
        ax2.set_title('Inference Time by Video')
        ax2.set_ylabel('Time (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Processing FPS
        ax3.bar(videos, processing_fps)
        ax3.set_title('Processing FPS by Video')
        ax3.set_ylabel('FPS')
        ax3.tick_params(axis='x', rotation=45)
        
        # Ground truth comparison
        # This part needs to be updated to use the new ground truth bbox system
        # For now, it will be simplified or removed if not directly applicable
        # as the ground_truth parameter is no longer a simple count.
        # I will remove this plot for now as it relies on the old ground_truth format.
        fig.delaxes(ax4) # Remove the empty subplot

        plt.tight_layout()
        plt.savefig('output/test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Results visualization saved to: output/test_results.png")
