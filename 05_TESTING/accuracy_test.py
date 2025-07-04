#!/usr/bin/env python3
"""
Accuracy Testing Script for YOLOv8 Vehicle Detection System
Tests detection accuracy, tracking performance, and false positive/negative rates
"""

import sys
import os
import time
import cv2
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "01_CORE"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_DIR))

from src.config import Config
from src.vehicle_detector import VehicleDetector
from src.video_processor import VideoProcessor

class AccuracyTester:
    """Comprehensive accuracy testing for vehicle detection system"""
    
    def __init__(self, config_path=None):
        """Initialize accuracy tester"""
        self.config = Config(config_path or "02_CONFIG/config.yaml")
        self.detector = VehicleDetector(self.config)
        self.processor = VideoProcessor(self.detector, self.config)
        
        # Test results storage
        self.test_results = {
            'detection_accuracy': {},
            'tracking_performance': {},
            'false_positives': {},
            'false_negatives': {},
            'processing_speed': {},
            'overall_stats': {}
        }
        
        # Ground truth data (you can customize this)
        self.ground_truth = {
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
    
    def test_detection_accuracy(self, video_path, video_name):
        """Test detection accuracy for a single video"""
        print(f"\n🎯 Testing Detection Accuracy: {video_name}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Test metrics
        detection_counts = []
        processing_times = []
        tracking_accuracy = []
        frame_count = 0
        
        # Ground truth for this video
        gt = self.ground_truth.get(video_name, {})
        expected_total = gt.get('total_vehicles', 0)
        critical_frames = gt.get('critical_frames', [])
        
        print(f"📊 Expected vehicles: {expected_total}")
        print(f"🎬 Total frames: {total_frames}, FPS: {fps:.1f}")
        
        # Reset tracker for clean test
        self.detector.tracker.reset_tracker()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Test only on sample frames to speed up testing
            if frame_count % 30 != 0 and frame_count not in critical_frames:
                continue
            
            # Resize frame
            frame_resized = cv2.resize(frame, (self.config.RESIZE_WIDTH // 2, self.config.RESIZE_HEIGHT // 2))
            
            # Use full screen area for testing
            area_points = [(0, 0), (frame_resized.shape[1], 0), 
                          (frame_resized.shape[1], frame_resized.shape[0]), (0, frame_resized.shape[0])]
            
            # Measure detection time
            start_time = time.time()
            
            if hasattr(self.detector, 'tracker') and self.config.TRACKING_ENABLED:
                parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
                    frame_resized, area_points)
                total_detected = len(detections)
            else:
                total_detected, detections = self.detector.detect_vehicles(frame_resized, area_points)
                parked_count = total_detected
                moving_count = 0
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            detection_counts.append(total_detected)
            
            # Calculate tracking accuracy for critical frames
            if frame_count in critical_frames:
                accuracy = min(total_detected / expected_total, 1.0) if expected_total > 0 else 0
                tracking_accuracy.append(accuracy)
                print(f"Frame {frame_count}: Detected {total_detected}, Expected {expected_total}, Accuracy: {accuracy:.2f}")
        
        cap.release()
        
        # Get final state from tracker
        final_parked_vehicles = []
        final_moving_vehicles = []
        total_final_vehicles = 0
        
        if hasattr(self.detector, 'tracker') and self.detector.tracker.tracked_vehicles:
            final_parked_vehicles = self.detector.tracker.get_parked_vehicles()
            final_moving_vehicles = self.detector.tracker.get_moving_vehicles()
            total_final_vehicles = len(self.detector.tracker.tracked_vehicles)
        
        # Calculate metrics
        avg_detection = np.mean(detection_counts) if detection_counts else 0
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        avg_accuracy = np.mean(tracking_accuracy) if tracking_accuracy else 0
        detection_variance = np.std(detection_counts) if detection_counts else 0
        
        results = {
            'video_name': video_name,
            'expected_vehicles': expected_total,
            'average_detected': avg_detection,
            'final_parked': len(final_parked_vehicles),
            'final_moving': len(final_moving_vehicles),
            'final_total': total_final_vehicles,
            'detection_accuracy': avg_accuracy,
            'detection_variance': detection_variance,
            'average_processing_time': avg_processing_time,
            'frames_tested': len(detection_counts),
            'fps_performance': 1 / avg_processing_time if avg_processing_time > 0 else 0
        }
        
        print(f"✅ DETECTION RESULTS:")
        print(f"   Average detected per frame: {avg_detection:.1f}")
        print(f"   Detection accuracy: {avg_accuracy:.2%}")
        print(f"   Processing time: {avg_processing_time:.3f}s")
        print(f"   Performance FPS: {results['fps_performance']:.1f}")
        print(f"   Detection variance: {detection_variance:.2f}")
        print()
        print(f"🅿️ FINAL PARKING STATUS:")
        print(f"   🟢 Kendaraan PARKIR terdeteksi: {len(final_parked_vehicles)}")
        print(f"   🔴 Kendaraan BERGERAK terdeteksi: {len(final_moving_vehicles)}")
        print(f"   📊 Total kendaraan dalam sistem: {total_final_vehicles}")
        print(f"   🎯 Expected total: {expected_total}")
        
        # Accuracy assessment
        if total_final_vehicles > 0:
            final_accuracy = min(total_final_vehicles / expected_total, 1.0) if expected_total > 0 else 0
            print(f"   📈 Final accuracy: {final_accuracy:.2%}")
        
        # Show individual vehicles if available
        if final_parked_vehicles:
            print(f"\n🚗 DETAIL KENDARAAN PARKIR:")
            for i, vehicle in enumerate(final_parked_vehicles):
                print(f"   Vehicle {vehicle.id}: PARKIR ({vehicle.parking_duration:.1f}s)")
        
        if final_moving_vehicles:
            print(f"\n🚗 DETAIL KENDARAAN BERGERAK:")
            for i, vehicle in enumerate(final_moving_vehicles):
                print(f"   Vehicle {vehicle.id}: BERGERAK ({vehicle.moving_duration:.1f}s)")
        
        return results
    
    def test_tracking_performance(self, video_path, video_name):
        """Test tracking consistency and performance"""
        print(f"\n🎯 Testing Tracking Performance: {video_name}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Reset tracker
        self.detector.tracker.reset_tracker()
        
        track_consistency = []
        status_changes = defaultdict(int)
        vehicle_lifetimes = defaultdict(list)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Test every 20 frames
            if frame_count % 20 != 0:
                continue
            
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            # Get tracking results
            parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
                frame_resized, area_points)
            
            # Track vehicle consistency
            current_vehicles = set()
            for detection in detections:
                vehicle_id = detection.get('id', None)
                if vehicle_id is not None:
                    current_vehicles.add(vehicle_id)
                    vehicle_lifetimes[vehicle_id].append(frame_count)
                    
                    # Count status changes
                    status = "parked" if detection.get('is_parked', True) else "moving"
                    status_changes[f"{vehicle_id}_{status}"] += 1
            
            track_consistency.append(len(current_vehicles))
        
        cap.release()
        
        # Get final tracking state
        final_tracked_vehicles = {}
        final_parked_count = 0
        final_moving_count = 0
        
        if hasattr(self.detector, 'tracker') and self.detector.tracker.tracked_vehicles:
            final_tracked_vehicles = self.detector.tracker.tracked_vehicles.copy()
            final_parked_count = len(self.detector.tracker.get_parked_vehicles())
            final_moving_count = len(self.detector.tracker.get_moving_vehicles())
        
        # Calculate tracking metrics
        avg_tracked_vehicles = np.mean(track_consistency) if track_consistency else 0
        tracking_variance = np.std(track_consistency) if track_consistency else 0
        
        # Calculate average vehicle lifetime
        avg_lifetime = 0
        if vehicle_lifetimes:
            lifetimes = [len(frames) for frames in vehicle_lifetimes.values()]
            avg_lifetime = np.mean(lifetimes)
        
        results = {
            'video_name': video_name,
            'average_tracked_vehicles': avg_tracked_vehicles,
            'tracking_variance': tracking_variance,
            'unique_vehicles_detected': len(vehicle_lifetimes),
            'average_vehicle_lifetime': avg_lifetime,
            'total_status_changes': sum(status_changes.values()),
            'frames_processed': len(track_consistency),
            'final_parked': final_parked_count,
            'final_moving': final_moving_count,
            'final_total': len(final_tracked_vehicles)
        }
        
        print(f"✅ TRACKING PERFORMANCE:")
        print(f"   Average tracked per frame: {avg_tracked_vehicles:.1f}")
        print(f"   Unique vehicles detected: {len(vehicle_lifetimes)}")
        print(f"   Average lifetime: {avg_lifetime:.1f} frames")
        print(f"   Tracking variance: {tracking_variance:.2f}")
        print(f"   Total status changes: {sum(status_changes.values())}")
        print()
        print(f"🅿️ FINAL TRACKING STATE:")
        print(f"   🟢 Vehicles PARKIR: {final_parked_count}")
        print(f"   🔴 Vehicles BERGERAK: {final_moving_count}")
        print(f"   📊 Total vehicles tracked: {len(final_tracked_vehicles)}")
        
        # Show individual vehicle tracking details
        if final_tracked_vehicles:
            print(f"\n🔍 DETAIL TRACKING SETIAP KENDARAAN:")
            for vehicle_id, vehicle in final_tracked_vehicles.items():
                status = "PARKIR" if vehicle.is_parked else "BERGERAK"
                duration = vehicle.parking_duration if vehicle.is_parked else vehicle.moving_duration
                lifetime_frames = len(vehicle_lifetimes.get(vehicle_id, []))
                print(f"   Vehicle {vehicle_id}: {status} ({duration:.1f}s) - Tracked {lifetime_frames} frames")
        
        return results
    
    def test_false_positive_negative(self, video_path, video_name):
        """Test for false positives and false negatives"""
        print(f"\n🎯 Testing False Positives/Negatives: {video_name}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        gt = self.ground_truth.get(video_name, {})
        expected_total = gt.get('total_vehicles', 0)
        critical_frames = gt.get('critical_frames', [])
        
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        
        frame_count = 0
        self.detector.tracker.reset_tracker()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only test on critical frames
            if frame_count not in critical_frames:
                continue
            
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
                frame_resized, area_points)
            
            detected_count = len(detections)
            
            # Simple false positive/negative calculation
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
        
        # Calculate metrics
        total_predictions = true_positives + false_positives + false_negatives + true_negatives
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'video_name': video_name,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        print(f"✅ FP/FN Results:")
        print(f"   True Positives: {true_positives}")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1_score:.2%}")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive accuracy testing on all videos"""
        print("🧪 COMPREHENSIVE ACCURACY TESTING")
        print("=" * 60)
        print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_results = {
            'detection_results': [],
            'tracking_results': [],
            'fp_fn_results': [],
            'test_timestamp': datetime.now().isoformat(),
            'config_used': {
                'conf_threshold': self.config.CONF_THRESHOLD,
                'iou_threshold': self.config.IOU_THRESHOLD,
                'min_parking_time': self.config.MIN_PARKING_TIME,
                'min_moving_time': self.config.MIN_MOVING_TIME,
                'max_movement_threshold': self.config.MAX_MOVEMENT_THRESHOLD
            }
        }
        
        # Test each video
        for video_path in self.config.VIDEO_PATHS:
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping missing video: {video_path}")
                continue
            
            video_name = os.path.basename(video_path)
            print(f"\n📹 Testing Video: {video_name}")
            
            # Run all tests for this video
            detection_result = self.test_detection_accuracy(video_path, video_name)
            tracking_result = self.test_tracking_performance(video_path, video_name)
            fp_fn_result = self.test_false_positive_negative(video_path, video_name)
            
            if detection_result:
                overall_results['detection_results'].append(detection_result)
            if tracking_result:
                overall_results['tracking_results'].append(tracking_result)
            if fp_fn_result:
                overall_results['fp_fn_results'].append(fp_fn_result)
        
        # Calculate overall statistics
        self.calculate_overall_stats(overall_results)
        
        # Save results
        self.save_test_results(overall_results)
        
        return overall_results
    
    def calculate_overall_stats(self, results):
        """Calculate overall statistics from all test results"""
        print(f"\n📊 OVERALL STATISTICS")
        print("=" * 50)
        
        # Detection accuracy stats
        detection_accuracies = [r['detection_accuracy'] for r in results['detection_results']]
        avg_detection_accuracy = np.mean(detection_accuracies) if detection_accuracies else 0
        
        # Processing performance stats
        processing_times = [r['average_processing_time'] for r in results['detection_results']]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Tracking stats
        tracking_variances = [r['tracking_variance'] for r in results['tracking_results']]
        avg_tracking_variance = np.mean(tracking_variances) if tracking_variances else 0
        
        # F1 scores
        f1_scores = [r['f1_score'] for r in results['fp_fn_results']]
        avg_f1_score = np.mean(f1_scores) if f1_scores else 0
        
        # Precision and Recall
        precisions = [r['precision'] for r in results['fp_fn_results']]
        recalls = [r['recall'] for r in results['fp_fn_results']]
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        overall_stats = {
            'average_detection_accuracy': avg_detection_accuracy,
            'average_processing_time': avg_processing_time,
            'average_tracking_variance': avg_tracking_variance,
            'average_f1_score': avg_f1_score,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'videos_tested': len(results['detection_results'])
        }
        
        results['overall_stats'] = overall_stats
        
        print(f"🎯 Average Detection Accuracy: {avg_detection_accuracy:.2%}")
        print(f"⚡ Average Processing Time: {avg_processing_time:.3f}s")
        print(f"📈 Average F1 Score: {avg_f1_score:.2%}")
        print(f"🎯 Average Precision: {avg_precision:.2%}")
        print(f"🔍 Average Recall: {avg_recall:.2%}")
        print(f"📊 Average Tracking Variance: {avg_tracking_variance:.2f}")
        print(f"📹 Videos Tested: {len(results['detection_results'])}")
        
        # Performance rating
        if avg_f1_score >= 0.9:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif avg_f1_score >= 0.8:
            rating = "VERY GOOD ⭐⭐⭐⭐"
        elif avg_f1_score >= 0.7:
            rating = "GOOD ⭐⭐⭐"
        elif avg_f1_score >= 0.6:
            rating = "FAIR ⭐⭐"
        else:
            rating = "NEEDS IMPROVEMENT ⭐"
        
        print(f"\n🏆 Overall Performance Rating: {rating}")
    
    def save_test_results(self, results):
        """Save test results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure output directory exists
        output_dir = Path(self.config.OUTPUT_DIR).parent / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"accuracy_test_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_file = output_dir / f"accuracy_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("YOLOv8 Vehicle Detection - Accuracy Test Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            overall = results['overall_stats']
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  Detection Accuracy: {overall['average_detection_accuracy']:.2%}\n")
            f.write(f"  Processing Time: {overall['average_processing_time']:.3f}s\n")
            f.write(f"  F1 Score: {overall['average_f1_score']:.2%}\n")
            f.write(f"  Precision: {overall['average_precision']:.2%}\n")
            f.write(f"  Recall: {overall['average_recall']:.2%}\n")
            f.write(f"  Videos Tested: {overall['videos_tested']}\n\n")
            
            f.write("PER-VIDEO RESULTS:\n")
            for result in results['detection_results']:
                f.write(f"  {result['video_name']}:\n")
                f.write(f"    Detection Accuracy: {result['detection_accuracy']:.2%}\n")
                f.write(f"    Processing Time: {result['average_processing_time']:.3f}s\n")
                f.write(f"    Average Detected: {result['average_detected']:.1f}\n\n")
        
        print(f"💾 Test results saved:")
        print(f"   📄 Detailed: {results_file}")
        print(f"   📝 Summary: {summary_file}")

def main():
    """Main function to run accuracy testing"""
    print("🧪 YOLOv8 Vehicle Detection - Accuracy Testing")
    print("=" * 60)
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    try:
        # Initialize tester
        tester = AccuracyTester()
        
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        print(f"\n✅ Testing completed successfully!")
        print(f"📊 Check output directory for detailed results")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
