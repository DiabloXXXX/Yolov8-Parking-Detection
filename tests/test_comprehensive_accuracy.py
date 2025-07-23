import pytest
import os
import time
import cv2
import json
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

# Import modules from src
from src.config import Config
from src.vehicle_detector import VehicleDetector

# Define PROJECT_ROOT for consistent pathing in tests
PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="module")
def config_and_detector():
    """Fixture to provide a Config and VehicleDetector instance"""
    config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
    detector = VehicleDetector(config)
    return config, detector

class TestComprehensiveAccuracy:
    """Advanced testing class with multiple runs and full screen mode using pytest"""

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


    def test_comprehensive_accuracy_all_videos(self, config_and_detector):
        """Run comprehensive testing on all videos (multi-run, auto video selection, save CSV)"""
        import csv
        config, detector = config_and_detector
        runs_per_video = 5  # You can change this value as needed
        print("\n🚀 COMPREHENSIVE ACCURACY TESTING - FULL SCREEN MODE (Pytest)")
        print("=" * 80)
        print(f"🔄 {runs_per_video} runs per video for maximum accuracy")
        print(f"📺 Full resolution processing (no frame skipping)")
        print("=" * 80)
        test_results = []
        for video_path in config.VIDEO_PATHS:
            if not os.path.exists(video_path):
                print(f"⚠️ Skipping missing video: {video_path}")
                continue
            video_name = os.path.basename(video_path)
            cap = cv2.VideoCapture(video_path)
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            area_points = [(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)]
            result = self._test_video_multiple_runs(detector, config, video_path, area_points, runs_per_video)
            if result:
                result['runs_per_video'] = runs_per_video  # Store for reporting
                test_results.append(result)
                print(f"✅ Completed testing: {video_name}")
            else:
                print(f"❌ Failed testing: {video_name}")
        assert len(test_results) > 0, "No videos were tested in comprehensive accuracy test."
        self._generate_comprehensive_report(test_results, runs_per_video)
        self._save_comprehensive_results(test_results, config.OUTPUT_DIR, runs_per_video)
        # Save summary to CSV
        csv_path = Path(config.OUTPUT_DIR) / "comprehensive_test_results" / "comprehensive_accuracy_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Video", "Avg Detection", "Avg Parked", "Avg Moving", "Consistency", "Reliability", "FPS", "Runs"
            ])
            for result in test_results:
                video_name = os.path.basename(result['video_name'])
                avg_parked = result.get('avg_parked', 0.0)
                avg_moving = result.get('avg_moving', 0.0)
                writer.writerow([
                    video_name,
                    result['overall_avg_detection'],
                    avg_parked,
                    avg_moving,
                    result['consistency_score'],
                    result['reliability_level'],
                    result['avg_processing_fps'],
                    runs_per_video
                ])
        print(f"\n💾 CSV summary saved to: {csv_path}")

    def _test_video_multiple_runs(self, detector, config, video_path: str, area_points: List[Tuple[int, int]], runs_per_video: int) -> Dict:
        """Test single video with multiple runs for better accuracy"""
        print(f"\n🎬 COMPREHENSIVE TESTING: {video_path}")
        print(f"🔄 Running {runs_per_video} tests per video for accuracy")
        print("=" * 70)
        all_run_results = []
        for run_num in range(1, runs_per_video + 1):
            print(f"\n🏃 Run {run_num}/{runs_per_video}")
            run_result = self._single_test_run(detector, config, video_path, area_points, run_num)
            if run_result:
                all_run_results.append(run_result)
                print(f"   ✅ Run {run_num}: {run_result['avg_detection']:.1f} vehicles detected")
            else:
                print(f"   ❌ Run {run_num}: Failed")
        if not all_run_results:
            print("❌ All test runs failed!")
            return None
        combined_result = self._calculate_comprehensive_stats(video_path, all_run_results)
        return combined_result
    
    def _single_test_run(self, detector, config, video_path: str, area_points: List[Tuple[int, int]], 
                        run_num: int) -> Dict:
        """Single test run with full screen processing"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Get original video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use full resolution for better accuracy
        detection_counts = []
        inference_times = []
        
        # Metrics for accuracy
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        frame_count = 0
        processed_frames = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame for full screen mode (no skipping)
            processed_frames += 1
            
            # Use original resolution for maximum accuracy
            inference_start = time.time()
            vehicle_count, detections = detector.detect_vehicles(frame, area_points)
            inference_time = time.time() - inference_start
            
            detection_counts.append(vehicle_count)
            inference_times.append(inference_time)

            # Load ground truth bounding boxes for the current frame
            # Each gt_bbox: [xmin, ymin, xmax, ymax, class_id]
            gt_bboxes = self._load_ground_truth_bboxes(os.path.basename(video_path), frame_count, original_width, original_height)
            # Each detection: {'bbox': [xmin, ymin, xmax, ymax], 'class': class_id}
            detected_bboxes = [(d['bbox'], d.get('class', None)) for d in detections]

            # COCO class indices for 4-wheeled vehicles: car=2, bus=5, truck=7, train=6
            VEHICLE_CLASSES = {2, 5, 6, 7}

            matched_gt = [False] * len(gt_bboxes)
            matched_det = [False] * len(detected_bboxes)

            for i, gt in enumerate(gt_bboxes):
                # gt: [xmin, ymin, xmax, ymax, class_id] or [xmin, ymin, xmax, ymax] if no class
                if len(gt) == 5:
                    gt_box, gt_class = gt[:4], gt[4]
                else:
                    gt_box, gt_class = gt[:4], None
                for j, (det_box, det_class) in enumerate(detected_bboxes):
                    if not matched_det[j]:
                        # Only match if both classes are in VEHICLE_CLASSES and equal
                        if gt_class is not None and det_class is not None:
                            if gt_class in VEHICLE_CLASSES and det_class in VEHICLE_CLASSES and gt_class == det_class:
                                if self._iou(gt_box, det_box) >= 0.5:
                                    total_true_positives += 1
                                    matched_gt[i] = True
                                    matched_det[j] = True
                                    break
                        else:
                            # Fallback: if no class info, match only by IoU (legacy)
                            if self._iou(gt_box, det_box) >= 0.5:
                                total_true_positives += 1
                                matched_gt[i] = True
                                matched_det[j] = True
                                break

            total_false_positives += sum(1 for m in matched_det if not m)
            total_false_negatives += sum(1 for m in matched_gt if not m)
            
            # Progress indicator every 100 frames
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"      Progress: {progress:.1f}% ({processed_frames}/{total_frames})")
        
        cap.release()
        total_time = time.time() - start_time
        
        # Calculate run statistics
        avg_detection = statistics.mean(detection_counts) if detection_counts else 0
        median_detection = statistics.median(detection_counts) if detection_counts else 0
        mode_detection = statistics.mode(detection_counts) if detection_counts else 0
        std_dev = statistics.stdev(detection_counts) if len(detection_counts) > 1 else 0
        
        # Nonaktifkan evaluasi akurasi, hanya statistik deteksi dan performa
        return {
            'run_number': run_num,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'avg_detection': avg_detection,
            'median_detection': median_detection,
            'mode_detection': mode_detection,
            'std_deviation': std_dev,
            'max_detection': max(detection_counts) if detection_counts else 0,
            'min_detection': min(detection_counts) if detection_counts else 0,
            'avg_inference_time': statistics.mean(inference_times) if inference_times else 0,
            'total_processing_time': total_time,
            'processing_fps': processed_frames / total_time if total_time > 0 else 0,
            'resolution': f"{original_width}x{original_height}",
            'detection_counts': detection_counts[:100]  # Store first 100 for analysis
        }
    
    def _calculate_comprehensive_stats(self, video_path: str, run_results: List[Dict]) -> Dict:
        """Calculate comprehensive statistics from multiple runs"""
        
        # Extract metrics from all runs
        avg_detections = [r['avg_detection'] for r in run_results]
        median_detections = [r['median_detection'] for r in run_results]
        inference_times = [r['avg_inference_time'] for r in run_results]
        processing_times = [r['total_processing_time'] for r in run_results]

        # Statistik deteksi frame-per-frame (perbandingan antar run)
        # Buat frame_stats: frame_idx -> [deteksi_run1, deteksi_run2, ...]
        frame_stats = {}
        max_frames = max(len(r['detection_counts']) for r in run_results)
        for frame_idx in range(max_frames):
            frame_stats[frame_idx+1] = [
                r['detection_counts'][frame_idx] if frame_idx < len(r['detection_counts']) else None
                for r in run_results
            ]

        # Statistik waktu inferensi per frame (gabungan semua run)
        all_inference_times = []
        for r in run_results:
            if 'inference_times' in r:
                all_inference_times.extend(r['inference_times'])
        # Jika tidak ada, fallback ke avg_inference_time
        if not all_inference_times:
            all_inference_times = [r['avg_inference_time'] for r in run_results if 'avg_inference_time' in r]

        if all_inference_times:
            inf_time_avg = round(statistics.mean(all_inference_times) * 1000, 2)
            inf_time_min = round(min(all_inference_times) * 1000, 2)
            inf_time_max = round(max(all_inference_times) * 1000, 2)
            inf_time_std = round(statistics.stdev(all_inference_times) * 1000, 2) if len(all_inference_times) > 1 else 0
            # Frame tercepat/terlambat (run, frame_idx, waktu)
            slowest = sorted([
                (run_idx+1, idx+1, t*1000)
                for run_idx, r in enumerate(run_results)
                for idx, t in enumerate(r.get('inference_times', []))
            ], key=lambda x: -x[2])[:5]
            fastest = sorted([
                (run_idx+1, idx+1, t*1000)
                for run_idx, r in enumerate(run_results)
                for idx, t in enumerate(r.get('inference_times', []))
            ], key=lambda x: x[2])[:5]
        else:
            inf_time_avg = inf_time_min = inf_time_max = inf_time_std = 0
            slowest = fastest = []

        # Log frame deteksi 0 (kosong) dan outlier (Q3 + 1.5*IQR)
        all_detections = [d for r in run_results for d in r['detection_counts']]
        zero_frames = [(run_idx+1, idx+1) for run_idx, r in enumerate(run_results) for idx, d in enumerate(r['detection_counts']) if d == 0]
        # Outlier detection: hanya jika data cukup (minimal 4 data untuk quantiles n=4)
        if len(all_detections) >= 4:
            quant = statistics.quantiles(all_detections, n=4)
            q3 = quant[2]  # Q3
            q4 = quant[3]  # Max
            iqr = q4 - q3
            outlier_thresh = q3 + 1.5 * iqr
            outlier_frames = [(run_idx+1, idx+1, d) for run_idx, r in enumerate(run_results) for idx, d in enumerate(r['detection_counts']) if d > outlier_thresh]
        else:
            outlier_frames = []

        # Hanya statistik deteksi dan performa, tanpa akurasi
        overall_avg = statistics.mean(avg_detections)
        overall_median = statistics.median(avg_detections)
        overall_std = statistics.stdev(avg_detections) if len(avg_detections) > 1 else 0
        consistency_score = (1 - (overall_std / overall_avg)) * 100 if overall_avg > 0 else 0
        return {
            'video_name': video_path,
            'total_runs': len(run_results),
            'resolution': run_results[0]['resolution'],
            'total_frames': run_results[0]['total_frames'],
            # Detection Statistics
            'overall_avg_detection': round(overall_avg, 2),
            'overall_median_detection': round(overall_median, 2),
            'detection_std_dev': round(overall_std, 2),
            'consistency_score': round(consistency_score, 1),
            'min_avg_detection': round(min(avg_detections), 2),
            'max_avg_detection': round(max(avg_detections), 2),
            # Performance Statistics
            'avg_inference_time_ms': inf_time_avg,
            'min_inference_time_ms': inf_time_min,
            'max_inference_time_ms': inf_time_max,
            'std_inference_time_ms': inf_time_std,
            'slowest_frames': slowest,
            'fastest_frames': fastest,
            'avg_processing_time': round(statistics.mean(processing_times), 2),
            'avg_processing_fps': round(statistics.mean([r['processing_fps'] for r in run_results]), 2),
            # Reliability only
            'reliability_level': self._assess_accuracy(0, overall_std, consistency_score)['reliability_level'],
            # Frame-level logs
            'zero_detection_frames': zero_frames,
            'outlier_detection_frames': outlier_frames,
            # Perbandingan antar run (deteksi per frame)
            'frame_stats': frame_stats,
            # Detailed Results
            'run_details': run_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_accuracy(self, f1_score: float, std_dev: float, consistency: float) -> Dict:
        """Assess detection accuracy and consistency based on F1-score, std_dev, and consistency"""
        # Accuracy Level based on F1-score
        if f1_score >= 0.95:
            accuracy_level = "Excellent (>95% F1)"
        elif f1_score >= 0.90:
            accuracy_level = "Very Good (90-95% F1)"
        elif f1_score >= 0.80:
            accuracy_level = "Good (80-90% F1)"
        elif f1_score >= 0.70:
            accuracy_level = "Fair (70-80% F1)"
        elif f1_score >= 0.60:
            accuracy_level = "Poor (60-70% F1)"
        else:
            accuracy_level = "Very Poor (<60% F1)"
        
        # Reliability Level based on consistency score
        if consistency >= 95:
            reliability_level = "Very Reliable (>95% consistent)"
        elif consistency >= 90:
            reliability_level = "Reliable (90-95% consistent)"
        elif consistency >= 80:
            reliability_level = "Moderately Reliable (80-90% consistent)"
        elif consistency >= 70:
            reliability_level = "Somewhat Reliable (70-80% consistent)"
        else:
            reliability_level = "Unreliable (<70% consistent)"
        
        return {'accuracy_level': accuracy_level, 'reliability_level': reliability_level}
    
    def _print_comprehensive_result(self, result: Dict):
        """Print detailed comprehensive test results"""
        print(f"\n📊 COMPREHENSIVE RESULTS - {os.path.basename(result['video_name'])}")
        print("=" * 70)
        
        print(f"🎥 Video Info:")
        print(f"   Resolution: {result['resolution']}")
        print(f"   Total Frames: {result['total_frames']}")
        print(f"   Test Runs: {result['total_runs']}")
        
        print(f"\n🎯 Detection Results:")
        print(f"   Average Detection: {result['overall_avg_detection']} vehicles")
        print(f"   Median Detection: {result['overall_median_detection']} vehicles")
        print(f"   Range: {result['min_avg_detection']} - {result['max_avg_detection']} vehicles")
        print(f"   Standard Deviation: {result['detection_std_dev']}")
        print(f"   Consistency Score: {result['consistency_score']}% ")
        
        print(f"\n⚡ Performance:")
        print(f"   Avg Inference Time: {result['avg_inference_time_ms']} ms/frame")
        print(f"   Avg Processing FPS: {result['avg_processing_fps']} fps")
        print(f"   Avg Processing Time: {result['avg_processing_time']} seconds")
        
        print(f"\n🎯 Reliability Assessment:")
        print(f"   Reliability: {result['reliability_level']}")
    
    def _generate_comprehensive_report(self, test_results, runs_per_video):
        """Generate comprehensive report for all videos"""
        if not test_results:
            print("❌ No test results available for report")
            return
        print(f"\n📋 FINAL COMPREHENSIVE ACCURACY REPORT")
        print("=" * 100)
        print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Videos Tested: {len(test_results)}")
        print(f"🔄 Runs per Video: {runs_per_video}")
        print(f"📺 Mode: Full Screen Resolution")
        # Summary table
        print(f"\n📊 SUMMARY TABLE:")
        print("+" + "-" * 154 + "+")
        print(f"| {'Video':<15} | {'Avg Det':<9} | {'Avg Parked':<11} | {'Avg Moving':<11} | {'Consistency':<11} | {'Reliability':<15} | {'FPS':<5} |")
        print("+" + "-" * 154 + "+")
        for result in test_results:
            video_name = os.path.basename(result['video_name'])[:15]
            avg_parked = result.get('avg_parked', 0.0)
            avg_moving = result.get('avg_moving', 0.0)
            print(f"| {video_name:<15} | {result['overall_avg_detection']:<9.2f} | {avg_parked:<11.2f} | {avg_moving:<11.2f} | {result['consistency_score']:<11.1f} | {result['reliability_level'][:15]:<15} | {result['avg_processing_fps']:<5.1f} |")
        print("+" + "-" * 154 + "+")
        # Overall statistics
        all_consistency = [r['consistency_score'] for r in test_results]
        all_inference = [r['avg_inference_time_ms'] for r in test_results]
        all_fps = [r['avg_processing_fps'] for r in test_results]
        print(f"\n📈 OVERALL SYSTEM PERFORMANCE:")
        print(f"   Average Consistency: {statistics.mean(all_consistency):.1f}%")
        print(f"   Average Inference Time: {statistics.mean(all_inference):.1f} ms/frame")
        print(f"   Average Processing FPS: {statistics.mean(all_fps):.1f}")
        print(f"   Total Test Runs: {len(test_results) * runs_per_video}")
    
    def _save_comprehensive_results(self, test_results, output_dir, runs_per_video, filename: str = "comprehensive_accuracy_results.json"):
        """Save comprehensive results to JSON file"""
        output_dir = Path(output_dir) / "comprehensive_test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        # Prepare data for JSON export
        export_data = {
            'test_metadata': {
                'runs_per_video': runs_per_video,
                'full_screen_mode': True,
                'total_videos': len(test_results),
                'total_test_runs': len(test_results) * runs_per_video,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': test_results
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Comprehensive results saved to: {filepath}")