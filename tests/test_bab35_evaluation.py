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

# Define PROJECT_ROOT for consistent pathing in tests
PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="module")
def config_and_detector():
    """Fixture to provide a Config and VehicleDetector instance"""
    config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
    detector = VehicleDetector(config)
    return config, detector

class TestSystemEvaluation:
    """Evaluator untuk sistem deteksi kendaraan parkir menggunakan pytest"""
    
    # Ground Truth data untuk setiap video
    ground_truth = {
        'park1.mp4': {
            'total_kendaraan': 5,
            'kondisi': 'Pencahayaan normal, kepadatan rendah',
            'posisi_kendaraan': ['tengah kiri', 'tengah', 'tengah kanan', 'kanan atas', 'kanan bawah']
        },
        'park2.mp4': {
            'total_kendaraan': 8,
            'kondisi': 'Pencahayaan terang, kepadatan tinggi',
            'posisi_kendaraan': ['tersebar merata di seluruh area']
        },
        'park3.mp4': {
            'total_kendaraan': 4,
            'kondisi': 'Pencahayaan sedang, kepadatan rendah',
            'posisi_kendaraan': ['kiri', 'tengah kiri', 'tengah kanan', 'kanan']
        },
        'park4.mp4': {
            'total_kendaraan': 7,
            'kondisi': 'Pencahayaan bervariasi, kepadatan sedang',
            'posisi_kendaraan': ['tersebar dengan jarak bervariasi']
        }
    }

    def test_detection_accuracy(self, config_and_detector):
        """Evaluasi 1: Ketepatan deteksi kendaraan"""
        config, detector = config_and_detector
        
        print("\n📊 EVALUASI 1: KETEPATAN DETEKSI KENDARAAN")
        print("=" * 50)
        
        all_results = []
        for video_path in config.VIDEO_PATHS:
            video_name = os.path.basename(video_path)
            print(f"\n📹 Video: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Tidak dapat membuka video: {video_path}")
                continue
            
            gt = self.ground_truth.get(video_name, {})
            expected_vehicles = gt.get('total_kendaraan', 0)
            
            detector.tracker.reset_tracker()
            
            detection_counts = []
            correct_detections = 0
            total_frames_tested = 0
            frame_count = 0
            
            test_frames = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count not in test_frames:
                    continue
                
                total_frames_tested += 1
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, area_points)
                
                total_detected = len(detections)
                detection_counts.append(total_detected)
                
                if abs(total_detected - expected_vehicles) <= 1:
                    correct_detections += 1
                
                print(f"Frame {frame_count}: Terdeteksi {total_detected}, Expected {expected_vehicles}")
            
            cap.release()
            
            accuracy = (correct_detections / total_frames_tested * 100) if total_frames_tested > 0 else 0
            avg_detected = np.mean(detection_counts) if detection_counts else 0
            
            results = {
                'video_name': video_name,
                'ground_truth': expected_vehicles,
                'accuracy_percentage': accuracy,
                'average_detected': avg_detected,
            }
            all_results.append(results)
            
            print(f"\n✅ HASIL EVALUASI 1 for {video_name}:")
            print(f"   Akurasi deteksi: {accuracy:.1f}%")
            print(f"   Rata-rata terdeteksi: {avg_detected:.1f} kendaraan")
        
        assert len(all_results) > 0, "No videos were tested for detection accuracy."

    def test_counting_accuracy(self, config_and_detector):
        """Evaluasi 2: Keakuratan sistem dalam menghitung jumlah kendaraan"""
        config, detector = config_and_detector
        
        print("\n📊 EVALUASI 2: KEAKURATAN PENGHITUNGAN KENDARAAN")
        print("=" * 50)
        
        all_results = []
        for video_path in config.VIDEO_PATHS:
            video_name = os.path.basename(video_path)
            print(f"\n📹 Video: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Tidak dapat membuka video: {video_path}")
                continue
            
            gt = self.ground_truth.get(video_name, {})
            expected_vehicles = gt.get('total_kendaraan', 0)
            
            detector.tracker.reset_tracker()
            
            frame_count = 0
            counting_results = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % 30 != 0:
                    continue
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, area_points)
                
                total_counted = parked_count + moving_count
                counting_results.append({
                    'frame': frame_count,
                    'counted': total_counted,
                    'parked': parked_count,
                    'moving': moving_count,
                    'expected': expected_vehicles
                })
            
            cap.release()
            
            if counting_results:
                final_result = counting_results[-1]
                final_counted = final_result['counted']
                absolute_error = abs(final_counted - expected_vehicles)
                relative_error = (absolute_error / expected_vehicles * 100) if expected_vehicles > 0 else 0
                counting_accuracy = max(0, 100 - relative_error)
            else:
                final_counted = absolute_error = relative_error = counting_accuracy = 0
            
            results = {
                'video_name': video_name,
                'expected_vehicles': expected_vehicles,
                'final_counted': final_counted,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'counting_accuracy': counting_accuracy,
            }
            all_results.append(results)
            
            print(f"\n✅ HASIL EVALUASI 2 for {video_name}:")
            print(f"   Expected: {expected_vehicles} kendaraan")
            print(f"   Terhitung: {final_counted} kendaraan")
            print(f"   Akurasi penghitungan: {counting_accuracy:.1f}%")
        
        assert len(all_results) > 0, "No videos were tested for counting accuracy."

    def test_processing_performance(self, config_and_detector):
        """Evaluasi 3: Kinerja sistem berdasarkan waktu inferensi per frame"""
        config, detector = config_and_detector
        
        print("\n📊 EVALUASI 3: KINERJA WAKTU PEMROSESAN")
        print("=" * 50)
        
        all_results = []
        for video_path in config.VIDEO_PATHS:
            video_name = os.path.basename(video_path)
            print(f"\n📹 Video: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Tidak dapat membuka video: {video_path}")
                continue
            
            detector.tracker.reset_tracker()
            
            processing_times = []
            inference_times = []
            frame_count = 0
            test_frames = 100
            
            while frame_count < test_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % 5 != 0:
                    continue
                
                frame_resized = cv2.resize(frame, (640, 480))
                area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
                
                total_start = time.time()
                
                inference_start = time.time()
                results_yolo = detector.model(frame_resized)
                inference_time = time.time() - inference_start
                
                parked_count, moving_count, detections = detector.detect_vehicles_with_tracking(
                    frame_resized, area_points)
                
                total_processing_time = time.time() - total_start
                
                inference_times.append(inference_time)
                processing_times.append(total_processing_time)
            
            cap.release()
            
            if inference_times and processing_times:
                avg_inference_time = np.mean(inference_times)
                avg_total_time = np.mean(processing_times)
                inference_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
                total_fps = 1 / avg_total_time if avg_total_time > 0 else 0
                realtime_capable = total_fps >= 15
            else:
                avg_inference_time = avg_total_time = 0
                inference_fps = total_fps = 0
                realtime_capable = False
            
            results = {
                'video_name': video_name,
                'avg_inference_time': avg_inference_time,
                'avg_total_processing_time': avg_total_time,
                'inference_fps': inference_fps,
                'total_fps': total_fps,
                'realtime_capable': realtime_capable
            }
            all_results.append(results)
            
            print(f"\n✅ HASIL EVALUASI 3 for {video_name}:")
            print(f"   Rata-rata waktu inferensi: {avg_inference_time:.3f} detik")
            print(f"   FPS total sistem: {total_fps:.1f}")
            print(f"   Kelayakan real-time: {'✅ YA' if realtime_capable else '❌ TIDAK'}")
        
        assert len(all_results) > 0, "No videos were tested for processing performance."