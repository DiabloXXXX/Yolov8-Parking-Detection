#!/usr/bin/env python3
"""
Sistem Evaluasi untuk Bab 3.5 - Pengujian dan Evaluasi Sistem
Mengevaluasi kemampuan model YOLOv8 dalam mendeteksi kendaraan pada area parkir

Tujuan Pengujian:
1. Menilai kemampuan model YOLOv8 dalam mendeteksi kendaraan pada area parkir 
   yang bervariasi dari segi posisi, pencahayaan, dan kepadatan objek
2. Mengevaluasi keakuratan sistem dalam menghitung jumlah kendaraan 
   berdasarkan deteksi bounding box
3. Mengetahui waktu pemrosesan per frame untuk menilai kelayakan sistem 
   dalam konteks real-time

Kriteria Evaluasi:
1. Ketepatan deteksi kendaraan (benar terdeteksi atau tidak terdeteksi)
2. Jumlah kendaraan terhitung dibandingkan dengan jumlah aktual dalam gambar (Ground Truth)
3. Kinerja sistem berdasarkan waktu inferensi (inference time) per frame
"""

import sys
import os
from pathlib import Path
import cv2
import time
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

# Setup paths
project_root = Path(__file__).parent.parent
core_dir = project_root / "01_CORE"
src_dir = core_dir / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(core_dir))
sys.path.insert(0, str(src_dir))

# Import modules from src directory
import config
import vehicle_detector
from config import Config
from vehicle_detector import VehicleDetector

class SystemEvaluator:
    """Evaluator untuk sistem deteksi kendaraan parkir"""
    
    def __init__(self):
        """Initialize evaluator"""
        print("🎯 SISTEM EVALUASI BAB 3.5 - PENGUJIAN DAN EVALUASI SISTEM")
        print("=" * 70)
        
        # Change to project root
        os.chdir(project_root)
        
        # Load config dan detector
        self.config = Config('02_CONFIG/config.yaml')
        self.detector = VehicleDetector(self.config)
        
        # Ground Truth data untuk setiap video
        self.ground_truth = {
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
        
        # Hasil evaluasi
        self.evaluation_results = {}
        
    def evaluate_detection_accuracy(self, video_path, video_name):
        """
        Evaluasi 1: Ketepatan deteksi kendaraan
        Mengukur kemampuan model dalam mendeteksi kendaraan dengan benar
        """
        print(f"\n📊 EVALUASI 1: KETEPATAN DETEKSI KENDARAAN")
        print(f"Video: {video_name}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Tidak dapat membuka video: {video_path}")
            return None
        
        # Ambil ground truth
        gt = self.ground_truth.get(video_name, {})
        expected_vehicles = gt.get('total_kendaraan', 0)
        kondisi_pengujian = gt.get('kondisi', 'Tidak diketahui')
        
        print(f"Ground Truth: {expected_vehicles} kendaraan")
        print(f"Kondisi pengujian: {kondisi_pengujian}")
        
        # Reset tracker
        self.detector.tracker.reset_tracker()
        
        # Variabel evaluasi
        detection_counts = []
        correct_detections = 0
        total_frames_tested = 0
        frame_count = 0
        
        # Test pada frame-frame tertentu untuk evaluasi
        test_frames = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Test hanya pada frame tertentu
            if frame_count not in test_frames:
                continue
                
            total_frames_tested += 1
            
            # Resize frame
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            # Deteksi kendaraan
            parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
                frame_resized, area_points)
            
            total_detected = len(detections)
            detection_counts.append(total_detected)
            
            # Hitung akurasi (toleransi ±1 kendaraan dianggap benar)
            if abs(total_detected - expected_vehicles) <= 1:
                correct_detections += 1
                
            print(f"Frame {frame_count}: Terdeteksi {total_detected}, Expected {expected_vehicles}")
        
        cap.release()
        
        # Hitung metrik
        accuracy = (correct_detections / total_frames_tested * 100) if total_frames_tested > 0 else 0
        avg_detected = np.mean(detection_counts) if detection_counts else 0
        detection_variance = np.std(detection_counts) if detection_counts else 0
        
        results = {
            'video_name': video_name,
            'kondisi_pengujian': kondisi_pengujian,
            'ground_truth': expected_vehicles,
            'frames_tested': total_frames_tested,
            'correct_detections': correct_detections,
            'accuracy_percentage': accuracy,
            'average_detected': avg_detected,
            'detection_variance': detection_variance,
            'detection_stability': (1 - detection_variance / avg_detected * 100) if avg_detected > 0 else 0
        }
        
        print(f"\n✅ HASIL EVALUASI 1:")
        print(f"   Akurasi deteksi: {accuracy:.1f}%")
        print(f"   Rata-rata terdeteksi: {avg_detected:.1f} kendaraan")
        print(f"   Stabilitas deteksi: {results['detection_stability']:.1f}%")
        print(f"   Variansi deteksi: {detection_variance:.2f}")
        
        return results
    
    def evaluate_counting_accuracy(self, video_path, video_name):
        """
        Evaluasi 2: Keakuratan sistem dalam menghitung jumlah kendaraan
        Membandingkan jumlah kendaraan terhitung dengan Ground Truth
        """
        print(f"\n📊 EVALUASI 2: KEAKURATAN PENGHITUNGAN KENDARAAN")
        print(f"Video: {video_name}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Ground truth
        gt = self.ground_truth.get(video_name, {})
        expected_vehicles = gt.get('total_kendaraan', 0)
        
        # Reset tracker dan proses seluruh video
        self.detector.tracker.reset_tracker()
        
        frame_count = 0
        counting_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Test setiap 30 frame
            if frame_count % 30 != 0:
                continue
            
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            # Deteksi dengan tracking
            parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
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
        
        # Analisis hasil penghitungan
        if counting_results:
            final_result = counting_results[-1]  # Hasil akhir
            final_counted = final_result['counted']
            final_parked = final_result['parked']
            final_moving = final_result['moving']
            
            # Hitung error
            absolute_error = abs(final_counted - expected_vehicles)
            relative_error = (absolute_error / expected_vehicles * 100) if expected_vehicles > 0 else 0
            
            # Akurasi penghitungan
            counting_accuracy = max(0, 100 - relative_error)
        else:
            final_counted = final_parked = final_moving = 0
            absolute_error = relative_error = 0
            counting_accuracy = 0
        
        results = {
            'video_name': video_name,
            'expected_vehicles': expected_vehicles,
            'final_counted': final_counted,
            'final_parked': final_parked,
            'final_moving': final_moving,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'counting_accuracy': counting_accuracy,
            'frames_processed': len(counting_results)
        }
        
        print(f"\n✅ HASIL EVALUASI 2:")
        print(f"   Expected: {expected_vehicles} kendaraan")
        print(f"   Terhitung: {final_counted} kendaraan")
        print(f"   - Parkir: {final_parked} kendaraan")
        print(f"   - Bergerak: {final_moving} kendaraan")
        print(f"   Error absolut: {absolute_error} kendaraan")
        print(f"   Error relatif: {relative_error:.1f}%")
        print(f"   Akurasi penghitungan: {counting_accuracy:.1f}%")
        
        return results
    
    def evaluate_processing_performance(self, video_path, video_name):
        """
        Evaluasi 3: Kinerja sistem berdasarkan waktu inferensi per frame
        Mengukur kelayakan sistem untuk aplikasi real-time
        """
        print(f"\n📊 EVALUASI 3: KINERJA WAKTU PEMROSESAN")
        print(f"Video: {video_name}")
        print("-" * 50)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Reset tracker
        self.detector.tracker.reset_tracker()
        
        # Variabel untuk mengukur performa
        processing_times = []
        inference_times = []
        total_processing_times = []
        frame_count = 0
        test_frames = 100  # Test 100 frame
        
        print("Mengukur waktu pemrosesan...")
        
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip beberapa frame untuk efisiensi
            if frame_count % 5 != 0:
                continue
            
            # Resize frame
            frame_resized = cv2.resize(frame, (640, 480))
            area_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
            
            # Ukur waktu total pemrosesan
            total_start = time.time()
            
            # Ukur waktu inferensi model saja
            inference_start = time.time()
            results = self.detector.model(frame_resized)
            inference_time = time.time() - inference_start
            
            # Proses deteksi lengkap
            parked_count, moving_count, detections = self.detector.detect_vehicles_with_tracking(
                frame_resized, area_points)
            
            total_processing_time = time.time() - total_start
            
            # Simpan hasil
            inference_times.append(inference_time)
            total_processing_times.append(total_processing_time)
            
            if frame_count % 20 == 0:
                print(f"   Frame {frame_count}: Inference {inference_time:.3f}s, Total {total_processing_time:.3f}s")
        
        cap.release()
        
        # Hitung metrik performa
        if inference_times and total_processing_times:
            avg_inference_time = np.mean(inference_times)
            avg_total_time = np.mean(total_processing_times)
            max_inference_time = np.max(inference_times)
            max_total_time = np.max(total_processing_times)
            
            # Hitung FPS
            inference_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
            total_fps = 1 / avg_total_time if avg_total_time > 0 else 0
            
            # Evaluasi kelayakan real-time (minimal 15 FPS untuk real-time)
            realtime_capable = total_fps >= 15
        else:
            avg_inference_time = avg_total_time = 0
            max_inference_time = max_total_time = 0
            inference_fps = total_fps = 0
            realtime_capable = False
        
        results = {
            'video_name': video_name,
            'frames_tested': len(total_processing_times),
            'avg_inference_time': avg_inference_time,
            'avg_total_processing_time': avg_total_time,
            'max_inference_time': max_inference_time,
            'max_total_processing_time': max_total_time,
            'inference_fps': inference_fps,
            'total_fps': total_fps,
            'realtime_capable': realtime_capable
        }
        
        print(f"\n✅ HASIL EVALUASI 3:")
        print(f"   Rata-rata waktu inferensi: {avg_inference_time:.3f} detik")
        print(f"   Rata-rata waktu total: {avg_total_time:.3f} detik")
        print(f"   FPS inferensi: {inference_fps:.1f}")
        print(f"   FPS total sistem: {total_fps:.1f}")
        print(f"   Kelayakan real-time: {'✅ YA' if realtime_capable else '❌ TIDAK'}")
        
        return results
    
    def run_comprehensive_evaluation(self):
        """Jalankan evaluasi komprehensif untuk semua video"""
        print("\n🚀 MEMULAI EVALUASI KOMPREHENSIF SISTEM")
        print("=" * 70)
        
        # Hasil evaluasi untuk semua video
        all_results = {}
        summary_metrics = {
            'total_videos': 0,
            'avg_detection_accuracy': 0,
            'avg_counting_accuracy': 0,
            'avg_fps': 0,
            'realtime_capable_count': 0
        }
        
        # Test setiap video
        for video_path in self.config.VIDEO_PATHS:
            video_name = Path(video_path).name
            
            if not Path(video_path).exists():
                print(f"⚠️ Video tidak ditemukan: {video_path}")
                continue
            
            print(f"\n{'='*20} EVALUASI VIDEO: {video_name} {'='*20}")
            
            # Jalankan 3 evaluasi
            eval1_result = self.evaluate_detection_accuracy(video_path, video_name)
            eval2_result = self.evaluate_counting_accuracy(video_path, video_name)
            eval3_result = self.evaluate_processing_performance(video_path, video_name)
            
            # Simpan hasil
            all_results[video_name] = {
                'detection_evaluation': eval1_result,
                'counting_evaluation': eval2_result,
                'performance_evaluation': eval3_result
            }
            
            # Update summary
            if eval1_result and eval2_result and eval3_result:
                summary_metrics['total_videos'] += 1
                summary_metrics['avg_detection_accuracy'] += eval1_result['accuracy_percentage']
                summary_metrics['avg_counting_accuracy'] += eval2_result['counting_accuracy']
                summary_metrics['avg_fps'] += eval3_result['total_fps']
                if eval3_result['realtime_capable']:
                    summary_metrics['realtime_capable_count'] += 1
        
        # Hitung rata-rata
        if summary_metrics['total_videos'] > 0:
            summary_metrics['avg_detection_accuracy'] /= summary_metrics['total_videos']
            summary_metrics['avg_counting_accuracy'] /= summary_metrics['total_videos']
            summary_metrics['avg_fps'] /= summary_metrics['total_videos']
        
        # Simpan hasil ke file
        self.save_evaluation_results(all_results, summary_metrics)
        
        # Tampilkan ringkasan
        self.display_evaluation_summary(summary_metrics)
        
        return all_results, summary_metrics
    
    def save_evaluation_results(self, results, summary):
        """Simpan hasil evaluasi ke file JSON"""
        output_dir = Path('08_LOGS_OUTPUT/evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        # Simpan hasil detail
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'evaluation_results_{timestamp}.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON
        clean_results = {}
        for video, data in results.items():
            clean_results[video] = {}
            for eval_type, eval_data in data.items():
                if eval_data:
                    clean_results[video][eval_type] = {k: convert_numpy(v) for k, v in eval_data.items()}
        
        clean_summary = {k: convert_numpy(v) for k, v in summary.items()}
        
        evaluation_data = {
            'timestamp': timestamp,
            'summary': clean_summary,
            'detailed_results': clean_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Hasil evaluasi disimpan: {results_file}")
    
    def display_evaluation_summary(self, summary):
        """Tampilkan ringkasan evaluasi"""
        print(f"\n📊 RINGKASAN EVALUASI SISTEM BAB 3.5")
        print("=" * 70)
        print(f"Total video yang dievaluasi: {summary['total_videos']}")
        print()
        print("📈 METRIK KINERJA SISTEM:")
        print(f"   1. Rata-rata akurasi deteksi: {summary['avg_detection_accuracy']:.1f}%")
        print(f"   2. Rata-rata akurasi penghitungan: {summary['avg_counting_accuracy']:.1f}%")
        print(f"   3. Rata-rata FPS sistem: {summary['avg_fps']:.1f}")
        print(f"   4. Video yang capable real-time: {summary['realtime_capable_count']}/{summary['total_videos']}")
        print()
        
        # Penilaian keseluruhan
        overall_score = (summary['avg_detection_accuracy'] + summary['avg_counting_accuracy']) / 2
        
        if overall_score >= 85:
            grade = "SANGAT BAIK ⭐⭐⭐⭐⭐"
        elif overall_score >= 75:
            grade = "BAIK ⭐⭐⭐⭐"
        elif overall_score >= 65:
            grade = "CUKUP ⭐⭐⭐"
        else:
            grade = "PERLU PERBAIKAN ⭐⭐"
        
        print(f"🏆 PENILAIAN KESELURUHAN: {grade}")
        print(f"   Skor rata-rata: {overall_score:.1f}%")
        
        # Rekomendasi
        print(f"\n💡 REKOMENDASI UNTUK PENELITIAN:")
        if summary['avg_detection_accuracy'] < 80:
            print("   - Tingkatkan akurasi deteksi dengan fine-tuning model")
        if summary['avg_counting_accuracy'] < 80:
            print("   - Perbaiki algoritma counting untuk mengurangi error")
        if summary['avg_fps'] < 15:
            print("   - Optimasi kode untuk meningkatkan performa real-time")
        if summary['realtime_capable_count'] < summary['total_videos']:
            print("   - Pertimbangkan hardware yang lebih powerful untuk deployment")
        
        print(f"\n✅ Evaluasi sistem selesai!")

def main():
    """Main function"""
    try:
        evaluator = SystemEvaluator()
        evaluator.run_comprehensive_evaluation()
    except Exception as e:
        print(f"❌ Error dalam evaluasi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
