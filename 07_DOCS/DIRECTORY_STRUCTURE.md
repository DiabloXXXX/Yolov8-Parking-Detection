# 📁 YOLOv8 Vehicle Detection - Directory Structure

## 🗂️ **Organized File Structure**

### **01_CORE** 📦 **(UTAMA - FILES WAJIB)**
File-file inti sistem yang diperlukan untuk menjalankan aplikasi:
- `main.py` - Entry point utama aplikasi
- `start.py` - Script starter alternatif
- `src/` - Source code utama
  - `vehicle_detector.py` - Core detection engine
  - `vehicle_tracker.py` - Tracking system
  - `video_processor.py` - Video processing logic
  - `config.py` - Configuration loader
  - `utils.py` - Utility functions

### **02_CONFIG** ⚙️ **(KONFIGURASI PENTING)**
File konfigurasi dan pengaturan sistem:
- `config.yaml` - Konfigurasi utama sistem
- `requirements.txt` - Dependencies Python
- `setup.py` - Setup dan instalasi

### **03_MODELS** 🤖 **(MODEL YOLO)**
Model YOLO yang digunakan untuk deteksi:
- `yolov8s.pt` - YOLOv8 Small (recommended)
- `yolov8m.pt` - YOLOv8 Medium
- `yolov8l.pt` - YOLOv8 Large
- `class_list.txt` - Daftar kelas deteksi

### **04_DATA** 🎬 **(VIDEO & DATA)**
Data input dan area parkir:
- `parking_area/` - Area dan video parkir
  - `video/` - Video untuk testing
  - `class_list.txt` - Konfigurasi kelas
- `output/` - Hasil deteksi (akan dipindah ke 08_LOGS_OUTPUT)

### **05_TESTING** 🧪 **(TESTING & DEBUG)**
Script untuk testing dan debugging:
- `debug_simple.py` - Debug tracking sederhana
- `debug_tracking.py` - Debug tracking advanced
- `test_*.py` - Script testing berbagai fitur
- `comprehensive_accuracy_test.py` - Test akurasi komprehensif

### **06_TOOLS** 🔧 **(UTILITY & TOOLS)**
Tools tambahan dan utility:
- `analyze_videos.py` - Analisis video
- `check_videos.py` - Validasi video
- `video_info.py` - Informasi video
- `src/advanced_tester.py` - Testing lanjutan
- `src/comprehensive_tester.py` - Testing komprehensif
- `src/realtime_monitor.py` - Monitor real-time

### **07_DOCS** 📚 **(DOKUMENTASI)**
Dokumentasi dan catatan pengembangan:
- `README.md` - Dokumentasi utama
- `LOGIC_TRACKING_FIXED.md` - Dokumentasi logic tracking
- `PERUBAHAN_FULL_SCREEN.md` - Catatan perubahan
- `TABEL_IMPLEMENTASI.md` - Tabel implementasi
- `QUICK_TABLE.md` - Referensi cepat

### **08_LOGS_OUTPUT** 📊 **(OUTPUT & LOGS)**
Log dan hasil output sistem:
- `output/` - Hasil deteksi dan video
- `vehicle_detection.log` - Log sistem
- `analysis_results/` - Hasil analisis

### **09_ARCHIVE** 🗃️ **(ARCHIVE & BACKUP)**
File lama dan backup:
- `Detection2.ipynb` - Jupyter notebook lama
- `.ipynb_checkpoints/` - Checkpoint notebook
- `old_scripts/` - Script lama

---

## 🚀 **Quick Start Guide**

### **Menjalankan Sistem:**
```bash
# Mode interaktif
python 01_CORE/main.py --mode interactive

# Mode otomatis
python 01_CORE/main.py --mode auto

# Mode testing
python 01_CORE/main.py --mode test
```

### **Debugging:**
```bash
# Debug sederhana
python 05_TESTING/debug_simple.py

# Debug tracking
python 05_TESTING/debug_tracking.py
```

### **Analisis Video:**
```bash
# Analisis video
python 06_TOOLS/analyze_videos.py

# Informasi video
python 06_TOOLS/video_info.py
```

---

## 📋 **File Priority Level**

### **🔴 CRITICAL (Wajib Ada)**
- `01_CORE/main.py`
- `01_CORE/src/vehicle_detector.py`
- `01_CORE/src/vehicle_tracker.py` 
- `01_CORE/src/video_processor.py`
- `02_CONFIG/config.yaml`
- `03_MODELS/yolov8s.pt`

### **🟡 IMPORTANT (Sangat Direkomendasikan)**
- `01_CORE/src/config.py`
- `01_CORE/src/utils.py`
- `02_CONFIG/requirements.txt`
- `04_DATA/parking_area/`

### **🟢 OPTIONAL (Tambahan)**
- `05_TESTING/*` - Script testing
- `06_TOOLS/*` - Tools utility
- `07_DOCS/*` - Dokumentasi

### **⚪ ARCHIVE (Bisa Dihapus)**
- `09_ARCHIVE/*` - File lama dan backup

---

## 🔧 **Maintenance Notes**

- Pastikan `01_CORE` dan `02_CONFIG` selalu ter-backup
- `03_MODELS` berisi file besar, backup terpisah jika perlu
- `08_LOGS_OUTPUT` dapat dibersihkan secara berkala
- `09_ARCHIVE` dapat dihapus jika tidak diperlukan

**Last Updated:** 2025-01-05
