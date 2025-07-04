# 🚗 YOLOv8 Vehicle Detection System

> **Sistema deteksi kendaraan parkir menggunakan YOLOv8 dengan tracking dan monitoring real-time**

## 📁 **NEW ORGANIZED STRUCTURE**

```
YoloV8-Detection/
├── 📦 01_CORE/              # FILES UTAMA (WAJIB)
│   ├── main.py              # Entry point utama
│   ├── start.py             # Starter alternatif  
│   └── src/                 # Source code inti
│       ├── vehicle_detector.py    # Core detection
│       ├── vehicle_tracker.py     # Tracking system
│       ├── video_processor.py     # Video processing
│       ├── config.py              # Config loader
│       └── utils.py               # Utilities
│
├── ⚙️ 02_CONFIG/            # KONFIGURASI
│   ├── config.yaml          # Konfigurasi utama
│   ├── requirements.txt     # Dependencies
│   └── setup.py             # Setup script
│
├── 🤖 03_MODELS/            # MODEL YOLO
│   ├── yolov8s.pt          # Model small (recommended)
│   ├── yolov8m.pt          # Model medium
│   └── yolov8l.pt          # Model large
│
├── 🎬 04_DATA/              # DATA & VIDEO
│   └── parking_area/       # Area dan video parkir
│       ├── class_list.txt  # Daftar kelas deteksi
│       └── video/          # Video testing
│
├── 🧪 05_TESTING/           # TESTING & DEBUG
│   ├── debug_simple.py     # Debug sederhana
│   ├── debug_tracking.py   # Debug tracking
│   └── test_*.py           # Script testing
│
├── 🔧 06_TOOLS/             # UTILITY TOOLS
│   ├── analyze_videos.py   # Analisis video
│   ├── advanced_tester.py  # Testing lanjutan
│   └── realtime_monitor.py # Monitor real-time
│
├── 📚 07_DOCS/              # DOKUMENTASI
│   ├── README.md           # Dokumentasi utama
│   └── *.md               # Dokumentasi lainnya
│
├── 📊 08_LOGS_OUTPUT/       # OUTPUT & LOGS
│   ├── output/             # Hasil deteksi
│   └── *.log              # Log sistem
│
└── 🗃️ 09_ARCHIVE/           # ARCHIVE
    └── old_files/          # File lama/backup
```

## 🚀 **QUICK START**

### **1. Persiapan**
```bash
# Install dependencies
pip install -r 02_CONFIG/requirements.txt

# Pastikan model tersedia
# File yolov8s.pt harus ada di 03_MODELS/
```

### **2. Menjalankan Sistem**

#### **🤖 AUTO MODE (Recommended)**
```bash
python auto_detect.py
```
- Deteksi otomatis semua video
- Menggunakan default settings
- No user interaction required

#### **🔧 DEBUG MODE**
```bash
python debug.py
```
- Melihat status tracking real-time
- Debug movement detection
- Performance monitoring

#### **🎮 INTERACTIVE MODE**
```bash
python run.py --mode interactive
```
- Pilih video manual
- Pilih area deteksi
- Real-time visualization

### **3. Testing & Analysis**
```bash
# Test komprehensif
python run.py --mode test

# Monitor real-time
python run.py --mode monitor

# Statistical analysis
python run.py --mode stats
```

## ⚙️ **KONFIGURASI**

### **File: `02_CONFIG/config.yaml`**
```yaml
# Model settings
model_path: 03_MODELS/yolov8s.pt
class_list_path: 04_DATA/parking_area/class_list.txt

# Target vehicle classes (TANPA MOTOR)
target_vehicle_classes:
  2: car     # Mobil
  5: bus     # Bus  
  7: truck   # Truk

# Video paths
video_paths:
  - 04_DATA/parking_area/video/park1.mp4
  - 04_DATA/parking_area/video/park2.mp4
  - 04_DATA/parking_area/video/park3.mp4

# Detection parameters
conf_threshold: 0.3           # Confidence threshold
iou_threshold: 0.5           # IoU threshold
min_area: 500                # Minimum area
max_area: 50000              # Maximum area

# Tracking settings
max_movement_threshold: 10   # Movement threshold (pixels)
min_moving_time: 0.8        # Min time to consider moving (seconds)
min_parking_time: 2.0       # Min time to consider parked (seconds)
```

## 🎯 **FITUR UTAMA**

### **✅ Vehicle Detection**
- ✅ Deteksi mobil, bus, truk (tanpa motor)
- ✅ YOLOv8 model dengan confidence threshold
- ✅ Filter berdasarkan area minimum/maksimum
- ✅ Validasi resolusi video minimal 640x360

### **🎯 Vehicle Tracking**
- ✅ State awal kendaraan: DIAM
- ✅ Status BERGERAK jika bergerak ≥0.8 detik
- ✅ Status PARKIR jika diam ≥2 detik
- ✅ Filter goyangan kamera dan false movement
- ✅ Reset tracker otomatis saat ganti video

### **📍 Area Detection**
- ✅ Default full screen jika tidak ada area dipilih
- ✅ Interactive area selection
- ✅ Support multiple area shapes

### **📊 Real-time Visualization**
- ✅ Bounding box dengan warna status
- ✅ Tracking trail (jejak pergerakan)
- ✅ Counter parkir vs bergerak
- ✅ Status duration display

## 🔧 **TROUBLESHOOTING**

### **❌ Error: Model tidak ditemukan**
```bash
# Pastikan model ada di direktori yang benar
ls 03_MODELS/yolov8s.pt

# Download model jika belum ada
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
mv yolov8s.pt 03_MODELS/
```

### **❌ Error: Video tidak ditemukan**
```bash
# Periksa path video di config
cat 02_CONFIG/config.yaml | grep video_paths

# Pastikan video ada
ls 04_DATA/parking_area/video/
```

### **❌ Error: Import module**
```bash
# Install dependencies
pip install -r 02_CONFIG/requirements.txt

# Periksa Python path
python -c "import sys; print(sys.path)"
```

## 📈 **PERFORMANCE TIPS**

### **🚀 Optimasi Kecepatan**
- Gunakan `yolov8s.pt` untuk speed
- Gunakan `yolov8m.pt` untuk balance
- Gunakan `yolov8l.pt` untuk accuracy
- Sesuaikan `frame_skip` di config

### **🎯 Optimasi Akurasi**
- Turunkan `conf_threshold` untuk deteksi lebih sensitif
- Atur `min_area` dan `max_area` sesuai ukuran kendaraan
- Sesuaikan `max_movement_threshold` untuk area Anda

### **⚡ Optimasi Memory**
- Gunakan `resize_width` dan `resize_height` lebih kecil
- Batasi `cleanup_timeout` untuk tracking
- Gunakan mode auto untuk batch processing

## 📝 **CHANGELOG**

### **v2.0.0 - Directory Restructure**
- ✅ Reorganisasi struktur direktori
- ✅ Simplified launchers (`auto_detect.py`, `debug.py`)
- ✅ Updated path configurations
- ✅ Improved documentation

### **v1.9.0 - Movement Detection Fix**
- ✅ Fixed movement detection logic
- ✅ Added camera shake filtering
- ✅ Improved tracking consistency
- ✅ Enhanced delay management

## 💡 **TIPS & TRICKS**

1. **Auto Mode adalah yang paling stabil** - Gunakan `python auto_detect.py`
2. **Debug Mode untuk troubleshooting** - Gunakan `python debug.py`
3. **Sesuaikan threshold di config** - Edit `02_CONFIG/config.yaml`
4. **Monitor log untuk errors** - Cek `08_LOGS_OUTPUT/vehicle_detection.log`
5. **Backup config sebelum perubahan** - Copy `02_CONFIG/config.yaml`

---

**🔗 Author:** Vehicle Detection System  
**📅 Last Updated:** 2025-01-05  
**📧 Support:** Check documentation in `07_DOCS/`
