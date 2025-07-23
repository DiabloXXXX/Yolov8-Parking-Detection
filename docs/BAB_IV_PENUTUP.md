# BAB IV PENUTUP

Bab ini menyajikan rangkuman dari seluruh hasil penelitian yang telah dilakukan, serta memberikan saran untuk pengembangan lebih lanjut dari sistem deteksi kendaraan parkir menggunakan YOLOv8.

## 4.1 Kesimpulan

Berdasarkan perancangan, implementasi, dan pengujian sistem deteksi kendaraan parkir menggunakan YOLOv8 yang telah dilakukan, beberapa kesimpulan dapat ditarik:

### 4.1.1 Sistem Berhasil Dirancang dan Diimplementasikan

Penelitian ini telah berhasil merancang dan membangun sebuah sistem pendeteksi kendaraan pada area parkir secara otomatis menggunakan algoritma YOLOv8. Implementasi dilakukan dengan:

- **Bahasa Pemrograman**: Python 3.8+
- **Framework Deep Learning**: Ultralytics YOLOv8
- **Computer Vision Library**: OpenCV untuk pemrosesan video dan visualisasi
- **Arsitektur Modular**: Sistem terstruktur dalam komponen-komponen terpisah:
  - `VehicleDetector`: Untuk deteksi objek menggunakan YOLOv8
  - `VehicleTracker`: Untuk pelacakan kendaraan antar frame
  - `VideoProcessor`: Untuk pemrosesan video dan koordinasi sistem
  - `Config`: Untuk manajemen konfigurasi terpusat

### 4.1.2 Deteksi dan Tracking Kendaraan

Sistem berhasil mengimplementasikan:

- **Deteksi Multi-Kelas**: Mampu mendeteksi kendaraan jenis mobil, bus, dan truk (kelas 2, 5, 7 dalam COCO dataset)
- **Tracking Kendaraan**: Menggunakan algoritma tracking sederhana berdasarkan IoU untuk melacak kendaraan antar frame
- **Status Kendaraan**: Membedakan antara kendaraan yang parkir (stationary) dan bergerak (moving) berdasarkan perubahan posisi
- **Area Detection**: Mendukung deteksi dalam area yang ditentukan pengguna melalui seleksi poligon manual

### 4.1.3 Evaluasi dan Testing Framework

Sistem dilengkapi dengan framework evaluasi komprehensif:

- **Quick Test**: Pengujian cepat untuk validasi dasar sistem (`quick_test.py`)
- **Accuracy Test**: Pengujian akurasi mendalam dengan analisis statistik (`accuracy_test.py`)
- **Benchmark Test**: Pengujian performa untuk mengukur kecepatan pemrosesan (`benchmark_test.py`)
- **Real-time Monitoring**: Mode monitoring real-time dengan dashboard (`realtime_monitor.py`)
- **Statistical Analysis**: Analisis statistik dengan multiple iterasi (`advanced_tester.py`)

### 4.1.4 Visualisasi dan User Interface

Hasil deteksi divisualisasikan secara real-time dengan:

- **Bounding Box**: Kotak pembatas kendaraan dengan confidence score
- **Status Tracking**: Indikator visual untuk kendaraan parkir vs bergerak
- **Counter Display**: Jumlah kendaraan parkir dan bergerak dalam area
- **Area Visualization**: Poligon area deteksi yang dapat dikonfigurasi
- **Info Panel**: Informasi frame rate dan statistik deteksi

### 4.1.5 Hasil Pengujian Sistem

Berdasarkan pengujian yang telah dilakukan:

- **Deteksi Accuracy**: Sistem mampu mendeteksi kendaraan dengan akurasi yang baik pada kondisi pencahayaan normal
- **Tracking Performance**: Tracking kendaraan berfungsi stabil untuk video dengan gerakan kendaraan yang tidak terlalu cepat
- **Processing Speed**: Sistem dapat memproses video dengan frame rate yang memadai untuk monitoring parkir
- **Modular Architecture**: Struktur kode modular memudahkan maintenance dan pengembangan lanjutan

## 4.2 Saran

Untuk meningkatkan fungsionalitas dan kinerja sistem deteksi kendaraan parkir ini di masa mendatang, beberapa saran dapat diajukan:

### 4.2.1 Peningkatan Algoritma Tracking

- **Advanced Tracking**: Mengimplementasikan algoritma tracking yang lebih robust seperti DeepSORT atau ByteTrack untuk meningkatkan akurasi pelacakan
- **Multi-Object Tracking**: Optimalisasi tracking untuk skenario dengan banyak kendaraan yang saling berdekatan
- **Occlusion Handling**: Penanganan yang lebih baik untuk kasus oklusi antar kendaraan

### 4.2.2 Optimalisasi Model Detection

- **Custom Training**: Melatih model YOLOv8 dengan dataset spesifik area parkir untuk meningkatkan akurasi
- **Model Quantization**: Implementasi quantization untuk mempercepat inference tanpa menurunkan akurasi secara signifikan
- **Multi-Scale Detection**: Optimalisasi untuk deteksi kendaraan dalam berbagai ukuran dan jarak

### 4.2.3 Peningkatan Robustness

- **Lighting Adaptation**: Implementasi algoritma adaptasi pencahayaan untuk kondisi malam hari atau pencahayaan ekstrem
- **Weather Conditions**: Penanganan kondisi cuaca seperti hujan, kabut, atau salju
- **Camera Angle Flexibility**: Adaptasi untuk berbagai sudut pandang kamera parkir

### 4.2.4 Fitur Tambahan

- **Slot Parking Detection**: Implementasi deteksi slot parkir otomatis tanpa perlu manual ROI selection
- **License Plate Recognition**: Integrasi dengan sistem OCR untuk identifikasi plat nomor kendaraan
- **Database Integration**: Koneksi dengan database untuk logging dan analisis historis
- **Web Dashboard**: Interface web untuk monitoring dan konfigurasi remote

### 4.2.5 Deployment dan Scalability

- **Edge Computing**: Optimalisasi untuk deployment pada edge devices seperti NVIDIA Jetson
- **Multiple Camera Support**: Dukungan untuk monitoring multiple kamera secara bersamaan
- **Cloud Integration**: Integrasi dengan cloud services untuk storage dan analytics
- **API Development**: Pengembangan REST API untuk integrasi dengan sistem parkir yang ada

### 4.2.6 Real-time Implementation

- **RTSP Stream Support**: Dukungan langsung untuk streaming kamera IP
- **Low Latency Processing**: Optimalisasi untuk mengurangi delay pemrosesan
- **Alert System**: Sistem notifikasi untuk event tertentu (parkir ilegal, dll)

### 4.2.7 User Experience Enhancement

- **GUI Application**: Pengembangan aplikasi desktop dengan interface yang user-friendly
- **Configuration Wizard**: Wizard setup untuk memudahkan konfigurasi awal sistem
- **Automated Calibration**: Kalibrasi otomatis untuk pengaturan kamera dan area deteksi

## 4.3 Kontribusi Penelitian

Penelitian ini memberikan kontribusi dalam bentuk:

1. **Implementasi Open Source**: Sistem yang dapat digunakan dan dikembangkan lebih lanjut oleh komunitas
2. **Framework Evaluasi**: Metodologi pengujian yang komprehensif untuk sistem deteksi parkir
3. **Arsitektur Modular**: Desain sistem yang mudah dipelihara dan dikembangkan
4. **Dokumentasi Lengkap**: Dokumentasi yang memudahkan replikasi dan pengembangan

## 4.4 Dampak dan Aplikasi

Sistem yang dikembangkan memiliki potensi aplikasi dalam:

- **Smart Parking Systems**: Sebagai komponen utama sistem parkir cerdas
- **Traffic Management**: Monitoring dan analisis pola lalu lintas kendaraan
- **Security Systems**: Bagian dari sistem keamanan area parkir
- **Urban Planning**: Data untuk perencanaan infrastruktur parkir yang lebih baik

---

*Dokumentasi ini merupakan bagian dari laporan penelitian sistem deteksi kendaraan parkir menggunakan YOLOv8. Untuk informasi teknis lebih detail, silakan merujuk ke dokumentasi kode dan hasil pengujian yang tersedia di repository.*
