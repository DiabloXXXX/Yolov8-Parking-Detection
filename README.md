# YOLOv8 Parking Detection & Evaluation

## Short Description
YOLOv8-based vehicle detection and parking area evaluation system. Modular Python code for automated multi-video testing, detailed statistics, and export to Excel/CSV. Suitable for smart parking analytics and research.

## Features
- Vehicle detection in parking area videos using YOLOv8
- Modular, robust Python code structure
- Automated multi-run, multi-video evaluation
- Detailed frame/run-level statistics (average, median, std, outlier, FPS, inference time)
- Export results to Excel and CSV
- Easy integration and reproducibility

## Folder Structure
```
├── 01_CORE/           # Core detection modules
├── 02_CONFIG/         # Config files
├── 05_TESTING/        # Test scripts
├── 06_TOOLS/          # Utility scripts
├── output_logs/       # Output results (JSON, CSV, Excel)
├── parking_area/      # Sample data & videos
├── tests/             # Automated test scripts
├── yolov8s.pt         # YOLOv8 model (if included)
└── README.md          # This file
```

## Installation
1. Clone this repo:
   ```bash
   git clone https://github.com/DiabloXXXX/Yolov8-Parking-Detection.git
   cd Yolov8-Parking-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Pastikan Python 3.8+ dan ultralytics, opencv-python, pandas, xlsxwriter sudah terinstall)

## Usage
- Jalankan deteksi dan evaluasi otomatis:
  ```bash
  python tests/test_comprehensive_accuracy.py
  ```
- Ekspor hasil ke Excel:
  ```bash
  python output_logs/output/comprehensive_test_results/export_accuracy_to_excel.py
  ```
- Lihat hasil di folder `output_logs/output/comprehensive_test_results/`

## Example Output
- JSON: comprehensive_accuracy_results.json
- Excel: comprehensive_accuracy_report.xlsx
- CSV: comprehensive_accuracy_summary.csv

## License
MIT

## Contact
- Author: DiabloXXXX
- Email: yugiindra40@gmail.com

---
Feel free to contribute, fork, or open issues!
