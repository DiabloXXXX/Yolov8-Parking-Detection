# YOLOv8 Vehicle Detection System

## Overview
A Python-based vehicle detection system using YOLOv8 for parking area monitoring. This system specifically detects cars, buses, and trucks while excluding motorcycles and bicycles.

## Features
- ✅ **Vehicle Detection**: Cars, buses, trucks only (NO motorcycles)
- ✅ **Video Processing**: Multiple video format support
- ✅ **Interactive Area Selection**: Click-based polygon selection
- ✅ **Real-time Processing**: Live detection with visualization
- ✅ **Comprehensive Testing**: Automated testing framework
- ✅ **Configuration Management**: YAML-based configuration
- ✅ **Resolution Validation**: Minimum 640x360 requirement

## Project Structure
```
YoloV8-Detection/
├── main.py                 # Main entry point
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── vehicle_detector.py # Core detection logic
│   ├── video_processor.py  # Video processing
│   ├── tester.py          # Testing framework
│   └── utils.py           # Utility functions
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── parking_area/          # Data directory
│   ├── class_list.txt     # COCO class names
│   ├── yolov8s.pt         # YOLO model
│   └── video/             # Video files
│       ├── park1.mp4
│       ├── park2.mp4
│       ├── park3.mp4
│       └── park4.mp4
├── output/                # Results output
└── requirements.txt       # Dependencies
```

## Installation

### 1. Clone or setup the project structure as shown above

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLO model (if not already present)
```bash
# The model will be downloaded automatically on first run
# Or manually download yolov8s.pt to the project root
```

## Usage

### Interactive Mode (Recommended)
```bash
python main.py --mode interactive
```
This will:
1. Show available videos for selection
2. Allow interactive area selection
3. Process video with real-time detection

### Automatic Mode
```bash
python main.py --mode auto
```
Processes all videos with default settings.

### Testing Mode
```bash
python main.py --mode test
```
Runs comprehensive testing with metrics and reporting.

### Specific Video
```bash
python main.py --video "parking_area/video/park1.mp4"
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Detection settings
conf_threshold: 0.3    # Detection confidence
iou_threshold: 0.5     # Non-max suppression
frame_skip: 3          # Process every Nth frame

# Video validation  
min_width: 640         # Minimum video width
min_height: 360        # Minimum video height

# Target vehicles (COCO IDs)
target_vehicle_classes:
  2: car
  5: bus  
  7: truck
```

## Key Features

### 1. Vehicle Filtering
- **Primary Method**: COCO class ID filtering (2=car, 5=bus, 7=truck)
- **Fallback Method**: Keyword matching with motorcycle exclusion
- **Excluded**: Motorcycle (ID:3), Bicycle (ID:1)

### 2. Video Validation
- Automatic resolution checking (minimum 640x360)
- Support for MP4, AVI and other OpenCV-supported formats
- Frame rate and duration analysis

### 3. Detection Area
- Interactive 4-point polygon selection
- Real-time visualization of selection
- Default coordinates for automated testing

### 4. Real-time Processing
- Frame skipping for performance optimization
- Live detection visualization
- Pause/resume functionality (press 'p')
- Quit anytime (press 'q')

### 5. Testing Framework
- Automated testing on all videos
- Performance metrics (inference time, FPS)
- Accuracy comparison with ground truth
- JSON export of results
- Visualization plots

## Controls

### During Video Processing:
- **'q'**: Quit
- **'p'**: Pause/Resume
- **ESC**: Use default area (during area selection)
- **ENTER**: Confirm area selection

### During Area Selection:
- **Left Click**: Add point (up to 4 points)
- **ESC**: Use default coordinates
- **ENTER**: Confirm selection when 4 points selected

## Output

### Console Output
- Real-time detection counts
- Processing statistics
- Performance metrics

### File Output
- `output/test_results.json`: Detailed test results
- `output/test_results.png`: Visualization plots
- `vehicle_detection.log`: System logs

## Troubleshooting

### Common Issues:

1. **"Cannot open video"**
   - Check video file path
   - Ensure video file exists and is not corrupted
   - Try different video format

2. **"Resolution too small"**
   - Videos must be at least 640x360
   - Use higher resolution videos

3. **"No detections"**
   - Check detection area selection
   - Adjust confidence threshold in config
   - Ensure vehicles are in the defined area

4. **Poor performance**
   - Increase frame_skip value
   - Use smaller resize dimensions
   - Check GPU availability for YOLO

### Performance Tips:
- Use frame skipping (process every 3rd frame)
- Resize frames for faster processing
- Adjust detection thresholds based on your needs
- Use GPU acceleration if available

## Technical Details

### Detection Logic:
1. Load YOLOv8 model with COCO classes
2. Filter detections to target vehicle classes only
3. Apply spatial filtering (point-in-polygon test)
4. Validate detection area and confidence
5. Draw results and count vehicles

### Performance Optimization:
- Frame skipping for real-time processing
- Adaptive resolution scaling
- Efficient polygon testing
- Optimized drawing operations

## License
This project is for educational and research purposes.

## Support
For issues and questions, please check the troubleshooting section or review the code documentation.
