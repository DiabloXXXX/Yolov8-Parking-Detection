# TECHNICAL SPECIFICATIONS

## Arsitektur Sistem

### 1. Struktur Proyek

```
YoloV8-Detection/
├── 01_CORE/                    # Core system components
│   ├── main.py                 # Main application entry point
│   ├── start.py                # System startup script
│   └── src/                    # Source code modules
│       ├── config.py           # Configuration management
│       ├── vehicle_detector.py # YOLO detection engine
│       ├── vehicle_tracker.py  # Object tracking system
│       ├── video_processor.py  # Video processing coordinator
│       └── utils.py            # Utility functions
├── 02_CONFIG/                  # Configuration files
│   ├── config.yaml             # Main configuration
│   └── setup.py                # Configuration setup script
├── 03_MODELS/                  # Model storage
│   └── yolov8s.pt             # YOLOv8 model weights
├── 04_DOCUMENTATION/          # Project documentation
├── 05_TESTING/                # Testing framework
│   ├── quick_test.py          # Quick validation test
│   ├── accuracy_test.py       # Comprehensive accuracy test
│   └── benchmark_test.py      # Performance benchmark
├── 06_TOOLS/                  # Additional tools
│   ├── realtime_monitor.py    # Real-time monitoring
│   ├── advanced_tester.py     # Statistical analysis
│   └── tester.py              # General testing utilities
└── parking_area/              # Input data
    ├── video/                 # Video files for testing
    └── class_list.txt         # COCO class definitions
```

### 2. Core Components

#### VehicleDetector (`vehicle_detector.py`)
```python
class VehicleDetector:
    """
    YOLOv8-based vehicle detection engine
    
    Features:
    - Multi-class vehicle detection (car, bus, truck)
    - Configurable confidence thresholds
    - GPU acceleration support
    - Real-time inference optimization
    """
    
    def __init__(self, config):
        # Initialize YOLO model
        # Setup device (CPU/GPU)
        # Configure detection parameters
    
    def detect_vehicles(self, frame):
        # Perform object detection
        # Filter by vehicle classes
        # Return detection results
```

#### VehicleTracker (`vehicle_tracker.py`)
```python
class VehicleTracker:
    """
    Simple IoU-based vehicle tracking system
    
    Features:
    - Track assignment using IoU overlap
    - Parked vs moving vehicle classification
    - Track lifecycle management
    - Configurable tracking parameters
    """
    
    def __init__(self, config):
        # Initialize tracking parameters
        # Setup track management
    
    def update_tracks(self, detections):
        # Update existing tracks
        # Create new tracks
        # Remove lost tracks
        # Classify movement status
```

#### VideoProcessor (`video_processor.py`)
```python
class VideoProcessor:
    """
    Main video processing and coordination system
    
    Features:
    - Video input/output handling
    - Detection area (ROI) management
    - Real-time visualization
    - Processing mode coordination
    """
    
    def __init__(self, detector, config):
        # Initialize components
        # Setup video parameters
    
    def process_video(self, video_path, area_points):
        # Main processing loop
        # Coordinate detection and tracking
        # Handle visualization and output
```

### 3. Configuration System

#### Config Class (`config.py`)
```python
class Config:
    """
    Centralized configuration management
    
    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Dynamic path resolution
    - Validation and defaults
    """
    
    # Model Configuration
    MODEL_PATH: str
    CONFIDENCE_THRESHOLD: float
    IOU_THRESHOLD: float
    
    # Detection Configuration
    TARGET_CLASSES: List[int]  # [2, 5, 7] for car, bus, truck
    MAX_DETECTIONS: int
    
    # Tracking Configuration
    TRACKING_IOU_THRESHOLD: float
    MAX_DISAPPEARED: int
    MOVEMENT_THRESHOLD: float
    
    # Video Configuration
    VIDEO_PATHS: List[str]
    OUTPUT_PATH: str
    RESIZE_FACTOR: float
    
    # UI Configuration
    COLORS: Dict
    FONT_SCALE: float
    LINE_THICKNESS: int
```

### 4. Testing Framework

#### Test Architecture
```
Testing Framework
├── Unit Tests
│   ├── Detection accuracy per class
│   ├── Tracking algorithm validation
│   └── Configuration validation
├── Integration Tests
│   ├── End-to-end video processing
│   ├── Multi-component interaction
│   └── Performance benchmarking
└── System Tests
    ├── Real-world scenario testing
    ├── Edge case handling
    └── Stress testing
```

#### Performance Metrics
```python
class PerformanceMetrics:
    """
    Comprehensive performance evaluation
    
    Metrics:
    - Detection: Precision, Recall, F1-Score, mAP
    - Tracking: MOTA, MOTP, ID Switches
    - Performance: FPS, Processing Time, Memory Usage
    - System: CPU Usage, GPU Utilization
    """
```

## Implementation Details

### 1. Detection Pipeline

```python
# Detection Flow
frame → preprocessing → YOLO inference → post-processing → results

# Preprocessing
- Resize for optimal inference speed
- Color space conversion (BGR → RGB)
- Normalization

# YOLO Inference  
- Forward pass through YOLOv8 model
- Multi-scale detection
- Class probability calculation

# Post-processing
- Non-Maximum Suppression (NMS)
- Confidence filtering
- Class filtering (vehicles only)
```

### 2. Tracking Algorithm

```python
# Tracking Flow
detections → IoU calculation → assignment → track update → status classification

# IoU Assignment
for track in existing_tracks:
    for detection in new_detections:
        iou = calculate_iou(track.bbox, detection.bbox)
        if iou > threshold:
            assign(track, detection)

# Status Classification
if track.movement_distance < threshold:
    status = "PARKED"
else:
    status = "MOVING"
```

### 3. Real-time Processing

```python
# Optimization Strategies
- Frame skipping for high FPS videos
- Asynchronous processing
- Memory management
- GPU acceleration when available

# Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Detection
    detections = detector.detect_vehicles(frame)
    
    # Tracking
    tracks = tracker.update_tracks(detections)
    
    # Visualization
    frame = visualizer.draw_results(frame, tracks)
    
    # Output
    writer.write(frame)
```

## Dependencies dan Requirements

### Core Dependencies
```txt
# Deep Learning & Computer Vision
ultralytics>=8.0.0          # YOLOv8 framework
opencv-python>=4.5.0        # Computer vision operations
torch>=1.9.0                # PyTorch backend
torchvision>=0.10.0         # Vision transformations

# Data Processing
numpy>=1.21.0               # Numerical operations
pandas>=1.3.0               # Data manipulation
scipy>=1.7.0                # Scientific computing

# Visualization
matplotlib>=3.3.0           # Plotting and charts
seaborn>=0.11.0             # Statistical visualization

# Configuration & Utilities
pyyaml>=5.4.0              # YAML configuration files
pathlib                    # Path operations (built-in)
argparse                   # Command line parsing (built-in)
datetime                   # Time operations (built-in)
```

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8+
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **CPU**: Intel i5 8th gen / AMD Ryzen 5 3600

#### Recommended Requirements  
- **RAM**: 16 GB
- **GPU**: NVIDIA GTX 1060 6GB / RTX 3060
- **CUDA**: 11.0+ (for GPU acceleration)
- **Storage**: SSD for optimal performance

## Performance Characteristics

### Benchmark Results

| Configuration | Resolution | FPS | CPU Usage | RAM Usage |
|---------------|------------|-----|-----------|-----------|
| CPU Only      | 720p       | 8-12| 75-85%    | 2-3 GB    |
| CPU Only      | 1080p      | 5-8 | 85-95%    | 3-4 GB    |
| GPU Accelerated| 720p      | 25-35| 45-55%   | 4-5 GB    |
| GPU Accelerated| 1080p     | 15-25| 55-65%   | 5-6 GB    |

### Optimization Guidelines

1. **Video Resolution**: Use 720p for real-time applications
2. **Confidence Threshold**: 0.3-0.5 for optimal precision/recall balance
3. **Batch Processing**: Process multiple videos sequentially for better throughput
4. **Memory Management**: Monitor RAM usage for long-running processes

## Deployment Considerations

### Production Deployment
- Use Docker containers for consistent environment
- Implement proper logging and monitoring
- Setup automated testing pipeline
- Consider edge computing for local processing

### Integration Points
- REST API endpoints for external integration
- Database connectivity for historical data
- Message queue support for high-throughput scenarios
- Cloud storage integration for video archives

---

*Technical specifications version 1.0 - Updated for YOLOv8 Vehicle Detection System*
