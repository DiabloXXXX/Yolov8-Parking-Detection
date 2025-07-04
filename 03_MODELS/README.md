# YOLOv8 Models

This directory should contain YOLOv8 model files. Due to file size limitations, models are not included in the repository and must be downloaded separately.

## Download Instructions

### Option 1: Direct Download (Recommended)

```bash
# YOLOv8 Small (fastest, ~6MB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# YOLOv8 Medium (balanced, ~25MB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# YOLOv8 Large (most accurate, ~87MB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
```

### Option 2: Using Python

```python
from ultralytics import YOLO

# This will automatically download the model
model = YOLO('yolov8s.pt')  # or yolov8m.pt, yolov8l.pt
```

### Option 3: Manual Download

1. Visit [Ultralytics YOLOv8 Releases](https://github.com/ultralytics/assets/releases)
2. Download your preferred model:
   - `yolov8s.pt` - Small model (recommended for most cases)
   - `yolov8m.pt` - Medium model (good balance)
   - `yolov8l.pt` - Large model (highest accuracy)
3. Place the downloaded `.pt` file in this directory

## Model Comparison

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| YOLOv8s | 6MB | Fast | 44.9 | Real-time applications |
| YOLOv8m | 25MB | Medium | 50.2 | Balanced performance |
| YOLOv8l | 87MB | Slow | 52.9 | High accuracy needs |

## Configuration

After downloading, update the model path in `02_CONFIG/config.yaml`:

```yaml
model_path: 03_MODELS/yolov8s.pt  # Change to your downloaded model
```

## Troubleshooting

- **File not found error**: Ensure the model file is in this directory
- **Permission denied**: Check file permissions and directory access
- **Network issues**: Try downloading manually if wget fails

For more information, visit the [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/).
