"""
Utility functions for the vehicle detection system
"""

import os
import logging
import yaml
from pathlib import Path
from typing import List

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vehicle_detection.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")

def validate_video_files(video_paths: List[str]) -> bool:
    """
    Validate that video files exist
    
    Args:
        video_paths: List of video file paths
        
    Returns:
        True if at least one video file exists
    """
    valid_count = 0
    
    for video_path in video_paths:
        if os.path.exists(video_path):
            valid_count += 1
            print(f"✅ Found: {video_path}")
        else:
            print(f"❌ Missing: {video_path}")
    
    print(f"📊 Valid videos: {valid_count}/{len(video_paths)}")
    return valid_count > 0

def create_output_directory(output_dir: str = "output"):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory ready: {output_dir}")

def get_video_info(video_path: str) -> dict:
    """
    Get video information without OpenCV dependency in utils
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def save_config_template(output_path: str = "config/config_template.yaml"):
    """Save a configuration template file"""
    config_template = {
        "model_path": "models/yolov8s.pt",
        "class_list_path": "data/parking_area/class_list.txt",
        "target_vehicle_classes": {
            2: "car",
            5: "bus", 
            7: "truck"
        },
        "parking_vehicle_keywords": ["car", "bus", "truck"],
        "video_paths": [
            "data/parking_area/video/park1.mp4",
            "data/parking_area/video/park2.mp4",
            "data/parking_area/video/park3.mp4",
            "data/parking_area/video/park4.mp4"
        ],
        "min_width": 640,
        "min_height": 360,
        "conf_threshold": 0.3,
        "iou_threshold": 0.5,
        "min_area": 500,
        "max_area": 50000,
        "frame_skip": 3,
        "resize_width": 1020,
        "resize_height": 500,
        "output_dir": "output_logs/output",
        "save_results": True,
        "show_display": True,
        "ground_truth_data": {
            "park1.mp4": 5,
            "park2.mp4": 8,
            "park3.mp4": 4,
            "park4.mp4": 7
        },
        "default_test_points": {
            "park1.mp4": [[100, 200], [800, 200], [800, 400], [100, 400]],
            "park2.mp4": [[150, 180], [850, 180], [850, 420], [150, 420]],
            "park3.mp4": [[120, 160], [900, 160], [900, 380], [120, 380]],
            "park4.mp4": [[80, 220], [920, 220], [920, 450], [80, 450]]
        }
    }
    
    # Create config directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_template, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📄 Configuration template saved to: {output_path}")

def print_system_info():
    """Print system information"""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        print("🖥️ SYSTEM INFORMATION")
        print("=" * 40)
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Python version: {os.sys.version}")
        print("=" * 40)
        
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'cv2', 'numpy', 'pandas', 'ultralytics', 
        'PIL', 'matplotlib', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True
