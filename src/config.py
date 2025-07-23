"""
Configuration module for YOLOv8 Vehicle Detection System
"""

import os
import yaml
from pathlib import Path

class Config:
    """Configuration class for the vehicle detection system"""
    
    def __init__(self, config_path=None):
        """Initialize configuration"""
        # Set project root (go up 2 levels from src to project root)
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        self.load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        elif config_path:
            # Try absolute path from project root or relative from current working dir
            full_config_path = self.PROJECT_ROOT / config_path
            if full_config_path.exists():
                self.load_from_file(str(full_config_path))
    
    def get_absolute_path(self, relative_path):
        """Convert relative path to absolute path from project root"""
        return str(self.PROJECT_ROOT / relative_path)
    
    def load_default_config(self):
        """Load default configuration values"""
        # Model settings (with absolute paths from project root)
        self.MODEL_PATH = self.get_absolute_path('models/yolov8s.pt')
        self.CLASS_LIST_PATH = self.get_absolute_path('data/parking_area/class_list.txt')
        
        # Target vehicle classes (TANPA MOTOR)
        self.TARGET_VEHICLE_CLASSES = {
            2: 'car',
            5: 'bus',
            7: 'truck'
        }
        
        # Keywords untuk fallback filtering
        self.PARKING_VEHICLE_KEYWORDS = ['car', 'bus', 'truck']
        
        # Video settings (with absolute paths from project root)
        self.VIDEO_PATHS = [
            self.get_absolute_path('data/parking_area/video/park1.mp4'),
            self.get_absolute_path('data/parking_area/video/park2.mp4'), 
            self.get_absolute_path('data/parking_area/video/park3.mp4'),
            self.get_absolute_path('data/parking_area/video/park4.mp4')
        ]
        
        # Video validation
        self.MIN_WIDTH = 640
        self.MIN_HEIGHT = 360
        
        # Detection parameters
        self.CONF_THRESHOLD = 0.3
        self.IOU_THRESHOLD = 0.5
        self.MIN_AREA = 500
        self.MAX_AREA = 50000
        
        # Processing settings
        self.FRAME_SKIP = 3
        self.RESIZE_WIDTH = 1920
        self.RESIZE_HEIGHT = 1080
        self.FULLSCREEN_MODE = True
        
        # Tracking settings for parking detection
        self.TRACKING_ENABLED = True
        self.MAX_DISTANCE_THRESHOLD = 100
        self.MIN_PARKING_TIME = 2.0
        self.MIN_MOVING_TIME = 0.5  # Reduced for better responsiveness
        self.MAX_MOVEMENT_THRESHOLD = 5   # Very low threshold for testing
        self.CLEANUP_TIMEOUT = 5.0
        
        # Output settings (with absolute paths from project root)
        self.OUTPUT_DIR = self.get_absolute_path('output_logs/output')
        self.SAVE_RESULTS = True
        self.SHOW_DISPLAY = True
        
        
    
    def load_from_file(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # Handle path conversion for specific keys
            path_keys = ['model_path', 'class_list_path', 'output_dir']
            
            for key, value in config_data.items():
                key_upper = key.upper()
                
                # Convert relative paths to absolute for specific keys
                if key in path_keys and isinstance(value, str):
                    if not os.path.isabs(value):
                        value = self.get_absolute_path(value)
                elif key == 'video_paths' and isinstance(value, list):
                    value = [self.get_absolute_path(path) if not os.path.isabs(path) else path for path in value]
                
                if hasattr(self, key_upper):
                    setattr(self, key_upper, value)
                    
            print(f"✅ Configuration loaded from: {config_path}")
        except Exception as e:
            print(f"⚠️ Could not load config file: {e}")
            print("Using default configuration")
    
    def save_to_file(self, config_path):
        """Save current configuration to YAML file"""
        config_data = {}
        
        for attr in dir(self):
            if attr.isupper() and not attr.startswith('_'):
                config_data[attr.lower()] = getattr(self, attr)
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"💾 Configuration saved to: {config_path}")
    
    def print_config(self):
        """Print current configuration"""
        print("\n📋 CURRENT CONFIGURATION:")
        print("=" * 50)
        print(f"🤖 Model: {self.MODEL_PATH}")
        print(f"🚗 Target vehicles: {list(self.TARGET_VEHICLE_CLASSES.values())}")
        print(f"📊 Min resolution: {self.MIN_WIDTH}x{self.MIN_HEIGHT}")
        print(f"🎯 Confidence threshold: {self.CONF_THRESHOLD}")
        print(f"📹 Frame skip: {self.FRAME_SKIP}")
        print(f"📁 Output dir: {self.OUTPUT_DIR}")
        print(f"🖥️ Display: {self.RESIZE_WIDTH}x{self.RESIZE_HEIGHT}")
        print(f"⏱️ Parking time: {self.MIN_PARKING_TIME}s minimum")
        print("=" * 50)
