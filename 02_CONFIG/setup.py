#!/usr/bin/env python3
"""
Setup script for YOLOv8 Vehicle Detection System
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run shell command with description"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {description} completed")
            return True
        else:
            print(f"   ❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🛠️ YOLOv8 Vehicle Detection System - Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    print("\n1️⃣ Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️ Please install manually: pip install -r requirements.txt")
    
    # Create necessary directories
    print("\n2️⃣ Creating directories...")
    directories = ["output", "config", "parking_area/video"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   📁 {directory}")
    
    # Check for model file
    print("\n3️⃣ Checking YOLO model...")
    model_paths = ["yolov8s.pt", "parking_area/yolov8s.pt"]
    model_found = False
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"   ✅ Model found: {model_path}")
            model_found = True
            break
    
    if not model_found:
        print("   ⚠️ YOLOv8 model not found. It will be downloaded automatically on first run.")
    
    # Check for class list
    print("\n4️⃣ Checking class list...")
    class_list_path = "parking_area/class_list.txt"
    
    if os.path.exists(class_list_path):
        print(f"   ✅ Class list found: {class_list_path}")
    else:
        print("   📝 Creating default class list...")
        # Create default COCO class list
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        
        os.makedirs(os.path.dirname(class_list_path), exist_ok=True)
        with open(class_list_path, 'w', encoding='utf-8') as f:
            for class_name in coco_classes:
                f.write(class_name + '\n')
        
        print(f"   ✅ Class list created: {class_list_path}")
    
    # Check for videos
    print("\n5️⃣ Checking video files...")
    video_dir = "parking_area/video"
    video_files = []
    
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(file)
                print(f"   ✅ {file}")
    
    if not video_files:
        print("   ⚠️ No video files found in parking_area/video/")
        print("   📁 Please add your parking video files to: parking_area/video/")
    
    # Test configuration
    print("\n6️⃣ Testing configuration...")
    try:
        sys.path.append('src')
        from src.config import Config
        config = Config('config/config.yaml')
        print("   ✅ Configuration loaded successfully")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    
    if video_files:
        print("✅ System ready to use!")
        print("\n🚀 Quick start options:")
        print("   python start.py              # Quick start menu")
        print("   python main.py --mode interactive  # Interactive mode")
        print("   python main.py --mode auto         # Auto mode")
        print("   python main.py --mode test         # Test mode")
    else:
        print("⚠️ Setup complete, but no video files found.")
        print("📁 Please add video files to parking_area/video/ directory")
        print("   Supported formats: .mp4, .avi, .mov, .mkv")
    
    print(f"\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
