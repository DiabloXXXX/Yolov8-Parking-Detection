#!/usr/bin/env python3
"""
Quick start script for YOLOv8 Vehicle Detection System
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 YOLOv8 Vehicle Detection System - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src") or not os.path.exists("main.py"):
        print("❌ Please run this script from the YoloV8-Detection directory")
        return
    
    # Check dependencies
    try:
        from src.utils import check_dependencies, validate_video_files
        from src.config import Config
        
        print("1️⃣ Checking dependencies...")
        if not check_dependencies():
            print("❌ Please install missing dependencies: pip install -r requirements.txt")
            return
        
        print("\n2️⃣ Loading configuration...")
        config = Config('config/config.yaml')
        
        print("\n3️⃣ Validating video files...")
        if not validate_video_files(config.VIDEO_PATHS):
            print("⚠️ No valid video files found. Please check parking_area/video/ directory")
            print("📁 Expected videos:")
            for video in config.VIDEO_PATHS:
                print(f"   - {video}")
            return
        
        print("\n4️⃣ System ready! Choose running mode:")
        print("   1. Interactive Mode (recommended)")
        print("   2. Auto Mode (all videos with defaults)")
        print("   3. Test Mode (comprehensive testing)")
        print("   4. Monitor Mode (real-time dashboard)")
        print("   5. Stats Mode (statistical analysis)")
        
        choice = input("\n🎯 Select mode (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Interactive Mode...")
            os.system("python main.py --mode interactive")
        elif choice == "2":
            print("\n🤖 Starting Auto Mode...")
            os.system("python main.py --mode auto")
        elif choice == "3":
            print("\n🧪 Starting Test Mode...")
            os.system("python main.py --mode test")
        elif choice == "4":
            print("\n📊 Starting Monitor Mode...")
            os.system("python main.py --mode monitor")
        elif choice == "5":
            iterations = input("Enter number of iterations per video (default 10): ").strip()
            iterations = iterations if iterations.isdigit() else "10"
            print(f"\n📈 Starting Statistical Analysis ({iterations} iterations)...")
            os.system(f"python main.py --mode stats --iterations {iterations}")
        else:
            print("⚠️ Invalid choice, starting Interactive Mode...")
            os.system("python main.py --mode interactive")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
