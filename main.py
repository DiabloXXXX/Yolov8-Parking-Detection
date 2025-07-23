#!/usr/bin/env python3
"""
YOLOv8 Vehicle Detection System for Parking Areas
Main entry point for the application

Author: Vehicle Detection System
Date: 2025-01-05
"""

import sys
import argparse
from pathlib import Path

# Add src to path (for direct imports)
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import core modules
from src.config import Config
from src.vehicle_detector import VehicleDetector
from src.video_processor import VideoProcessor
from src.utils import setup_logging, validate_video_files

# Import tool modules
from tools.realtime_monitor import RealTimeMonitor
from tools.advanced_tester import AdvancedTester
from tools.tester_util import VehicleDetectionTester

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Vehicle Detection for Parking Areas')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--output', type=str, default='output_logs/output', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['interactive', 'auto', 'test', 'monitor', 'stats'], 
                       default='interactive', help='Running mode')
    parser.add_argument('--iterations', type=int, default=10, help='Number of test iterations for stats mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = Config(args.config)
    
    # Validate video files
    if not validate_video_files(config.VIDEO_PATHS):
        print("❌ No valid video files found!")
        return
    
    # Initialize detector
    detector = VehicleDetector(config)
    
    # Initialize video processor
    processor = VideoProcessor(detector, config)
    
    if args.mode == 'interactive':
        processor.run_interactive_mode(args.video)
    elif args.mode == 'auto':
        processor.run_auto_mode()
    elif args.mode == 'test':
        tester = VehicleDetectionTester(detector, config)
        tester.run_comprehensive_testing()
    elif args.mode == 'monitor':
        # Real-time monitoring mode
        monitor = RealTimeMonitor(detector, config)
        
        # Select video and area
        selected_video = processor.select_video(args.video)
        if processor.validate_video(selected_video):
            area_points = processor.select_detection_area(selected_video)
            monitor.start_monitoring(selected_video, area_points)
    elif args.mode == 'stats':
        # Statistical analysis mode
        tester = AdvancedTester(detector, config)
        
        print(f"🧪 STATISTICAL ANALYSIS MODE ({args.iterations} iterations per video)")
        print("=" * 70)
        
        for video_path in config.VIDEO_PATHS:
            if os.path.exists(video_path):
                tester.run_statistical_test(video_path, args.iterations)
        
        tester.generate_statistical_report()
        tester.plot_statistical_analysis()
    
    print("✅ Detection completed successfully!")

if __name__ == "__main__":
    main()
