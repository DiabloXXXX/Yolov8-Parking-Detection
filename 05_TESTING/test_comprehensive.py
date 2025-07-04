#!/usr/bin/env python3
"""
Comprehensive testing script with multiple runs and fullscreen display
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("🧪 COMPREHENSIVE TESTING - 10 RUNS PER VIDEO")
    print("=" * 60)
    
    try:
        from src.config import Config
        from src.vehicle_detector import VehicleDetector
        from src.comprehensive_tester import AdvancedVehicleDetectionTester
        from src.utils import setup_logging, validate_video_files
        
        # Setup
        setup_logging()
        config = Config('config/config.yaml')
        
        # Check videos
        if not validate_video_files(config.VIDEO_PATHS):
            print("❌ No valid video files found!")
            return
        
        # Initialize system
        detector = VehicleDetector(config)
        tester = AdvancedVehicleDetectionTester(detector, config)
        
        print(f"\n📋 TEST CONFIGURATION:")
        print(f"   🎯 Target vehicles: {list(config.TARGET_VEHICLE_CLASSES.values())}")
        print(f"   🖥️ Fullscreen mode: YES")
        print(f"   🔄 Runs per video: 10")
        print(f"   📊 Min resolution: {config.MIN_WIDTH}x{config.MIN_HEIGHT}")
        
        # Ask user confirmation
        print(f"\n⚠️ IMPORTANT NOTES:")
        print(f"   - Each video will be tested 10 times")
        print(f"   - Tests will run in fullscreen mode")
        print(f"   - Press 'q' to skip a run, 's' to speed up")
        print(f"   - Total estimated time: 10-20 minutes")
        
        confirm = input(f"\n🚀 Start comprehensive testing? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("❌ Testing cancelled")
            return
        
        # Run comprehensive testing
        tester.run_comprehensive_testing(runs_per_video=10, fullscreen=True)
        
        print(f"\n🎉 COMPREHENSIVE TESTING COMPLETED!")
        print(f"📁 Results saved to: output/comprehensive_test_results.json")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
    except KeyboardInterrupt:
        print(f"\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
