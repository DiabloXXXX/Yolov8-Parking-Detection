#!/usr/bin/env python3
"""
Test Launcher for YOLOv8 Vehicle Detection System
Easy launcher for all testing options
"""

import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
TESTING_DIR = PROJECT_ROOT / "tests"

def show_menu():
    """Show test menu"""
    print("[TEST] YOLOv8 Vehicle Detection - TEST MENU")
    print("=" * 50)
    print("1. [RUN] Quick Test         - Fast accuracy check")
    print("2. [STATS] Full Accuracy Test - Comprehensive analysis")
    print("3. [PERF] Benchmark Test     - Performance testing")
    print("4. [DEBUG] Debug Test         - Real-time tracking debug")
    print("5. [ALL] Run All Tests      - Complete test suite")
    print("0. [ERROR] Exit")
    print("-" * 50)

def run_test(test_type):
    """Run specified test"""
    os.chdir(PROJECT_ROOT)
    
    test_files = {
        1: "quick_test.py",
        2: "accuracy_test.py", 
        3: "benchmark_test.py",
        4: "debug_simple.py"
    }
    
    if test_type in test_files:
        test_file = TESTING_DIR / test_files[test_type]
        if test_file.exists():
            print(f"\n🚀 Running: {test_files[test_type]}")
            print("=" * 40)
            exec(open(test_file).read())
        else:
            print(f"❌ Test file not found: {test_file}")
    elif test_type == 5:
        # Run all tests
        print(f"\n🏃 RUNNING ALL TESTS")
        print("=" * 40)
        for i in [1, 2, 3]:
            test_file = TESTING_DIR / test_files[i]
            if test_file.exists():
                print(f"\n--- {test_files[i]} ---")
                try:
                    exec(open(test_file).read())
                except Exception as e:
                    print(f"❌ Error in {test_files[i]}: {e}")
                print("\n" + "="*40)
    else:
        print("❌ Invalid option")

def main():
    """Main function"""
    while True:
        try:
            show_menu()
            choice = input("Select test option (0-5): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            
            try:
                test_num = int(choice)
                if 1 <= test_num <= 5:
                    run_test(test_num)
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Please enter a number between 0-5")
            except ValueError:
                print("❌ Please enter a valid number")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
