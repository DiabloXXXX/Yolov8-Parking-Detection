import pytest
import os
from pathlib import Path

# Import modules from src
from src.config import Config
from src.vehicle_detector import VehicleDetector

# Import the utility tester
from tools.tester_util import VehicleDetectionTester

# Define PROJECT_ROOT for consistent pathing in tests
PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture(scope="module")
def config_and_detector():
    """Fixture to provide a Config and VehicleDetector instance"""
    config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
    detector = VehicleDetector(config)
    return config, detector

def test_comprehensive_testing_integration(config_and_detector):
    """Integration test for VehicleDetectionTester comprehensive testing"""
    config, detector = config_and_detector
    
    tester = VehicleDetectionTester(detector, config)
    tester.run_comprehensive_testing()
    
    # Assert that some results were generated
    assert len(tester.test_results) > 0, "No test results were generated by comprehensive testing."
    
    # Optionally, assert on specific metrics if ground truth is well-defined
    # For example, check if average detection is within a reasonable range
    # assert all(r['avg_detection'] > 0 for r in tester.test_results)
