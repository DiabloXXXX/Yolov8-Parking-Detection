"""
Initialization file for the vehicle detection package
"""

from .config import Config
from .vehicle_detector import VehicleDetector
from .video_processor import VideoProcessor
from . import utils

__version__ = "2.0.0"
__author__ = "Vehicle Detection System"

__all__ = [
    'Config',
    'VehicleDetector', 
    'VideoProcessor',
    'utils'
]
