import pytest
from tools.video_info import display_video_info

def test_display_video_info(capsys):
    """Test display_video_info function"""
    display_video_info()
    captured = capsys.readouterr()
    
    # Assert that some output is generated
    assert "INFORMASI DETAIL VIDEO PARKIR" in captured.out
    assert "Total video files" in captured.out
    assert "Valid videos" in captured.out
