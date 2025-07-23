"""
Video Information Display Script
Shows detailed information for all parking videos
"""

import cv2
import os
from pathlib import Path
from datetime import datetime

# Import Config from src
from src.config import Config

# Define PROJECT_ROOT for consistent pathing
PROJECT_ROOT = Path(__file__).parent.parent

def get_detailed_video_info(video_path):
    """Get comprehensive video information"""
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Get file size
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    
    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    cap.release()
    
    return {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'file_size_mb': file_size_mb,
        'codec': codec,
        'resolution': f"{width}x{height}",
        'aspect_ratio': round(width/height, 2) if height > 0 else 0
    }

def format_duration(seconds):
    """Format duration in readable format"""
    if seconds < 60:
        return f"{seconds:.1f} detik"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} menit {secs:.1f} detik"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} jam {minutes} menit {secs:.1f} detik"

def display_video_info():
    """Display information for all videos"""
    
    config = Config(str(PROJECT_ROOT / "config" / "config.yaml"))
    video_paths = config.VIDEO_PATHS
    
    print("🎬 INFORMASI DETAIL VIDEO PARKIR")
    print("=" * 80)
    
    all_videos_info = []
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n📹 VIDEO {i}: {os.path.basename(video_path)}")
        print("-" * 60)
        
        info = get_detailed_video_info(video_path)
        
        if info:
            all_videos_info.append(info)
            
            print(f"• Jumlah Frame    : {info['total_frames']:,} frames")
            print(f"• Durasi Video    : {format_duration(info['duration'])}")
            print(f"• Kecepatan FPS   : {info['fps']:.2f} fps")
            print(f"• Size Gambar     : {info['resolution']} pixels")
            print(f"• Aspect Ratio    : {info['aspect_ratio']}:1")
            print(f"• File Size       : {info['file_size_mb']:.1f} MB")
            print(f"• Codec           : {info['codec']}")
            print(f"• Path            : {info['path']}")
            
            # Validation status
            if info['width'] >= config.MIN_WIDTH and info['height'] >= config.MIN_HEIGHT:
                print(f"• Status          : ✅ Valid (memenuhi minimum {config.MIN_WIDTH}x{config.MIN_HEIGHT})")
            else:
                print(f"• Status          : ❌ Invalid (di bawah minimum {config.MIN_WIDTH}x{config.MIN_HEIGHT})")
                
        else:
            print(f"❌ Error: Tidak dapat membaca video {video_path}")
    
    # Summary table
    if all_videos_info:
        print(f"\n📊 RINGKASAN SEMUA VIDEO")
        print("=" * 80)
        
        # Table header
        print("| No | Video     | Frames    | Durasi    | FPS   | Resolusi  | Size    |")
        print("|----|-----------|-----------|-----------|-------|-----------|---------|")
        
        total_frames = 0
        total_duration = 0
        total_size = 0
        
        for i, info in enumerate(all_videos_info, 1):
            video_name = info['filename'][:10]
            frames_str = f"{info['total_frames']:,}"
            duration_str = f"{info['duration']:.1f}s"
            fps_str = f"{info['fps']:.1f}"
            resolution_str = info['resolution']
            size_str = f"{info['file_size_mb']:.1f}MB"
            
            print(f"| {i:2} | {video_name:<9} | {frames_str:<9} | {duration_str:<9} | {fps_str:<5} | {resolution_str:<9} | {size_str:<7} |")
            
            total_frames += info['total_frames']
            total_duration += info['duration']
            total_size += info['file_size_mb']
        
        # Summary statistics
        avg_fps = sum(info['fps'] for info in all_videos_info) / len(all_videos_info)
        
        print("|----|-----------|-----------|-----------|-------|-----------|---------|")
        print(f"| TOT| {len(all_videos_info)} videos  | {total_frames:,} | {format_duration(total_duration)} | {avg_fps:.1f} | Mixed     | {total_size:.1f}MB |")
        
        print(f"\n📈 STATISTIK TAMBAHAN:")
        print(f"• Total video files      : {len(all_videos_info)}")
        print(f"• Total frames           : {total_frames:,}")
        print(f"• Total duration         : {format_duration(total_duration)}")
        print(f"• Average FPS            : {avg_fps:.2f}")
        print(f"• Total file size        : {total_size:.1f} MB")
        print(f"• Processing time est.   : {total_duration/3:.1f}s (dengan frame skip 3)")
        
        # Resolution analysis
        resolutions = list(set(info['resolution'] for info in all_videos_info))
        print(f"• Unique resolutions     : {len(resolutions)} ({', '.join(resolutions)})")
        
        # Validation summary
        valid_videos = sum(1 for info in all_videos_info if info['width'] >= config.MIN_WIDTH and info['height'] >= config.MIN_HEIGHT)
        print(f"• Valid videos (≥{config.MIN_WIDTH}x{config.MIN_HEIGHT}): {valid_videos}/{len(all_videos_info)}")

if __name__ == "__main__":
    display_video_info()
