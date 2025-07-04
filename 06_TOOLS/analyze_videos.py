#!/usr/bin/env python3
"""
Video analyzer script for parking detection system
"""

import sys
import os
sys.path.append('src')

from src.config import Config
from src.utils import get_video_info

def main():
    print("🎬 ANALISIS VIDEO PARKIR - INFORMASI DETAIL")
    print("=" * 60)
    
    # Load configuration
    config = Config('config/config.yaml')
    
    print(f"📊 Minimum resolution requirement: {config.MIN_WIDTH}x{config.MIN_HEIGHT}")
    print("=" * 60)
    
    total_frames = 0
    total_duration = 0
    valid_videos = 0
    
    for i, video_path in enumerate(config.VIDEO_PATHS, 1):
        video_name = os.path.basename(video_path)
        print(f"\n📹 VIDEO {i}: {video_name}")
        print("-" * 50)
        
        if os.path.exists(video_path):
            info = get_video_info(video_path)
            
            if info:
                # Display detailed information
                print(f"• Jumlah Frame    : {info['total_frames']:,} frames")
                print(f"• Durasi Video    : {info['duration']:.1f} detik ({info['duration']/60:.1f} menit)")
                print(f"• Kecepatan FPS   : {info['fps']:.2f} fps")
                print(f"• Size Gambar     : {info['width']}x{info['height']} pixels")
                
                # Additional info
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                print(f"• File Size       : {file_size:.1f} MB")
                
                # Validation
                if info['width'] >= config.MIN_WIDTH and info['height'] >= config.MIN_HEIGHT:
                    print(f"• Status          : ✅ VALID (memenuhi requirement)")
                    valid_videos += 1
                else:
                    print(f"• Status          : ❌ INVALID (di bawah minimum)")
                
                # Processing estimation
                processing_time = info['duration'] / config.FRAME_SKIP
                print(f"• Est. Process    : {processing_time:.1f} detik (dengan frame skip {config.FRAME_SKIP})")
                
                total_frames += info['total_frames']
                total_duration += info['duration']
            else:
                print("❌ Gagal membaca informasi video")
        else:
            print("❌ File tidak ditemukan")
    
    # Summary
    print(f"\n📊 RINGKASAN")
    print("=" * 60)
    print(f"• Total Videos    : {len(config.VIDEO_PATHS)}")
    print(f"• Valid Videos    : {valid_videos}/{len(config.VIDEO_PATHS)}")
    print(f"• Total Frames    : {total_frames:,}")
    print(f"• Total Duration  : {total_duration:.1f} detik ({total_duration/60:.1f} menit)")
    print(f"• Est. Process    : {total_duration/config.FRAME_SKIP:.1f} detik")
    
    if valid_videos > 0:
        avg_fps = sum(get_video_info(v)['fps'] for v in config.VIDEO_PATHS if os.path.exists(v) and get_video_info(v)) / valid_videos
        print(f"• Average FPS     : {avg_fps:.2f}")
    
    print(f"\n🎯 SISTEM SIAP: {valid_videos > 0}")

if __name__ == "__main__":
    main()
