import cv2
import os

print("🎬 INFORMASI VIDEO PARKIR")
print("=" * 50)

videos = [
    'parking_area/video/park1.mp4',
    'parking_area/video/park2.mp4', 
    'parking_area/video/park3.mp4',
    'parking_area/video/park4.mp4'
]

for i, video_path in enumerate(videos, 1):
    print(f"\n📹 VIDEO {i}: {os.path.basename(video_path)}")
    print("-" * 40)
    
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps
            
            print(f"• Jumlah Frame : {frames:,} frames")
            print(f"• Durasi Video : {duration:.1f} detik")
            print(f"• Kecepatan FPS: {fps:.2f} fps")
            print(f"• Size Gambar  : {width}x{height} pixels")
            
            cap.release()
        else:
            print("❌ Gagal membuka video")
    else:
        print("❌ File tidak ditemukan")
