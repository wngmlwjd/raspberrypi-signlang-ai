# utils.py
"""
MediaPipe 처리에 필요한 공통 함수들
"""

import os
import cv2

def extract_frames_from_video(video_path, output_dir):
    """
    영상에서 프레임을 추출해 output_dir에 연속 이미지로 저장
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    return frame_count

def get_video_frame_count(video_path):
    """비디오 파일에서 총 프레임 개수를 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return -1
    # cv2.CAP_PROP_FRAME_COUNT 속성을 사용하여 총 프레임 개수를 가져옴
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count