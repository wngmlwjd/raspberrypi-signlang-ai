"""
MediaPipe 전처리 관련 유틸
"""

import os
import cv2
from landmark_extractor.config import RAW_DIR, VIDEO_LIST_PATH
from utils import log_message  # 루트 폴더 utils.py의 로그 함수 사용

def extract_frames_from_video(video_path, output_dir):
    """
    영상에서 프레임 추출 후 output_dir에 저장
    """
    os.makedirs(output_dir, exist_ok=True)

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
    log_message(f"[Frames] {video_path} -> {frame_count} frames saved to {output_dir}")
    return frame_count
