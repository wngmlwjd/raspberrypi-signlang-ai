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

# def get_video_frame_count(video_path):
#     """비디오 총 프레임 수 반환"""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         log_message(f"[Error] Cannot open video {video_path}")
#         return -1

#     count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return count

# def save_video_list(root_video_dir=RAW_DIR, save_path=VIDEO_LIST_PATH):
#     """
#     RAW_DIR 하위 모든 mp4/avi 영상 목록 txt로 저장
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     video_list = []
#     for root, _, files in os.walk(root_video_dir):
#         for f in sorted(files):
#             if f.lower().endswith((".mp4", ".avi")):
#                 full_path = os.path.join(root, f)
#                 rel_path = os.path.relpath(full_path, root_video_dir)
#                 video_list.append(rel_path)

#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(video_list))

#     log_message(f"[VideoList] Saved {len(video_list)} videos to {save_path}")
