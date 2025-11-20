"""
MediaPipe 세팅 및 경로 정의
"""

import os

# ===============================
# MediaPipe Hands 설정
# ===============================
MEDIAPIPE_HANDS_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# ===============================
# 경로 설정
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
PREPROCESSED_DIR = os.path.join(DATASET_ROOT, "processed")

# 원본 비디오
RAW_DIR = os.path.join(DATASET_ROOT, "수어 영상")

# 임시 프레임 저장
FRAME_DIR = os.path.join(PREPROCESSED_DIR, "temp_frames")

# 랜드마크 저장
LANDMARKS_DIR = os.path.join(PREPROCESSED_DIR, "landmarks")

# 랜드마크 시각화
DRAW_IMG_DIR = os.path.join(PREPROCESSED_DIR, "drawn_frames")

# ===============================
# 비디오 처리 상태 파일
# ===============================
METADATA_DIR = os.path.join(PREPROCESSED_DIR, "metadata")
os.makedirs(METADATA_DIR, exist_ok=True)

VIDEO_LIST_PATH = os.path.join(METADATA_DIR, "video_list.txt")  # 전체 영상 목록
TOPROCESS_LIST_PATH = os.path.join(METADATA_DIR, "to_process.txt")  # 처리할 영상 목록
PROCESSED_LIST_PATH = os.path.join(METADATA_DIR, "processed.txt")  # 처리 완료
EXTRACTED_LIST_PATH = os.path.join(METADATA_DIR, "extracted.txt")  # 프레임 추출 완료
FAILED_VIDEOS_PATH = os.path.join(METADATA_DIR, "failed.txt")  # 처리 실패
