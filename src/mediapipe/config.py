# config.py
"""
MediaPipe 세팅 및 파라미터 정의
"""

# MediaPipe Hands 설정
MEDIAPIPE_HANDS_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# 영상 프레임 저장 경로 (전처리 이미지 저장 폴더 등)
FRAME_OUTPUT_DIR = "dataset/processed/frames/"