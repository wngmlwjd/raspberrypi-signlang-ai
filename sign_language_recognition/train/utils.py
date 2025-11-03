import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
from time import perf_counter
import numpy as np # pad_landmark_array에서 사용되므로 임포트 유지

# --- 경로 및 환경 설정 ---
BASE_DIR = Path('./dataset')
# 데이터셋의 루트 경로를 설정합니다. (사용자 환경에 맞게 수정 가능)
JSON_ROOT_DIR = BASE_DIR / '수어 영상/1.Training/morpheme'
NPY_ROOT_DIR = BASE_DIR / 'processed/landmarks'
OUTPUT_DIR = BASE_DIR
MODEL_DIR = Path('./sign_language_recognition') / 'models'
PROCESSED_DIR = BASE_DIR / 'preprocessed_npy' # NPY 저장 전용 폴더

# 로그 및 처리 상태 파일 경로
# 훈련 경로/파일명
TRAIN_MANIFEST_PATH = BASE_DIR / 'labels_train.csv'
TRAIN_X_NPY_PATH = PROCESSED_DIR / 'X_train.npy'
TRAIN_Y_NPY_PATH = PROCESSED_DIR / 'Y_train.npy'
ENCODER_PATH = Path('sign_language_recognition/models/encoder_classes.json') # 인코더는 하나만 사용

# 검증 경로/파일명
VAL_MANIFEST_PATH = BASE_DIR / 'labels_val.csv'
VAL_X_NPY_PATH = PROCESSED_DIR / 'X_val.npy'
VAL_Y_NPY_PATH = PROCESSED_DIR / 'Y_val.npy'

MODEL_CHECKPOINT_PATH = Path('./sign_language_recognition/models/best_sign_model.h5')

# 매니페스트 생성 시 사용했던 처리 완료 항목 기록 파일을 로드하기 위해 경로 추가
PROCESSED_MANIFEST_TRAIN = BASE_DIR / 'processed_manifest_data_train.txt'
PROCESSED_MANIFEST_VAL = BASE_DIR / 'processed_manifest_data_val.txt'

# 훈련 및 데이터 설정
SEQUENCE_LENGTH = 30  # 시퀀스 길이 (프레임 수)
# 오버랩을 위한 시퀀스 추출 간격 (preprocess.py에서 사용)
# SEQUENCE_LENGTH=30에 대해 SEQUENCE_STEP=10을 사용하면 20 프레임씩 오버랩됩니다.
SEQUENCE_STEP = 10
MAX_HANDS = 2         # 최대 손 개수 (패딩에 사용)

# 화자 폴더 (01, 02, 03)
JSON_SIGNER_KEYS = ['01', '02', '03']
NPY_SIGNER_SUBDIRS = ['01', '01-1', '02', '02-1', '03', '03-1']

# 매니페스트 파일 헤더
MANIFEST_HEADER = [
    'signer_id', 'word_label', 'start_time', 'end_time',
    'landmark_folder_relative_path', 'original_video_name'
]

# --- 유틸리티 함수 ---

def log_message(message: str):
    """콘솔에 출력합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)

def load_processed_data_set(path: Path) -> Set[str]:
    """이미 매니페스트에 추가된 데이터 목록을 로드합니다."""
    if not path.exists():
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def append_processed_data_line(path: Path, line: str):
    """매니페스트에 추가된 데이터를 기록합니다."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        
def pad_landmark_array(landmark_array: np.ndarray | None):
    """랜드마크 배열을 (MAX_HANDS, 21, 3) 형태로 패딩합니다."""
    
    if landmark_array is None or (isinstance(landmark_array, np.ndarray) and landmark_array.size == 0):
        return np.zeros((MAX_HANDS, 21, 3), dtype=np.float32)
    
    arr = np.array(landmark_array)
    if arr.ndim == 2:  # (21, 3) => 손 1개인 경우
        arr = arr[np.newaxis, ...]  # (1, 21, 3)
    
    padded = np.zeros((MAX_HANDS, 21, 3), dtype=np.float32)
    limit = min(arr.shape[0], MAX_HANDS)
    padded[:limit] = arr[:limit]
    return padded
