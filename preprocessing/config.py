import os

# ===============================
# 기본 설정
# ===============================
FPS = 30
USE_WORD_NUM = 30   # 사용할 라벨 개수

# ===============================
# 디렉토리 경로 설정
# ===============================
# 현재 파일 (config.py)의 상위 폴더의 상위 폴더를 BASE_DIR로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
PROCESSED_DIR = os.path.join(DATASET_ROOT, "processed")
LANDMARKS_DIR = os.path.join(PROCESSED_DIR, "landmarks")

# 원본 비디오 디렉토리
RAW_DIR = os.path.join(DATASET_ROOT, "수어 영상")

# 형태소 기반 영상 (Training / Validation) 상위 디렉토리
MORPHEMES_DIR = [
    os.path.join(RAW_DIR, "1.Training", "morpheme"),
    os.path.join(RAW_DIR, "2.Validation", "morpheme"),
]

# ===============================
# 화자 필터링 설정
# ===============================
# Training 에 사용할 화자 번호 리스트
TRAINING_VIDEOS_LIST = [i for i in range(1, 5)]  # ex: 1, 2, 3

# Validation/Test 에 사용할 화자 번호 리스트
VALIDATION_VIDEOS_LIST = [i for i in range(17, 19)]  # ex: 17

# ===============================
# ★★★ 경로 조합 로직 추가 (화자 폴더까지의 경로 생성) ★★★

# Training 화자 디렉토리 경로
TRAINING_DIRS = [
    os.path.join(MORPHEMES_DIR[0], f"{speaker_id:02d}")
    for speaker_id in TRAINING_VIDEOS_LIST
]

# Validation 화자 디렉토리 경로
VALIDATION_DIRS = [
    os.path.join(MORPHEMES_DIR[1], f"{speaker_id:02d}")
    for speaker_id in VALIDATION_VIDEOS_LIST
]

# 전처리 시 이 두 리스트(TRAINING_DIRS, VALIDATION_DIRS)를 병합하여 사용
ALL_MORPHEME_DIRS = TRAINING_DIRS + VALIDATION_DIRS

# ===============================
# 라벨 리스트 설정
# ===============================
# 전체 라벨 목록 파일
LABEL_LIST_PATH = os.path.join(LABELS_DIR, "labels_list.txt")

# USE_WORD_NUM 만큼 필터링된 라벨 파일
USE_LABELS_LIST_PATH = os.path.join(LABELS_DIR, f"labels_{USE_WORD_NUM}.txt")
# → 이 라벨 리스트에 포함된 라벨만 사용함

# ===============================
# 전처리 후 저장될 경로
# ===============================
TRAIN_DATASET_DIR = os.path.join(PROCESSED_DIR, "train")
TEST_DATASET_DIR = os.path.join(PROCESSED_DIR, "test")

TRAIN_FEATURES_DIR = os.path.join(TRAIN_DATASET_DIR, "features")   # 입력 데이터
TRAIN_LABELS_DIR = os.path.join(TRAIN_DATASET_DIR, "labels")       # 라벨 데이터

TEST_FEATURES_DIR = os.path.join(TEST_DATASET_DIR, "features")     # 입력 데이터
TEST_LABELS_DIR = os.path.join(TEST_DATASET_DIR, "labels")         # 라벨 데이터

# ===============================
# 데이터 분할 및 시퀀스 설정
# ===============================
TRAIN_TEST_SPLIT = 0.8     # Train 80% / Test 20% 비율
SEQUENCE_LENGTH = 30       # 시퀀스 길이
SEQUENCE_STEP = 5          # 슬라이딩 윈도우 이동 크기