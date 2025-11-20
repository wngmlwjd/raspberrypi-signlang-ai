# ----------------------------
# test_config.py
# ----------------------------

import os

# ----------------------------
# 1. 환경 설정
# ----------------------------
SEED = 42
BATCH_SIZE = 32

# ----------------------------
# 2. 데이터셋 경로
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

TEST_DIR = os.path.join(PROCESSED_DIR, "test")

TEST_FEATURES_DIR = os.path.join(TEST_DIR, "features")
TEST_LABELS_DIR = os.path.join(TEST_DIR, "labels")

# ----------------------------
# 3. 모델 경로
# ----------------------------
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models") 

SEQUENCE_LENGTH = 30  # 테스트 시퀀스 길이
