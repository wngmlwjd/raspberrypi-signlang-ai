import os

# ===============================
# 학습 시드 및 하이퍼파라미터
# ===============================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100

# ===============================
# 시퀀스 관련
# ===============================
SEQUENCE_LENGTH = 30   # 시퀀스 길이
SEQUENCE_STEP = 5      # 슬라이딩 윈도우 이동 크기

# ===============================
# 클래스 개수
# ===============================
NUM_CLASSES = 30  # USE_WORD_NUM과 매칭

# ===============================
# 전처리된 데이터 경로
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

TRAIN_FEATURES_DIR = os.path.join(PROCESSED_DIR, "train", "features")
TRAIN_LABELS_DIR   = os.path.join(PROCESSED_DIR, "train", "labels")

TEST_FEATURES_DIR  = os.path.join(PROCESSED_DIR, "test", "features")
TEST_LABELS_DIR    = os.path.join(PROCESSED_DIR, "test", "labels")

# ===============================
# LSTM 모델 하이퍼파라미터
# ===============================
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS  = 64
DROPOUT_RATE = 0.3

# ===============================
# 모델 저장 경로
# ===============================
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
