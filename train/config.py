import os

# ===============================
# 학습 시드 및 하이퍼파라미터
# ===============================
SEED = 123                     # 재현성 확보용 시드
BATCH_SIZE = 64                 # 배치 사이즈
EPOCHS = 200                     # 학습 에폭 수
LEARNING_RATE = 3e-4            # Adam 학습률
DROPOUT_RATE = 0.25              # Dropout 비율

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
# ConvGRU 모델 하이퍼파라미터
# ===============================
INPUT_SIZE = 42          # 입력 feature 차원 (x,y 좌표 등)
SEQ_LEN = 30             # 시퀀스 길이
CONV_CHANNELS = [64, 128, 128]       # Conv1D 출력 채널
CONV_KERNEL = 3          # Conv1D 커널 크기
GRU_HIDDEN = 160         # GRU hidden size
GRU_LAYERS = 3           # GRU 레이어 수

# ===============================
# 모델 저장 경로
# ===============================
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "conv1d+gru")
