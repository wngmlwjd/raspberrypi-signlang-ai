import os
import glob
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from train.config import *
from models.model_v1 import build_lstm_model

# ----------------------------
# 0. NPY 데이터 로드
# ----------------------------
def load_npy_data(features_dir, labels_dir, sequence_length):
    """
    NPY 시퀀스를 로드하고, 길이가 sequence_length와 다르면 패딩 또는 스킵.
    """
    X, y = [], []
    feature_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    
    for feat_path in feature_files:
        arr = np.load(feat_path)
        
        # 길이가 sequence_length보다 짧으면 0으로 패딩
        if arr.shape[0] < sequence_length:
            pad_len = sequence_length - arr.shape[0]
            pad_shape = (pad_len, arr.shape[1])
            arr = np.vstack([arr, np.zeros(pad_shape, dtype=arr.dtype)])
        elif arr.shape[0] > sequence_length:
            # 길면 슬라이싱
            arr = arr[:sequence_length]

        # 길이 맞춤 후 체크
        if arr.shape[0] != sequence_length:
            continue  # 혹시 예상치 못한 문제 발생 시 skip

        X.append(arr)
        
        lbl_path = feat_path.replace(features_dir, labels_dir).replace(".npy", ".txt")
        with open(lbl_path, "r") as f:
            y.append(int(f.read().strip()))
    
    X = np.array(X)
    y = np.array(y)
    
    # 로드 확인
    print(f"✅ Loaded {len(X)} samples. Shape: {X.shape}")
    return X, y

# ----------------------------
# 1. 데이터 로드 및 train/val split
# ----------------------------
X, y = load_npy_data(TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR, SEQUENCE_LENGTH)

# train/validation split (0.8 / 0.2)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)

# ----------------------------
# 2. 모델 정의
# ----------------------------
model = build_lstm_model(
    input_timesteps=X_train.shape[1],
    input_features=X_train.shape[2],
    num_classes=NUM_CLASSES,
    lstm_units_1=LSTM_UNITS_1,
    lstm_units_2=LSTM_UNITS_2,
    dense_units=DENSE_UNITS,
    dropout_rate=DROPOUT_RATE
)

model.summary()

# ----------------------------
# 3. 콜백 설정
# ----------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 새 버전 폴더 생성
existing = [d for d in os.listdir(MODEL_SAVE_DIR) if os.path.isdir(os.path.join(MODEL_SAVE_DIR, d)) and d.startswith("v")]
if not existing:
    next_idx = 1
else:
    nums = [int(d.replace("v", "")) for d in existing if d.replace("v", "").isdigit()]
    next_idx = max(nums) + 1 if nums else 1

model_dir = os.path.join(MODEL_SAVE_DIR, f"v{next_idx}")
os.makedirs(model_dir, exist_ok=True)

checkpoint_path = os.path.join(model_dir, "best_model.h5")
checkpoint_cb = ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ----------------------------
# 4. 모델 학습
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# ----------------------------
# 5. 모델 저장
# ----------------------------
model_path = os.path.join(model_dir, "model.h5")
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

# ----------------------------
# 6. 학습 설정 저장
# ----------------------------
config_to_save = {
    "SEED": SEED,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
    "SEQUENCE_STEP": SEQUENCE_STEP,
    "NUM_CLASSES": NUM_CLASSES,
    "LSTM_UNITS_1": LSTM_UNITS_1,
    "LSTM_UNITS_2": LSTM_UNITS_2,
    "DENSE_UNITS": DENSE_UNITS,
    "DROPOUT_RATE": DROPOUT_RATE,
    "EarlyStopping_monitor": "val_loss",
    "EarlyStopping_patience": 10,
    "EarlyStopping_restore_best_weights": True,
    "ReduceLROnPlateau_monitor": "val_loss",
    "ReduceLROnPlateau_factor": 0.5,
    "ReduceLROnPlateau_patience": 5,
    "ReduceLROnPlateau_min_lr": 1e-6,
    "ModelCheckpoint_monitor": "val_loss",
    "ModelCheckpoint_save_best_only": True
}

config_path = os.path.join(model_dir, "config.txt")
with open(config_path, "w") as f:
    for key, value in config_to_save.items():
        f.write(f"{key}: {value}\n")
print(f"✅ Training config (with callbacks) saved to: {config_path}")
