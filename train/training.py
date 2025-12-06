import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

from preprocess.config import SEQUENCE_LENGTH, TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR, USE_WORD_NUM
from train.config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    INPUT_SIZE, SEQ_LEN, CONV_CHANNELS, CONV_KERNEL, GRU_HIDDEN, GRU_LAYERS,
    MODEL_SAVE_DIR
)
from models.gru_4 import build_convgru_model_keras


# =====================================================
# 랜덤 시드 고정
# =====================================================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# Helper: 화자 ID 추출
# =====================================================
def extract_speaker_id(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.isdigit() and len(p) <= 4:
            return p
    return None

# =====================================================
# Helper: label.txt 읽기
# =====================================================
def load_label(label_path):
    with open(label_path, "r") as f:
        return int(f.read().strip())

# =====================================================
# 데이터 수집/그룹화/분할
# =====================================================
def collect_all_samples():
    data = []
    for root, _, files in os.walk(TRAIN_FEATURES_DIR):
        for f in sorted(files):
            if not f.endswith(".npy"):
                continue
            feat_path = os.path.join(root, f)
            speaker = extract_speaker_id(feat_path)
            label_path = feat_path.replace("features","labels").replace(".npy",".txt")
            if not os.path.exists(label_path):
                print(f"[WARN] Label missing for {feat_path}")
                continue
            class_id = load_label(label_path)
            data.append((feat_path, label_path, speaker, class_id))
    return data

def group_by_speaker_and_class(samples):
    grouped = {}
    for feat, label, speaker, cls in samples:
        grouped.setdefault(speaker,{})
        grouped[speaker].setdefault(cls,[])
        grouped[speaker][cls].append((feat,label))
    return grouped

def stratified_split(grouped):
    train_list = []
    val_list = []
    for speaker, cls_dict in grouped.items():
        for cls, items in cls_dict.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n*0.8)
            train_list.extend(items[:n_train])
            val_list.extend(items[n_train:])
    return train_list, val_list

# =====================================================
# Dataset 생성
# =====================================================
def load_data(samples, max_joints):
    X, y = [], []
    for feat_path, label_path in samples:
        features = np.load(feat_path)
        # 패딩/자르기
        seq_len, feat_dim = features.shape
        target_dim = max_joints * 3
        if feat_dim < target_dim:
            pad_width = target_dim - feat_dim
            features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
        elif feat_dim > target_dim:
            features = features[:, :target_dim]

        X.append(features)
        y.append(load_label(label_path))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

# =====================================================
# 날짜 폴더 생성 + 중복 방지
# =====================================================
def create_date_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    existing = [d for d in os.listdir(base_dir) if d.startswith(today)]
    if not existing:
        folder_name = f"{today}_1"
    else:
        nums = [int(d.split("_")[-1]) for d in existing if "_" in d]
        next_num = max(nums) + 1 if nums else 1
        folder_name = f"{today}_{next_num}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# =====================================================
# 학습
# =====================================================
def train():
    print("📌 Collecting data...")
    samples = collect_all_samples()
    grouped = group_by_speaker_and_class(samples)
    train_samples, val_samples = stratified_split(grouped)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    # 최대 joint 수
    max_joints = max(np.load(feat_path).shape[1]//3 for feat_path, *_ in samples)
    print(f"Max joints: {max_joints}")

    X_train, y_train = load_data(train_samples, max_joints)
    X_val, y_val = load_data(val_samples, max_joints)
    USE_WORD_NUM = len(np.unique(y_train))

    # 모델 생성
    model = build_convgru_model_keras(
        input_size=max_joints*3,
        seq_len=SEQUENCE_LENGTH,
        num_classes=USE_WORD_NUM,
        conv_channels=CONV_CHANNELS,  # 리스트로 변경
        conv_kernel=CONV_KERNEL,
        gru_hidden=GRU_HIDDEN,
        gru_layers=GRU_LAYERS,
        dropout=DROPOUT_RATE
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 결과 저장 폴더 생성
    save_dir = create_date_folder(MODEL_SAVE_DIR)
    save_path = os.path.join(save_dir, "best_model.h5")
    print(f"Saving models to: {save_path}")

    # 콜백 정의
    checkpoint_cb = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
    reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    
    # =====================================================
    # config.txt 저장 (콜백 정보 포함)
    # =====================================================
    config_txt_path = os.path.join(save_dir, "config.txt")
    with open(config_txt_path, "w") as f:
        f.write("===== 기본 설정 =====\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"EPOCHS={EPOCHS}\n")
        f.write(f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}\n")
        f.write(f"TRAIN_FEATURES_DIR={TRAIN_FEATURES_DIR}\n")
        f.write(f"TRAIN_LABELS_DIR={TRAIN_LABELS_DIR}\n\n")

        f.write("===== 모델/학습 하이퍼파라미터 =====\n")
        f.write(f"MODEL_TYPE=ConvGRUSignModel (Keras)\n")
        f.write(f"INPUT_SIZE={max_joints*3}\n")
        f.write(f"NUM_CLASSES={USE_WORD_NUM}\n")
        f.write(f"MAX_JOINTS={max_joints}\n")
        f.write(f"DROP_OUT={DROPOUT_RATE}\n")
        f.write(f"OPTIMIZER=Adam\n")
        f.write(f"LEARNING_RATE={LEARNING_RATE}\n")
        f.write(f"LOSS_FUNCTION=sparse_categorical_crossentropy\n\n")

        f.write("===== 데이터 정보 =====\n")
        f.write(f"TRAIN_SAMPLES={len(train_samples)}\n")
        f.write(f"VAL_SAMPLES={len(val_samples)}\n\n")

        f.write("===== 콜백 정보 =====\n")
        f.write(f"ModelCheckpoint:\n")
        f.write(f"  monitor={checkpoint_cb.monitor}\n")
        f.write(f"  save_best_only={checkpoint_cb.save_best_only}\n")
        f.write(f"  verbose={checkpoint_cb.verbose}\n")
        f.write(f"ReduceLROnPlateau:\n")
        f.write(f"  monitor={reduce_cb.monitor}\n")
        f.write(f"  factor={reduce_cb.factor}\n")
        f.write(f"  patience={reduce_cb.patience}\n")
        f.write(f"  verbose={reduce_cb.verbose}\n")
        f.write(f"  min_lr={reduce_cb.min_lr}\n")
        f.write(f"EarlyStopping:\n")
        f.write(f"  monitor={earlystop_cb.monitor}\n")
        f.write(f"  patience={earlystop_cb.patience}\n")
        f.write(f"  verbose={earlystop_cb.verbose}\n")
        f.write(f"  restore_best_weights={earlystop_cb.restore_best_weights}\n\n")

        f.write(f"DATE={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Config saved to {config_txt_path}")

    # 학습
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, reduce_cb, earlystop_cb],
        verbose=1
    )

    print("Training completed.")

if __name__ == "__main__":
    train()
