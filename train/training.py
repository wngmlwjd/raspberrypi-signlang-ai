import os
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

from preprocess.config import SEQUENCE_LENGTH, TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR, USE_WORD_NUM
from train.config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    CONV_CHANNELS, CONV_KERNEL, GRU_HIDDEN, GRU_LAYERS,
    MODEL_SAVE_DIR, LABEL_ENCODER_SAVE_DIR
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
# Helper: 라벨 인코더 저장
# =====================================================
def build_label_encoder(label_dir):
    labels = set()

    for root, _, files in os.walk(label_dir):
        for f in files:
            if not f.endswith(".txt"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fp:
                labels.add(fp.read().strip())

    labels = sorted(list(labels))
    label_to_int = {label: idx for idx, label in enumerate(labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    return label_to_int, int_to_label, labels


def save_label_encoder(save_dir, file_name, label_to_int, int_to_label, labels):
    os.makedirs(save_dir, exist_ok=True)

    encoder_path = os.path.join(save_dir, f"{file_name}.pkl")
    list_path = os.path.join(save_dir, f"{file_name}.txt")

    with open(encoder_path, "wb") as f:
        pickle.dump({"label_to_int": label_to_int, "int_to_label": int_to_label}, f)

    with open(list_path, "w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(lbl + "\n")

    print(f"📌 Label encoder saved: {encoder_path}")
    return encoder_path


# =====================================================
# Helper: label.txt 읽기 (문자 → 정수)
# =====================================================
def load_label(label_path, label_to_int):
    with open(label_path, "r", encoding="utf-8") as f:
        label_str = f.read().strip()
    return label_to_int[label_str]


# =====================================================
# 데이터 수집/그룹화/분할
# =====================================================
def collect_all_samples(label_to_int):
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

            class_id = load_label(label_path, label_to_int)
            data.append((feat_path, label_path, speaker, class_id))

    return data


def group_by_speaker_and_class(samples):
    grouped = {}
    for feat, label, speaker, cls in samples:
        grouped.setdefault(speaker, {})
        grouped[speaker].setdefault(cls, [])
        grouped[speaker][cls].append((feat, label))
    return grouped


def stratified_split(grouped):
    train_list = []
    val_list = []
    for speaker, cls_dict in grouped.items():
        for cls, items in cls_dict.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * 0.8)
            train_list.extend(items[:n_train])
            val_list.extend(items[n_train:])
    return train_list, val_list


# =====================================================
# Dataset 생성
# =====================================================
def load_data(samples, max_joints, label_to_int):
    X, y = [], []
    for feat_path, label_path in samples:
        features = np.load(feat_path)

        seq_len, feat_dim = features.shape
        target_dim = max_joints * 3

        # 패딩/자르기
        if feat_dim < target_dim:
            features = np.pad(features, ((0, 0), (0, target_dim - feat_dim)), mode='constant')
        else:
            features = features[:, :target_dim]

        X.append(features)
        y.append(load_label(label_path, label_to_int))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


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
    return folder_path, folder_name


# =====================================================
# 학습
# =====================================================
def train():
    save_dir, folder_name = create_date_folder(MODEL_SAVE_DIR)
    
    print("📌 Building label encoder from training labels...")
    label_to_int, int_to_label, labels = build_label_encoder(TRAIN_LABELS_DIR)

    # 라벨 인코더 저장
    encoder_path = save_label_encoder(LABEL_ENCODER_SAVE_DIR, folder_name, label_to_int, int_to_label, labels)

    print("📌 Collecting feature-label pairs...")
    samples = collect_all_samples(label_to_int)

    grouped = group_by_speaker_and_class(samples)
    train_samples, val_samples = stratified_split(grouped)

    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    max_joints = max(np.load(feat_path).shape[1] // 3 for feat_path, *_ in samples)
    print(f"Max joints: {max_joints}")
    
    maxj_path = os.path.join(save_dir, "max_joints.txt")
    with open(maxj_path, "w") as f:
        f.write(str(max_joints))
    print(f"📌 max_joints saved to {maxj_path}")

    X_train, y_train = load_data(train_samples, max_joints, label_to_int)
    X_val, y_val = load_data(val_samples, max_joints, label_to_int)

    num_classes = len(labels)

    # 모델 생성
    model = build_convgru_model_keras(
        input_size=max_joints * 3,
        seq_len=SEQUENCE_LENGTH,
        num_classes=num_classes,
        conv_channels=CONV_CHANNELS,
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
    
    save_path = os.path.join(save_dir, "best_model.h5")

    print(f"📌 Saving models to: {save_path}")

    checkpoint_cb = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
    reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

    # =====================================================
    # config.txt 저장
    # =====================================================
    config_txt_path = os.path.join(save_dir, "config.txt")
    with open(config_txt_path, "w") as f:
        f.write("===== 기본 설정 =====\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"EPOCHS={EPOCHS}\n")
        f.write(f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}\n\n")

        f.write("===== 모델 하이퍼파라미터 =====\n")
        f.write(f"INPUT_SIZE={max_joints * 3}\n")
        f.write(f"NUM_CLASSES={USE_WORD_NUM}\n")
        f.write(f"MAX_JOINTS={max_joints}\n")
        f.write(f"CONV_CHANNELS={CONV_CHANNELS}\n")
        f.write(f"CONV_KERNEL={CONV_KERNEL}\n")
        f.write(f"GRU_HIDDEN={GRU_HIDDEN}\n")
        f.write(f"GRU_LAYERS={GRU_LAYERS}\n")
        f.write(f"DROPOUT_RATE={DROPOUT_RATE}\n")
        f.write(f"LEARNING_RATE={LEARNING_RATE}\n\n")

        f.write("===== 데이터 정보 =====\n")
        f.write(f"TRAIN_SAMPLES={len(train_samples)}\n")
        f.write(f"VAL_SAMPLES={len(val_samples)}\n\n")

        f.write("===== 콜백 정보 =====\n")
        f.write(f"ModelCheckpoint: monitor={checkpoint_cb.monitor}, save_best_only={checkpoint_cb.save_best_only}\n")
        f.write(f"ReduceLROnPlateau: monitor={reduce_cb.monitor}, factor={reduce_cb.factor}, patience={reduce_cb.patience}, min_lr={reduce_cb.min_lr}\n")
        f.write(f"EarlyStopping: monitor={earlystop_cb.monitor}, patience={earlystop_cb.patience}, restore_best_weights={earlystop_cb.restore_best_weights}\n\n")

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

    print("✅ Training completed.")


if __name__ == "__main__":
    train()
