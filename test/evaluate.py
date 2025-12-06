import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess.config import SEQUENCE_LENGTH, TEST_FEATURES_DIR, TEST_LABELS_DIR
from train.config import (
    INPUT_SIZE, CONV_CHANNELS, CONV_KERNEL, GRU_HIDDEN, GRU_LAYERS, DROPOUT_RATE,
    MODEL_SAVE_DIR
)

# =====================================================
# Helper: label.txt 읽기
# =====================================================
def load_label(label_path):
    with open(label_path, "r") as f:
        return int(f.read().strip())

# =====================================================
# 테스트 데이터 로드
# =====================================================
def collect_test_samples():
    data = []
    for root, _, files in os.walk(TEST_FEATURES_DIR):
        for f in sorted(files):
            if not f.endswith(".npy"):
                continue
            feat_path = os.path.join(root, f)
            label_path = feat_path.replace("features","labels").replace(".npy",".txt")
            if not os.path.exists(label_path):
                print(f"[WARN] Label missing for {feat_path}")
                continue
            data.append((feat_path, label_path))
    return data

def load_test_data(samples, max_joints):
    X, y = [], []
    for feat_path, label_path in samples:
        features = np.load(feat_path)
        seq_len, feat_dim = features.shape
        target_dim = max_joints * 3
        if feat_dim < target_dim:
            features = np.pad(features, ((0,0),(0,target_dim - feat_dim)), mode='constant')
        elif feat_dim > target_dim:
            features = features[:, :target_dim]
        X.append(features)
        y.append(load_label(label_path))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =====================================================
# 혼동 행렬 저장
# =====================================================
def save_confusion_matrix(cm, save_dir, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def save_confusion_matrix_txt(cm, save_dir, filename="confusion_matrix.txt"):
    np.savetxt(os.path.join(save_dir, filename), cm, fmt='%d')

# =====================================================
# 테스트
# =====================================================
def test(model_path):
    print("📌 Collecting test data...")
    samples = collect_test_samples()
    if not samples:
        print("No test samples found.")
        return

    max_joints = max(np.load(feat_path).shape[1]//3 for feat_path, _ in samples)
    X_test, y_test = load_test_data(samples, max_joints)

    print(f"📌 Number of test samples: {len(X_test)}")

    save_dir = os.path.dirname(model_path)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # 예측
    y_pred_prob = model.predict(X_test, batch_size=32, verbose=1)  # verbose=1로 진행 상황 표시
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 정확도 계산 및 출력
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # 결과 저장
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_confusion_matrix(cm, save_dir)
    save_confusion_matrix_txt(cm, save_dir)

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        for cls, metrics in report.items():
            if cls.isdigit():
                f.write(f"Class {cls}: ")
                f.write(", ".join(f"{k}={v:.4f}" for k,v in metrics.items()))
                f.write("\n")
        f.write("\nOverall:\n")
        if isinstance(report["accuracy"], dict):
            for k, v in report["accuracy"].items():
                f.write(f"{k}: {v}\n")
        else:
            f.write(f"accuracy: {report['accuracy']}\n")

if __name__ == "__main__":
    model_path = os.path.join(MODEL_SAVE_DIR, "20251205_3", "best_model.h5")
    test(model_path)
