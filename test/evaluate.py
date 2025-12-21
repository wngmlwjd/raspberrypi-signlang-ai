import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.metrics import top_k_categorical_accuracy 

from preprocess.config import TEST_FEATURES_DIR, TEST_LABELS_DIR
from train.config import MODEL_SAVE_DIR, LABEL_ENCODER_SAVE_DIR

# =====================================================
# label.txt 읽기 (단어 라벨)
# =====================================================
def load_label_word(label_path):
    with open(label_path, "r") as f:
        return f.read().strip()

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

            # 라벨 파일 이름 변환
            label_name = f.replace(".npy", ".txt")
            label_path = os.path.join(TEST_LABELS_DIR, label_name)

            if not os.path.exists(label_path):
                print(f"[WARN] Label missing for {feat_path}")
                continue

            data.append((feat_path, label_path))
    return data

def load_test_data(samples, max_joints, label_encoder):
    X, y = [], []
    for feat_path, label_path in samples:
        features = np.load(feat_path)

        seq_len, feat_dim = features.shape
        target_dim = max_joints * 3

        # pad / cut
        if feat_dim < target_dim:
            features = np.pad(features, ((0,0),(0,target_dim - feat_dim)), mode='constant')
        elif feat_dim > target_dim:
            features = features[:, :target_dim]

        X.append(features)

        word_label = load_label_word(label_path)
        y.append(label_encoder.transform([word_label])[0])   # 단어 → 정수 변환

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =====================================================
# 혼동 행렬 저장
# =====================================================
def save_confusion_matrix(cm, classes, save_dir, filename="confusion_matrix.png"):
    plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def save_confusion_matrix_txt(cm, save_dir, filename="confusion_matrix.txt"):
    np.savetxt(os.path.join(save_dir, filename), cm, fmt='%d')

# =====================================================
# Encoder Loader (통합 버전)
# =====================================================
def load_label_encoder_safely(path):
    enc = joblib.load(path)

    # Case 1: 이미 LabelEncoder 객체
    if isinstance(enc, LabelEncoder):
        return enc

    # Case 2: dict 구조 → classes 혹은 다른 키 여부 확인
    if isinstance(enc, dict):
        if "classes" in enc:
            le = LabelEncoder()
            le.classes_ = np.array(enc["classes"])
            return le

        # 예전 방식: label_to_int 기반
        if "label_to_int" in enc and "int_to_label" in enc:
            labels = [enc["int_to_label"][i] for i in sorted(enc["int_to_label"].keys())]
            le = LabelEncoder()
            le.fit(labels)
            return le

    raise ValueError("❌ Unsupported label encoder format.")

# =====================================================
# 테스트
# =====================================================
def test(date):
    model_path = os.path.join(MODEL_SAVE_DIR, date, "best_model.h5")
    
    print("📌 Loading Label Encoder...")
    encoder_path = os.path.join(LABEL_ENCODER_SAVE_DIR, f"{date}.pkl")
    label_encoder = load_label_encoder_safely(encoder_path)
    print(f"Loaded Label Encoder from: {encoder_path}")

    print("📌 Collecting test data...")
    samples = collect_test_samples()
    if not samples:
        print("No test samples found.")
        return

    max_joints = max(np.load(fp).shape[1]//3 for fp, _ in samples)
    X_test, y_test = load_test_data(samples, max_joints, label_encoder)
    print(f"📌 Number of test samples: {len(X_test)}")

    save_dir = os.path.dirname(model_path)

    print(f"📌 Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # 예측
    y_pred_prob = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Top-5 Accuracy 계산
    top5_correct = top_k_categorical_accuracy(tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_)), y_pred_prob, k=5).numpy()
    top5_acc = np.mean(top5_correct)
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    # 정확도
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = label_encoder.classes_
    save_confusion_matrix(cm, classes, save_dir)
    save_confusion_matrix_txt(cm, save_dir)

    # classification report
    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True
    )

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        for cls in classes:
            cls_report = report[cls]
            f.write(f"{cls}: ")
            f.write(", ".join(f"{k}={v:.4f}" for k, v in cls_report.items()))
            f.write("\n")

        f.write("\nOverall Accuracy:\n")
        f.write(f"{report['accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy:\n{top5_acc:.4f}\n")   # Top-5 저장

    print(f"📄 Saved confusion matrix + report to {save_dir}")
    
if __name__ == "__main__":
    DATE = "20251208_2"
    test(DATE)
