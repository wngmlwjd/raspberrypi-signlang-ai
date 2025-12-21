import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.metrics import top_k_categorical_accuracy
import time

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
            label_path = os.path.join(TEST_LABELS_DIR, f.replace(".npy", ".txt"))

            if not os.path.exists(label_path):
                print(f"[WARN] Label missing for {feat_path}")
                continue

            data.append((feat_path, label_path))
    return data


def load_test_data(samples, label_encoder):
    X, y = [], []

    for feat_path, label_path in samples:
        features = np.load(feat_path)  # 이미 (GLOBAL_MAX_FRAMES, JOINTS*3)
        X.append(features)

        word_label = load_label_word(label_path)
        y.append(label_encoder.transform([word_label])[0])

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
# Encoder loader
# =====================================================
def load_label_encoder_safely(path):
    enc = joblib.load(path)

    if isinstance(enc, LabelEncoder):
        return enc

    if isinstance(enc, dict):
        if "classes" in enc:
            le = LabelEncoder()
            le.classes_ = np.array(enc["classes"])
            return le

        if "int_to_label" in enc:
            labels = [enc["int_to_label"][i] for i in sorted(enc["int_to_label"])]
            le = LabelEncoder()
            le.fit(labels)
            return le

    raise ValueError("Unsupported label encoder format.")


# =====================================================
# 테스트 실행
# =====================================================
def test(date):
    model_path = os.path.join(MODEL_SAVE_DIR, date, "best_model.h5")
    encoder_path = os.path.join(LABEL_ENCODER_SAVE_DIR, f"{date}.pkl")
    save_dir = os.path.dirname(model_path)

    print("📌 Loading Label Encoder...")
    label_encoder = load_label_encoder_safely(encoder_path)
    print(f"Loaded Label Encoder: {encoder_path}")

    print("📌 Collecting test data...")
    samples = collect_test_samples()
    if not samples:
        print("No test samples found.")
        return

    X_test, y_test = load_test_data(samples, label_encoder)
    print(f"📌 Number of test samples: {len(X_test)}")

    print(f"📌 Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # 예측 + 추론 시간 측정
    print("⏳ Measuring inference time...")

    start = time.perf_counter()
    y_pred_prob = model.predict(X_test, batch_size=32, verbose=1)
    end = time.perf_counter()

    total_infer_time = end - start
    avg_infer_time = total_infer_time / len(X_test)

    print(f"Total inference time: {total_infer_time:.4f} sec")
    print(f"Average inference time per sample: {avg_infer_time:.6f} sec")

    y_pred = np.argmax(y_pred_prob, axis=1)

    # top-5 accuracy
    y_true_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))
    top5_tensor = top_k_categorical_accuracy(y_true_onehot, y_pred_prob, k=5)
    top5_acc = float(np.mean(top5_tensor.numpy()))
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    # 🔥 top-3 accuracy
    top3_tensor = top_k_categorical_accuracy(y_true_onehot, y_pred_prob, k=3)
    top3_acc = float(np.mean(top3_tensor.numpy()))
    print(f"Top-3 Accuracy: {top3_acc:.4f}")

    # 정확도
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # 혼동행렬 저장
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, label_encoder.classes_, save_dir)
    save_confusion_matrix_txt(cm, save_dir)

    # classification report 저장
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        for cls in label_encoder.classes_:
            cls_report = report[cls]
            f.write(f"{cls}: ")
            f.write(", ".join(f"{k}={v:.4f}" for k, v in cls_report.items()))
            f.write("\n")

        f.write("\nOverall Accuracy:\n")
        f.write(f"{report['accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy:\n{top5_acc:.4f}\n")
        f.write(f"Top-3 Accuracy:\n{top3_acc:.4f}\n")
        f.write(f"Average inference time per sample:\n{avg_infer_time:.6f} sec\n")

    print(f"📄 Saved confusion matrix + report to {save_dir}")


if __name__ == "__main__":
    DATE = "20251210_10"
    test(DATE)
