# evaluate.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------
# 1. 설정 불러오기
# ----------------------------
from test.config import TEST_FEATURES_DIR, TEST_LABELS_DIR, MODEL_SAVE_DIR, SEQUENCE_LENGTH

# ----------------------------
# 2. 테스트 데이터 로드 (패딩/슬라이싱 포함)
# ----------------------------
def load_npy_data(features_dir, labels_dir, sequence_length):
    X, y = [], []
    feature_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    
    for feat_path in feature_files:
        arr = np.load(feat_path)
        
        # 시퀀스 길이 맞추기
        if arr.shape[0] < sequence_length:
            pad_len = sequence_length - arr.shape[0]
            pad_shape = (pad_len, arr.shape[1])
            arr = np.vstack([arr, np.zeros(pad_shape, dtype=arr.dtype)])
        elif arr.shape[0] > sequence_length:
            arr = arr[:sequence_length]

        X.append(arr)
        
        lbl_path = os.path.join(labels_dir, os.path.basename(feat_path).replace(".npy", ".txt"))
        with open(lbl_path, "r") as f:
            y.append(int(f.read().strip()))
    
    X = np.array(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} samples. Shape: {X.shape}")
    return X, y

# ----------------------------
# 3. 최신 모델 경로 가져오기
# ----------------------------
def get_latest_model_path(model_base_dir):
    versions = [d for d in os.listdir(model_base_dir) 
                if os.path.isdir(os.path.join(model_base_dir, d)) and d.startswith("v")]
    if not versions:
        raise FileNotFoundError(f"No model version folder found in {model_base_dir}")
    versions.sort(key=lambda x: int(x.replace("v", "")))
    latest_version = versions[-1]
    model_path = os.path.join(model_base_dir, latest_version, "best_model.h5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path, latest_version

# ----------------------------
# 4. 평가 함수 및 저장
# ----------------------------
def evaluate_model(model_path, X_test, y_test, save_dir):
    model = load_model(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    y_pred_prob = model.predict(X_test, batch_size=32)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # 성능 지표 계산
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    print("\n--- Test Performance ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    
    # ----------------------------
    # 성능 지표 저장 (txt)
    # ----------------------------
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("--- Test Performance ---\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
    print(f"✅ Metrics saved to {metrics_path}")
    
    # ----------------------------
    # Confusion Matrix 저장 (이미지)
    # ----------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {cm_path}")
    
    return acc, precision, recall, f1

# ----------------------------
# 5. 실행
# ----------------------------
if __name__ == "__main__":
    X_test, y_test = load_npy_data(TEST_FEATURES_DIR, TEST_LABELS_DIR, SEQUENCE_LENGTH)
    model_path, latest_version = get_latest_model_path(MODEL_SAVE_DIR)
    print(f"🔹 Evaluating model from version: {latest_version}")
    
    save_dir = os.path.join(MODEL_SAVE_DIR, latest_version)
    evaluate_model(model_path, X_test, y_test, save_dir)
