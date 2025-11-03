import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import json
from sign_language_recognition.train.utils import log_message, MODEL_CHECKPOINT_PATH, X_NPY_PATH, Y_NPY_PATH, ENCODER_PATH
from sklearn.model_selection import train_test_split


def load_label_encoder_map(encoder_path: Path) -> dict:
    """
    레이블 인코더 파일을 로드하고, 인덱스(int)를 단어(str)로 매핑하는 딕셔너리를 반환합니다.
    JSON 파일 구조: {"classes": ["word1", "word2", ...]}
    """
    if not encoder_path.exists():
        log_message(f"Error: Label encoder file not found at {encoder_path}")
        return {}
    
    with open(encoder_path, 'r', encoding='utf-8') as f:
        try:
            # 원본 JSON 로드: {"classes": [word, ...]} 형태
            data = json.load(f)
            
            # "classes" 키에서 단어 리스트 추출
            classes = data.get('classes', [])
            
            # 인덱스(0, 1, 2, ...)를 키로, 단어를 값으로 매핑
            index_to_label = {i: label for i, label in enumerate(classes)}
            
            return index_to_label
            
        except Exception as e:
            log_message(f"Error loading or parsing label encoder JSON: {repr(e)}")
            return {}


def test_sign_language_model(num_samples_to_test: int = -1, validation_split: float = 0.2):
    """
    Loads the trained model and evaluates its performance on the validation set.

    Args:
        num_samples_to_test (int): Number of samples to use for testing (-1 for all).
        validation_split (float): The validation split used during training (to replicate the split).
    """
    log_message("--- Model Test and Evaluation Start ---")

    # 1. Load Model Checkpoint
    if not MODEL_CHECKPOINT_PATH.exists():
        log_message(f"Error: Trained model not found at {MODEL_CHECKPOINT_PATH}. Cannot proceed with testing.")
        return

    try:
        model = tf.keras.models.load_model(str(MODEL_CHECKPOINT_PATH))
        log_message(f"Successfully loaded model from: {MODEL_CHECKPOINT_PATH.name}")
    except Exception as e:
        log_message(f"Error loading model: {repr(e)}")
        return

    # 2. Load Data and Label Encoder
    if not X_NPY_PATH.exists() or not Y_NPY_PATH.exists():
        log_message("Error: Processed data (x.npy or y.npy) not found. Run preprocessing first.")
        return

    try:
        X = np.load(X_NPY_PATH)
        Y = np.load(Y_NPY_PATH)
        log_message(f"Data loaded. X shape: {X.shape}, Y shape: {Y.shape}")
        
        index_to_label_map = load_label_encoder_map(ENCODER_PATH)
        
        # 클래스 이름 리스트 생성 (정렬된 인덱스 순서대로)
        sorted_indices = sorted(index_to_label_map.keys())
        class_names = [index_to_label_map.get(i, f"Class_{i}") for i in sorted_indices]
        
    except Exception as e:
        # 이전에 발생했던 numpy 로드 오류나 다른 예외를 처리
        log_message(f"Error loading numpy data or encoder: {repr(e)}")
        return

    # 3. Data Split (Replicating Training Split)
    # 훈련 시 사용한 random_state=42와 stratify=Y를 그대로 사용해야 합니다.
    _, X_val, _, Y_val = train_test_split(
        X, Y, test_size=validation_split, shuffle=True, random_state=42, stratify=Y
    )
    log_message(f"Validation dataset size: {X_val.shape[0]} samples.")
    
    # 4. Determine Test Samples
    if num_samples_to_test > 0 and num_samples_to_test < X_val.shape[0]:
        X_test = X_val[:num_samples_to_test]
        Y_true = Y_val[:num_samples_to_test]
        log_message(f"Using a subset of {num_samples_to_test} samples for testing.")
    else:
        X_test = X_val
        Y_true = Y_val
        log_message(f"Using the entire validation set ({X_test.shape[0]} samples) for testing.")

    # 5. Model Prediction
    log_message("Starting prediction...")
    Y_pred_proba = model.predict(X_test, verbose=0)
    Y_pred = np.argmax(Y_pred_proba, axis=1)

    # 6. Evaluation and Reporting
    
    # Calculate Overall Accuracy
    accuracy = accuracy_score(Y_true, Y_pred)
    log_message(f"\n--- Overall Test Accuracy: {accuracy * 100:.2f}% ---")

    # Generate Classification Report (Precision, Recall, F1-Score)
    # 클래스 개수가 너무 많지 않은 경우에만 상세 리포트 출력
    unique_true_classes = np.unique(Y_true)
    if len(unique_true_classes) > 1 and len(unique_true_classes) < 50 and class_names:
        # unique_true_classes에 해당하는 이름만 필터링하여 사용
        # class_names는 전체 인덱스(0 ~ max_index)에 대한 단어명을 담고 있습니다.
        # Y_true는 인덱스이므로, Y_true에 있는 인덱스만 사용하여 target_names를 구성합니다.
        target_names = [index_to_label_map.get(i, f"Class_{i}") for i in unique_true_classes]
        log_message("\n--- Detailed Classification Report ---")
        print(classification_report(
            Y_true, 
            Y_pred, 
            labels=unique_true_classes, # <--- 💡 Y_true에 포함된 실제 인덱스 리스트
            target_names=target_names,   # <--- 💡 이 인덱스에 해당하는 이름 리스트
            zero_division=0
        ))
    else:
        log_message(f"\nDetailed Classification Report skipped (Classes in test set: {len(unique_true_classes)}).")


if __name__ == '__main__':
    # 검증 데이터 중 처음 10개 샘플로 테스트:
    test_sign_language_model(num_samples_to_test=10)
    
    log_message("--- Model Test Complete ---")
