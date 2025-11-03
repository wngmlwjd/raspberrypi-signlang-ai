import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import re
from pathlib import Path

# 현재 프로젝트 구조에 맞게 경로 수정
from sign_language_recognition.train.utils import log_message, SEQUENCE_LENGTH, MODEL_DIR
from sign_language_recognition.train.preprocess import prepare_and_load_datasets
from sign_language_recognition.models.model import build_lstm_model
from sklearn.preprocessing import LabelEncoder

def get_latest_version(model_dir: Path, base_model_name: str):
    # ... (생략, 변경 없음) ...
    if not model_dir.exists():
        return 0
    pattern = re.compile(rf"{re.escape(base_model_name)}_v(\d+)\.h5$")
    versions = []
    for file in model_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) if versions else 0

def train_sign_language_model(epochs: int = 50, batch_size: int = 32, validation_split: float = 0.5, retrain: bool = False):
    """
    수어 인식 모델 훈련을 위한 메인 함수입니다.
    파일에서 로드한 훈련 데이터를 모두 학습에 사용하고, 
    로드된 검증/테스트 데이터(X_val)를 분할하여 일부는 Keras 검증에, 나머지는 최종 테스트에 사용합니다.
    
    Args:
        epochs: 훈련 에포크 수
        batch_size: 배치 크기
        validation_split: 로드된 '검증/테스트 데이터(X_val)'를 Keras '검증'과 '최종 테스트'로 나눌 비율.
        (예: 0.5면 반반)
        retrain: True면 새 모델 생성 및 버전 업, False면 기존 체크포인트가 있으면 불러와 추가 학습
    """
    base_model_name = "best_sign_model"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log_message("--- 수어 인식 모델 훈련 시작 ---")

    # 1. 데이터 로딩 및 전처리 
    # prepare_and_load_datasets는 (X_train, Y_train, X_val, Y_val, encoder)를 반환합니다.
    # 여기서 X_train/Y_train은 모두 훈련에 사용하고, X_val/Y_val은 검증/테스트 풀로 사용합니다.
    X_train, Y_train, X_test_full, Y_test_full, encoder = prepare_and_load_datasets(force_reprocess=False)
    
    if X_train is None or encoder is None:
        log_message("훈련 데이터 로드 실패 또는 LabelEncoder 누락. 종료합니다.")
        return

    num_classes = len(encoder.classes_)
    
    # 2. 검증/테스트 데이터 풀 (X_test_full)을 Keras 검증용과 최종 테스트용으로 분할 (수정된 부분)
    log_message(f"로드된 전체 훈련 데이터 크기: {X_train.shape[0]} (전부 훈련에 사용)")
    log_message(f"로드된 검증/테스트 풀 크기: {X_test_full.shape[0]}")
    log_message(f"검증/테스트 풀 분할 비율 (Keras 검증용:최종 테스트용): {validation_split:.2f}:{1.0 - validation_split:.2f}")

    if X_test_full.shape[0] > 0 and validation_split > 0 and validation_split < 1.0:
        # X_test_full을 최종 테스트용과 Keras 검증용으로 분할
        # test_size=1.0 - validation_split: validation_split만큼 Keras 검증용으로 남깁니다.
        X_test_final, X_val_keras, Y_test_final, Y_val_keras = train_test_split(
            X_test_full, Y_test_full, test_size=validation_split, # validation_split만큼을 Keras 검증용으로 사용
            shuffle=True, random_state=42, stratify=Y_test_full
        )
        validation_data_tuple = (X_val_keras, Y_val_keras)
    else:
        # X_test_full이 비어 있거나 validation_split이 부적절한 경우
        X_test_final, Y_test_final = X_test_full, Y_test_full
        X_val_keras, Y_val_keras = None, None
        validation_data_tuple = None

    # 3. 데이터 크기 확인
    if X_train.shape[0] == 0:
        log_message("훈련 데이터가 비어 있습니다. 종료합니다.")
        return
        
    log_message(f"실제 훈련 데이터 크기: {X_train.shape[0]}")
    if X_val_keras is not None:
        log_message(f"Keras 검증 데이터 크기: {X_val_keras.shape[0]}")
    log_message(f"최종 테스트 데이터 크기 (훈련에 사용되지 않음): {X_test_final.shape[0]}")

    input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
    model = None
    initial_epoch = 0

    latest_version = get_latest_version(MODEL_DIR, base_model_name)
    if retrain:
        # retrain=True면 새 버전 생성
        new_version = latest_version + 1
        versioned_model_path = MODEL_DIR / f"{base_model_name}_v{new_version}.h5"
        log_message(f"retrain=True -> 새 모델 버전 {new_version} 생성: {versioned_model_path.name}")
        model = build_lstm_model(input_shape=input_shape, num_classes=num_classes)
        save_path = versioned_model_path
    else:
        # retrain=False이고 이전 버전 있으면 가장 최신 버전 로드
        # ... (기존 로직 유지) ...
        if latest_version > 0:
            latest_model_path = MODEL_DIR / f"{base_model_name}_v{latest_version}.h5"
            try:
                model = tf.keras.models.load_model(str(latest_model_path), compile=False)
                log_message(f"기존 모델 버전 {latest_version} 로드: {latest_model_path.name}")
                save_path = latest_model_path
            except Exception as e:
                log_message(f"기존 모델 로드 실패({repr(e)}), 새 모델 생성")
                model = build_lstm_model(input_shape=input_shape, num_classes=num_classes)
                save_path = MODEL_DIR / f"{base_model_name}_v1.h5"
        else:
            log_message("체크포인트 없음 -> 새 모델 생성")
            model = build_lstm_model(input_shape=input_shape, num_classes=num_classes)
            save_path = MODEL_DIR / f"{base_model_name}_v1.h5"

    # 항상 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 콜백 설정
    monitor_metric = 'accuracy'
    if validation_data_tuple:
        monitor_metric = 'val_accuracy'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=str(save_path),
        save_weights_only=False,
        monitor=monitor_metric, 
        mode='max',
        save_best_only=True,
        verbose=1
    )
    monitor_loss = 'loss' if not validation_data_tuple else 'val_loss'
    early_stopping_callback = EarlyStopping(
        monitor=monitor_loss,
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor=monitor_loss,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    log_message(f"모델 훈련 시작 (Epochs: {epochs}, Batch Size: {batch_size})")
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_tuple, # ✅ 분할된 Keras 검증 데이터를 사용
        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=1,
        initial_epoch=initial_epoch
    )

    log_message("--- 모델 훈련 완료 ---")
    best_acc = max(history.history[monitor_metric])
    log_message(f"최고 {monitor_metric} 정확도: {best_acc:.4f}")
    log_message(f"모델 저장 위치: {save_path.name}")
    log_message(f"최종 테스트용 데이터 (X_test_final) 크기: {X_test_final.shape[0]}")

    # 로드된 테스트 데이터(X_test_final, Y_test_final)를 함께 반환하여 최종 평가에 사용
    return history, X_test_final, Y_test_final