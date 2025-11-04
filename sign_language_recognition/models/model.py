import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sign_language_recognition.train.utils import log_message

def build_lstm_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    용량을 4배 확장하고 드롭아웃을 완화한 CNN-BiLSTM 모델을 구축합니다.
    (2,737개 클래스 미달 학습 해소를 위한 최종 구조)

    Args:
        input_shape (tuple): (SEQUENCE_LENGTH, features_per_frame)
        num_classes (int): 분류할 클래스 개수
    """
    log_message(f"모델 Input Shape: {input_shape}, Output Classes: {num_classes}")
    
    model = Sequential([
        # 💡 Conv1D 필터 수 확장: 128 -> 256
        Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2), # 규제 완화

        # 💡 Bidirectional LSTM 유닛 수 확장: 128 -> 256
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.2)),
        
        # 💡 Bidirectional LSTM 유닛 수 확장: 128 -> 256
        Bidirectional(LSTM(256, return_sequences=False, dropout=0.2)), 
        Dropout(0.2), # 규제 완화

        # 💡 Dense 계층 유닛 수 확장: 128 -> 256
        Dense(256, activation='relu'),
        Dropout(0.2), # 규제 완화
        
        # 최종 출력 계층
        Dense(num_classes, activation='softmax')
    ])
    
    # 모델 빌드 및 요약
    try:
        model.build(input_shape=(None, *input_shape))
        log_message("모델 빌드 완료.")
    except Exception as e:
        log_message(f"경고: 모델 강제 빌드 실패. {repr(e)}")


    return model
