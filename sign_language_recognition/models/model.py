import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
# L2 정규화가 사용되지 않으므로 import에서 제거합니다.
# from tensorflow.keras.regularizers import l2
from sign_language_recognition.train.utils import log_message

def build_lstm_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    실시간 수어 통역 시스템에 적합한 경량화된 Bidirectional LSTM 기반 모델을 구성합니다.
    (컴파일은 train.py에서 수행합니다.)

    Args:
        input_shape (tuple): (SEQUENCE_LENGTH, features_per_frame)
        num_classes (int): 분류할 클래스 개수
    """
    log_message(f"모델 Input Shape: {input_shape}, Output Classes: {num_classes}")
    
    model = Sequential([
        Conv1D(128, 5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.3)),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # 💡 model.summary()를 출력하기 전에 명시적으로 빌드
    try:
        model.build(input_shape=(None, *input_shape))
        log_message("모델 빌드 완료.")
    except Exception as e:
        log_message(f"경고: 모델 강제 빌드 실패. {repr(e)}")


    return model
