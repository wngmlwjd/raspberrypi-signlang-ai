import tensorflow as tf
import numpy as np
from sign_language_recognition.train.preprocess import get_calibration_data
from sign_language_recognition.train.utils import log_message

# 1. 모델 로드
H5_MODEL_PATH = "sign_language_recognition/models/best_sign_model_v5.h5" 
# V11이든 V5든, 가장 성능이 좋았던 모델 버전을 로드합니다.
model = tf.keras.models.load_model(H5_MODEL_PATH)
log_message(f"Keras 모델 로드 완료: {H5_MODEL_PATH}")

# 2. Calibration Data (보정 데이터) 정의
# INT8 양자화를 위해서는 모델의 가중치와 활성화 값 범위를 결정할 보정 데이터셋이 필요합니다.
# 이 함수는 실제 훈련 데이터셋의 일부(100개 배치 정도)를 제공해야 합니다.
def representative_dataset_gen():
    """TFLite Converter에 사용할 보정 데이터 제너레이터."""
    # preprocess.py에서 정의된 함수라고 가정합니다.
    for data_point in get_calibration_data(num_samples=100): 
        yield [data_point.astype(np.float32)]

# 3. TFLite Converter 설정
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # 기본 최적화 (양자화 포함) 적용

# 정수 양자화 설정 (가장 높은 성능을 위해 권장)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen

# 4. 변환 및 저장
tflite_model = converter.convert()
TFLITE_MODEL_PATH = "sign_language_recognition/models/best_sign_model_v5_int8.tflite"

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

log_message(f"TFLite INT8 모델 저장 완료: {TFLITE_MODEL_PATH}")
