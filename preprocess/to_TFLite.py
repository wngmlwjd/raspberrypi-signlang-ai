import tensorflow as tf
import os

from train.config import MODEL_SAVE_DIR

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, '20251205_3', 'best_model.h5')
TFLITE_PATH = os.path.join(MODEL_SAVE_DIR, '20251205_3', 'best_model.tflite')

print(f"1. Keras 모델 로드 중: {MODEL_PATH}")
try:
    # 1. Keras 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH)
except FileNotFoundError:
    print(f"\n[오류] 지정된 경로에 모델 파일이 없습니다: {MODEL_PATH}")
    print("경로 설정 (MODEL_SAVE_DIR, MODEL_SUB_DIR, MODEL_FILENAME)을 확인해 주세요.")
    exit()

print("2. TFLite 컨버터 설정 및 변환 시작...")
# 2. TFLite 컨버터 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Bi-GRU/CNN 모델 변환을 위한 핵심 설정
# Bidirectional GRU와 같은 복잡한 연산을 TFLite가 처리할 수 있도록 Select TF Ops를 허용
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # TFLite의 기본 연산
    tf.lite.OpsSet.SELECT_TF_OPS    # TFLite에 없는 Keras/TensorFlow 연산을 포함
]

# TensorList 관련 오류 방지를 위해 실험적 기능인 자동 변환 비활성화
converter._experimental_lower_tensor_list_ops = False

# (선택 사항) 최적화: 모델 크기 축소 및 속도 향상을 위한 양자화
# converter.optimizations = [tf.lite.Optimize.DEFAULT]


# 4. 모델 변환
try:
    tflite_model = converter.convert()
    print("3. 변환 성공!")
except Exception as e:
    print(f"\n[오류] TFLite 변환 중 치명적인 오류 발생:")
    print(e)
    print("\nSelect TF Ops를 사용하도록 설정했음에도 불구하고 오류가 발생했다면,")
    print("모델의 특정 사용자 정의 계층(Custom Layer)이나 동적 Shape 문제가 해결되지 않은 것일 수 있습니다.")
    exit()

# 5. TFLite 모델 저장
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"\n4. TFLite 모델 저장 완료: {TFLITE_PATH}")