import tensorflow as tf

# -----------------------
# 모델 경로 설정
# -----------------------
MODEL_PATH = "models/conv1d+gru/20251208_1/best_model.h5"  

# -----------------------
# 모델 로드
# -----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------
# 모델 구조 출력
# -----------------------
model.summary()

# 필요하면 층 목록도 출력
# for i, layer in enumerate(model.layers):
#     print(f"{i:02d} : {layer.name} — {layer.__class__.__name__}")
