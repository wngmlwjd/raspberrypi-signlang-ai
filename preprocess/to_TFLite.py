import tensorflow as tf
import tf2onnx
import numpy as np
import os

def convert_h5_to_onnx(h5_path, onnx_path, input_shape):
    # Load Keras model
    model = tf.keras.models.load_model(h5_path)
    print("Loaded Keras model")

    # Dummy input for model signature
    dummy_input = tf.TensorSpec(input_shape, tf.float32, name="input")

    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=[dummy_input],
        opset=13  # Hailo recommended
    )

    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Saved ONNX model: {onnx_path}")


if __name__ == "__main__":
    h5_path = "/Users/wngmlwjd/workspace/github/raspberrypi-signlang-ai/models/cnn/20251210_7/best_model.h5"
    onnx_path = "/Users/wngmlwjd/workspace/github/raspberrypi-signlang-ai/models/cnn/20251210_7/best_model.onnx"

    # 너 모델 입력이 (1, 1, 63)이니 batch 제외 → (1, 63)
    convert_h5_to_onnx(h5_path, onnx_path, input_shape=(1, 1, 63))
