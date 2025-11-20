import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_timesteps, input_features, num_classes, lstm_units_1=128, lstm_units_2=64, dense_units=64, dropout_rate=0.3):
    """
    LSTM 기반 시퀀스 분류 모델 정의
    """
    model = models.Sequential()
    model.add(layers.LSTM(lstm_units_1, input_shape=(input_timesteps, input_features), return_sequences=True))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.LSTM(lstm_units_2))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
