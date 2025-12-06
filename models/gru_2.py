from tensorflow.keras import layers, models

def build_convgru_model_keras(input_size=42, seq_len=30, num_classes=20, conv_channels=[64, 128], conv_kernel=3, gru_hidden=128, gru_layers=2, dropout=0.3):
    """
    개선된 Conv1D + GRU hybrid model for sign language recognition
    - Conv1D 레이어 2개 이상
    - GRU 레이어 2개 이상
    - Fully connected 강화
    """
    inputs = layers.Input(shape=(seq_len, input_size))  # [batch, seq_len, input_size]

    x = inputs
    # Conv1D 블록 여러 개
    for filters in conv_channels:
        x = layers.Conv1D(filters=filters, kernel_size=conv_kernel, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    # GRU layers
    for i in range(gru_layers):
        return_seq = True if i < gru_layers - 1 else False
        x = layers.GRU(gru_hidden, return_sequences=return_seq, dropout=dropout)(x)

    # Fully connected
    x = layers.Dense(gru_hidden, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(gru_hidden // 2, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
