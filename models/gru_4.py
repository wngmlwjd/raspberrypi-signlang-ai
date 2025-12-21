from tensorflow.keras import layers, models

def build_convgru_model_keras(
        input_size, seq_len, num_classes,
        conv_channels, conv_kernel,
        gru_hidden, gru_layers, dropout):

    inputs = layers.Input(shape=(seq_len, input_size))
    x = inputs

    # ============================
    # 1) Multi-Scale Conv Block
    # ============================
    # conv_kernel = [3, 5, 7] 형태를 지원
    for filters in conv_channels:

        conv_outputs = []
        for k in conv_kernel:
            c = layers.Conv1D(filters=filters, kernel_size=k, padding='same', activation='relu')(x)
            conv_outputs.append(c)

        # 3개 Conv 출력 병합
        x = layers.Concatenate()(conv_outputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    # ============================
    # 2) Bi-GRU Stack
    # ============================
    for i in range(gru_layers):
        return_seq = True if i < gru_layers - 1 else False
        x = layers.Bidirectional(
            layers.GRU(
                gru_hidden,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=0.0
            )
        )(x)

    # ============================
    # 3) Dense + Residual
    # ============================
    h = layers.Dense(gru_hidden, activation='relu')(x)
    h = layers.Dropout(dropout)(h)

    h = layers.Dense(gru_hidden // 2, activation='relu')(h)
    h = layers.Dropout(dropout)(h)

    # GRU 출력 + Dense 블록 결합
    x = layers.Concatenate()([x, h])

    # ============================
    # 4) Output
    # ============================
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)
