from tensorflow.keras import layers, models

def build_convgru_model_keras(
        input_size, seq_len, num_classes,
        conv_channels, conv_kernel,
        gru_hidden, gru_layers, dropout):

    inputs = layers.Input(shape=(seq_len, input_size))
    x = inputs

    # Conv1D 블록
    for filters in conv_channels:
        x = layers.Conv1D(filters=filters, kernel_size=conv_kernel, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    # Bi-GRU 스택: 앞 레이어들은 시퀀스 반환, 마지막은 단일 출력
    for i in range(gru_layers):
        return_seq = True if i < gru_layers - 1 else False
        x = layers.Bidirectional(layers.GRU(gru_hidden, return_sequences=return_seq, dropout=dropout, recurrent_dropout=0.0))(x)

    # Fully connected layers
    x = layers.Dense(gru_hidden, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(gru_hidden // 2, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)
