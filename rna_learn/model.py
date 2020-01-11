import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


def rnn_regression_model(alphabet_size, n_timesteps=None, n_hidden=100, dropout=0.5, n_lstm=1):
    mask_value = np.array([0] * alphabet_size, dtype=np.float32)

    inputs = keras.Input(shape=(n_timesteps, alphabet_size), name='sequence')

    x = keras.layers.Masking(mask_value=mask_value)(inputs)

    for i in range(n_lstm):
        not_last = i + 1 != n_lstm
        x = keras.layers.LSTM(
            n_hidden, 
            recurrent_dropout=dropout,
            return_sequences=not_last,
        )(x)

    x = keras.layers.Dense(n_hidden, activation='relu', name='logits')(x)
    x = keras.layers.Dropout(dropout)(x)

    x_loc = keras.layers.Dense(1, name='loc')(x)
    x_scale = keras.layers.Dense(1, activation='softplus', name='scale')(x)

    x_normal = keras.layers.concatenate([x_loc, x_scale], axis=1)

    outputs = tfp.layers.IndependentNormal(1)(x_normal)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_regression_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y, normal_dist: -normal_dist.log_prob(y),
    )


def rnn_classification_model(
    alphabet_size, 
    n_classes, 
    n_timesteps=None, 
    n_hidden=100, 
    dropout=0.5,
    n_lstm=1,
):
    mask_value = np.array([0] * alphabet_size, dtype=np.float32)

    inputs = keras.Input(shape=(n_timesteps, alphabet_size), name='sequence')

    x = keras.layers.Masking(mask_value=mask_value)(inputs)

    for i in range(n_lstm):
        not_last = i + 1 != n_lstm
        x = keras.layers.LSTM(
            n_hidden, 
            recurrent_dropout=dropout,
            return_sequences=not_last,
        )(x)

    x = keras.layers.Dense(n_hidden, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def conv1d_classification_model(
    alphabet_size, 
    n_classes,

    n_conv_1=2,
    n_filters_1=100, 
    kernel_size_1=10,

    n_conv_2=2,
    n_filters_2=100, 
    kernel_size_2=10,

    l2_reg=1e-3,
    dropout=0.5,
):
    inputs = keras.Input(shape=(None, alphabet_size), name='sequence')

    x = inputs
    for _ in range(n_conv_1):
        x = keras.layers.Conv1D(
            filters=n_filters_1, 
            kernel_size=kernel_size_1, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l=l2_reg),
        )(x)

    for _ in range(n_conv_2):
        x = keras.layers.Conv1D(
            filters=n_filters_2, 
            kernel_size=kernel_size_2, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l=l2_reg),
        )(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(
        n_classes, 
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(l=l2_reg),
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_classification_model(model, learning_rate, epsilon=1e-07):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=epsilon,
            amsgrad=True,
        ),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
