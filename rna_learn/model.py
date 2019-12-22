import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


def rnn_regression_model(alphabet_size, n_timesteps=None, n_hidden=100, dropout=0.5, n_lstm=1):
    mask_value = np.array([0] * alphabet_size, dtype=np.float32)

    inputs = keras.Input(shape=(n_timesteps, alphabet_size), name='sequence')

    x = keras.layers.Masking(mask_value=mask_value)(inputs)

    for i in range(n_lstm):
        x = keras.layers.LSTM(
            n_hidden, 
            return_sequences=i + 1 != n_lstm,
        )(x)

    x = keras.layers.Dense(n_hidden, activation='relu', name='logits')(x)
    x = keras.layers.Dropout(dropout)(x)

    x_loc = keras.layers.Dense(1, name='loc')(x)
    x_scale = keras.layers.Dense(1, activation='softplus', name='scale')(x)

    x_normal = keras.layers.concatenate([x_loc, x_scale], axis=1)

    outputs = tfp.layers.IndependentNormal(1)(x_normal)

    return keras.Model(inputs=inputs, outputs=outputs)


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
        not_first = i > 0
        not_last = i + 1 != n_lstm
        x = keras.layers.LSTM(
            n_hidden, 
            dropout=dropout if not_first else 0.,
            return_sequences=not_last,
        )(x)

    x = keras.layers.Dense(n_hidden, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_regression_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y, normal_dist: -normal_dist.log_prob(y),
    )


def compile_classification_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
