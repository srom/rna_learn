import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


def rnn_regression_model(alphabet_size, n_timesteps=None, dropout=0.5, mask_value=None):
    if mask_value is None:
        mask_value = np.array([0] * alphabet_size, dtype=np.float32)

    inputs = keras.Input(shape=(n_timesteps, alphabet_size), name='sequence')

    x = keras.layers.Masking(mask_value=mask_value)(inputs)

    x = keras.layers.GRU(100, recurrent_dropout=dropout)(x)

    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(2)(x)

    outputs = tfp.layers.IndependentNormal(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y, normal_dist: -normal_dist.log_prob(y),
        metrics=[keras.losses.MAE],
    )
