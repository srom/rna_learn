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

    l2_reg=1e-4,
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


def conv1d_regression_model(
    alphabet_size, 

    n_conv_1=2,
    n_filters_1=100, 
    kernel_size_1=10,

    n_conv_2=2,
    n_filters_2=100, 
    kernel_size_2=10,

    l2_reg=1e-4,
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
    x = keras.layers.Dense(
        units=2, 
        kernel_regularizer=keras.regularizers.l2(l=l2_reg),
    )(x)

    outputs = tfp.layers.IndependentNormal(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def conv1d_densenet_regression_model(
    alphabet_size,
    growth_rate,
    n_layers,
    kernel_sizes,
    l2_reg=1e-4,
    dropout=0.5,
):
    if len(kernel_sizes) != n_layers:
        raise ValueError('Kernel sizes argument must specify one kernel size per layer')

    inputs = keras.Input(shape=(None, alphabet_size), name='sequence')

    x = inputs
    for l in range(n_layers):
        kernel_size = kernel_sizes[l]

        out = keras.layers.Conv1D(
            filters=growth_rate, 
            kernel_size=kernel_size, 
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l=l2_reg),
            name=f'conv_{l+1}'
        )(x)

        x = keras.layers.concatenate([x, out], axis=2, name=f'concat_{l+1}')

    x = keras.layers.GlobalAveragePooling1D(name='logits')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(
        units=2, 
        kernel_regularizer=keras.regularizers.l2(l=l2_reg),
        name='mean_and_std'
    )(x)

    outputs = tfp.layers.IndependentNormal(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_regression_model(model, learning_rate, metrics=None):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y, normal_dist: -normal_dist.log_prob(y),
        metrics=metrics,
    )


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


class MeanAbsoluteError(tf.keras.losses.Loss):

    def __init__(self, mean=None, std=None):
        if mean is not None and std is None:
            raise ValueError('Specify both mean and std, or none')
        elif mean is None and std is not None:
            raise ValueError('Specify both mean and std, or none')

        super().__init__()

        self.name = 'mae'
        self.mean = mean
        self.std = std

    def call(self, y_true, y_pred):
        if self.mean is not None:
            y_t = (y_true * self.std) + self.mean
            y_p = (y_pred * self.std) + self.mean
        else:
            y_t = y_true
            y_p = y_pred

        return tf.keras.backend.mean(
            tf.keras.backend.abs(y_t - y_p),
            axis=-1,
        )
