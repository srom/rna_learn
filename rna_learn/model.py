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
    strides=None,
    l2_reg=1e-4,
    dropout=0.5,
    masking=False,
):
    if len(kernel_sizes) != n_layers:
        raise ValueError('Kernel sizes argument must specify one kernel size per layer')

    if strides is None:
        strides = [1] * n_layers

    if len(strides) != n_layers:
        raise ValueError('Strides argument must specify one stride per layer')

    inputs = keras.Input(shape=(None, alphabet_size), name='sequence')

    mask = None
    if masking:
        mask_value = np.array([0.] * alphabet_size)
        mask = keras.layers.Masking(mask_value=mask_value).compute_mask(inputs)

    x = inputs
    for l in range(n_layers):
        kernel_size = kernel_sizes[l]
        stride = strides[l]

        out = keras.layers.Conv1D(
            filters=growth_rate, 
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l=l2_reg),
            name=f'conv_{l+1}'
        )(x)

        x = keras.layers.concatenate([x, out], axis=2, name=f'concat_{l+1}')

    x = keras.layers.GlobalAveragePooling1D(name='logits')(x, mask=mask)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(
        units=2, 
        kernel_regularizer=keras.regularizers.l2(l=l2_reg),
        name='mean_and_std'
    )(x)

    outputs = tfp.layers.IndependentNormal(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def variational_conv1d_densenet(
    encoding_size,
    alphabet_size,
    growth_rate,
    n_layers,
    kernel_sizes,
    strides=None,
    decoder_n_hidden=100,
    l2_reg=1e-4,
    dropout=0.5,
):
    prior = tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.zeros(encoding_size), scale=1),
        reinterpreted_batch_ndims=1,
    )

    encoder = variational_conv1d_densenet_encoder(
        encoding_size, 
        prior, 
        alphabet_size,
        growth_rate,
        n_layers,
        kernel_sizes,
        strides=strides,
        l2_reg=l2_reg,
        dropout=dropout,
    )

    decoder = variational_conv1d_densenet_decoder(
        encoding_size,
        n_hidden=decoder_n_hidden,
        dropout=dropout,
    )

    variational_model = keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    return prior, encoder, decoder, variational_model


def variational_conv1d_densenet_encoder(
    encoding_size,
    prior,
    alphabet_size,
    growth_rate,
    n_layers,
    kernel_sizes,
    strides=None,
    l2_reg=1e-4,
    dropout=0.5,
):
    if len(kernel_sizes) != n_layers:
        raise ValueError('Kernel sizes argument must specify one kernel size per layer')

    if strides is None:
        strides = [1] * n_layers

    if len(strides) != n_layers:
        raise ValueError('Strides argument must specify one stride per layer')

    n_mvn_params = tfp.layers.MultivariateNormalTriL.params_size(encoding_size)

    inputs = keras.Input(shape=(None, alphabet_size), name='sequence')

    mask_value = np.array([0.] * alphabet_size)
    mask = keras.layers.Masking(mask_value=mask_value).compute_mask(inputs)

    x = inputs
    for l in range(n_layers):
        kernel_size = kernel_sizes[l]
        stride = strides[l]

        out = keras.layers.Conv1D(
            filters=growth_rate, 
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l=l2_reg),
            name=f'encoder_conv_{l+1}'
        )(x)

        x = keras.layers.concatenate([x, out], axis=2, name=f'concat_{l+1}')

    x = keras.layers.GlobalAveragePooling1D(name='logits')(x, mask=mask)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(
        units=n_mvn_params, 
        kernel_regularizer=keras.regularizers.l2(l=l2_reg),
        name='encoder_dense_linear'
    )(x)

    encoded_outputs = tfp.layers.MultivariateNormalTriL(
        encoding_size,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0)
    )(x)

    return keras.Model(inputs=inputs, outputs=encoded_outputs)


def variational_conv1d_densenet_decoder(
    encoding_size,
    n_layers=1,
    n_hidden=100,
    dropout=0.5,
):
    encoded_inputs = keras.layers.Input(shape=(encoding_size,))

    x = encoded_inputs
    for n in range(n_layers):
        x = keras.layers.Dense(n_hidden, activation='relu', name=f'decoder_dense_relu_{n+1}')(x)
        x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(2, name='decoder_dense_linear')(x)

    decoded_outputs = tfp.layers.IndependentNormal(1)(x)

    return keras.Model(inputs=encoded_inputs, outputs=decoded_outputs)


def compile_variational_model(model, learning_rate, metrics=None):
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
        loss=negative_log_likelihood,
        metrics=metrics,
    )


def dual_stream_conv1d_densenet_regression(
    alphabet_size_1,
    alphabet_size_2,
    growth_rate,
    n_layers,
    kernel_sizes,
    strides=None,
    l2_reg=1e-4,
    dropout=0.5,
):
    if len(kernel_sizes) != n_layers:
        raise ValueError('Kernel sizes argument must specify one kernel size per layer')

    inputs = keras.Input(shape=(None, alphabet_size_1 + alphabet_size_2), name='sequence')

    x_stream_1 = inputs[..., :alphabet_size_1]
    x_stream_2 = inputs[..., alphabet_size_1:]

    features = []
    for i, x_stream in enumerate([x_stream_1, x_stream_2]):
        for l in range(n_layers):
            kernel_size = kernel_sizes[l]

            out = keras.layers.Conv1D(
                filters=growth_rate, 
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l=l2_reg),
                name=f'conv_{i}_{l+1}'
            )(x_stream)

            x_stream = keras.layers.concatenate([x_stream, out], axis=2, name=f'concat_{i}_{l+1}')

        xi = keras.layers.GlobalAveragePooling1D(name=f'logits_{i}')(x_stream)
        xi = keras.layers.Dropout(dropout)(xi)
        features.append(xi)

    x = keras.layers.concatenate(features, axis=1, name=f'concat_final')
    x = keras.layers.Dense(
        units=2, 
        kernel_regularizer=keras.regularizers.l2(l=l2_reg),
        name='mean_and_std'
    )(x)

    outputs = tfp.layers.IndependentNormal(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def compile_regression_model(model, learning_rate, metrics=None):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
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
