import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


def variational_autoencoder(n_inputs, encoding_size, n_hidden, n_layers=1, dropout=0.5):
    prior = tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.zeros(encoding_size), scale=1),
        reinterpreted_batch_ndims=1,
    )
    encoder = make_encoder(n_inputs, encoding_size, n_hidden, n_layers, dropout, prior)
    decoder = make_decoder(n_inputs, encoding_size, n_hidden, n_layers, dropout)
    vae = keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    return prior, encoder, decoder, vae


def make_encoder(n_inputs, encoding_size, n_hidden, n_layers, dropout, prior):
    n_mvn_params = tfp.layers.MultivariateNormalTriL.params_size(encoding_size)

    inputs = keras.layers.Input(shape=(n_inputs,))

    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(n_hidden, activation='relu')(inputs)
        x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(n_mvn_params)(x)

    encoded_outputs = tfp.layers.MultivariateNormalTriL(
        encoding_size,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1.0)
    )(x)

    return keras.Model(inputs=inputs, outputs=encoded_outputs)


def make_decoder(n_inputs, encoding_size, n_hidden, n_layers, dropout):
    n_normal_params = n_inputs * 2

    encoded_inputs = keras.layers.Input(shape=(encoding_size,))

    x = encoded_inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(n_hidden, activation='relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(n_normal_params)(x)

    decoded_outputs = tfp.layers.IndependentNormal(n_inputs)(x)

    return keras.Model(inputs=encoded_inputs, outputs=decoded_outputs)


def compile_vae(vae, learning_rate):
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    vae.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=negative_log_likelihood,
    )
