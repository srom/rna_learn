import argparse
import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import kerastuner as kt
from kerastuner.tuners import Hyperband

from .alphabet import ALPHABET_DNA, ALPHABET_PROTEIN
from .transform import (
    sequence_embedding, 
    normalize, denormalize,
    one_hot_encode_classes,
    split_train_test_set,
    combine_sequences,
)
from .model import MeanAbsoluteError
from .utilities import generate_random_run_id


logger = logging.getLogger(__name__)


def hyperband_densenet_model(n_inputs, dropout=0.5, metrics=None):

    def build_model(hp):
        inputs = keras.Input(shape=(None, n_inputs), name='sequence')

        n_layers = hp.Int(
            'n_layers',
            min_value=2,
            max_value=20,
            step=2,
        )
        growth_rate = hp.Int(
            'growth_rate',
            min_value=4,
            max_value=20,
            step=2,
        )
        l2_reg_conv = hp.Choice(
            'l2_reg_conv',
            values=(1e-3, 5e-4, 1e-4, 5e-5, 1e-5),
        )
        l2_reg_mean_std = hp.Choice(
            'l2_reg_mean_std',
            values=(1e-3, 5e-4, 1e-4, 5e-5, 1e-5),
        )
        learning_rate = hp.Choice(
            'learning_rate',
            values=(1e-3, 5e-4, 1e-4, 5e-5, 1e-5),
        )

        x = inputs
        for l in range(n_layers):
            kernel_size = hp.Int(
                f'kernel_size_l{l+1}',
                min_value=2,
                max_value=5,
                step=1,
            )

            out = keras.layers.Conv1D(
                filters=growth_rate, 
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l=l2_reg_conv),
                name=f'conv_{l+1}'
            )(x)

            x = keras.layers.concatenate([x, out], axis=2, name=f'concat_{l+1}')

        x = keras.layers.GlobalAveragePooling1D(name='logits')(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Dense(
            units=2, 
            kernel_regularizer=keras.regularizers.l2(l=l2_reg_mean_std),
            name='mean_and_std'
        )(x)

        outputs = tfp.layers.IndependentNormal(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
            loss=lambda y, normal_dist: -normal_dist.log_prob(y),
            metrics=metrics,
        )

        return model

    return build_model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    logger.info('Hyperparameters optimisation with Hyperband')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--hyperband_iterations', type=int, default=1)
    parser.add_argument('--run_id', type=str, default=None)
    args = parser.parse_args()

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    factor = args.factor
    hyperband_iterations = args.hyperband_iterations
    run_id = args.run_id
    seed = 444

    if run_id is None:
        run_id = generate_random_run_id()

    logger.info(f'Loading data (run_id: {run_id})')

    input_path = os.path.join(os.getcwd(), 'data/gtdb/dataset_train.csv')
    dataset_df = pd.read_csv(input_path)

    alphabet = ALPHABET_PROTEIN
    raw_sequences = dataset_df['amino_acid_sequence'].values

    x = sequence_embedding(raw_sequences, alphabet, dtype='float32')
    y = dataset_df['temperature'].values.astype('float32')

    logger.info('Split train and test set')
    x_train, y_train, x_test, y_test, train_idx, test_idx = split_train_test_set(
        x, y, test_ratio=0.2, return_indices=True, seed=seed)

    mean, std = np.mean(y), np.std(y)
    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    logger.info('Hyperparameters optimisation')

    metrics = [MeanAbsoluteError(mean, std)]

    build_model_fn = hyperband_densenet_model(n_inputs=len(alphabet), metrics=metrics)

    hypermodel = Hyperband(
        build_model_fn,
        max_epochs=n_epochs,
        objective=kt.Objective('val_loss', 'min'),
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        project_name=f'hyperband_logs/{run_id}',
    )

    hypermodel.search(
        x_train,
        y_train_norm,
        validation_data=(x_test, y_test_norm),
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=5)]
    )

    logger.info('DONE')


if __name__ == '__main__':
    main()
