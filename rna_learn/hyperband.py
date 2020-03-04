import argparse
import os
import logging
import math

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
from .utilities import generate_random_run_id, BioSequence


logger = logging.getLogger(__name__)


def hyperband_densenet_model(
    n_inputs,
    dropout=0.5,
    metrics=None,
    n_extras=0,
    init_n_layers=None,
    init_growth_rate=None,
    init_l2_reg=None,
    init_learning_rate=None,
):

    def build_model(hp):
        inputs = keras.Input(shape=(None, n_inputs), name='sequence')

        inputs_extra = None
        if n_extras > 0:
            inputs_extra = keras.Input(shape=(n_extras,), name='x_extra')

        if init_n_layers is None:
            n_layers = hp.Int(
                'n_layers',
                min_value=2,
                max_value=15,
                step=2,
            )
        else:
            n_layers = init_n_layers

        if init_growth_rate is None:
            growth_rate = hp.Int(
                'growth_rate',
                min_value=4,
                max_value=15,
                step=2,
            )
        else:
            growth_rate = init_growth_rate

        if init_l2_reg is None:
            l2_reg = hp.Choice(
                'l2_reg',
                values=(5e-4, 1e-4, 5e-5, 1e-5, 5e-6),
            )
        else:
            l2_reg = init_l2_reg

        if init_learning_rate is None:
            learning_rate = hp.Choice(
                'learning_rate',
                values=(5e-3, 1e-3, 5e-4, 1e-4, 5e-5),
            )
        else:
            learning_rate = init_learning_rate

        mask_value = np.array([0.] * n_inputs)
        mask = keras.layers.Masking(mask_value=mask_value).compute_mask(inputs)

        x = inputs
        for l in range(n_layers):
            kernel_size = hp.Int(
                f'kernel_size_l{l+1}',
                min_value=2,
                max_value=10,
                step=1,
            )

            out = keras.layers.Conv1D(
                filters=growth_rate, 
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l=l2_reg),
                name=f'conv_{l+1}'
            )(x)

            x = keras.layers.concatenate([x, out], axis=2, name=f'concat_{l+1}')

        x = keras.layers.GlobalAveragePooling1D(name='logits')(x, mask=mask)

        if n_extras > 0:
            x = keras.layers.concatenate([x, inputs_extra], axis=1, name=f'concat_extras')

        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Dense(
            units=2, 
            kernel_regularizer=keras.regularizers.l2(l=l2_reg),
            name='mean_and_std'
        )(x)

        outputs = tfp.layers.IndependentNormal(1)(x)

        if n_extras > 0:
            final_inputs = [inputs, inputs_extra]
        else:
            final_inputs = inputs

        model = keras.Model(inputs=final_inputs, outputs=outputs)

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
    parser.add_argument('alphabet_type', choices=['dna', 'protein'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--hyperband_iterations', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=444)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--extra_prot_run_id', type=str)
    args = parser.parse_args()

    alphabet_type = args.alphabet_type
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    factor = args.factor
    hyperband_iterations = args.hyperband_iterations
    run_id = args.run_id
    dataset_path = args.dataset_path
    seed = args.random_seed
    verbose = args.verbose
    dtype = args.dtype
    extra_prot_run_id = args.extra_prot_run_id

    if run_id is None:
        run_id = generate_random_run_id()

    logger.info(f'Alphabet: {alphabet_type} | run_id: {run_id}')

    if dataset_path is None:
        dataset_path = os.path.join(os.getcwd(), 'data/gtdb/dataset_full_train.csv')

    logger.info(f'Loading data')
    dataset_df = pd.read_csv(dataset_path)

    if alphabet_type == 'dna':
        alphabet = ALPHABET_DNA
        raw_sequences = dataset_df['nucleotide_sequence'].values
    else:
        alphabet = ALPHABET_PROTEIN
        raw_sequences = np.array([
            s[:-1]  # Removing stop amino acid at the end
            for s in dataset_df['amino_acid_sequence'].values
        ])

    y = dataset_df['temperature'].values.astype(dtype)

    logger.info('Split train and test set')
    x_raw_train, y_train, x_raw_test, y_test, train_idx, test_idx = split_train_test_set(
        raw_sequences, y, test_ratio=0.2, return_indices=True, seed=seed)

    mean, std = np.mean(y), np.std(y)
    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    x_test = sequence_embedding(x_raw_test, alphabet, dtype=dtype)

    x_extras_train, x_extras_test = None, None
    if extra_prot_run_id is not None:
        extra_model_path = os.path.join(
            os.getcwd(), 
            f'hyperband_logs/protein/{extra_prot_run_id}/best_model.h5',
        )
        logger.info(f'Loading extra protein logits from {extra_model_path}')
        extra_logits_model = tf.keras.models.load_model(extra_model_path)

        logger.info('Computing forward pass from extra logits model')

        extra_seq = np.array([
            s[:-1]
            for s in dataset_df['amino_acid_sequence'].values
        ])
        x_seq_extras = sequence_embedding(extra_seq, ALPHABET_PROTEIN, dtype=dtype)

        x_extras = load_extra_logits(extra_logits_model, x_seq_extras)

        x_extras_train = x_extras[train_idx]
        x_extras_test = x_extras[test_idx]

    train_sequence = BioSequence(x_raw_train, y_train_norm, batch_size, alphabet, x_extras=x_extras_train)

    if extra_prot_run_id is None:
        validation_data = (x_test, y_test_norm)
    else:
        validation_data = ((x_test, x_extras_test), y_test_norm)

    logger.info('Hyperparameters optimisation')

    metrics = [MeanAbsoluteError(mean, std)]

    tf.keras.backend.set_floatx(dtype)

    build_model_fn = hyperband_densenet_model(
        n_inputs=len(alphabet), 
        metrics=metrics,
        n_extras=x_extras_train.shape[1],
        init_n_layers=12,
        init_growth_rate=12,
        init_l2_reg=1e-5,
        init_learning_rate=1e-03,
    )

    hypermodel = Hyperband(
        build_model_fn,
        max_epochs=n_epochs,
        objective=kt.Objective('val_loss', 'min'),
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        project_name=f'hyperband_logs/{alphabet_type}/{run_id}',
    )

    hypermodel.search(
        train_sequence,
        validation_data=validation_data,
        epochs=n_epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=5)],
        verbose=verbose,
    )

    logger.info('DONE')


def load_extra_logits(extra_logits_model, x_seq_extras, n_parts=20):
    batch_size = int(math.ceil(len(x_seq_extras) / 20))

    outputs = []
    for i in range(n_parts):
        logger.info(f'Part {i+1} / {n_parts}')
        a = i * batch_size
        b = (i + 1) * batch_size

        o = extra_logits_model(x_seq_extras[a:b])
        outputs.append(o)

    return np.concatenate(outputs, axis=0)


if __name__ == '__main__':
    main()
