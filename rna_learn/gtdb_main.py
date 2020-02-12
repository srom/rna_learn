import argparse
import os
import logging
import string
import sys
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from .transform import (
    sequence_embedding, 
    normalize, denormalize,
    one_hot_encode_classes,
    split_train_test_set,
    combine_sequences,
)
from .model import (
    conv1d_regression_model, 
    conv1d_densenet_regression_model, 
    dual_stream_conv1d_densenet_regression,
    compile_regression_model, 
    MeanAbsoluteError,
)
from .utilities import SaveModelCallback, generate_random_run_id


logger = logging.getLogger(__name__)


ALPHABET_DNA = [
    'A', 'C', 'G', 'T',
]
ALPHABET_PROTEIN = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 
    'S', 'T', 'V', 'W', 'Y',
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('alphabet_type', choices=['dna', 'protein'])
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    alphabet_type = args.alphabet_type
    run_id = args.run_id
    resume = args.resume
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    if run_id is None and resume:
        logger.error('Specify --run_id to resume run')
        sys.exit(1)
    elif run_id is None and not resume:
        run_id = generate_random_run_id()

    logger.info(f'Run {run_id}')

    input_path = os.path.join(os.getcwd(), 'data/gtdb/dataset_train.csv')
    output_folder = os.path.join(os.getcwd(), f'saved_models_gtdb/{run_id}/')
    model_path = os.path.join(output_folder, f'model.h5')
    metadata_path = os.path.join(output_folder, f'metadata.json')
    log_dir = os.path.join(os.getcwd(), f'summary_log/gtdb/{run_id}')

    for dir_path in [output_folder, log_dir]:
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

    if alphabet_type == 'dna':
        alphabet = ALPHABET_DNA
    else:
        alphabet = ALPHABET_PROTEIN

    if resume:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        initial_epoch = 0
        dropout_rate = 0.5
        seed = np.random.randint(0, 9999)

        growth_rate = 15
        kernel_sizes = [2, 3, 5, 5, 10]
        n_layers = len(kernel_sizes)
        l2_reg = 5e-4

        metadata = {
            'run_id': run_id,
            'alphabet': alphabet,
            'alphabet_type': alphabet_type,
            'growth_rate': growth_rate,
            'n_layers': n_layers,
            'kernel_sizes': kernel_sizes,
            'l2_reg': l2_reg,
            'dropout': dropout_rate,
            'n_epochs': initial_epoch,
            'seed': seed,
        }

    logger.info('Loading data')
    dataset_df = pd.read_csv(input_path)

    if alphabet_type == 'dna':
        raw_sequences = dataset_df['mrna_candidate_sequence'].values
    else:
        raw_sequences = dataset_df['amino_acid_sequence'].values

    x = sequence_embedding(raw_sequences, alphabet, dtype='float64')

    if alphabet_type == 'dna':
        aa_raw = dataset_df['amino_acid_sequence'].values
        aa = sequence_embedding(aa_raw, ALPHABET_PROTEIN, dtype='float64')
        x = combine_sequences(x, aa)
        alphabet_size_1 = len(ALPHABET_DNA)
        alphabet_size_2 = len(ALPHABET_PROTEIN)
    else:
        alphabet_size = len(alphabet)

    y = dataset_df['temperature'].values.astype('float64')

    logger.info('Split train and test set')
    x_train, y_train, x_test, y_test, train_idx, test_idx = split_train_test_set(
        x, y, test_ratio=0.2, return_indices=True, seed=metadata['seed'])

    mean, std = np.mean(y), np.std(y)
    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    if alphabet_type == 'dna':
        model = dual_stream_conv1d_densenet_regression(
            alphabet_size_1=alphabet_size_1, 
            alphabet_size_2=alphabet_size_2, 
            growth_rate=metadata['growth_rate'],
            n_layers=metadata['n_layers'],
            kernel_sizes=metadata['kernel_sizes'],
            l2_reg=metadata['l2_reg'],
            dropout=metadata['dropout'],
        )
    else:
        model = conv1d_densenet_regression_model(
            alphabet_size=alphabet_size, 
            growth_rate=metadata['growth_rate'],
            n_layers=metadata['n_layers'],
            kernel_sizes=metadata['kernel_sizes'],
            l2_reg=metadata['l2_reg'],
            dropout=metadata['dropout'],
        )

    compile_regression_model(
        model, 
        learning_rate=learning_rate,
        metrics=[MeanAbsoluteError(mean, std)],
    )

    if resume:
        logger.info(f'Resuming from {model_path}')
        model.load_weights(model_path)

    initial_epoch = 0
    epochs = n_epochs
    if resume:
        initial_epoch = metadata['n_epochs']
        epochs += initial_epoch

    logger.info(f'Training run {run_id}')
    model.fit(
        x_train,
        y_train_norm,
        validation_data=(x_test, y_test_norm),
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=False,
                update_freq='epoch',
                embeddings_freq=0,
            ),
            SaveModelCallback(
                model_path=model_path,
                metadata_path=metadata_path,
                metadata=metadata,
            ),
        ],
    )

    logger.info('DONE')


if __name__ == '__main__':
    main()
