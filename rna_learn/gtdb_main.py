import argparse
import os
import logging
import string
import sys
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from .alphabet import ALPHABET_DNA, ALPHABET_PROTEIN
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
    compile_regression_model, 
    MeanAbsoluteError,
)
from .utilities import (
    SaveModelCallback, 
    generate_random_run_id,
    BioSequence,
)


logger = logging.getLogger(__name__)


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

    logger.info(f'Run {run_id} - {alphabet_type}')

    input_path = os.path.join(os.getcwd(), 'data/gtdb/dataset_full_train.csv')
    output_folder = os.path.join(os.getcwd(), f'saved_models_gtdb/{alphabet_type}/{run_id}/')
    model_path = os.path.join(output_folder, f'model.h5')
    metadata_path = os.path.join(output_folder, f'metadata.json')
    log_dir = os.path.join(os.getcwd(), f'summary_log/gtdb/{alphabet_type}/{run_id}')

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
        raw_sequences = dataset_df['nucleotide_sequence'].values
    else:
        raw_sequences = np.array([
            s[:-1]  # Removing stop amino acid at the end
            for s in dataset_df['amino_acid_sequence'].values
        ])

    y = dataset_df['temperature'].values.astype('float64')

    logger.info('Split train and test set')
    x_raw_train, y_train, x_raw_test, y_test, train_idx, test_idx = split_train_test_set(
        raw_sequences, y, test_ratio=0.2, return_indices=True, seed=metadata['seed'])

    mean, std = np.mean(y), np.std(y)
    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    x_test = sequence_embedding(x_raw_test, alphabet, dtype='float64')

    train_sequence = BioSequence(x_raw_train, y_train_norm, batch_size, alphabet)

    model = conv1d_densenet_regression_model(
        alphabet_size=len(alphabet), 
        growth_rate=metadata['growth_rate'],
        n_layers=metadata['n_layers'],
        kernel_sizes=metadata['kernel_sizes'],
        l2_reg=metadata['l2_reg'],
        dropout=metadata['dropout'],
        masking=True,
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
        train_sequence,
        validation_data=(x_test, y_test_norm),
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
