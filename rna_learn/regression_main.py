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
    make_dataset_balanced,
    one_hot_encode_classes,
    split_train_test_set,
)
from .model import conv1d_regression_model, compile_regression_model, MeanAbsoluteError
from .load import load_dataset
from .utilities import SaveModelCallback, generate_random_run_id


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

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

    input_path = os.path.join(os.getcwd(), 'data/dataset_train.csv')
    output_folder = os.path.join(os.getcwd(), f'saved_models_regression/{run_id}/')
    model_path = os.path.join(output_folder, f'model.h5')
    metadata_path = os.path.join(output_folder, f'metadata.json')
    log_dir = os.path.join(os.getcwd(), f'summary_log/regression/{run_id}')

    for dir_path in [output_folder, log_dir]:
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

    alphabet = ['A', 'T', 'G', 'C']
    classes = ['psychrophilic', 'mesophilic', 'thermophilic']

    logger.info('Loading data')
    dataset_df = load_dataset(input_path, alphabet)

    y, dataset_df = make_dataset_balanced(
        dataset_df,
        cat_name='temperature_range',
        output_col='temperature', 
        classes=classes, 
    )
    y = y.astype(np.float64)

    sequences = dataset_df['sequence'].values
    x = sequence_embedding(sequences, alphabet)

    logger.info('Split train and test set')
    x_train, y_train, x_test, y_test, train_idx, test_idx = split_train_test_set(
        x, y, test_ratio=0.2, return_indices=True)

    mean, std = np.mean(y), np.std(y)
    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    if resume:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'run_id': run_id,
            'alphabet': alphabet,
            'classes': classes,
            'n_epochs': 0,
            'n_conv_1': 3,
            'n_filters_1': 88, 
            'kernel_size_1': 29,
            'n_conv_2': 1,
            'n_filters_2': 54, 
            'kernel_size_2': 44,
            'l2_reg': 1e-4,
            'dropout': 0.5,
        }

    model = conv1d_regression_model(
        alphabet_size=len(alphabet), 
        n_conv_1=metadata['n_conv_1'],
        n_filters_1=metadata['n_filters_1'], 
        kernel_size_1=metadata['kernel_size_1'],
        n_conv_2=metadata['n_conv_2'],
        n_filters_2=metadata['n_filters_2'], 
        kernel_size_2=metadata['kernel_size_2'],
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


if __name__ == '__main__':
    main()
