import argparse
import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from .model import (
    rnn_regression_model, 
    rnn_classification_model,
    compile_regression_model,
    compile_classification_model,
)
from .transform import (
    sequence_embedding, 
    normalize, denormalize,
    make_dataset_balanced,
    one_hot_encode_classes,
)
from .load import load_rna_structure_dataset


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_type', choices=['regression', 'classification'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=444)
    args = parser.parse_args()

    learning_type = args.learning_type
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    seed = args.seed

    np.random.seed(seed)

    alphabet = ['.', '(', ')']

    if learning_type == 'regression':
        regression(alphabet, learning_rate, batch_size, n_epochs)
    else:
        classification(alphabet, learning_rate, batch_size, n_epochs)


def regression(alphabet, learning_rate, batch_size, n_epochs):
    alphabet_size = len(alphabet)

    metadata_path = 'data/tab/mrna.tab'
    sequences_folder = 'data/seq/'

    output_path = os.path.join(os.getcwd(), 'saved_models', 'mrna_regression.h5')

    logger.info('Building model')
    model = rnn_regression_model(alphabet_size=alphabet_size)
    compile_regression_model(model, learning_rate=learning_rate)

    logger.info('Loading data')
    metadata = pd.read_csv(metadata_path, delimiter='\t')
    metadata['category'] = metadata['temp.cat']

    sequences = load_rna_structure_dataset(metadata, sequences_folder)
    x = sequence_embedding(sequences, alphabet)
    y = metadata['temp'].values.astype(np.float32)

    logger.info('Split train and test set')
    n_seq = len(sequences)
    test_idx = np.random.choice(range(n_seq), size=int(0.2 * n_seq), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = np.array([idx for idx in range(n_seq) if idx not in test_idx_set])

    mean, std = np.mean(y), np.std(y)

    x_test, y_test = x[test_idx], y[test_idx]
    x_train, y_train = x[train_idx], y[train_idx]

    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    logger.info('Training')
    model.fit(
        x_train,
        y_train_norm,
        validation_data=(x_test, y_test_norm),
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
    )

    model.save(output_path)
    logger.info(f'Model saved to {output_path}')


def classification(alphabet, learning_rate, batch_size, n_epochs):
    alphabet_size = len(alphabet)
    classes = ['psychrophile', 'mesophile', 'thermophile', 'hyperthermophile']
    n_classes = len(classes)

    metadata_path = 'data/tab/mrna.tab'
    sequences_folder = 'data/seq/'

    output_path = os.path.join(os.getcwd(), 'saved_models', 'mrna_classification.h5')

    logger.info('Building model')
    model = rnn_classification_model(alphabet_size=alphabet_size, n_classes=n_classes, n_lstm=2)
    compile_classification_model(model, learning_rate=learning_rate)

    logger.info('Loading data')
    metadata = pd.read_csv(metadata_path, delimiter='\t')
    metadata['category'] = metadata['temp.cat']

    y_str, metadata = make_dataset_balanced(metadata)
    y = one_hot_encode_classes(y_str, classes)
    
    sequences = load_rna_structure_dataset(metadata, sequences_folder)
    x = sequence_embedding(sequences, alphabet)

    logger.info('Split train and test set')
    n_seq = len(sequences)
    test_idx = np.random.choice(range(n_seq), size=int(0.2 * n_seq), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = np.array([idx for idx in range(n_seq) if idx not in test_idx_set])

    x_test, y_test = x[test_idx], y[test_idx]
    x_train, y_train = x[train_idx], y[train_idx]

    logger.info('Training')
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
    )

    model.save(output_path)
    logger.info(f'Model saved to {output_path}')


if __name__ == '__main__':
    main()
