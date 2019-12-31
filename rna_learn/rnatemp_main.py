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
    split_train_test_set,
)
from .load import load_rna_structure_dataset, load_rna_nucleotides_dataset


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_type', choices=['regression', 'classification'])
    parser.add_argument('rna_type', choices=['mrna', 'trna', 'rrna'])
    parser.add_argument('alphabet', choices=['nucleotides', '2d_structure'])
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=444)
    args = parser.parse_args()

    learning_type = args.learning_type
    rna_type = args.rna_type
    alphabet_type = args.alphabet
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    resume = args.resume
    seed = args.seed

    np.random.seed(seed)

    if alphabet_type == 'nucleotides':
        alphabet = ['A', 'T', 'G', 'C']
    else:
        alphabet = ['.', '(', ')']

    if learning_type == 'regression':
        regression(rna_type, alphabet, learning_rate, batch_size, n_epochs, resume, seed)
    else:
        classification(rna_type, alphabet, learning_rate, batch_size, n_epochs, resume, seed)


def regression(rna_type, alphabet, learning_rate, batch_size, n_epochs, resume, seed):
    alphabet_size = len(alphabet)

    metadata_path = f'data/tab/{rna_type}.tab'
    sequences_folder = 'data/seq/'

    output_folder = os.path.join(os.getcwd(), 'saved_models_rnatemp', f'seed_{seed}')
    output_path = os.path.join(output_folder, f'{rna_type}_regression.h5')

    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass

    log_dir = os.path.join(os.getcwd(), 'summary_log', f'seed_{seed}', 'regression')
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass

    logger.info('Building model')
    model = rnn_regression_model(alphabet_size=alphabet_size, n_lstm=2)
    compile_regression_model(model, learning_rate=learning_rate)

    if resume > 0:
        logger.info(f'Resuming from {output_path}')
        model.load_weights(output_path)

    logger.info('Loading data')
    metadata = pd.read_csv(metadata_path, delimiter='\t')
    metadata['category'] = metadata['temp.cat']

    y, metadata = make_dataset_balanced(
        metadata, 
        output_col='temp', 
    )
    y = y.astype(np.float32)

    if 'A' in alphabet:
        sequences = load_rna_nucleotides_dataset(metadata, sequences_folder)
    else:
        sequences = load_rna_structure_dataset(metadata, sequences_folder)

    x = sequence_embedding(sequences, alphabet)

    logger.info('Split train and test set')
    x_train, y_train, x_test, y_test = split_train_test_set(x, y, test_ratio=0.2)

    mean, std = np.mean(y), np.std(y)

    y_test_norm = normalize(y_test, mean, std)
    y_train_norm = normalize(y_train, mean, std)

    initial_epoch = 0
    epochs = n_epochs
    if resume > 0:
        initial_epoch = resume
        epochs += initial_epoch

    logger.info('Training')
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
        ],
    )

    model.save(output_path)
    logger.info(f'Model saved to {output_path}')


def classification(rna_type, alphabet, learning_rate, batch_size, n_epochs, resume, seed):
    alphabet_size = len(alphabet)
    classes = ['psychrophile', 'mesophile', 'thermophile', 'hyperthermophile']
    n_classes = len(classes)

    metadata_path = f'data/tab/{rna_type}.tab'
    sequences_folder = 'data/seq/'

    output_folder = os.path.join(os.getcwd(), 'saved_models_rnatemp', f'seed_{seed}')
    output_path = os.path.join(output_folder, f'{rna_type}_classification.h5')
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass

    log_dir = os.path.join(os.getcwd(), 'summary_log', f'seed_{seed}', 'classification')
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass

    logger.info('Building model')
    model = rnn_classification_model(alphabet_size=alphabet_size, n_classes=n_classes, n_lstm=2)
    compile_classification_model(model, learning_rate=learning_rate)

    if resume > 0:
        logger.info(f'Resuming from {output_path}')
        model.load_weights(output_path)

    logger.info('Loading data')
    metadata = pd.read_csv(metadata_path, delimiter='\t')
    metadata['category'] = metadata['temp.cat']

    n_entries_per_class = 153
    y_str, metadata = make_dataset_balanced(metadata)
    y = one_hot_encode_classes(y_str, classes)
    
    if 'A' in alphabet:
        sequences = load_rna_nucleotides_dataset(metadata, sequences_folder)
    else:
        sequences = load_rna_structure_dataset(metadata, sequences_folder)

    x = sequence_embedding(sequences, alphabet)

    logger.info('Split train and test set')
    x_train, y_train, x_test, y_test = split_train_test_set(x, y, test_ratio=0.2)

    initial_epoch = 0
    epochs = n_epochs
    if resume > 0:
        initial_epoch = resume
        epochs += initial_epoch

    logger.info('Training')
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
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
        ],
    )

    model.save(output_path)
    logger.info(f'Model saved to {output_path}')


if __name__ == '__main__':
    main()
