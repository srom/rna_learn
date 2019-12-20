import os
import logging

import numpy as np
import tensorflow as tf

from .model import rnn_regression_model, compile_model
from .transform import sequence_embedding, normalize, denormalize
from .load import load_rna_structure_dataset


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    np.random.seed(444)

    metadata_path = 'data/tab/mrna.tab'
    sequences_folder = 'data/seq/'

    alphabet = ['.', '(', ')']
    alphabet_size = len(alphabet)

    learning_rate = 1e-5
    batch_size = 32
    n_epochs = 2

    output_path = os.path.join(os.getcwd(), 'saved_models', 'mrna_learn.h5')

    logger.info('Building model')
    model = rnn_regression_model(alphabet_size=alphabet_size)
    compile_model(model, learning_rate=learning_rate)

    logger.info('Loading data')
    sequences, metadata = load_rna_structure_dataset(metadata_path, sequences_folder)
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


if __name__ == '__main__':
    main()
