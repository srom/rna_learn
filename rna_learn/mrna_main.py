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
from .load import load_mrna_model, load_dataset


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_type', choices=['regression', 'classification'])
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    run_id = args.run_id
    resume = args.resume
    learning_type = args.learning_type
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    if run_id is None and resume:
        logger.error('Specify --run_id to resume run')
        sys.exit(1)
    elif run_id is None and not resume:
        run_id = generate_random_run_id()

    logger.info(f'Run {run_id}')
    run(run_id, learning_type, learning_rate, batch_size, n_epochs, resume)


def run(run_id, learning_type, learning_rate, batch_size, n_epochs, resume):
    input_path = os.path.join(os.getcwd(), 'data/ncbi/dataset.csv')
    output_folder = os.path.join(os.getcwd(), f'saved_models_mrna/{run_id}/')
    model_path = os.path.join(output_folder, f'{learning_type}_model.h5')
    metadata_path = os.path.join(output_folder, f'{learning_type}_metadata.json')
    log_dir = os.path.join(os.getcwd(), f'summary_log/{run_id}')

    for dir_path in [output_folder, log_dir]:
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

    if resume:
        logger.info(f'Resuming from {model_path}')

    logger.info('Building model')
    model, metadata = load_mrna_model(run_id, learning_type, learning_rate, model_path, metadata_path, resume)

    alphabet = metadata['alphabet']
    classes = metadata['classes']

    logger.info('Loading data')
    dataset_df = load_dataset(input_path, alphabet)

    if learning_type == 'regression':
        y, dataset_df = make_dataset_balanced(
            dataset_df,
            cat_name='temperature_range',
            output_col='temperature', 
            classes=classes, 
        )
        y = y.astype(np.float32)
    else:
        y_str, dataset_df = make_dataset_balanced(
            dataset_df, 
            cat_name='temperature_range',
            classes=classes,
        )
        y = one_hot_encode_classes(y_str, classes)

    sequences = dataset_df['sequence'].values
    x = sequence_embedding(sequences, alphabet)

    logger.info('Split train and test set')
    x_train, y_train, x_test, y_test = split_train_test_set(x, y, test_ratio=0.2)

    if learning_type == 'regression':
        mean, std = np.mean(y), np.std(y)
        y_test_norm = normalize(y_test, mean, std)
        y_train_norm = normalize(y_train, mean, std)
    else:
        y_test_norm = y_test
        y_train_norm = y_train

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


class SaveModelCallback(tf.keras.callbacks.Callback):

    def __init__(
        self, 
        model_path, 
        metadata_path, 
        metadata, 
        monitor='val_loss',
        metrics=('val_accuracy',),
    ):
        self.monitor = monitor
        self.metrics = metrics
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.best_loss = metadata.get(self.monitor, np.inf)

    def on_epoch_end(self, epoch, logs):
        val_loss = logs[self.monitor]

        if val_loss < self.best_loss:
            self.model.save(self.model_path)

            self.metadata['n_epochs'] = epoch + 1
            self.metadata[self.monitor] = val_loss

            for metric in self.metrics or []:
                if metric in logs:
                    self.metadata[metric] = float(logs[metric])

            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)

            self.best_loss = val_loss


def generate_random_run_id():
    chars = [c for c in string.ascii_lowercase + string.digits]
    random_slug_chars = np.random.choice(chars, size=5, replace=True)
    random_slug = ''.join(random_slug_chars)
    return f'run_{random_slug}'


if __name__ == '__main__':
    main()
