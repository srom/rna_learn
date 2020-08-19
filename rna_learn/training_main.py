import argparse
import os
import logging
import string
import sys
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine

from .alphabet import ALPHABET_DNA
from .model import (
    variational_conv1d_densenet,
    compile_variational_model,
    conv1d_densenet_regression_model,
    compile_regression_model,
    DenormalizedMAE,
)
from .load_sequences import (
    TrainingSequence,
    TestingSequence,
    load_growth_temperatures,
    assign_weight_to_batch_values,
    compute_inverse_probability_weights,
)
from .utilities import (
    SaveModelCallback, 
    generate_random_run_id,
)
from .validation import validate_model_on_test_set


DB_PATH = 'data/condensed_traits/db/seq.db'

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--variational', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--max_queue_size', type=int, default=10)
    parser.add_argument('--max_sequence_length', type=int, default=int(1e4))
    parser.add_argument('--dtype', type=str, default='float32')
    args = parser.parse_args()

    run_id = args.run_id
    resume = args.resume
    variational = args.variational
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    db_path = args.db_path
    verbose = args.verbose
    max_queue_size = args.max_queue_size
    max_sequence_length = args.max_sequence_length
    dtype = args.dtype

    if run_id is None and resume:
        logger.error('Specify --run_id to resume run')
        sys.exit(1)
    elif run_id is None and not resume:
        run_id = generate_random_run_id()

    if db_path is None:
        db_path = os.path.join(os.getcwd(), DB_PATH)

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    logger.info(f'Run {run_id}')

    output_folder = os.path.join(os.getcwd(), f'saved_models/{run_id}/')
    model_path = os.path.join(output_folder, 'model.h5')
    metadata_path = os.path.join(output_folder, 'metadata.json')
    validation_output_path = os.path.join(output_folder, 'validation.csv')
    log_dir = os.path.join(os.getcwd(), f'summary_log/{run_id}')

    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass

    if resume:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        initial_epoch = 0
        dropout_rate = 0.5
        seed = np.random.randint(0, 9999)

        encoding_size = 20
        decoder_n_hidden = 100
        growth_rate = 15
        kernel_sizes = [3] + [5] * 9
        strides = None
        dilation_rates = None
        n_layers = len(kernel_sizes)
        l2_reg = 1e-5

        metadata = {
            'run_id': run_id,
            'alphabet': ALPHABET_DNA,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'encoding_size': encoding_size,
            'decoder_n_hidden': decoder_n_hidden,
            'growth_rate': growth_rate,
            'n_layers': n_layers,
            'kernel_sizes': kernel_sizes,
            'strides': strides,
            'dilation_rates': dilation_rates,
            'l2_reg': l2_reg,
            'dropout': dropout_rate,
            'n_epochs': initial_epoch,
            'seed': seed,
        }

    logger.info('Loading data')
    tmps, mean, std = load_growth_temperatures(engine)

    training_sequence = TrainingSequence(
        engine, 
        batch_size=batch_size, 
        temperatures=tmps,
        mean=mean,
        std=std,
        dtype=dtype,
        alphabet=metadata['alphabet'], 
        max_sequence_length=max_sequence_length,
        random_seed=metadata['seed'],
    )
    testing_sequence = TestingSequence(
        engine, 
        batch_size=batch_size, 
        temperatures=tmps,
        mean=mean,
        std=std,
        dtype=dtype,
        alphabet=metadata['alphabet'],
        max_sequence_length=max_sequence_length, 
        random_seed=metadata['seed'],
    )

    if variational:
        _, _, _, model = variational_conv1d_densenet(
            encoding_size=metadata['encoding_size'],
            alphabet_size=len(metadata['alphabet']),
            growth_rate=metadata['growth_rate'],
            n_layers=metadata['n_layers'],
            kernel_sizes=metadata['kernel_sizes'],
            strides=metadata.get('strides'),
            dilation_rates=metadata.get('dilation_rates'),
            l2_reg=metadata['l2_reg'],
            dropout=metadata['dropout'],
            decoder_n_hidden=metadata['decoder_n_hidden'],
        )
        compile_fn = compile_variational_model
    else:
        model = conv1d_densenet_regression_model(
            alphabet_size=len(metadata['alphabet']),
            growth_rate=metadata['growth_rate'],
            n_layers=metadata['n_layers'],
            kernel_sizes=metadata['kernel_sizes'],
            strides=metadata.get('strides'),
            dilation_rates=metadata.get('dilation_rates'),
            l2_reg=metadata['l2_reg'],
            dropout=metadata['dropout'],
            masking=True,
        )
        compile_fn = compile_regression_model

    compile_fn(model, learning_rate)

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
        training_sequence,
        validation_data=testing_sequence,
        max_queue_size=max_queue_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=verbose,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=False,
                update_freq=1000,
                embeddings_freq=0,
            ),
            SaveModelCallback(
                model_path=model_path,
                metadata_path=metadata_path,
                metadata=metadata,
            ),
        ],
    )
    logger.info('Training completed')

    logger.info('Validating on test set')
    validation_df = validate_model_on_test_set(
        engine, 
        model,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        max_sequence_length=max_sequence_length,
    )
    validation_df.to_csv(validation_output_path)

    logger.info('DONE')


if __name__ == '__main__':
    main()
