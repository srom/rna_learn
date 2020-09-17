import os
import json
import timeit

import pandas as pd
import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine

from rna_learn.alphabet import ALPHABET_DNA
from rna_learn.load_sequences import (
    load_growth_temperatures, 
    compute_inverse_effective_sample,
    assign_weight_to_batch_values,
    SpeciesSequence,
)
from rna_learn.model import conv1d_densenet_regression_model, compile_regression_model


def main():
    run_id = 'run_yb64o'
    model_path = os.path.join(os.getcwd(), f'saved_models/{run_id}/model.h5')
    metadata_path = os.path.join(os.getcwd(), f'saved_models/{run_id}/metadata.json')

    db_path = os.path.join(os.getcwd(), 'data/condensed_traits/db/seq.db')
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    with open(metadata_path) as f:
        metadata = json.load(f)

    temperatures, mean, std = load_growth_temperatures(engine)

    model = conv1d_densenet_regression_model(
        alphabet_size=len(metadata['alphabet']), 
        growth_rate=metadata['growth_rate'],
        n_layers=metadata['n_layers'],
        kernel_sizes=metadata['kernel_sizes'],
        dilation_rates=metadata['dilation_rates'],
        l2_reg=metadata['l2_reg'],
        dropout=metadata['dropout'],
    )
    model.load_weights(model_path)

    species_taxid = 167
    max_sequence_length = 9999
    species_seq = SpeciesSequence(
        engine, 
        species_taxid=species_taxid,
        batch_size=64, 
        temperatures=temperatures,
        mean=mean,
        std=std,
        alphabet=ALPHABET_DNA, 
        max_sequence_length=max_sequence_length,
        random_seed=metadata['seed'],
    )

    print('fast:', timeit.timeit(lambda: evaluate_seq_fast(model, species_seq), number=1))
    print('fast 2:', timeit.timeit(lambda: evaluate_seq_fast_2(model, species_seq), number=1))
    print('slow:', timeit.timeit(lambda: evaluate_seq(model, species_seq), number=1))


def evaluate_seq(model, species_seq):
    iterator = species_seq.__iter__()
    for _ in range(len(species_seq)):
        batch_x, _, _ = next(iterator)
        _ = model(batch_x)


@tf.function(experimental_relax_shapes=True)
def evaluate_seq_fast(model, species_seq):
    iterator = species_seq.__iter__()
    for _ in tf.range(len(species_seq)):
        batch_x, _, _ = next(iterator)
        _ = model(batch_x)


def evaluate_seq_fast_2(model, species_seq):

    def evaluate_batch(batch_x):
        model(batch_x)

    fn = tf.function(evaluate_batch, experimental_relax_shapes=True)

    iterator = species_seq.__iter__()
    for _ in range(len(species_seq)):
        batch_x, _, _ = next(iterator)
        fn(batch_x)


if __name__ == '__main__':
    main()
