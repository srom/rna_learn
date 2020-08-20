import os

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

if os.getcwd().endswith('notebook'):
    os.chdir('..')

from rna_learn.alphabet import ALPHABET_DNA
from rna_learn.load_sequences import (
    load_growth_temperatures, 
    compute_inverse_effective_sample,
    assign_weight_to_batch_values,
    SpeciesSequence,
)
from rna_learn.model import conv1d_densenet_regression_model, compile_regression_model


def main():
    db_path = os.path.join(os.getcwd(), 'data/condensed_traits/db/seq.db')
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    temperatures, mean, std = load_growth_temperatures(engine)

    species_seq = SpeciesSequence(
        engine, 
        species_taxid=7,
        batch_size=64, 
        temperatures=temperatures,
        mean=mean,
        std=std,
        alphabet=ALPHABET_DNA, 
        max_sequence_length=5000,
        random_seed=444,
    )

    for i in range(len(species_seq)):
        x_batch, y_norm, sample_weights = species_seq[i]


if __name__ == '__main__':
    main()
