import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine

from .alphabet import ALPHABET_DNA
from .load_sequences import SpeciesSequence
from .model import (
    conv1d_densenet_regression_model,
    compile_regression_model,
)
from .validation import validate_model_for_species


class TestValidation(unittest.TestCase):

    def test_validate_model_for_species(self):
        """
        Warning: uses the actual database, may take a few second to run.
        """
        db_path = os.path.join(
            os.getcwd(), 
            'data/condensed_traits/db/seq.db',
        )
        engine = create_engine(f'sqlite+pysqlite:///{db_path}')

        tf.random.set_seed(444)

        alphabet = ALPHABET_DNA
        model = conv1d_densenet_regression_model(
            alphabet_size=len(alphabet),
            growth_rate=5,
            n_layers=2,
            kernel_sizes=[3, 2],
            masking=True,
        )
        compile_regression_model(model, learning_rate=1e-4)

        output_df = validate_model_for_species(
            engine, 
            model, 
            species_taxids=[14],
            batch_size=64,
            max_sequence_length=5000,
            max_queue_size=10,
        )

        self.assertEqual(len(output_df), 1)

        species_tax_id = output_df.index[0]
        actual, prediction, std = output_df.values[0]

        self.assertEqual(species_tax_id, 14)
        self.assertEqual(actual, 74.15)
        self.assertEqual(prediction, 37.9)
        self.assertEqual(std, 6.0)
