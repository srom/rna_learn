import os
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from .alphabet import ALPHABET_DNA
from .load_sequences import SpeciesSequence


class TestTrainingSequence(unittest.TestCase):

    def test_training_sequence(self):
        """
        This test uses the actual database, therefore it is not a 
        unittest per se but rather some sort of functional test.
        """
        db_path = os.path.join(
            os.getcwd(), 
            'data/condensed_traits/db/seq.db',
        )
        engine = create_engine(f'sqlite+pysqlite:///{db_path}')

        batch_size = 10
        tf_sequence = SpeciesSequence(
            engine, 
            species_taxid=7,
            batch_size=batch_size, 
            alphabet=ALPHABET_DNA, 
            max_sequence_length=100,
            random_seed=444,
        )

        self.assertGreater(len(tf_sequence), 0)
        self.assertEqual(
            len(tf_sequence), 
            int(np.ceil(len(tf_sequence.rowids) / batch_size)),
        )

        x_a, y_a, _ = tf_sequence[1]
        x_b, y_b, _ = tf_sequence[5]

        self.assertEqual(len(x_a), len(y_a))
        self.assertEqual(len(x_b), len(y_b))
        self.assertEqual(len(x_a), len(x_b))
        self.assertEqual(len(x_a), 10)
        self.assertEqual(x_a.shape[-1], 4)
        self.assertLessEqual(x_a.shape[1], 100)
        self.assertIsInstance(y_a[0], np.float32)
        self.assertIsInstance(y_b[0], np.float32)

        first_rowid = tf_sequence.rowids[0][0]

        tf_sequence.on_epoch_end()

        self.assertNotEqual(first_rowid, tf_sequence.rowids[0][0])

        x_a, y_a, _ = tf_sequence[1]
        self.assertEqual(len(x_a), 10)
        self.assertEqual(x_a.shape[-1], 4)
        self.assertLessEqual(x_a.shape[1], 100)
        self.assertEqual(y_a.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
