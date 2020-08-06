import os
import unittest

import numpy as np
from sqlalchemy import create_engine

from .alphabet import ALPHABET_DNA
from .load_sequences import TrainingSequence


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

        tf_sequence = TrainingSequence(
            engine, 
            batch_size=10, 
            alphabet=ALPHABET_DNA, 
            random_seed=444
        )

        self.assertGreater(len(tf_sequence), 0)
        self.assertEqual(
            len(tf_sequence), 
            int(np.ceil(len(tf_sequence.rowids) / 10)),
        )

        x_a, y_a, _ = tf_sequence[20]
        x_b, y_b, _ = tf_sequence[55]

        self.assertEqual(len(x_a), len(y_a))
        self.assertEqual(len(x_b), len(y_b))
        self.assertEqual(len(x_a), len(x_b))
        self.assertEqual(len(x_a), 10)
        self.assertNotEqual(x_a[0], x_b[0])
        self.assertIsInstance(y_a[0], float)
        self.assertIsInstance(y_b[0], float)

        first_rowid = tf_sequence.rowids[0]

        tf_sequence.on_epoch_end()

        self.assertNotEqual(first_rowid, tf_sequence.rowids[0])


if __name__ == '__main__':
    unittest.main()
