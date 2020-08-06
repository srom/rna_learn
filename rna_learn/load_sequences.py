import numpy as np
import pandas as pd
import tensorflow as tf

from .transform import sequence_embedding


LOAD_ROWIDS_QUERY = """
select rowid from sequences
where species_taxid in (
    select species_taxid from train_test_split
    where in_test_set = ?
)
"""


def load_train_sequence_rowids(engine):
    df = pd.read_sql(LOAD_ROWIDS_QUERY, engine, params=(0,))
    return df['rowid'].values


def load_test_sequence_rowids(engine):
    df = pd.read_sql(LOAD_ROWIDS_QUERY, engine, params=(1,))
    return df['rowid'].values


def load_batch_dataframe(engine, batch_rowids):
    batch_rowids_str = ','.join([str(r) for r in batch_rowids])
    query = f"""
    select s.sequence, t.growth_tmp
    from sequences as s
    inner join species_traits as t
    on t.species_taxid = s.species_taxid
    where s.rowid in ({batch_rowids_str})
    """
    return pd.read_sql(query, engine)


class TrainingSequence(tf.keras.utils.Sequence):

    def __init__(self, engine, batch_size, alphabet, dtype='float32', random_seed=None):
        self.rs = np.random.RandomState(random_seed)
        rowids = load_train_sequence_rowids(engine)
        self.rowids = self.rs.choice(
            rowids, 
            size=len(rowids), 
            replace=False,
        )
        self.batch_size = batch_size
        self.alphabet = alphabet
        self.engine = engine
        self.dtype = dtype

    def __len__(self):
        return int(np.ceil(len(self.rowids) / self.batch_size))

    def __getitem__(self, batch_ix):
        a = batch_ix * self.batch_size
        b = (batch_ix + 1) * self.batch_size

        batch_rowids = self.rowids[a:b]
        batch_df = load_batch_dataframe(self.engine, batch_rowids)

        batch_x = sequence_embedding(
            batch_df['sequence'].values, 
            self.alphabet, 
            dtype=self.dtype,
        )
        batch_y = batch_df['growth_tmp'].values

        return batch_x, batch_y, [None]

    def on_epoch_end(self):
        """
        Re-shuffle ahead of next epoch.
        """
        self.rowids = self.rs.choice(
            self.rowids, 
            size=len(self.rowids), 
            replace=False,
        )
