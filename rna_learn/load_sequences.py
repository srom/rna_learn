import numpy as np
import pandas as pd
import tensorflow as tf

from .alphabet import ALPHABET_DNA
from .transform import sequence_embedding, normalize


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


def load_growth_temperatures(engine):
    query = "select growth_tmp from species_traits"
    growth_tmps = pd.read_sql(query, engine)['growth_tmp'].values
    return growth_tmps, np.mean(growth_tmps), np.std(growth_tmps)


def compute_inverse_probability_weights(growth_temperatures, step=3):
    min_ = int(np.floor(np.min(growth_temperatures)))
    max_ = int(np.ceil(np.max(growth_temperatures)))
    bins = list(range(min_, max_, step)) + [max_]
    total = len(growth_temperatures)
    values, _ = np.histogram(growth_temperatures, bins)
    weights_dict = {
        b: total / values[i]
        for i, b in enumerate(bins[:-1])
    }
    return weights_dict, bins


def assign_weight_to_batch_values(growth_temperatures, weights_dict, bins, dtype):
    index = np.digitize(growth_temperatures, bins)
    weights_u = np.array(
        [weights_dict[bins[ix-1]] for ix in index],
        dtype=dtype,
    )
    return weights_u / np.sum(weights_u)


def load_partial_test_set(
    engine, 
    alphabet=ALPHABET_DNA,
    dtype='float32', 
    n_records=int(1e4), 
    seed=123,
):
    rs = np.random.RandomState(seed)
    rowids = load_test_sequence_rowids(engine)
    _, mean, std = load_growth_temperatures(engine)
    selected_rowids = rs.choice(rowids, size=n_records, replace=False)
    df = load_batch_dataframe(engine, selected_rowids)

    x_test = sequence_embedding(
        df['sequence'].values, 
        alphabet, 
        dtype=dtype,
    )
    y_test = df['growth_tmp'].values
    y_test_norm = normalize(y_test, mean, std)
    return x_test, y_test, y_test_norm, mean, std


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

        # Compute inverse probability weights based on the frequency 
        # of growth temperatures in the whole dataset.
        temperatures, mean, std = load_growth_temperatures(engine)
        bin_to_weights, bins = compute_inverse_probability_weights(temperatures) 
        self.bin_to_weights = bin_to_weights
        self.bins = bins
        self.mean = mean
        self.std = std

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
        batch_y_norm = normalize(
            batch_y, 
            self.mean,
            self.std,
        )
        sample_weights = assign_weight_to_batch_values(
            batch_y, 
            self.bin_to_weights, 
            self.bins,
            dtype=self.dtype,
        )
        return batch_x, batch_y_norm, sample_weights

    def on_epoch_end(self):
        """
        Re-shuffle ahead of next epoch.
        """
        self.rowids = self.rs.choice(
            self.rowids, 
            size=len(self.rowids), 
            replace=False,
        )
