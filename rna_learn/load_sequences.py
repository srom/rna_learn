import numpy as np
import pandas as pd
import tensorflow as tf

from .alphabet import ALPHABET_DNA
from .transform import sequence_embedding, normalize


LOAD_ROWIDS_QUERY = """
select rowid, length from sequences
where species_taxid in (
    select species_taxid from train_test_split
    where in_test_set = ?
)
"""

LOAD_ROWIDS_PER_SPECIES_QUERY = """
select rowid, length from sequences
where species_taxid = ?
"""


def load_train_sequence_rowids(engine):
    df = pd.read_sql(LOAD_ROWIDS_QUERY, engine, params=(0,))
    return df['rowid'].values, df['length'].values


def load_test_sequence_rowids(engine):
    df = pd.read_sql(LOAD_ROWIDS_QUERY, engine, params=(1,))
    return df['rowid'].values, df['length'].values


def load_species_rowids(engine, species_taxid):
    df = pd.read_sql(
        LOAD_ROWIDS_PER_SPECIES_QUERY, 
        engine, 
        params=(species_taxid,),
    )
    return df['rowid'].values, df['length'].values


def load_growth_temperatures(engine):
    query = "select growth_tmp from species_traits"
    growth_tmps = pd.read_sql(query, engine)['growth_tmp'].values
    return growth_tmps, np.mean(growth_tmps), np.std(growth_tmps)


def compute_inverse_probability_weights(growth_temperatures, bins_step=3):
    min_ = int(np.floor(np.min(growth_temperatures)))
    max_ = int(np.ceil(np.max(growth_temperatures)))
    bins = list(range(min_, max_, bins_step)) + [max_]
    total = len(growth_temperatures)
    values, _ = np.histogram(growth_temperatures, bins)
    weights_dict = {
        b: total / values[i]
        for i, b in enumerate(bins[:-1])
    }
    return weights_dict, bins


def compute_inverse_effective_sample(
    growth_temperatures, 
    batch_size,
    bins,
    beta=0.99,
):
    """
    Class-balanced weighting based on inverse effective sample.
    Effective sample = (1 - beta^n) / (1 - beta)
    Beta is an parameter; 0.99 is a good value in practice.
    https://arxiv.org/abs/1901.05555
    """
    values, _ = np.histogram(growth_temperatures, bins)
    inv_effective_sample_fn = lambda n: (1 - beta) / (1 - beta**n)
    inv_effective_weights = np.apply_along_axis(
        inv_effective_sample_fn, 
        axis=0, 
        arr=values,
    )
    ###
    # Factor inferred experimentally such that for a typical
    # batch, the sum of weights will equal the batch size.
    # A widely different distribution of temperatures would
    # lead to a different factor.
    factor = 0.341
    alpha = factor * batch_size
    ###
    weights_sum = np.sum(inv_effective_weights)
    weights_dict = {
        b: alpha * inv_effective_weights[i] / weights_sum
        for i, b in enumerate(bins[:-1])
    }
    return weights_dict


def assign_weight_to_batch_values(
    batch_temperatures, 
    weights_dict, 
    bins, 
    dtype='float32',
):
    index = np.digitize(batch_temperatures, bins)
    return np.array(
        [weights_dict[bins[ix-1]] for ix in index],
        dtype=dtype,
    )


def load_batch_dataframe(engine, batch_rowids):
    batch_rowids_str = ','.join([str(r) for r in batch_rowids])
    query = f"""
    select s.sequence, s.length, t.growth_tmp
    from sequences as s
    inner join species_traits as t
    on t.species_taxid = s.species_taxid
    where s.rowid in ({batch_rowids_str})
    """
    return pd.read_sql(query, engine)


def make_rowid_groups(rowids, lengths, bins):
    groups = []
    for min_v, max_v in zip(bins, bins[1:]):
        group_ix = [
            i for i, length in enumerate(lengths)
            if length >= min_v and length < max_v
        ]
        groups.append(rowids[group_ix])

    return groups


def shuffle_rowid_groups(groups, rs):
    for i in range(len(groups)):
        groups[i] = rs.choice(
            groups[i], 
            size=len(groups[i]), 
            replace=False,
        )


def get_batched_rowids(groups, batch_size, rs):
    batched_rowids = []
    leftover_rowids = []
    for group in groups:
        n_batches = int(np.ceil(len(group) / batch_size))
        for i in range(n_batches):
            a = i * batch_size
            b = (i + 1) * batch_size
            batch = group[a:b]
            if len(batch) == batch_size:
                batched_rowids.append(batch)
            else:
                leftover_rowids += batch.tolist()

    ix = list(range(len(batched_rowids)))
    batch_ids = rs.choice(ix, size=len(ix), replace=False)

    rowids = []
    for i in batch_ids:
        rowids += batched_rowids[i].tolist()

    rowids += leftover_rowids

    return np.array(rowids)


def expand_inputs_when_exceeding_length(batch_df, max_length):
    sequences = batch_df['sequence'].values
    temperatures = batch_df['growth_tmp'].values
    lengths = batch_df['length'].values

    if batch_df['length'].max() <= max_length:
        return sequences, temperatures

    expanded_sequences, expanded_tmps = [], []
    for i, sequence in enumerate(sequences):
        temperature = temperatures[i]
        length = lengths[i]
        if length > max_length:
            chunks = int(np.ceil(len(sequence) / max_length))
            for i in range(chunks):
                a = i * max_length
                b = (i + 1) * max_length
                seq = sequence[a:b]
                expanded_sequences.append(sequence)
                expanded_tmps.append(temperature)
        else:
            expanded_sequences.append(sequence)
            expanded_tmps.append(temperature)

    return (
        np.array(expanded_sequences, dtype=sequences.dtype),,
        np.array(expanded_tmps, dtype=temperatures.dtype),
    )


class SequenceBase(tf.keras.utils.Sequence):

    def __init__(self, 
        engine,
        batch_size, 
        alphabet, 
        temperatures=None,
        mean=None, 
        std=None,
        max_sequence_length=None,
        dtype='float32', 
        random_seed=None,
    ):
        self.rs = np.random.RandomState(random_seed)
        self.engine = engine

        rowids, lengths = self.fetch_rowids_and_lengths()

        group_bins = [1, 250, 500, 1000, 2000, 3000, 4000, 5000, np.inf]
        self.num_batches = int(np.ceil(len(rowids) / batch_size))
        self.rowid_groups = make_rowid_groups(rowids, lengths, group_bins)
        shuffle_rowid_groups(self.rowid_groups, self.rs)
        self.rowids = get_batched_rowids(
            self.rowid_groups, 
            batch_size,
            self.rs,
        )

        if temperatures is None:
            temperatures, mean, std = load_growth_temperatures(engine)

        bins = np.array([0, 20, 30, 40, 50, 60, 70, 80, 90, 105])
        bin_to_weights = compute_inverse_effective_sample(
            temperatures,
            batch_size,
            bins,
        )
        self.bin_to_weights = bin_to_weights
        self.bins = bins
        self.mean = mean
        self.std = std

        self.batch_size = batch_size
        self.alphabet = alphabet
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_ix):
        a = batch_ix * self.batch_size
        b = (batch_ix + 1) * self.batch_size

        batch_rowids = self.rowids[a:b]
        batch_df = load_batch_dataframe(self.engine, batch_rowids)

        x_raw, batch_y = expand_inputs_when_exceeding_length(
            batch_df,
            self.max_sequence_length,
        )
        batch_x = sequence_embedding(
            x_raw, 
            self.alphabet, 
            dtype=self.dtype,
        )
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
        shuffle_rowid_groups(self.rowid_groups, self.rs)
        self.rowids = get_batched_rowids(
            self.rowid_groups, 
            self.batch_size,
            self.rs,
        )

    def fetch_rowids_and_lengths(self):
        raise NotImplementedError()


class TrainingSequence(SequenceBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fetch_rowids_and_lengths(self):
        return load_train_sequence_rowids(self.engine)
            

class TestingSequence(SequenceBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fetch_rowids_and_lengths(self):
        return load_test_sequence_rowids(self.engine)


class SpeciesSequence(SequenceBase):

    def __init__(self, *args, **kwargs):
        if 'species_taxid' not in kwargs:
            raise ValueError('Species taxid not specified')
        else:
            self.species_taxid = kwargs.pop('species_taxid')

        super().__init__(*args, **kwargs)

    def fetch_rowids_and_lengths(self):
        return load_species_rowids(self.engine, self.species_taxid)    
