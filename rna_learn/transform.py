import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .alphabet import CODON_REDUNDANCY


def sequence_embedding(x, alphabet, unknown_chars=('-', 'X'), dtype='float32'):
    """
    Args:
      x: sequence of strings
      alphabet: list of unique characters
    """
    mask_value = np.array([0] * len(alphabet), dtype=dtype)

    x_one_hot = one_hot_encode_sequences(x, alphabet, unknown_chars)

    return tf.keras.preprocessing.sequence.pad_sequences(
        x_one_hot, 
        dtype=dtype, 
        padding='post', 
        value=mask_value,
    )


def one_hot_encode_sequences(x, alphabet, unknown_chars):
    lookup = {
        ord(unknown_char): [0.] * len(alphabet) 
        for unknown_char in unknown_chars
    }
    for i, l in enumerate(alphabet):
        key = ord(l)
        val = [0.] * len(alphabet)
        val[i] = 1.
        lookup[key] = val

    output = []
    for x_i in x:
        row = []
        for v in x_i:
            row.append(lookup[ord(v)])

        output.append(row)

    return output


def combine_sequences(x, aa, dtype='float64'):
    output = []
    for r, xi in enumerate(x):
        sequences = []
        for i, n in enumerate(xi):
            aa_idx = int(np.floor(i / 3))
            a = aa[r, aa_idx, :]

            combined = n.tolist() + a.tolist()

            sequences.append(combined)

        output.append(sequences)

    return np.array(output, dtype=dtype)


def one_hot_encode_classes(y, classes):
    return tf.keras.utils.to_categorical(
        [classes.index(c) for c in  y], 
        num_classes=len(classes),
    )


def make_dataset_balanced(
    metadata, 
    cat_name='rna.type', 
    output_col=None, 
    n_entries_per_class=None,
    classes=None,
    seed=None,
):
    rs = np.random.RandomState(seed)
    
    if classes is None:
        classes = sorted(metadata[cat_name].unique().tolist())

    if output_col is None:
        output_col = cat_name

    if n_entries_per_class is None:
        candidates = []
        for class_ in classes:
            metadata_slice = metadata[metadata[cat_name] == class_]
            n_entries = len(metadata_slice)
            candidates.append(n_entries)

        n_entries_per_class = min(candidates)

    output_data = []
    output_metadata = []

    for class_ in classes:
        metadata_slice = metadata[metadata[cat_name] == class_]
        n_entries = len(metadata_slice)
        values_idx = metadata_slice.index.tolist()

        indices = rs.choice(
            values_idx, 
            size=n_entries_per_class, 
            replace=n_entries < n_entries_per_class,
        )

        sorted_indices = sorted(indices.tolist())

        for idx in sorted_indices:
            m = metadata.loc[idx]
            output_data.append(m[output_col])
            output_metadata.append(m.values)

    return (
        np.stack(output_data),
        pd.DataFrame(output_metadata, columns=metadata.columns)
    )


def split_train_test_set(x, y, test_ratio=0.2, return_indices=False, seed=None):
    rs = np.random.RandomState(seed)

    n_seq = len(x)
    test_idx = rs.choice(range(n_seq), size=int(test_ratio * n_seq), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = np.array([idx for idx in range(n_seq) if idx not in test_idx_set])

    x_test, y_test = x[test_idx], y[test_idx]
    x_train, y_train = x[train_idx], y[train_idx]

    if not return_indices:
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test, train_idx, test_idx


def get_x_extras(dataset_df, extras):
    x_extras = []
    for extra in extras:
        x_extras.append(dataset_df[extra].values)
    return np.stack(x_extras).swapaxes(0, 1)


def normalize(y, mean, std):
    return (y - mean) / std


def denormalize(y_norm, mean, std):
    return (y_norm * std) + mean


def randomly_swap_redundant_codon(rna_seq, seed):
    rs = np.random.RandomState(seed)

    if len(rna_seq) % 3 != 0:
        raise ValueError('RNA sequence length must be a multiple of 3')

    # Start and end codons are left unmodified.
    if len(rna_seq) <= 6:
        return seq

    seq = rna_seq[3:-3]

    output_seq = rna_seq[:3]
    for codon_ix in range(0, int(len(seq) / 3) + 3, 3):
        codon = seq[codon_ix:codon_ix + 3]

        if codon not in CODON_REDUNDANCY:
            output_seq += codon
        else:
            r_map = CODON_REDUNDANCY[codon]

            candidates = [m[0] for m in r_map]
            probabilities = [m[1] for m in r_map]

            s = rs.choice(candidates, size=1, p=probabilities)

            output_seq += ''.join(s)

    output_seq += rna_seq[-3:]

    return output_seq
