import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def sequence_embedding(x, alphabet):
    """
    Args:
      x: sequence of strings
      alphabet: list of unique characters
    """
    mask_value = np.array([0] * len(alphabet), dtype=np.float32)

    x_one_hot = one_hot_encode_sequences(x, alphabet)

    return tf.keras.preprocessing.sequence.pad_sequences(
        x_one_hot, 
        dtype='float32', 
        padding='post', 
        value=mask_value,
    )


def one_hot_encode_sequences(x, alphabet):
    lookup = {ord(l): i for i, l in enumerate(alphabet)}
    output = []
    for x_i in x:
        x_one_hot = tf.keras.utils.to_categorical(
            [
                lookup[ord(v)] for v in x_i
            ],
            num_classes=len(alphabet)
        ).tolist()
        output.append(x_one_hot)
    return output


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
):
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

        indices = np.random.choice(
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


def split_train_test_set(x, y, test_ratio=0.2, return_indices=False):
    n_seq = len(x)
    test_idx = np.random.choice(range(n_seq), size=int(test_ratio * n_seq), replace=False)
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
