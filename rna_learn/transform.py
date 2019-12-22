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


def make_dataset_balanced(metadata, cat_name='rna.type', n_entries_per_class=1000):
    classes = sorted(metadata[cat_name].unique().tolist())

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
            output_data.append(m[cat_name])
            output_metadata.append(m.values)

    return (
        np.stack(output_data),
        pd.DataFrame(output_metadata, columns=metadata.columns)
    )


def normalize(y, mean, std):
    return (y - mean) / std


def denormalize(y_norm, mean, std):
    return (y_norm * std) + mean
