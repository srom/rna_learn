import numpy as np
import tensorflow as tf
from tensorflow import keras


def sequence_embedding(x, alphabet):
    """
    Args:
      x: sequence of strings
      alphabet: list of unique characters
    """
    mask_value = np.array([0] * len(alphabet), dtype=np.float32)

    x_one_hot = one_hot_encoding(x, alphabet)

    return tf.keras.preprocessing.sequence.pad_sequences(
        x_one_hot, 
        dtype='float32', 
        padding='post', 
        value=mask_value,
    )


def one_hot_encoding(x, alphabet):
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


def normalize(y, mean, std):
    return (y - mean) / std


def denormalize(y_norm, mean, std):
    return (y_norm * std) + mean
