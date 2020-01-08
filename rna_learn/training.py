import os
import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from .model import (
    conv1d_classification_model,
    compile_classification_model,
)
from .transform import (
    sequence_embedding, 
    one_hot_encode_classes,
    split_train_test_set,
)


logger = logging.getLogger(__name__)


def train_conv1d_with_hyperparameters(
    n_epochs,
    batch_size,
    learning_rate,
    adam_epsilon,
    n_conv_1,
    n_filters_1, 
    kernel_size_1,
    l2_reg_1,
    n_conv_2,
    n_filters_2, 
    kernel_size_2,
    l2_reg_2,
    dropout,
    train_set_path=None,
    verbose=1,
):
    alphabet = ['A', 'T', 'G', 'C']
    classes = ['psychrophilic', 'mesophilic', 'thermophilic']

    if train_set_path is None:
        train_set_path = os.path.join(os.getcwd(), 'data/dataset_train.csv')

    dataset_df = pd.read_csv(train_set_path)

    sequences = dataset_df['sequence'].values
    temperature_classes = dataset_df['temperature_range'].values

    x = sequence_embedding(sequences, alphabet)
    y = one_hot_encode_classes(temperature_classes, classes)

    x_train, y_train, x_test, y_test = split_train_test_set(x, y, test_ratio=0.2)

    start = time.time()

    model = conv1d_classification_model(
        alphabet_size=len(alphabet), 
        n_classes=len(classes),
        n_conv_1=n_conv_1,
        n_filters_1=n_filters_1, 
        kernel_size_1=kernel_size_1,
        l2_reg_1=l2_reg_1,
        n_conv_2=n_conv_2,
        n_filters_2=n_filters_2, 
        kernel_size_2=kernel_size_2,
        l2_reg_2=l2_reg_2,
        dropout=dropout,
    )
    compile_classification_model(model, learning_rate, adam_epsilon)

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=verbose,
    )
    metrics = model.evaluate(x_test, y_test, verbose=verbose)

    loss = metrics[1] if len(metrics) > 1 else metrics[0]

    elapsed = time.time() - start

    return float(loss), elapsed, model
