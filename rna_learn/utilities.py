import json
import math
import string

import numpy as np
import tensorflow as tf

from .transform import sequence_embedding


class SaveModelCallback(tf.keras.callbacks.Callback):

    def __init__(
        self, 
        model_path, 
        metadata_path, 
        metadata, 
        monitor='val_loss',
        metrics=('val_accuracy', 'val_mae'),
    ):
        self.monitor = monitor
        self.metrics = metrics
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.best_loss = metadata.get(self.monitor, np.inf)

    def on_epoch_end(self, epoch, logs):
        val_loss = logs[self.monitor]

        if val_loss < self.best_loss:
            self.model.save(self.model_path)

            self.metadata['n_epochs'] = epoch + 1
            self.metadata[self.monitor] = val_loss

            for metric in self.metrics or []:
                if metric in logs:
                    self.metadata[metric] = float(logs[metric])

            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)

            self.best_loss = val_loss


def generate_random_run_id():
    chars = [c for c in string.ascii_lowercase + string.digits]
    random_slug_chars = np.random.choice(chars, size=5, replace=True)
    random_slug = ''.join(random_slug_chars)
    return f'run_{random_slug}'


class BioSequence(tf.keras.utils.Sequence):

    def __init__(self, x_raw, y, batch_size, alphabet, x_extras=None, random_seed=None):
        self.rs = np.random.RandomState(random_seed)

        idx = list(range(len(x_raw)))
        shuffled_idx = self.rs.choice(idx, size=len(idx), replace=False)

        self.x_raw = x_raw[shuffled_idx]
        self.x_extras = x_extras
        self.y = y[shuffled_idx]

        self.batch_size = batch_size
        self.alphabet = alphabet

    def __len__(self):
        return math.ceil(len(self.x_raw) / self.batch_size)

    def __getitem__(self, idx):
        a = idx * self.batch_size
        b = (idx + 1) * self.batch_size

        batch_x_raw = self.x_raw[a:b]
        batch_y = self.y[a:b]

        batch_x = sequence_embedding(batch_x_raw, self.alphabet, dtype='float64')

        batch_x_extras = None
        if self.x_extras is not None:
            batch_x_extras = self.x_extras[a:b]

        if self.x_extras is None:
            return batch_x, batch_y, [None]
        else:
            return (batch_x, batch_x_extras), batch_y, [None]

    def on_epoch_end(self):
        """
        Re-shuffle ahead of next epoch.
        """
        idx = list(range(len(self.x_raw)))
        shuffled_idx = self.rs.choice(idx, size=len(idx), replace=False)

        self.x_raw = self.x_raw[shuffled_idx]
        self.y = self.y[shuffled_idx]

        if self.x_extras is not None:
            self.x_extras = self.x_extras[shuffled_idx]
