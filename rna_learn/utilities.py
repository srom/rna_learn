import string
import json

import numpy as np
import tensorflow as tf


class SaveModelCallback(tf.keras.callbacks.Callback):

    def __init__(
        self, 
        model_path, 
        metadata_path, 
        metadata, 
        monitor='val_loss',
        metrics=('val_accuracy',),
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
