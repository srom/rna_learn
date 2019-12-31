import os
import re

import pandas as pd
import numpy as np

from .model import (
    rnn_regression_model, 
    rnn_classification_model,
    compile_regression_model,
    compile_classification_model,
)


def load_dataset(input_path, alphabet):
    df = pd.read_csv(input_path)

    indices = []
    alphabet_set = set(alphabet)
    for i, tpl in enumerate(df.itertuples()):
        sequence = tpl.sequence
        if len(set(sequence) - alphabet_set) == 0:
            indices.append(i)
    return df.iloc[indices].reset_index(drop=True)


def load_rna_structure_dataset(metadata, sequence_folder_path):
    sequences = []
    for tpl in metadata.itertuples():
        rna_type, prot_type = getattr(tpl, 'category').split(' ')
        filename = tpl.sp.replace(' ', '_') + '.structure.txt'

        path = os.path.join(
            sequence_folder_path, 
            rna_type, 
            prot_type,
            filename,
        )
        with open(path) as f:
            content = f.read()

            # Remove free energy information at the end
            content = re.sub(
               r'\s+\([-0-9\.]+\)\s*$', 
               '', 
               content
            )
            content = content.strip()

            sequences.append(content)

    return sequences


def load_rna_nucleotides_dataset(metadata, sequence_folder_path):
    sequences = []
    for tpl in metadata.itertuples():
        rna_type, prot_type = getattr(tpl, 'category').split(' ')
        filename = tpl.sp.replace(' ', '_') + '.fasta'

        path = os.path.join(
            sequence_folder_path, 
            rna_type, 
            prot_type,
            filename,
        )
        with open(path) as f:
            while True:
                content = f.readline()
                if not content.startswith('>'):
                    break

            content = content.strip()
            sequences.append(content)

    return sequences


def load_mrna_model(run_id, learning_type, learning_rate, model_path, metadata_path, resume):
    if resume:
        if not os.path.isfile(model_path):
            raise ValueError(f'Model file does not exist: {model_path}')
        elif not os.path.isfile(metadata_path):
            raise ValueError(f'Metadata file does not exist: {metadata_path}')

        with open(metadata_path, 'r') as f:
            metadata = jons.load(f)

        n_lstm = metadata['n_lstm']
        alphabet = metadata['alphabet']
        classes = metadata['classes']
        random_seed = metadata['random_seed']
        np.random.seed(random_seed)
    else:
        n_lstm = 2
        alphabet = ['A', 'T', 'G', 'C']
        classes = ['psychrophilic', 'mesophilic', 'thermophilic']
        random_seed = np.random.randint(1, 10000)

        metadata = {
            'run_id': run_id,
            'learning_type': learning_type,
            'alphabet': alphabet,
            'n_lstm': n_lstm,
            'classes': classes,
            'random_seed': random_seed,
            'n_epochs': 0,
        }
        np.random.seed(random_seed)

    if learning_type == 'regression':
        model = rnn_regression_model(alphabet_size=len(alphabet), n_lstm=n_lstm)
        compile_regression_model(model, learning_rate=learning_rate)
    else:
        model = rnn_classification_model(alphabet_size=len(alphabet), n_classes=len(classes), n_lstm=n_lstm)
        compile_classification_model(model, learning_rate=learning_rate)

    if resume:
        model.load_weights(model_path)

    return model, metadata
