import os
import re
import json

import pandas as pd
import numpy as np

from .model import (
    rnn_regression_model, 
    rnn_classification_model,
    conv1d_classification_model,
    compile_regression_model,
    compile_classification_model,
)


def load_dataset(input_path, alphabet, secondary=True):
    output_df = pd.read_csv(input_path)

    if sorted(set(alphabet)) == ['A', 'C', 'G', 'T']:
        output_df['gc_content'] = output_df.apply(get_gc_content, axis=1)
        output_df['ag_content'] = output_df.apply(get_ag_content, axis=1)
        output_df['gt_content'] = output_df.apply(get_gt_content, axis=1)

    if secondary:
        output_df['secondary_structure'] = output_df.apply(get_secondary_structure, axis=1)
        output_df['paired_nucleotides'] = output_df.apply(get_paired_nucleotides, axis=1)

    return output_df


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
               r'\s+\(\s*[-0-9\.]+\s*\)\s*$', 
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
            metadata = json.load(f)

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
    if learning_type == 'classification_lstm':
        model = rnn_classification_model(alphabet_size=len(alphabet), n_classes=len(classes), n_lstm=n_lstm)
        compile_classification_model(model, learning_rate=learning_rate)
    if learning_type == 'classification_conv1d':
        model = conv1d_classification_model(alphabet_size=len(alphabet), n_classes=len(classes))
        compile_classification_model(model, learning_rate=learning_rate)
    else:
        raise ValueError(f'Unknown learning type: {learning_type}')

    if resume:
        model.load_weights(model_path)

    return model, metadata


def get_gc_content(row):
    return get_letters_content(row, {'G', 'C'})


def get_ag_content(row):
    return get_letters_content(row, {'A', 'G'})


def get_gt_content(row):
    return get_letters_content(row, {'G', 'T'})


def get_letters_content(row, letters):
    sequence = row['sequence']
    length = row['length']
    n = len([b for b in sequence if b in letters])
    return n / length


def get_secondary_structure(row):
    specie_name = row['specie_name'].replace(' ', '_').lower()
    seqid = row['seqid'].replace(' ', '_').lower()
    gene_name = row['gene_name'].replace(' ', '_').lower()

    f_name = f'{specie_name}_{seqid}_{gene_name}.fold'
    folder = 'data/ncbi/secondary_structure/'
    path = os.path.join(os.getcwd(), folder, f_name)

    with open(path) as f:
        lines = f.readlines()

        # Remove free energy information at the end
        content = re.sub(
           r'\s+\(\s*[-0-9\.]+\s*\)\s*$', 
           '', 
           lines[-1]
        )
        content = content.strip()

    return content


def get_paired_nucleotides(row):
    secondary_structure = row['secondary_structure']
    length = row['length']
    p = len([b for b in secondary_structure if b in {'(', ')'}])
    return p / length
