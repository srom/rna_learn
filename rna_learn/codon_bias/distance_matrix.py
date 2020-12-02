import os
import argparse
import logging

import pandas as pd
import numpy as np

from rna_learn.alphabet import CODON_REDUNDANCY
from rna_learn.codon_bias.distance import jensen_shannon_distance


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    args = parser.parse_args()

    input_path = args.input_path

    if input_path is None:
        input_path = os.path.join(os.getcwd(), 'data/codon_bias.csv')

    logger.info('Loading codon bias file')
    codon_frequency_df = pd.read_csv(input_path)

    logger.info('Aggregating per species')
    cols = codon_frequency_df.columns[1:]
    codon_df = codon_frequency_df[cols].groupby(
        ['species_taxid', 'in_test_set'],
    ).sum().reset_index().sort_values(
        'species_taxid',
    ).reset_index(drop=True)

    compute_codon_ratio(codon_df)
    ratio_columns = [c for c in codon_df.columns if c.endswith('_ratio')]

    logger.info('Exporting species codon ratio')
    output_columns = ['species_taxid', 'in_test_set'] + ratio_columns
    output_path = os.path.join(os.getcwd(), 'data/species_codon_ratios.csv')
    codon_df[output_columns].to_csv(output_path, index=False)

    logger.info('Computing distance matrix')
    distance_matrix = compute_distance_matrix(codon_df, ratio_columns)

    logger.info('Saving distance matrix')
    distance_matrix_path = os.path.join(os.getcwd(), 'data/distance_matrix.npy')
    np.save(distance_matrix_path, distance_matrix)

    logger.info('DONE')


def compute_distance_matrix(codon_df, ratio_columns):
    X = codon_df[ratio_columns].values
    n_species = len(X)

    distance_matrix = np.zeros((n_species, n_species))

    aa_indices = get_aa_indices()
    species_ix = list(range(n_species))

    for i in range(n_species - 1):
        if (i + 1) % 30 == 0:
            logger.info(f'{i + 1:,} / {n_species:,}')

        Y = np.roll(X, -(i+1), axis=0)

        roll_ix = np.roll(species_ix, -(i+1))

        d = np.zeros((n_species, len(aa_indices)))
        for aa_idx, idx in enumerate(aa_indices):
            p = X[:, idx]
            q = Y[:, idx]

            d[species_ix, aa_idx] = jensen_shannon_distance(p, q, axis=1)

        distance_matrix[species_ix, roll_ix] = np.mean(d, axis=1)

    return distance_matrix


def compute_codon_ratio(codon_df):
    processed_codons = set()
    for codon in sorted(CODON_REDUNDANCY.keys()):
        if codon in processed_codons:
            continue
        elif len(CODON_REDUNDANCY[codon]) <= 1:
            processed_codons.add(codon)
            continue
        
        synonymous_codons = CODON_REDUNDANCY[codon]
        for c, _ in synonymous_codons:
            processed_codons.add(c)
            codon_df[f'{c}_ratio'] = (
                codon_df[c] / sum((codon_df[cod] for cod, _ in synonymous_codons))
            )


def get_aa_indices():
    processed_codons = set()
    indices = []
    i = 0
    for codon in sorted(CODON_REDUNDANCY.keys()):
        if codon in processed_codons:
            continue

        elif len(CODON_REDUNDANCY[codon]) <= 1:
            processed_codons.add(codon)
            continue

        row = []
        synonymous_codons = CODON_REDUNDANCY[codon]
        for c, _ in synonymous_codons:
            processed_codons.add(c)
            row.append(i)
            i += 1

        indices.append(row)

    return indices


if __name__ == '__main__':
    main()
