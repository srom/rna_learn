import argparse
import collections
import itertools
import os
import json
import logging

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from rna_learn.alphabet import CODON_REDUNDANCY
from rna_learn.codon_bias.distance import jensen_shannon_distance


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path

    if db_path is None:
        db_path = os.path.join(os.getcwd(), 'data/db/seq.db')
    
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    possible_codons = sorted(CODON_REDUNDANCY.keys())

    assembly_bias_path = os.path.join(os.getcwd(), 'data/tri_nucleotide_bias_cds.csv')
    assembly_bias_df = pd.read_csv(assembly_bias_path, index_col='assembly_accession')

    assemblies = assembly_bias_df.index
    ratio_columns = [f'{codon}_ratio' for codon in possible_codons]

    for i, assembly in enumerate(assemblies):
        if i == 0 or (i+1) % 10 == 0:
            logger.info(f'{i+1:,} / {len(assemblies):,}')

        series = assembly_bias_df.loc[assembly]
        species_taxid = series['species_taxid']

        distances_df = compute_ratios_and_distance(
            engine, 
            assembly, 
            species_taxid,
            assembly_bias_df,
            possible_codons, 
            ratio_columns,
        )

        output_path = os.path.join(
            os.getcwd(), 
            f'data/cds_tri_nucleotide/{assembly}_bias.csv',
        )
        distances_df.to_csv(output_path, index=False)

    logger.info('DONE')


def compute_ratios_and_distance(
    engine, 
    assembly, 
    species_taxid,
    assembly_bias_df,
    possible_codons, 
    ratio_columns,
):
    data = {
        'assembly_accession': [],
        'species_taxid': [],
        'protein_id': [],
        'description': [],
        'length': [],
        'distance': [],
    }
    data.update({
        r: []
        for r in ratio_columns
    })

    query = """
    select metadata_json, sequence from sequences
    where assembly_accession = ? and sequence_type = 'CDS'
    """
    df = pd.read_sql(
        query, 
        engine, 
        params=(assembly,),
    )
    metadata_list = [
        json.loads(df.loc[ix, 'metadata_json']) 
        for ix in df.index
        if not pd.isnull(df.loc[ix, 'metadata_json'])
    ]
    sequences = [
        df.loc[ix, 'sequence']
        for ix in df.index
        if not pd.isnull(df.loc[ix, 'metadata_json'])
    ]

    for i in range(len(metadata_list)):
        metadata = metadata_list[i]
        sequence = sequences[i]

        protein_id = metadata.get('protein_id')
        protein_label = metadata.get('protein', 'Unknown')
        if pd.isnull(protein_id):
            continue

        ratios = compute_ratios(sequence, possible_codons)
        if ratios is None:
            continue

        assembly_bias = assembly_bias_df.loc[assembly][
            ratio_columns
        ].values.astype(float)

        distance = jensen_shannon_distance(assembly_bias, ratios)

        data['assembly_accession'].append(assembly)
        data['species_taxid'].append(species_taxid)
        data['protein_id'].append(protein_id)
        data['description'].append(protein_label)
        data['length'].append(len(sequence))
        data['distance'].append(distance)

        for j, r in enumerate(ratio_columns):
            data[r].append(ratios[j])

    output_df = pd.DataFrame.from_dict(data)
    distance_vector = output_df['distance'].values

    n_stds = 1.3
    thres = np.mean(distance_vector) - n_stds * np.std(distance_vector)
    while thres <= 0:
        n_stds -= 0.1
        thres = np.mean(distance_vector) - n_stds * np.std(distance_vector)

    for r in ratio_columns:
        output_df[r] = output_df[r].round(6)

    output_df['distance'] = output_df['distance'].round(6)
    output_df['threshold'] = np.round(thres, 6)
    output_df['below_threshold'] = output_df['distance'] <= thres

    return output_df


def compute_ratios(seq, possible_codons):
    if len(seq) < 3:
        return None

    possible_codons_set = set(possible_codons)
    triplet_count = collections.defaultdict(int)

    # Consider the 3 possible reading frames
    for frame in [0, 1, 2]:
        if len(seq) < 3 + frame:
            continue

        reminder = len(seq[frame:]) % 3
        if reminder > 0:
            sequence = seq[frame:-reminder]
        else:
            sequence = seq[frame:]

        assert len(sequence) % 3 == 0

        for pos in range(0, len(sequence), 3):
            codon = sequence[pos:pos+3]
            if codon not in possible_codons_set:
                # Ignore ambiguous letters (e.g. N, etc)
                continue
            
            triplet_count[codon] += 1

    total_sum = np.sum([v for v in triplet_count.values()])

    codon_counts = np.array([
        triplet_count.get(codon, 0)
        for codon in possible_codons
    ])

    return (codon_counts / total_sum).astype(float)


if __name__ == '__main__':
    main()
