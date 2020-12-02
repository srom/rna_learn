import collections
import os
import json
import logging

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from ..alphabet import CODON_REDUNDANCY
from .distance import compute_codon_bias_distance, get_aa_indices


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(levelname)s) %(message)s')
    
    db_path = os.path.join(os.getcwd(), 'data/db/seq.db')
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    species_codon_ratios_path = os.path.join(os.getcwd(), 'data/species_codon_ratios.csv')
    species_codon_df = pd.read_csv(species_codon_ratios_path)

    ratio_columns = [c for c in species_codon_df.columns if c.endswith('_ratio')]

    assembly_query = """
    select assembly_accession, species_taxid from assembly_source
    """
    assembly_df = pd.read_sql(assembly_query, engine)

    assembly_accessions = assembly_df['assembly_accession'].values
    species_taxids = assembly_df['species_taxid'].values

    for i, assembly_ix in enumerate(range(len(assembly_accessions))):
        if i == 0 or (i+1) % 10 == 0:
            logger.info(f'{i+1:,} / {len(assembly_accessions):,}')

        assembly_accession = assembly_accessions[assembly_ix]
        species_taxid = species_taxids[assembly_ix]

        output_path_1 = os.path.join(os.getcwd(), f'data/cds_codon_bias/all/{assembly_accession}_codon_bias.csv')
        output_path_2 = os.path.join(os.getcwd(), f'data/cds_codon_bias/below_threshold/{assembly_accession}_codon_bias.csv')

        distance_df = distance_to_species_ratios(engine, species_codon_df, ratio_columns, species_taxid, assembly_accession)

        distance_df.to_csv(output_path_1, index=False)
        distance_df[distance_df['below_threshold']].to_csv(output_path_2, index=False)

    logger.info('DONE')


def distance_to_species_ratios(engine, species_codon_df, ratio_columns, species_taxid, assembly_accession):
    query = """
    select metadata_json, length, sequence from sequences
    where assembly_accession = ? and sequence_type = 'CDS'
    """
    df = pd.read_sql(query, engine, params=(assembly_accession,))
    
    coding_sequences = df['sequence'].values
    metadata_values = df['metadata_json'].values
    lengths = df['length'].values
    
    target = species_codon_df[
        species_codon_df['species_taxid'] == species_taxid
    ].iloc[0][ratio_columns].values.astype(float)
    
    aa_indices = get_aa_indices()
    
    data = []
    for i, sequence in enumerate(coding_sequences):
        metadata_json = metadata_values[i]
        length = lengths[i]

        if metadata_json is None:
            continue
        else:
            metadata = json.loads(metadata_json)
            protein_id = metadata.get('protein_id')
            protein_description = metadata.get('protein')

        ratios = compute_codon_ratios(sequence, ratio_columns)

        row = [
            assembly_accession,
            species_taxid,
            protein_id,
            protein_description, 
            length
        ]
        row += ratios.tolist()
        data.append(row)
        
    columns = [
        'assembly_accession',
        'species_taxid',
        'protein_id',
        'description',
        'length',
    ]
    columns += ratio_columns
    cds_df = pd.DataFrame(data, columns=columns)

    distance_vector = compute_distance_vector(cds_df, ratio_columns, target)

    cds_df['distance'] = distance_vector

    cds_df_sorted = cds_df.sort_values('distance')

    n_stds = 1.3
    thres = np.mean(distance_vector) - n_stds * np.std(distance_vector)
    while thres <= 0:
        n_stds -= 0.1
        thres = np.mean(distance_vector) - n_stds * np.std(distance_vector)

    cds_df_sorted['threshold'] = thres
    cds_df_sorted['below_threshold'] = cds_df['distance'] <= thres

    return cds_df_sorted.reset_index(drop=True)


def compute_distance_vector(df, ratio_columns, target):
    X = df[ratio_columns].values
    return compute_codon_bias_distance(X, target)

        
def compute_codon_ratios(sequence, ratio_columns):
    codon_count = collections.defaultdict(int)
    assert len(sequence) % 3 == 0
    for pos in range(0, len(sequence), 3):
        codon = sequence[pos:pos+3]
        if codon not in CODON_REDUNDANCY:
            # Ignore ambiguous letters (e.g. N, etc)
            continue

        codon_count[codon] += 1
        
    codon_ratios = np.zeros((62,))
    
    ordered_codons = [c.replace('_ratio', '') for c in ratio_columns]
    
    processed_codons = set()
    for codon in sorted(CODON_REDUNDANCY.keys()):
        if codon in processed_codons:
            continue
        elif len(CODON_REDUNDANCY[codon]) <= 1:
            processed_codons.add(codon)
            continue
        
        synonymous_codons = CODON_REDUNDANCY[codon]
        
        aa_sum = np.sum([codon_count[cod] for cod, _ in synonymous_codons])
        
        for i, (c, _) in enumerate(synonymous_codons):
            processed_codons.add(c)
            
            ix = ordered_codons.index(c)
            
            if aa_sum > 0:
                codon_ratios[ix] = codon_count[c] / aa_sum
            else:
                codon_ratios[ix] = 1 / len(synonymous_codons)
            
    return codon_ratios


if __name__ == '__main__':
    main()
