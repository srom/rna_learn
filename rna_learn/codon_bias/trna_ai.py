'''
tRNA Adaptation Index
'''
import collections
import os
import json
import logging

import pandas as pd
import numpy as np
import scipy.stats.mstats
from sqlalchemy import create_engine

from ..alphabet import CODON_REDUNDANCY


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")
    
    db_path = os.path.join(os.getcwd(), 'data/db/seq.db')
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    possible_codons = sorted(CODON_REDUNDANCY.keys())

    test_set_query = """
    select assembly_accession, species_taxid from assembly_source
    """
    response_df = pd.read_sql(test_set_query, engine)
    assembly_accession_ids = response_df['assembly_accession'].values
    species_taxids = response_df['species_taxid'].values

    logger.info('Computing adaptation index for all strains')

    for i in range(len(assembly_accession_ids)):
        if (i + 1) % 100 == 0:
            logger.info(f'Assembly {i + 1:,} / {len(assembly_accession_ids):,}')

        assembly_accession = assembly_accession_ids[i]
        species_taxid = species_taxids[i]

        trna_count = compute_trna_count(engine, assembly_accession)

        weights = compute_trna_ai_weights(trna_count)

        output_df = compute_genes_adaptation_index(
            engine, 
            assembly_accession, 
            species_taxid, 
            weights,
        )

        output_df['adaptation_index'] /= output_df['adaptation_index'].max()

        output_df.sort_values('adaptation_index', ascending=False).to_csv(
            os.path.join(os.getcwd(), f'data/trn_adaptation_index/{assembly_accession}_tai.csv'), 
            index=False,
        )


def compute_genes_adaptation_index(engine, assembly_accession, species_taxid, weights):
    columns = [
        'assembly_accession',
        'species_taxid',
        'row_id',
        'protein_id',
        'protein',
        'gene',
        'adaptation_index',
    ]

    query = """
    select rowid, * from sequences
    where assembly_accession = ? and sequence_type = 'CDS'
    """
    coding_sequences = pd.read_sql(query, engine, params=(assembly_accession,))

    data = []
    for tpl in coding_sequences.itertuples():
        sequence = tpl.sequence
        metadata = (
            json.loads(tpl.metadata_json) 
            if tpl.metadata_json is not None 
            else {}
        )
        adaptation_index = compute_adaptation_index(sequence, weights)

        data.append([
            assembly_accession,
            species_taxid,
            tpl.rowid,
            metadata.get('protein_id'),
            metadata.get('protein'),
            metadata.get('gene'),
            adaptation_index,
        ])

    return pd.DataFrame(data, columns=columns)


def compute_trna_count(engine, assembly_accession):
    trna_codons = sorted(set(CODON_REDUNDANCY.keys()) - {'ATG', 'TGG', 'TGA', 'TAG', 'TAA'})
    trna_df = load_trnas(engine, assembly_accession)
    trna_count = collections.defaultdict(int)
    for tpl in trna_df.itertuples():
        if pd.isnull(tpl.codon):
            continue
        else:
            trna_count[tpl.codon] += 1

    for codon in trna_codons:
        if codon not in trna_count:
            trna_count[codon] = 0
        
    return dict(trna_count)


def load_trnas(engine, assembly_accession):
    trna_query = """
    select * from sequences where sequence_type = 'tRNA' and assembly_accession = ?
    """
    trna_df = pd.read_sql(trna_query, engine, params=(assembly_accession,))
    metadata = [json.loads(v) if v is not None else {} for v in trna_df['metadata_json'].values]
    trna_df['codon'] = [v.get('codon') for v in metadata]
    trna_df['anticodon'] = [v.get('anticodon') for v in metadata]
    trna_df['amino_acid'] = [v.get('amino_acid') for v in metadata]
    return trna_df


def compute_trna_ai_weights(trna_count, is_prokaryote=True):
    p = {'T': 0.59, 'C': 0.72, 'A': 0.0001, 'G': 0.32}
    isoleucine_p = 0.11

    # Consider all codons but the ones coding for Met, Trp or STP 
    codons = sorted(
        set(CODON_REDUNDANCY.keys()) - {'ATG', 'TGG', 'TGA', 'TAG', 'TAA'}
    )

    weights = {}
    for codon in codons:
        if codon not in CODON_REDUNDANCY:
            # Ignore ambiguous letters (e.g. N, etc)
            continue

        base = codon[:2]
        if codon[2] == 'T':                  # INN -> NNT, NNC, NNA
            weights[codon] = trna_count[codon] + p['T'] * trna_count[base + 'C']
        elif codon[2] == 'C':                # GNN -> NNT, NNC
            weights[codon] = trna_count[codon] + p['C'] * trna_count[base + 'T']
        elif codon[2] == 'A':                # TNN -> NNA, NNG
            weights[codon] = trna_count[codon] + p['A'] * trna_count[base + 'T']
        elif codon[2] == 'G':                # CNN -> NNG
            weights[codon] = trna_count[codon] + p['G'] * trna_count[base + 'A']

    # Modify isoleucine ATA codon for prokaryotes
    if is_prokaryote:
        weights['ATA'] = isoleucine_p

    keys = sorted(weights.keys())
    weights = np.array([weights[k] for k in keys])
    weights = weights / weights.max()

    geometric_mean = scipy.stats.mstats.gmean([v for v in weights if v != 0])

    for i in range(len(weights)):
        if weights[i] == 0:
            weights[i] = geometric_mean

    return {k: weights[i] for i, k in enumerate(keys)}


def compute_adaptation_index(sequence, weights):
    assert len(sequence) % 3 == 0, 'Sequence length must be divisible by 3'
    adaptation = np.zeros((int(len(sequence) / 3),))

    seq_codonds = list(range(0, len(sequence), 3))
    for i, pos in enumerate(seq_codonds):
        codon = sequence[pos:pos+3]
        if codon not in weights:
            continue

        adaptation[i] = np.log(weights[codon])

    adaptation_index = np.exp(np.sum(adaptation) / len(adaptation))

    if pd.isnull(adaptation_index):
        return 0.
    else:
        return adaptation_index


if __name__ == '__main__':
    main()
