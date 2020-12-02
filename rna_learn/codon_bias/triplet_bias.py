import collections
import itertools
import os
import json
import logging

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from rna_learn.alphabet import ALPHABET_DNA, CODON_REDUNDANCY


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")
    
    db_path = os.path.join(os.getcwd(), 'data/db/seq.db')
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    possible_codons = sorted(CODON_REDUNDANCY.keys())
    possible_doublets = [f'{a}{b}' for a, b in sorted(itertools.permutations(ALPHABET_DNA, 2))]

    test_set_query = """
    select assembly_accession, species_taxid from assembly_source
    """
    response_df = pd.read_sql(test_set_query, engine)
    assembly_accession_ids = response_df['assembly_accession'].values
    species_taxids = response_df['species_taxid'].values

    nucleotide_df, doublet_df, triplet_df = compute_frequency(
        engine, 
        assembly_accession_ids, 
        species_taxids, 
        possible_doublets,
        possible_codons,
    )

    compute_ratios(nucleotide_df, ALPHABET_DNA)
    compute_ratios(doublet_df, possible_doublets)
    compute_ratios(triplet_df, possible_codons)

    nucleotide_df.to_csv(
        os.path.join(os.getcwd(), 'data/nucleotide_bias.csv'), 
        index=False,
    )
    doublet_df.to_csv(
        os.path.join(os.getcwd(), 'data/doublet_bias.csv'), 
        index=False,
    )
    triplet_df.to_csv(
        os.path.join(os.getcwd(), 'data/triplet_bias.csv'), 
        index=False,
    )


def compute_frequency(engine, assembly_accession_ids, species_taxids, possible_doublets, possible_codons):
    columns = [
        'assembly_accession',
        'species_taxid',
        'in_test_set',
    ]
    columns_nucleotide = columns + ALPHABET_DNA
    columns_doublet = columns + possible_doublets
    columns_triplet = columns + possible_codons

    test_set_species_taxids_query = """
    select species_taxid from train_test_split
    where in_test_set = 1
    """
    test_set_species_taxids = set(pd.read_sql(
        test_set_species_taxids_query, 
        engine,
    )['species_taxid'].values)

    logger.info(f'Processing {len(assembly_accession_ids):,} assemblies')

    data_nucleotide = []
    data_doublet = []
    data_triplet = []
    for i, assembly_accession in enumerate(assembly_accession_ids):
        if (i + 1) % 100 == 0:
            logger.info(f'Assembly {i + 1:,} / {len(assembly_accession_ids):,}')

        species_taxid = species_taxids[i]
        in_test_set = species_taxid in test_set_species_taxids
        
        query = """
        select sequence from sequences
        where assembly_accession = ?
        """
        sequences = pd.read_sql(
            query, 
            engine, 
            params=(assembly_accession,),
        )['sequence'].values
        
        if len(sequences) == 0:
            logger.warning(f'No sequences found for assembly ID {assembly_accession}')
            continue
        
        nucleotide_count = collections.defaultdict(int)
        doublet_count = collections.defaultdict(int)
        triplet_count = collections.defaultdict(int)
        for seq in sequences:
            update_nucleotide_count(seq, nucleotide_count, ALPHABET_DNA)
            update_doublet_count(seq, doublet_count, possible_doublets)
            update_triplet_count(seq, triplet_count, possible_codons)

        row = [
            assembly_accession,
            species_taxid,
            in_test_set,
        ]

        row_nucleotide = row + [nucleotide_count[n] for n in ALPHABET_DNA]
        data_nucleotide.append(row_nucleotide)

        row_doublet = row + [doublet_count[d] for d in possible_doublets]
        data_doublet.append(row_doublet)

        row_triplet = row + [triplet_count[codon] for codon in possible_codons]
        data_triplet.append(row_triplet)
        
    return (
        pd.DataFrame(data_nucleotide, columns=columns_nucleotide),
        pd.DataFrame(data_doublet, columns=columns_doublet),
        pd.DataFrame(data_triplet, columns=columns_triplet),
    )


def update_nucleotide_count(sequence, nucleotide_count, possible_nucleotides):
    possible_nucleotides_set = set(possible_nucleotides)
    for pos in range(len(sequence)):
        nucleotide = sequence[pos]
        if nucleotide not in possible_nucleotides_set:
            continue
        
        nucleotide_count[nucleotide] += 1


def update_doublet_count(seq, doublet_count, possible_doublets):
    # Consider the 2 possible reading frames
    possible_doublets_set = set(possible_doublets)
    for frame in [0, 2]:
        if len(seq) < 2 + frame:
            continue

        reminder = len(seq[frame:]) % 2
        if reminder > 0:
            sequence = seq[frame:-reminder]
        else:
            sequence = seq[frame:]

        assert len(sequence) % 2 == 0

        for pos in range(0, len(sequence), 2):
            doublet = sequence[pos:pos+2]
            if doublet not in possible_doublets_set:
                continue
            
            doublet_count[doublet] += 1


def update_triplet_count(seq, triplet_count, possible_codons):
    # Consider the 3 possible reading frames
    possible_codons_set = set(possible_codons)
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


def compute_ratios(df, possible_values):
    for v in possible_values:
        count_sum = sum((df[val] for val in possible_values))
        df[f'{v}_ratio'] = df[v] / count_sum


if __name__ == '__main__':
    main()
