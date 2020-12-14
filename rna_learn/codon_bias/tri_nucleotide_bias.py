import argparse
import collections
import itertools
import os
import json
import logging
import gzip

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from Bio import SeqIO

from rna_learn.alphabet import CODON_REDUNDANCY


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['all', 'cds'], default='all')
    parser.add_argument('--db_path', type=str, default=None)
    args = parser.parse_args()

    method = args.method
    db_path = args.db_path

    if db_path is None:
        db_path = os.path.join(os.getcwd(), 'data/db/seq.db')
    
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    possible_codons = sorted(CODON_REDUNDANCY.keys())

    test_set_query = """
    select assembly_accession, species_taxid from assembly_source
    """
    response_df = pd.read_sql(test_set_query, engine)
    assembly_accession_ids = response_df['assembly_accession'].values
    species_taxids = response_df['species_taxid'].values

    tri_nucleotides_df = compute_frequency(
        engine, 
        assembly_accession_ids, 
        species_taxids, 
        possible_codons,
        method,
    )
    compute_ratios(tri_nucleotides_df, possible_codons)

    tri_nucleotides_df.to_csv(
        os.path.join(os.getcwd(), f'data/tri_nucleotide_bias_{method}.csv'), 
        index=False,
    )


def compute_frequency(engine, assembly_accession_ids, species_taxids, possible_codons, method):
    columns = [
        'assembly_accession',
        'species_taxid',
        'in_test_set',
    ]
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

    data_triplet = []
    for i, assembly_accession in enumerate(assembly_accession_ids):
        if (i + 1) % 100 == 0:
            logger.info(f'Assembly {i + 1:,} / {len(assembly_accession_ids):,}')

        species_taxid = species_taxids[i]
        in_test_set = species_taxid in test_set_species_taxids
        
        sequences = load_sequences(engine, assembly_accession, method)
        
        triplet_count = collections.defaultdict(int)
        for seq in sequences:
            update_triplet_count(seq, triplet_count, possible_codons)

        row = [
            assembly_accession,
            species_taxid,
            in_test_set,
        ]
        row_triplet = row + [triplet_count[codon] for codon in possible_codons]
        data_triplet.append(row_triplet)
        
    return pd.DataFrame(data_triplet, columns=columns_triplet)


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
    count_sum = sum((df[val] for val in possible_values))
    for v in possible_values:
        df[f'{v}_ratio'] = (df[v] / count_sum).round(6)


def load_sequences(engine, assembly_accession, method):
    if method == 'all':
        return load_sequences_from_fasta(assembly_accession)
    else:
        return load_sequences_from_cds(engine, assembly_accession)


def load_sequences_from_cds(engine, assembly_accession):
    query = """
    select sequence from sequences
    where assembly_accession = ? and sequence_type = 'CDS'
    """
    return pd.read_sql(
        query, 
        engine, 
        params=(assembly_accession,),
    )['sequence'].values


def load_sequences_from_fasta(assembly_accession):
    # Load whole genome from FASTA
    fasta_path = os.path.join(
        os.getcwd(), 
        f'data/sequences/{assembly_accession}/{assembly_accession}_genomic.fna.gz',
    )
    with gzip.open(fasta_path, mode='rt') as f:
        genome_dict = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))

    # For each chromosome or plasmid, extract sequence and 
    # its reverse complement as strings.
    sequences = []
    for genome_key in sorted(genome_dict.keys()):
        seq = genome_dict[genome_key].seq
        comp_seq = seq.reverse_complement()
        sequences.append(str(seq))
        sequences.append(str(comp_seq))

    return sequences


if __name__ == '__main__':
    main()
