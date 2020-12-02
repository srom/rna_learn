import collections
import os
import json
import logging

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from rna_learn.alphabet import CODON_REDUNDANCY


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

    output_df = compute_codon_frequency(engine, assembly_accession_ids, species_taxids, possible_codons)

    output_df.to_csv(
        os.path.join(os.getcwd(), 'data/codon_bias.csv'), 
        index=False,
    )


def compute_codon_frequency(engine, assembly_accession_ids, species_taxids, possible_codons):
    columns = [
        'assembly_accession',
        'species_taxid',
        'in_test_set',
        'n_total_codons',
    ]
    columns += [codon for codon in possible_codons]

    test_set_species_taxids_query = """
    select species_taxid from train_test_split
    where in_test_set = 1
    """
    test_set_species_taxids = set(pd.read_sql(test_set_species_taxids_query, engine)['species_taxid'].values)

    logger.info(f'Processing {len(assembly_accession_ids):,} assemblies')

    data = []
    for i, assembly_accession in enumerate(assembly_accession_ids):
        if (i + 1) % 100 == 0:
            logger.info(f'Sequence {i + 1:,} / {len(assembly_accession_ids):,}')

        species_taxid = species_taxids[i]
        in_test_set = species_taxid in test_set_species_taxids
        
        query = """
        select sequence from sequences
        where assembly_accession = ? and sequence_type = 'CDS'
        """
        coding_sequences = pd.read_sql(query, engine, params=(assembly_accession,))['sequence'].values
        
        if len(coding_sequences) == 0:
            logger.warning(f'No coding sequences found for assembly ID {assembly_accession}')
            continue
        
        n_total_codons = 0
        codon_count = collections.defaultdict(int)
        for sequence in coding_sequences:
            assert len(sequence) % 3 == 0
            for pos in range(0, len(sequence), 3):
                codon = sequence[pos:pos+3]
                if codon not in possible_codons:
                    # Ignore ambiguous letters (e.g. N, etc)
                    continue
                
                n_total_codons += 1
                codon_count[codon] += 1
                
        row = [
            assembly_accession,
            species_taxid,
            in_test_set,
            n_total_codons,   
        ]
        row += [codon_count[codon] for codon in possible_codons]
        data.append(row)
        
    return pd.DataFrame(data, columns=columns)


if __name__ == '__main__':
    main()
