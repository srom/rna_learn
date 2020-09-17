import os
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


# S can be G or C, see https://www.bioinformatics.org/sms/iupac.html
GC_SET = {'G', 'C', 'S'}

# Codons representative of amino acids I, V, Y, W, R, E, L
IVYWREL_codons = {
    'ATT',  # I
    'ATC',  # I
    'ATA',  # I
    'GTT',  # V
    'GTC',  # V
    'GTA',  # V
    'GTG',  # V
    'TAT',  # Y
    'TAC',  # Y
    'TGG',  # W
    'CGT',  # R
    'CGC',  # R
    'CGA',  # R
    'CGG',  # R
    'AGA',  # R
    'AGG',  # R
    'GAA',  # E
    'GAC',  # E
    'TTA',  # L
    'TTG',  # L
    'CTT',  # L
    'CTC',  # L
    'CTA',  # L
    'CTG',  # L
}


def compute_gc_content(nucleotide_sequence):
    """
    GC content is the ratio of the count of characters that are G or C
    over the total sequence length.
    """
    return compute_gc_count(nucleotide_sequence) / len(nucleotide_sequence)


def compute_gc_count(nucleotide_sequence):
    gc_count = 0
    for char in nucleotide_sequence:
        if char in GC_SET:
            gc_count += 1

    return gc_count


def compute_IVYWREL_content(cds_nucleotide_sequence):
    return compute_IVYWREL_count(cds_nucleotide_sequence) / len(cds_nucleotide_sequence)


def compute_IVYWREL_count(cds_nucleotide_sequence):
    assert len(cds_nucleotide_sequence) % 3 == 0

    IVYWREL_count = 0
    for i in range(0, len(cds_nucleotide_sequence), 3):
        codon = cds_nucleotide_sequence[i:i+3]
        if codon in IVYWREL_codons:
            IVYWREL_count += 1
    
    return IVYWREL_count


def compute_gc_and_IVYWREL_content_on_test_set(engine, chunksize=100, logging=True):
    total_rows_query = """
    select count(rowid) as c from sequences
    where species_taxid in (
        select species_taxid from train_test_split
        where in_test_set = 1
    )
    """
    n_sequences = pd.read_sql(total_rows_query, engine).iloc[0]['c']

    logger.info(f'Number of sequences: {n_sequences:,}')

    sequences_query = """
    select species_taxid, sequence, sequence_type from sequences
    where species_taxid in (
        select species_taxid from train_test_split
        where in_test_set = 1
    )
    """
    gc_acc = {}
    IVYWREL_acc = {}

    iterator = pd.read_sql(sequences_query, engine, chunksize=chunksize)

    count = 0
    for sequence_df in iterator:
        if logging and count % 100000 == 0:
            logger.info(f'{count:,} / {n_sequences:,}')

        for tpl in sequence_df.itertuples():
            species_taxid = tpl.species_taxid
            sequence = tpl.sequence
            sequence_type = tpl.sequence_type

            sequence_length = len(sequence)
            gc_count = compute_gc_count(sequence)

            if species_taxid not in gc_acc:
                gc_acc[species_taxid] = [gc_count, sequence_length]
            else:
                tpl = gc_acc[species_taxid]
                tpl[0] += gc_count
                tpl[1] += sequence_length

            if sequence_type == 'CDS':
                IVYWREL_count = compute_IVYWREL_count(sequence)

                if species_taxid not in IVYWREL_acc:
                    IVYWREL_acc[species_taxid] = [IVYWREL_count, sequence_length]
                else:
                    tpl = IVYWREL_acc[species_taxid]
                    tpl[0] += IVYWREL_count
                    tpl[1] += sequence_length

        count += chunksize

    specie_taxids = sorted(gc_acc.keys())
    output_data = []
    for species_taxid in specie_taxids:
        gc_count, gc_length = gc_acc[species_taxid]
        IVYWREL_count, IVYWREL_length = IVYWREL_acc[species_taxid]

        gc_content = gc_count / gc_length
        IVYWREL_content = IVYWREL_count / IVYWREL_length

        output_data.append([species_taxid, gc_content, IVYWREL_content])

    return pd.DataFrame(output_data, columns=['species_taxid', 'gc_content', 'IVYWREL_content'])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    db_path = os.path.join(os.getcwd(), 'data/condensed_traits/db/seq.db')
    output_path = os.path.join(
        os.getcwd(), 'data/condensed_traits/gc_content_IVYWREL_content.csv')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    logger.info('Computing GC content & IVYWREL content')

    output_df = compute_gc_and_IVYWREL_content_on_test_set(engine)

    output_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
