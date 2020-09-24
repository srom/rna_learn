import argparse
import os
import logging
import gzip
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from Bio import SeqIO

from .sequence_utils import (
    get_non_coding_records, 
    get_location_range,
)


DB_PATH = 'data/db/seq.db'
SEQUENCES_BASE_FOLDER = 'data/sequences'

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--sequences_base_folder', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path
    sequences_base_folder = args.sequences_base_folder

    if db_path is None:
        db_path = os.path.join(os.getcwd(), DB_PATH)
    if sequences_base_folder is None:
        sequences_base_folder = os.path.join(os.getcwd(), SEQUENCES_BASE_FOLDER)

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    dna_path_fmt = os.path.join(sequences_base_folder, '{0}/{0}_genomic.fna.gz')

    assembly_accession_query = 'select assembly_accession, species_taxid from assembly_source'
    source_df = pd.read_sql(assembly_accession_query, engine)

    logger.info(f'Importing non-coding for {len(source_df):,} strains')

    for i, tpl in enumerate(source_df.itertuples()):
        if i == 0 or (i + 1) % 100 == 0:
            logger.info(f'Strain {i+1:,} / {len(source_df):,}')

        assembly_accession = tpl.assembly_accession
        species_taxid = tpl.species_taxid

        dna_fasta_path = dna_path_fmt.format(assembly_accession)
        with gzip.open(dna_fasta_path, mode='rt') as f:
            dna_dict = SeqIO.to_dict(SeqIO.parse(f, "fasta"))

        chromosome_ids = sorted(dna_dict.keys())

        query = (
            "select chromosome_id, location_json, strand "
            "from sequences where assembly_accession = ?"
        )
        sequences = pd.read_sql(query, engine, params=(assembly_accession,))

        locations_per_chromosome_id = {c_id: [] for c_id in chromosome_ids}
        for tpl in sequences.itertuples():
            locations_per_chromosome_id[tpl.chromosome_id].append((
                json.loads(tpl.location_json),
                tpl.strand,
            ))

        records_to_import = []
        for chromosome_id in chromosome_ids:
            record = dna_dict[chromosome_id]
            sequence = str(record.seq)
            length = len(sequence)

            locations = locations_per_chromosome_id.get(chromosome_id, [])

            seen_locations = set()
            for location_list, strand in locations:
                for start, end in location_list:
                    seen_locations |= get_location_range(
                        start,
                        end, 
                        strand, 
                        length,
                    )

            full_set = set(range(1, length + 1))
            non_coding_ix = sorted(full_set - seen_locations)

            non_coding_records = get_non_coding_records(
                chromosome_id, 
                sequence, 
                non_coding_ix,
            )
            records_to_import.extend(non_coding_records)

        import_records(engine, assembly_accession, species_taxid, records_to_import)

    logger.info('DONE')


def import_records(engine, assembly_accession, species_taxid, records):
    columns = [
        'assembly_accession', 'species_taxid', 'sequence_type', 'chromosome_id', 
        'location_json', 'strand', 'length', 'description', 
        'metadata_json', 'sequence',
    ]
    sequence_type = 'non_coding'
    strand = '+'
    description = None
    metadata_json = None

    data = []
    for i, (chromosome_id, location_json, sequence) in enumerate(records):
        row = [
            assembly_accession,
            species_taxid,
            sequence_type,
            chromosome_id,
            location_json,
            strand,
            len(sequence),
            description,
            metadata_json,
            sequence,
        ]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_sql(
        'sequences',
        engine,
        if_exists='append',
        index=False,
        chunksize=100,
    )


if __name__ == '__main__':
    main()
