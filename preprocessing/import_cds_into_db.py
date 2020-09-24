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
    is_valid_cds,
    parse_location, 
    InvalidLocationError,
    parse_chromosome_id,
    parse_protein_information,
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

    cds_path_fmt = os.path.join(sequences_base_folder, '{0}/{0}_cds_from_genomic.fna.gz')

    assembly_accession_query = 'select assembly_accession, species_taxid from assembly_source'
    source_df = pd.read_sql(assembly_accession_query, engine)

    logger.info(f'Importing CDS for {len(source_df):,} strains')

    n_imported, n_seen_cds = 0, 0
    for i, tpl in enumerate(source_df.itertuples()):
        if i == 0 or (i + 1) % 100 == 0:
            r = 100 * n_imported / n_seen_cds if n_seen_cds > 0 else 100
            logger.info(f'Strain {i+1:,} / {len(source_df):,} | {r:.1f}% success rate')

        assembly_accession = tpl.assembly_accession
        species_taxid = tpl.species_taxid

        sequence_records_to_import = []
        cds_fasta_path = cds_path_fmt.format(assembly_accession)

        with gzip.open(cds_fasta_path, mode='rt') as f:
            cds_dict = SeqIO.to_dict(SeqIO.parse(f, "fasta"))

        sorted_sequence_ids = sorted(cds_dict.keys())

        for sequence_id in sorted_sequence_ids:
            sequence_record = cds_dict[sequence_id]

            n_seen_cds += 1
            if is_valid_cds(sequence_record.seq):
                n_imported += 1
                sequence_records_to_import.append(sequence_record)

        import_sequences(engine, assembly_accession, species_taxid, sequence_records_to_import)

    success_rate = 100 * n_imported / n_seen_cds
    logger.info(f'{n_imported:,} sequences imported | {success_rate:.1f}% success rate')
    logger.info('DONE')


def import_sequences(engine, assembly_accession, species_taxid, sequence_records):
    columns = [
        'assembly_accession', 'species_taxid', 'sequence_type', 
        'chromosome_id', 'location_json', 'strand', 'length', 
        'description', 'metadata_json', 'sequence',
    ]

    data = []
    for sequence_record in sequence_records:
        seq_id = sequence_record.id
        sequence = sequence_record.seq
        chromosome_id = parse_chromosome_id(seq_id)
        sequence_type = 'CDS'

        metadata_json = None
        metadata_dict = parse_protein_information(sequence_record)
        if metadata_dict is not None:
            metadata_json = json.dumps(metadata_dict, sort_keys=True)

        try:
            location_list, strand = parse_location(sequence_record)
            location_json = json.dumps(location_list, sort_keys=True)
        except InvalidLocationError as e:
            logger.warning(f'{assembly_accession} | Invalid location information: {e.message}')
            continue

        row = [
            assembly_accession,
            species_taxid,
            sequence_type,
            chromosome_id,
            location_json,
            strand,
            len(sequence),
            sequence_record.description,
            metadata_json,
            str(sequence),
        ]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_sql(
        'sequences',
        engine,
        if_exists='append',
        method='multi',
        index=False,
    )


if __name__ == '__main__':
    main()
