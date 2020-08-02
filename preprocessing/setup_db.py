import argparse
import os
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, Index


DB_PATH = 'data/condensed_traits/db/seq.db'
NCBI_SPECIES_PATH = 'data/condensed_traits/ncbi_species_final.csv'
SPECIES_TRAITS_PATH = 'data/condensed_traits/condensed_species_NCBI_with_ogt.csv'

logger = logging.getLogger(__name__)


def main():
    """
    Create and populate a few core tables in the sequence DB.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--ncbi_species_path', type=str, default=None)
    parser.add_argument('--species_traits_path', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path
    ncbi_species_path = args.ncbi_species_path
    species_traits_path = args.species_traits_path

    if db_path is None:
        db_path = os.path.join(os.getcwd(), DB_PATH)
    if ncbi_species_path is None:
        ncbi_species_path = os.path.join(os.getcwd(), NCBI_SPECIES_PATH)
    if species_traits_path is None:
        species_traits_path = os.path.join(os.getcwd(), SPECIES_TRAITS_PATH)

    logger.info(f'Database path: {db_path}')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    create_ncbi_species_metadata_table(engine, ncbi_species_path)
    create_species_traits_table(engine, species_traits_path)
    create_sequences_table(engine)


def create_ncbi_species_metadata_table(engine, ncbi_species_path):
    table_name = 'ncbi_species_metadata'

    if engine.dialect.has_table(engine, table_name):
        logger.info(f'Table {table_name} already exists, skipping')
        return
    else:
        logger.info(f'Creating table {table_name}')

    metadata = MetaData()
    ncbi_species_metadata = Table(
        table_name, 
        metadata,
        Column('species_taxid', Integer, primary_key=True),
        Column('taxid', Integer, nullable=False),
        Column('assembly_accession', String, nullable=False),
        Column('bioproject', String, nullable=False),
        Column('refseq_category', String, nullable=False),
        Column('organism_name', String, nullable=False),
        Column('infraspecific_name', String, nullable=True),
        Column('version_status', String, nullable=False),
        Column('assembly_level', String, nullable=False),
        Column('release_type', String, nullable=False),
        Column('genome_rep', String, nullable=False),
        Column('seq_rel_date', String, nullable=False),
        Column('asm_name', String, nullable=False),
        Column('submitter', String, nullable=True),
        Column('gbrs_paired_asm', String, nullable=False),
        Column('paired_asm_comp', String, nullable=False),
        Column('relation_to_type_material', String, nullable=True),
        Column('download_url_base', String, nullable=False),
    )
    metadata.create_all(engine)

    columns = [
        'species_taxid', 'taxid', 'assembly_accession', 'bioproject',
        'refseq_category', 'organism_name', 'infraspecific_name',
        'version_status', 'assembly_level', 'release_type', 'genome_rep',
        'seq_rel_date', 'asm_name', 'submitter', 'gbrs_paired_asm',
        'paired_asm_comp', 'relation_to_type_material', 'download_url_base'
    ]

    species_metadata = pd.read_csv(ncbi_species_path)
    species_metadata['seq_rel_date'] = species_metadata['seq_rel_date_processed']
    
    logger.info(f'Saving data to {table_name}')
    species_metadata[columns].to_sql(
        table_name,
        engine,
        if_exists='append',
        method='multi',
        index=False,
    )


def create_species_traits_table(engine, species_traits_path):
    table_name = 'species_traits'

    if engine.dialect.has_table(engine, table_name):
        logger.info(f'Table {table_name} already exists, skipping')
        return
    else:
        logger.info(f'Creating table {table_name}')

    species_traits_all = pd.read_csv(species_traits_path)
    species_traits_all['species_taxid'] = species_traits_all['species_tax_id']

    species_taxid_short_list = pd.read_sql(
        'select species_taxid from ncbi_species_metadata', 
        engine,
    )['species_taxid'].values

    species_traits = species_traits_all[
        species_traits_all['species_taxid'].isin(species_taxid_short_list)
    ].reset_index(drop=True)

    columns = ['species_taxid'] + species_traits.columns.tolist()[1:-1]

    def set_column_type(el):
        if isinstance(el, str):
            return String
        elif isinstance(el, np.int64):
            return Integer
        elif isinstance(el, float):
            return Float
        else:
            raise ValueError(f'Unknown type {type(el)}')

    metadata = MetaData()
    ncbi_species_metadata = Table(
        table_name, 
        metadata,
        Column('species_taxid', Integer, primary_key=True),
        *[
            Column(
                col_name, 
                set_column_type(species_traits[col_name].iloc[0]), 
                nullable=species_traits[col_name].isnull().any(),
            )
            for col_name in columns[1:]
        ],
    )
    metadata.create_all(engine)

    logger.info(f'Saving data to {table_name}')
    species_traits[columns].to_sql(
        table_name,
        engine,
        if_exists='append',
        method='multi',
        index=False,
    )


def create_sequences_table(engine):
    table_name = 'sequences'

    if engine.dialect.has_table(engine, table_name):
        logger.info(f'Table {table_name} already exists, skipping')
        return
    else:
        logger.info(f'Creating table {table_name}')

    metadata = MetaData()
    sequences_table = Table(
        table_name, 
        metadata,
        Column('sequence_id', String, nullable=False),
        Column('species_taxid', Integer, nullable=False),
        Column('sequence_type', String, nullable=False),
        Column('chromosome_id', String, nullable=False),
        Column('location_json', String, nullable=False),
        Column('strand', String, nullable=False),
        Column('length', String, nullable=False),
        Column('description', String, nullable=True),
        Column('metadata_json', String, nullable=True),
        Column('sequence', String, nullable=False),
        Index('idx_seq_species_taxid', 'species_taxid'),
    )
    sequences_table.create(engine)


if __name__ == '__main__':
    main()
