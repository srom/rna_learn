import argparse
import os
import logging
import re

from Bio import SeqIO
import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine, 
    Table, 
    Column, 
    Integer, 
    Float, 
    String,
    Boolean, 
    MetaData, 
    Index,
)


DB_PATH = 'data/condensed_traits/db/seq.db'
NCBI_SPECIES_PATH = 'data/condensed_traits/ncbi_species_final.csv'
SPECIES_TRAITS_PATH = 'data/condensed_traits/condensed_species_NCBI_with_ogt.csv'
TRNA_REFERENCE_FOLDER = 'data/condensed_traits/tRNADB-CE'
TRAIN_TEST_SPLIT_SEED = 444
TEST_RATIO = 0.2

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
    parser.add_argument('--trna_reference_folder', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path
    ncbi_species_path = args.ncbi_species_path
    species_traits_path = args.species_traits_path
    trna_reference_folder = args.trna_reference_folder

    if db_path is None:
        db_path = os.path.join(os.getcwd(), DB_PATH)
    if ncbi_species_path is None:
        ncbi_species_path = os.path.join(os.getcwd(), NCBI_SPECIES_PATH)
    if species_traits_path is None:
        species_traits_path = os.path.join(os.getcwd(), SPECIES_TRAITS_PATH)
    if trna_reference_folder is None:
        trna_reference_folder = os.path.join(os.getcwd(), TRNA_REFERENCE_FOLDER)

    logger.info(f'Database path: {db_path}')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    create_species_source_table(engine, ncbi_species_path)
    create_species_traits_table(engine, species_traits_path)
    create_trna_reference_table(engine, trna_reference_folder)
    create_sequences_table(engine)
    create_train_test_split_table(
        engine, 
        seed=TRAIN_TEST_SPLIT_SEED,
        test_ratio=TEST_RATIO,
    )


def create_species_source_table(engine, ncbi_species_path):
    table_name = 'species_source'

    if engine.dialect.has_table(engine, table_name):
        logger.info(f'Table {table_name} already exists, skipping')
        return
    else:
        logger.info(f'Creating table {table_name}')

    metadata = MetaData()
    species_source = Table(
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
        'select species_taxid from species_source', 
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
    species_traits_table = Table(
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


def create_trna_reference_table(engine, trna_reference_folder):
    table_name = 'trna_reference'

    if engine.dialect.has_table(engine, table_name):
        logger.info(f'Table {table_name} already exists, skipping')
        return
    else:
        logger.info(f'Creating table {table_name}')

    metadata = MetaData()
    trna_ref_table = Table(
        table_name, 
        metadata,
        Column('sequence_id', String, primary_key=True),
        Column('species', String, nullable=False),
        Column('phylum', String, nullable=False),
        Column('superkingdom', String, nullable=False),
        Column('amino_acid', String, nullable=False),
        Column('anticodon', String, nullable=False),
        Column('coding_sequence', String, nullable=False),
        Column('full_sequence', String, nullable=False),
    )
    trna_ref_table.create(engine)

    all_columns = [
        'sequence_id', 'genome_id', 'phylum', 'species', 
        'start_position','end_position', 
        'amino_acid', 'anticodon', 'first_intron_start_position',
        'first_intron_end_position', 'first_intron_seq',
        'second_intron_start_position', 'second_intron_end_position', 
        'second_intron_seq', 'comment', 'original_database', 'superkingdom', 
        'coding_sequence', 'full_sequence',
    ]
    columns_to_save = [
        'sequence_id', 'species', 'phylum', 'superkingdom', 
        'amino_acid', 'anticodon', 'coding_sequence', 'full_sequence',
    ]

    def populate_trna_ref(records, superkingdom):
        data = []
        for record in records:
            description = record.description
            row = list(description.split('|'))

            coding_seq = ''
            full_seq = str(record.seq)
            for char in full_seq:
                m = re.match('^[ATGC]$', char)
                if m is not None:
                    coding_seq += char
            
            row.append(superkingdom)
            row.append(coding_seq.upper())
            row.append(full_seq.upper())

            data.append(row)

        data_df = pd.DataFrame(
            data, 
            columns=all_columns,
        )
        data_df[columns_to_save].to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            chunksize=100,
        )

    logger.info(f'Saving data to {table_name} (Bacteria)')

    folder = trna_reference_folder
    bac_path = os.path.join(os.getcwd(), folder, 'trna_sequence_cmp_bac_1.fasta')
    with open(bac_path, 'r', encoding='cp1252') as f:
        bacteria_trnas = list(SeqIO.parse(f, 'fasta'))

    populate_trna_ref(bacteria_trnas, 'Bacteria')

    logger.info(f'Saving data to {table_name} (Archaea)')

    arc_path = os.path.join(os.getcwd(), folder, 'trna_sequence_cmp_arc_1.fasta')
    with open(arc_path, 'r', encoding='cp1252') as f:
        archaea_trnas = list(SeqIO.parse(f, 'fasta'))

    populate_trna_ref(archaea_trnas, 'Archaea')


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
        Column('length', Integer, nullable=False),
        Column('description', String, nullable=True),
        Column('metadata_json', String, nullable=True),
        Column('sequence', String, nullable=False),
        Index('idx_seq_species_taxid', 'species_taxid'),
    )
    sequences_table.create(engine)


def create_train_test_split_table(engine, seed, test_ratio):
    table_name = 'train_test_split'

    if engine.dialect.has_table(engine, table_name):
        logger.info(f'Table {table_name} already exists, skipping')
        return
    else:
        logger.info(f'Creating table {table_name}')

    rs = np.random.RandomState(seed)

    species_taxids = species_taxids = pd.read_sql(
        'select species_taxid from species_source', 
        engine
    )['species_taxid'].values

    test_set_size = int(np.ceil(test_ratio * len(species_taxids)))

    test_species_taxids = rs.choice(
        species_taxids, 
        size=test_set_size, 
        replace=False,
    )

    test_species_taxids_set = set(test_species_taxids.tolist())

    data = []
    for species_taxid in species_taxids:
        data.append([
            species_taxid,
            species_taxid in test_species_taxids_set,
        ])

    df = pd.DataFrame(data, columns=['species_taxid', 'in_test_set'])

    metadata = MetaData()
    train_test_split_table = Table(
        table_name, 
        metadata,
        Column('species_taxid', Integer, nullable=False),
        Column('in_test_set', Boolean, nullable=False),
    )
    train_test_split_table.create(engine)

    logger.info(f'Saving data to {table_name}')

    df.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        chunksize=100,
    )


if __name__ == '__main__':
    main()
