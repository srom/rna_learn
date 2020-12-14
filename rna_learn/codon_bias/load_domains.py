import argparse
import collections
import os
import json
import logging
import re
import pathlib

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=int(1e4))
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['codon_bias', 'tri_nucleotide_bias'], 
        default='codon_bias',
    )
    args = parser.parse_args()

    db_path = args.db_path
    input_folder = args.input_folder
    output_folder = args.output_folder
    batch_size = args.batch_size
    method = args.method

    if db_path is None:
        db_path = os.path.join(os.getcwd(), 'data/db/seq.db')

    if input_folder is None:
        input_folder = os.path.join(os.getcwd(), 'data/Large_EBMC_Bact_DB')

    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), f'data/domains/{method}')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    known_assemblies = set(pd.read_sql(
        'select assembly_accession from assembly_source', 
        engine,
    )['assembly_accession'].values)

    process_batch = process_batch_fn(engine, known_assemblies, output_folder, method)

    for batch_df in process_files(input_folder, known_assemblies, batch_size):
        process_batch(batch_df)

    logger.info('DONE')


def process_batch_fn(engine, known_assemblies, output_folder, method):
    pfam_folder = os.path.join(output_folder, 'pfam')
    tigr_folder = os.path.join(output_folder, 'tigr')

    seen_tigr_assemblies = set()
    seen_pfam_assemblies = set()
    assembly_to_protein_ids = {}
    assembly_to_top_protein_ids = {}

    def fn(batch_df):
        assemblies = set(batch_df.index.tolist())
        matching_assemblies = sorted(assemblies & known_assemblies)
        if len(matching_assemblies) == 0:
            return

        record_type = batch_df.iloc[0]['record_type']

        for assembly_accession in matching_assemblies:
            if assembly_accession not in assembly_to_protein_ids:
                all_protein_ids, top_protein_ids = load_protein_ids(assembly_accession, method)

                assembly_to_protein_ids[assembly_accession] = all_protein_ids
                assembly_to_top_protein_ids[assembly_accession] = set(top_protein_ids)

            all_protein_ids = assembly_to_protein_ids[assembly_accession]
            top_protein_ids = assembly_to_top_protein_ids[assembly_accession]

            df = batch_df.loc[[assembly_accession]].copy()

            if len(df) == 0:
                continue

            df['below_threshold'] = df['protein_id'].apply(lambda p: p in top_protein_ids)

            if record_type == 'pfam':
                pfam_path = os.path.join(pfam_folder, f'{assembly_accession}_protein_domains.csv')
                header = assembly_accession not in seen_pfam_assemblies
                seen_pfam_assemblies.add(assembly_accession)
                df.to_csv(pfam_path, index=True, mode='a', header=header)

            elif record_type == 'tigr':
                tigr_path = os.path.join(tigr_folder, f'{assembly_accession}_protein_domains.csv')
                header = assembly_accession not in seen_tigr_assemblies
                seen_tigr_assemblies.add(assembly_accession)
                df.to_csv(tigr_path, index=True, mode='a', header=header)

    return fn


def process_files(folder, known_assemblies, batch_size, skiplines=4, n_cols=19):
    files = []
    for p in pathlib.Path(folder).iterdir():
        if p.is_file() and ('pfam' in p.name.lower() or 'tigr' in p.name.lower()):
            files.append(p)

    logger.info(f'Processing {len(files)} files')

    p = r'\s+'.join([r'([^\s]+)' for _ in range(n_cols)])
    pattern = f'^{p}$'
    
    batch = []
    n_records = 0
    for i, path in enumerate(sorted(files, key=lambda p: p.name)):
        logger.info(f'Processing file {i+1} / {len(files)}: {path.name}')

        if 'pfam' in path.name.lower():
            record_type = 'pfam'
        else:
            record_type = 'tigr'

        line_nb = 0
        with path.open() as f:
            for line in f:
                line_nb += 1
                if line_nb < skiplines:
                    continue

                m = re.match(pattern, line)
                if m is None:
                    continue

                n_records += 1

                if n_records % int(1e5) == 0:
                    logger.info(f'{n_records:,} records processed')

                row = [m[i+1] for i in range(n_cols)]
                
                first_el = row[0]
                
                a, genome_accession = tuple(first_el.split('$'))

                if genome_accession not in known_assemblies:
                    continue

                _, protein_id = tuple(a.split('@'))
                protein_label = row[-1] if row[-1] != '-' else None
                
                pfam_query = row[2]
                pfam_accession = row[3]
                
                data_row = [
                    genome_accession,
                    protein_id,
                    record_type,
                    pfam_query,
                    pfam_accession,
                    protein_label,
                ]
                batch.append(data_row)
                
                if len(batch) >= batch_size:
                    batch_df = prepare_batch(batch)
                    yield batch_df
                    batch = []

        if len(batch) > 0:
            yield prepare_batch(batch)
            batch = []

    if len(batch) > 0:
        yield prepare_batch(batch)

    logger.info(f'Total number of records: {n_records:,}')

    return


def load_protein_ids(assembly_accession, method):
    if method == 'codon_bias':
        return load_codon_bias_protein_ids(assembly_accession)
    else:
        return load_tri_nucleotide_bias_protein_ids(assembly_accession)


def load_codon_bias_protein_ids(assembly_accession):
    path_all = os.path.join(os.getcwd(), f'data/cds_codon_bias/all/{assembly_accession}_codon_bias.csv')
    path_below = os.path.join(os.getcwd(), f'data/cds_codon_bias/below_threshold/{assembly_accession}_codon_bias.csv')
    all_protein_ids = [
        p_id.strip() for p_id in pd.read_csv(path_all)['protein_id'].unique()
        if not pd.isnull(p_id)
    ]
    top_protein_ids = [
        p_id.strip() for p_id in pd.read_csv(path_below)['protein_id'].unique()
        if not pd.isnull(p_id)
    ]
    return all_protein_ids, top_protein_ids


def load_tri_nucleotide_bias_protein_ids(assembly_accession):
    path = os.path.join(
        os.getcwd(), 
        f'data/cds_tri_nucleotide/{assembly_accession}_bias.csv',
    )
    df = pd.read_csv(path)

    all_protein_ids = [
        p_id.strip()
        for p_id in df['protein_id'].unique()
        if not pd.isnull(p_id)
    ]
    top_protein_ids = [
        p_id.strip()
        for p_id in df[df['below_threshold']]['protein_id'].unique()
        if not pd.isnull(p_id)
    ]
    return all_protein_ids, top_protein_ids


def prepare_batch(batch):
    return pd.DataFrame(batch, columns=[
        'assembly_accession',
        'protein_id',
        'record_type',
        'pfam_query',
        'pfam_accession',
        'protein_label',
    ]).set_index('assembly_accession')


if __name__ == '__main__':
    main()
