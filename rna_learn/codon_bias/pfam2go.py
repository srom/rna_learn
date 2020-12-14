import argparse
import collections
import os
import json
import logging
import re
import pathlib

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['codon_bias', 'tri_nucleotide_bias'], 
        default='codon_bias',
    )
    args = parser.parse_args()

    db_path = args.db_path
    method = args.method

    if db_path is None:
        db_path = os.path.join(os.getcwd(), 'data/db/seq.db')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    pfam2go_path = os.path.join(os.getcwd(), 'data/domains/Pfam2go.txt')
    domain_to_go = parse_pfam_to_go_file(pfam2go_path)

    assembly_query = """
    select assembly_accession, species_taxid from assembly_source
    """
    assembly_df = pd.read_sql(assembly_query, engine)
    assembly_accessions = assembly_df['assembly_accession'].values

    logger.info(f'Converting Pfam labels to GO labels for {len(assembly_accessions):,} assemblies')

    n_matches = 0
    total = 0
    for i, assembly in enumerate(assembly_accessions):
        if i == 0 or (i+1) % 10 == 0:
            logger.info(f'{i+1:,} / {len(assembly_accessions):,}')

        output_data = {
            'assembly_accession': [],
            'protein_id': [],
            'record_type': [],
            'pfam_query': [],
            'pfam_accession': [],
            'protein_label': [],
            'below_threshold': [],
        }
        pfam_domains_path = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/pfam/{assembly}_protein_domains.csv',
        )
        if not os.path.isfile(pfam_domains_path):
            continue

        df = pd.read_csv(pfam_domains_path, index_col='protein_id')

        for protein_id in df.index:
            seen_go_id = set()
            rows = df.loc[[protein_id]]

            for tpl in rows.itertuples():
                total += 1
                below_threshold = tpl.below_threshold
                pfam_query = tpl.pfam_query

                go_records = select_relevant_go_records(domain_to_go, pfam_query)

                if go_records is None:
                    continue
                else:
                    n_matches += 1

                for go_id, go_label in go_records:
                    if go_id in seen_go_id:
                        continue

                    output_data['assembly_accession'].append(assembly)
                    output_data['protein_id'].append(protein_id)
                    output_data['record_type'].append('go')
                    output_data['pfam_query'].append(go_id)
                    output_data['pfam_accession'].append(go_id)
                    output_data['protein_label'].append(go_label)
                    output_data['below_threshold'].append(below_threshold)

        output_path = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/go/{assembly}_protein_domains.csv',
        )
        output_df = pd.DataFrame.from_dict(output_data)
        output_df.to_csv(output_path, index=False)

    perc = 100 * n_matches / total
    logger.info(f'Match: {n_matches:,} / {total:,} ({perc:.2f}%)')
    logger.info('DONE')


def select_relevant_go_records(domain_to_go, pfam_query):
    if pfam_query in domain_to_go:
        return domain_to_go[pfam_query]

    # Try fuzzy matching by trying a different end number
    end_number_re = r'^(.+)_[0-9]+$'
    m = re.match(end_number_re, pfam_query)
    if m is not None:
        pfam_query_base = m[1]
    else:
        pfam_query_base = pfam_query

    if pfam_query_base in domain_to_go:
        return domain_to_go[pfam_query_base]

    for i in range(1, 10):
        query = f'{pfam_query_base}_{i}'
        if query in domain_to_go:
            return domain_to_go[query]

    return None


def parse_pfam_to_go_file(path):
    line_re = r'^Pfam:([^\s]+) ([^>]+) > GO:([^;]+) ; GO:([0-9]+)$'
    domain_to_go = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('!'):
                continue
                
            m = re.match(line_re, line)
            if m:
                pfam_id = m[1].strip()
                query =  m[2].strip()
                go_label = m[3].strip()
                go_id = m[4].strip()
                
                domain_to_go[query].append((go_id, go_label))
                
    return dict(domain_to_go)


if __name__ == '__main__':
    main()
