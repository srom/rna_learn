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

# Placeholder value when taking the log10 of 0 values.
# It is a way to avoid having to deal with −∞ values.
# This value is many time smaller than anything that would
# occur given in our dataset.
MIN_LOG_SCORE = -10


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('query_type', type=str, choices=['pfam', 'tigr', 'go'])
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['codon_bias', 'tri_nucleotide_bias'], 
        default='codon_bias',
    )
    args = parser.parse_args()

    query_type = args.query_type
    db_path = args.db_path
    method = args.method

    if db_path is None:
        db_path = os.path.join(os.getcwd(), 'data/db/seq.db')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    logger.info('Loading data')

    assemblies = load_matching_assemblies(engine, query_type, method)

    logger.info('Identify outliers with no protein domain information')
    matching_scores = check_protein_matching(engine, assemblies, query_type, method)
    outlier_threshold_percent = 90
    outlier_assemblies = {
        a for a in matching_scores.keys() if 
        matching_scores[a] < outlier_threshold_percent
    }
    assemblies = sorted(set(assemblies) - outlier_assemblies)

    phylum_to_assemblies = load_phyla(engine, assemblies)

    phyla = sorted(phylum_to_assemblies.keys())

    logger.info(f'Compute scores for {len(assemblies):,} assemblies')

    logger.info('Fetching labels')
    labels = compute_query_to_most_common_label(assemblies, query_type, method)

    results_per_phylum = {}
    seen_domains = set()
    for i, phylum in enumerate(phyla):
        logger.info(f'Processing phylum {i+1} / {len(phyla)}: {phylum}')
        phylum_assemblies = phylum_to_assemblies[phylum]
        phylum_df = compute_protein_domain_scores(
            engine, 
            phylum_assemblies, 
            query_type, 
            method,
        )

        phylum_labels = [labels[k] for k in phylum_df.index]
        phylum_df['label'] = phylum_labels

        output_path_phylum = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/phylum_results/',
            f'{query_type}_{phylum}_domains.csv',
        )
        phylum_df.to_csv(output_path_phylum)
        results_per_phylum[phylum] = phylum_df
        seen_domains |= set(phylum_df.index.tolist())

    logger.info('Computing aggregated results')

    output_df = compute_aggregated_results(
        results_per_phylum, 
        seen_domains,
        query_type, 
        labels,
    )

    output_path = os.path.join(os.getcwd(), f'data/domains/{method}/{query_type}.xlsx')
    output_df.to_excel(output_path)

    logger.info('DONE')


def compute_aggregated_results(results_per_phylum, seen_domains, query_type, labels):
    phyla = sorted(results_per_phylum.keys())
    query_col = f'{query_type}_query'
    output_data = {
        query_col: [],
        'label': [],
        'score': [],
        'score_random': [],
        'metric': [],
        'metric_log10': [],
        'evidence': [],
    }
    for domain in sorted(seen_domains):
        label = labels[domain]
        scores = []
        scores_random = []
        for phylum in phyla:
            res_df = results_per_phylum[phylum]
            if domain in res_df.index:
                scores.append(res_df.loc[domain, 'score'])
                scores_random.append(res_df.loc[domain, 'score_random'])
            else:
                scores.append(0.)
                scores_random.append(0.)

        score = np.mean(np.exp(scores))
        score_random = np.mean(np.exp(scores_random))

        metric = score / score_random
        metric_log10 = np.log10(metric) if metric > 0 else MIN_LOG_SCORE

        if metric_log10 <= 0:
            evidence = 'Negative'
        elif metric_log10 <= 0.5:
            evidence = 'Weak'
        elif 0.5 < metric_log10 <= 1:
            evidence = 'Substantial'
        elif 1 < metric_log10 <= 2:
            evidence = 'Strong'
        else:
            evidence = 'Decisive'
        
        output_data[query_col].append(domain)
        output_data['label'].append(label)
        output_data['score'].append(np.round(score, 6))
        output_data['score_random'].append(np.round(score_random, 6))
        output_data['metric'].append(np.round(metric, 6))
        output_data['metric_log10'].append(np.round(metric_log10, 6))
        output_data['evidence'].append(evidence)

    output_df = pd.DataFrame.from_dict(output_data).set_index(query_col)

    return output_df[
        output_df['metric_log10'] > 0
    ].sort_values(
        'metric_log10', 
        ascending=False,
    )


def compute_protein_domain_scores(engine, assemblies, query_type, method):
    n_assemblies = len(assemblies)
    
    domain_to_score = collections.defaultdict(int)
    domain_to_score_random = collections.defaultdict(int)

    domain_count = collections.defaultdict(int)
    domain_count_top = collections.defaultdict(int)

    for i, assembly in enumerate(assemblies):
        protein_domains_path = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/{query_type}/{assembly}_protein_domains.csv',
        )
        protein_domains = pd.read_csv(protein_domains_path)
        
        scores_df = compute_score(protein_domains, random=False)
        scores_df_random = compute_score(protein_domains, random=True)
        
        for pfam_query in scores_df.index:
            domain_to_score[pfam_query] += scores_df.loc[pfam_query, 'assembly_score']
            domain_to_score_random[pfam_query] += (
                scores_df_random.loc[pfam_query, 'assembly_score']
            )
            domain_count[pfam_query] += scores_df.loc[pfam_query, 'count_all']
            if scores_df.loc[pfam_query, 'count_below'] > 0:
                domain_count_top[pfam_query] += scores_df.loc[pfam_query, 'count_below']
        
    query_key = f'{query_type}_query'
    sorted_queries = sorted(domain_to_score.keys())
    data = {
        query_key: sorted_queries,
        'assembly_score_sum': [domain_to_score[k] for k in sorted_queries],
        'assembly_count': [domain_count[k] for k in sorted_queries],
        'assembly_count_top': [domain_count_top[k] for k in sorted_queries],
        'score': [domain_to_score[k] / n_assemblies for k in sorted_queries],
        'score_random': [domain_to_score_random[k] / n_assemblies for k in sorted_queries],
    }
    output_df = pd.DataFrame.from_dict(data).set_index(query_key)
    
    return output_df.sort_values(['score', 'assembly_count'], ascending=False)


def compute_score(protein_domains, random=False):
    if not random:
        return compute_actual_score(protein_domains)
    else:
        return compute_random_score(protein_domains)


def compute_actual_score(protein_domains):
    all_counts = protein_domains[
        ['pfam_query', 'pfam_accession']
    ].groupby('pfam_query').count()
    all_counts.columns = ['count_all']
    
    below_threshold_counts = protein_domains[
        protein_domains['below_threshold']
    ][
        ['pfam_query', 'pfam_accession']
    ].groupby('pfam_query').count()
    below_threshold_counts.columns = ['count_below']
    
    counts = pd.merge(
        all_counts,
        below_threshold_counts,
        how='left',
        on='pfam_query',
    )
    counts['count_below'] = counts['count_below'].fillna(0).astype(int)
    
    counts['frequency_weight'] = counts['count_below'] / counts['count_all']
    counts['assembly_score'] = counts['frequency_weight'].apply(
        lambda s: np.log10(s) if s > 0 else MIN_LOG_SCORE
    )
    return counts


def compute_random_score(protein_domains, iterations=100, seed=444):
    rs = np.random.RandomState(seed)
    df = protein_domains.copy()

    below_threshold_filter = df['below_threshold']
    n_below = len(df[below_threshold_filter])
    n_above = len(df[~below_threshold_filter])
    probabilities = [
        n_above / (n_above + n_below),
        n_below / (n_above + n_below),
    ]

    all_counts = df[
        ['pfam_query', 'pfam_accession']
    ].groupby('pfam_query').count()
    all_counts.columns = ['count_all']

    query_to_scores = collections.defaultdict(list)
    for _ in range(iterations):
        df['below_threshold'] = rs.choice(
            [False, True],
            size=len(df),
            replace=True,
            p=probabilities,
        )
        
        below_threshold_counts = df[
            df['below_threshold']
        ][
            ['pfam_query', 'pfam_accession']
        ].groupby('pfam_query').count()
        below_threshold_counts.columns = ['count_below']
        
        counts = pd.merge(
            all_counts,
            below_threshold_counts,
            how='left',
            on='pfam_query',
        )
        counts['count_below'] = counts['count_below'].fillna(0).astype(int)
        
        counts['frequency_weight'] = counts['count_below'] / counts['count_all']
        counts['assembly_score'] = counts['frequency_weight'].apply(
            lambda s: np.log10(s) if s > 0 else MIN_LOG_SCORE
        )

        for query in counts.index:
            query_to_scores[query].append(
                counts.loc[query, 'assembly_score']
            )

    output_data = {
        'pfam_query': [],
        'assembly_score': [],
    }
    for query in sorted(query_to_scores.keys()):
        output_data['pfam_query'].append(query)
        output_data['assembly_score'].append(np.mean(query_to_scores[query]))

    return pd.DataFrame.from_dict(output_data).set_index('pfam_query')


def load_matching_assemblies(engine, query_type, method):
    q = """
    select assembly_accession from assembly_source
    """
    assembly_accessions = pd.read_sql(q, engine)['assembly_accession'].values
    
    assemblies = []
    for assembly in assembly_accessions:
        protein_domains_path = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/{query_type}/{assembly}_protein_domains.csv',
        )
        if os.path.isfile(protein_domains_path):
            assemblies.append(assembly)
            
    return assemblies


def load_phyla(engine, assemblies):
    q = """
    select a.assembly_accession, s.phylum from assembly_source as a
    left join species_traits as s on s.species_taxid = a.species_taxid
    """
    df = pd.read_sql(q, engine, index_col='assembly_accession')
    phyla = df.loc[assemblies]['phylum'].values

    phylum_to_assemblies = collections.defaultdict(list)
    for i in range(len(assemblies)):
        phylum = phyla[i]
        if not pd.isnull(phylum):
            phylum_to_assemblies[phylum].append(assemblies[i])

    return dict(phylum_to_assemblies)


def check_protein_matching(engine, assemblies, query_type, method):
    matching_scores = {}
    for i, assembly in enumerate(assemblies):
        protein_domains_path = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/{query_type}/{assembly}_protein_domains.csv',
        )
        if not os.path.isfile(protein_domains_path):
            continue
        
        protein_domains = pd.read_csv(protein_domains_path)
        
        protein_query = """
        select metadata_json from sequences 
        where sequence_type = 'CDS' and assembly_accession = ?
        """
        cds_metadata_df = pd.read_sql(protein_query, engine, params=(assembly,))
        metadata = [
            json.loads(v) 
            for v in cds_metadata_df['metadata_json'].values 
            if not pd.isnull(v)
        ]
        
        cds_protein_ids = {
            m['protein_id'].strip() for m in metadata
            if m.get('protein_id') is not None
        }
        query_protein_ids = set([
            p.strip() 
            for p in protein_domains['protein_id'].values 
            if not pd.isnull(p)
        ])
        
        matching_score = 100 * (
            len(cds_protein_ids & query_protein_ids) / len(query_protein_ids)
        )
        matching_scores[assembly] = matching_score
        
    return matching_scores


def compute_query_to_most_common_label(assemblies, query_type, method):
    query_to_protein_labels = {}
    for i, assembly in enumerate(assemblies):
            
        protein_domains_path = os.path.join(
            os.getcwd(), 
            f'data/domains/{method}/{query_type}/{assembly}_protein_domains.csv',
        )
        if not os.path.isfile(protein_domains_path):
            continue
        
        protein_domains = pd.read_csv(protein_domains_path)
        
        for tpl in protein_domains.itertuples():
            query, label = tpl.pfam_query, tpl.protein_label
            
            if pd.isnull(label):
                label = 'Unknown'
                
            label = label.strip()
                
            if query not in query_to_protein_labels:
                query_to_protein_labels[query] = {
                    label: 1,
                }
            elif label not in query_to_protein_labels[query]:
                query_to_protein_labels[query][label] = 1
            else:
                query_to_protein_labels[query][label] += 1
    
    query_to_most_common_label = {}
    for query in sorted(query_to_protein_labels.keys()):
        label_counts = [(k, v) for k, v in query_to_protein_labels[query].items()]
        
        sorted_labels = sorted(label_counts, key=lambda t: t[1], reverse=True)
        
        query_to_most_common_label[query] = sorted_labels[0][0]
        
    return query_to_most_common_label


if __name__ == '__main__':
    main()
