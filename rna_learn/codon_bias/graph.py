import argparse
import os
import json
import logging
import string
import collections

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import networkx as nx

from rna_learn.alphabet import CODON_REDUNDANCY


logger = logging.getLogger(__name__)


def main():
    """
    Creates a networkx graph of species where the link between species is based on the 
    distance between "codon bias patterns". 

    A "codon bias pattern" is a vector of 62 values representing the usage ratio of each codon
    against synonymous codons (i.e. coding for the same amino acido or stop). There are 62 values
    and not 64 because the codons for Met and Trp are excluded (they do not have synonyms).

    An appropriate distance threshold is computed such that, given a species, only other 
    species with a distance lower than this threshold are linked.

    Various species attributes and traits are included as node metadata.
    Edges are weighted by the distance.

    Distance is based on the Jensen-Shannon metric, and is computed as follow:
    - Notice that for a given AA or stop, bias codon values form a distribution
    - Compute the Jensen-Shannon distance between each individual AA or stop distribution
    - Take the average of these 19 individual distances.

    Threshold is computed as follow:
    - For each species, consider the mean distance to other species
    - Substract 1.3 times the standard deviation (~10% tail for a normal distribution)
    - Compute the average local thresholds accross all species.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--distance_matrix_path', type=str, default=None)
    parser.add_argument('--codon_bias_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path
    distance_matrix_path = args.distance_matrix_path
    codon_bias_path = args.codon_bias_path
    output_path = args.output_path

    if db_path is None:
        db_path = db_path = os.path.join(os.getcwd(), 'data/db/seq.db')

    if distance_matrix_path is None:
        distance_matrix_path = os.path.join(os.getcwd(), 'data/distance_matrix.npy')

    if codon_bias_path is None:
        codon_bias_path = os.path.join(os.getcwd(), 'data/species_codon_ratios.csv')

    if output_path is None:
        output_path = os.path.join(os.getcwd(), 'data/codon_bias_graph.gpickle')
    
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    logger.info('Loading data')

    # Load strain & species ids
    assembly_accessions, species_taxids = load_strains_and_species(engine)

    species_to_assembly = map_species_to_assembly_list(assembly_accessions, species_taxids)

    unique_species_taxids = np.unique(species_taxids)
    n_species = len(unique_species_taxids)

    # Load distance matrix
    distance_matrix = np.load(distance_matrix_path, allow_pickle=True)
    assert distance_matrix.shape == (n_species, n_species)

    # Load codon bias
    species_codon_df = load_codon_bias(engine, codon_bias_path)

    # Compute distance threshold
    distance_thres = compute_appropriate_threshold_distance(species_codon_df, distance_matrix)

    # Build graph
    graph = build_graph(species_codon_df, distance_matrix, species_to_assembly, distance_thres)

    logger.info('Saving graph')
    nx.write_gpickle(graph, output_path)

    logger.info('DONE')


def build_graph(species_codon_df, distance_matrix, species_to_assembly, max_distance):
    graph = nx.Graph()
    
    logger.info('Buiding graph')
    n_neighbors = []
    for ix in range(len(species_codon_df)):
        if (ix + 1) % 200 == 0:
            logger.info(f'{ix + 1} / {len(species_codon_df)}')
            
        species_series = species_codon_df.loc[ix]
        
        species_taxid = species_series['species_taxid']
        if species_taxid not in graph.nodes:
            add_node(graph, species_series, species_to_assembly[species_taxid])

        nearby_ix = [
            i for i, v in enumerate(distance_matrix[ix])
            if i != ix and v <= max_distance
        ]

        n_neighbors.append(len(nearby_ix))
        
        for other_ix in nearby_ix:
            other_species_series = species_codon_df.loc[other_ix]
            other_species_taxid = other_species_series['species_taxid']
            
            if graph.has_edge(species_taxid, other_species_taxid):
                continue
            
            if other_species_taxid not in graph.nodes:
                add_node(graph, other_species_series, species_to_assembly[other_species_taxid])
                
            graph.add_edge(species_taxid, other_species_taxid, weight=distance_matrix[ix, other_ix])

    logger.info('Graph built')
    logger.info(f'Number of neighbors:')
    logger.info(f'Mean: {np.mean(n_neighbors):.2f}')
    logger.info(f'Std:  {np.std(n_neighbors):.2f}')
    logger.info(f'min:  {np.min(n_neighbors):.2f}')
    logger.info(f'max:  {np.max(n_neighbors):.2f}')

    return graph


def add_node(graph, species_series, assembly_accessions):
    attributes = [
        'species_taxid',
        'species',
        'growth_tmp',
        'genus', 
        'family', 
        'order',
        'class', 
        'phylum', 
        'superkingdom',
        'growth_tmp',
        'optimum_ph',
        'gram_stain',
        'metabolism',
        'sporulation',
        'motility',
        'range_salinity',
        'cell_shape',
        'isolation_source',
        'doubling_h',
        'genome_size',
        'gc_content',
        'coding_genes',
        'tRNA_genes',
        'rRNA16S_genes',
    ]
    attributes_dict = {
        attr: species_series[attr]
        for attr in attributes
    }
    attributes_dict['n_assemblies'] = len(assembly_accessions)
    attributes_dict['assembly_accessions'] = tuple(assembly_accessions)
    graph.add_node(species_series['species_taxid'], **attributes_dict)


def compute_appropriate_threshold_distance(species_codon_df, distance_matrix):    
    output_columns = [
        'species_taxid',
        'distance_mean',
        'distance_std',
    ]
    
    data = []
    for ix in range(len(species_codon_df)):
        species = species_codon_df.loc[ix]

        d = [
            v for i, v in enumerate(distance_matrix[ix])
            if i != ix
        ]

        data.append([
            species['species_taxid'],
            np.mean(d) if len(d) > 0 else 0.,
            np.std(d) if len(d) > 0 else 0.,
        ])
        
    distance_df = pd.DataFrame(data, columns=output_columns)
    
    return (distance_df['distance_mean'] - 1.3 * distance_df['distance_std']).mean()


def load_codon_bias(engine, codon_bias_path):
    return add_traits_information(engine, pd.read_csv(codon_bias_path))


def add_traits_information(engine, species_codon_df):
    q = """
    select 
        species_taxid, 
        species, 
        genus, 
        family, 
        "order", 
        class, 
        phylum, 
        superkingdom,
        growth_tmp,
        optimum_ph,
        gram_stain,
        metabolism,
        sporulation,
        case motility 
           when 'no' then 'no'
           else 'yes'
        end motility,
        range_salinity,
        cell_shape,
        isolation_source,
        doubling_h,
        genome_size,
        gc_content,
        coding_genes,
        tRNA_genes,
        rRNA16S_genes
    from species_traits
    """
    return pd.merge(
        species_codon_df,
        pd.read_sql(q, engine),
        how='inner',
        on='species_taxid',
    )


def load_strains_and_species(engine):
    assembly_query = """
    select assembly_accession, species_taxid from assembly_source
    """
    assembly_df = pd.read_sql(assembly_query, engine)

    assembly_accessions = assembly_df['assembly_accession'].values
    species_taxids = assembly_df['species_taxid'].values

    return assembly_accessions, species_taxids


def map_species_to_assembly_list(assembly_accessions, species_taxids):
    species_to_assembly_list = collections.defaultdict(list)

    for i in range(len(assembly_accessions)):
        species_to_assembly_list[species_taxids[i]].append(assembly_accessions[i])

    return dict(species_to_assembly_list)


if __name__ == '__main__':
    main()
