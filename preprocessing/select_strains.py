import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd


NCBI_ASSEMBLY_SUMMARY = 'data/NCBI_assembly_summary.txt'
SPECIES_TRAITS = 'data/condensed_traits/condensed_species_NCBI.csv'
OUTPUT_PATH = 'data/NCBI_selected_assemblies.csv'

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly_summary_path', type=str, default=None)
    parser.add_argument('--species_traits_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    assembly_summary_path = args.assembly_summary_path
    if assembly_summary_path is None:
        assembly_summary_path = os.path.join(os.getcwd(), NCBI_ASSEMBLY_SUMMARY)

    species_traits_path = args.species_traits_path
    if species_traits_path is None:
        species_traits_path = os.path.join(os.getcwd(), SPECIES_TRAITS)

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), OUTPUT_PATH)

    logger.info(f'Loading NCBI data from {assembly_summary_path}')
    assembly_summary = pd.read_csv(assembly_summary_path, sep='\t', skiprows=1)
    assembly_summary['seq_rel_date_processed'] = assembly_summary['seq_rel_date'].apply(
        lambda d_str: datetime.datetime.strptime(d_str, '%d/%m/%Y')
    )

    logger.info(f'Loading species traits from {species_traits_path}')
    species_traits_df = pd.read_csv(species_traits_path)

    species_with_ogt = species_traits_df[
        species_traits_df['growth_tmp'].notnull()
    ]['species_tax_id'].values

    logger.info(f'Found {len(species_with_ogt):,} species with non null growth temperature')

    assemblies = select_complete_assemblies(assembly_summary, species_with_ogt)

    logger.info(f'Found {len(assemblies):,} complete assemblies')

    more_assemblies = select_incomplete_assemblies_for_less_common_ogt(
        assembly_summary,
        species_traits_df,
        assemblies,
    )

    logger.info(f'Found {len(more_assemblies):,} other assemblies')

    assemblies += more_assemblies

    logger.info(f'Final number of assemblies: {len(assemblies):,}')

    final_df = assembly_summary[
        assembly_summary['assembly_accession'].isin(assemblies)
    ].reset_index(drop=True)

    final_df['download_url_base'] = final_df.apply(
        make_download_url, 
        axis=1,
    )

    logger.info(f'Exporting to {output_path}')

    final_df.to_csv(output_path, index=False)

    logger.info('DONE')


def select_complete_assemblies(assembly_summary, species_with_ogt):
    shortlist_1 = assembly_summary[
        (assembly_summary['assembly_level'] == 'Complete Genome') &
        (assembly_summary['genome_rep'] == 'Full') &
        (assembly_summary['species_taxid'].isin(species_with_ogt))
    ].reset_index(drop=True)

    shortlist_2, stats = select_representative_genome(shortlist_1)

    logger.info((
        f'Selection of {len(shortlist_2)} assemblies, including '
        f'{stats["reference genome"]} reference genomes and '
        f'{stats["representative genome"]} representative genomes'
    ))

    return shortlist_2['assembly_accession'].values.tolist()


def select_incomplete_assemblies_for_less_common_ogt(
    assembly_summary, 
    species_traits_df,
    seen_assemblies,
    low=20,
    high=45,
):
    candidate_taxids = species_traits_df[
        (species_traits_df['growth_tmp'].notnull())
        & (
            (species_traits_df['growth_tmp'] < low) |
            (species_traits_df['growth_tmp'] > high)
        )
    ]['species_tax_id'].values

    shortlist = assembly_summary[
        (assembly_summary['species_taxid'].isin(candidate_taxids)) &
        (assembly_summary['version_status'] == 'latest') &
        (assembly_summary['excluded_from_refseq'].isnull()) &
        (~assembly_summary['assembly_accession'].isin(seen_assemblies))
    ]

    return shortlist['assembly_accession'].values.tolist()


def select_representative_genome(assembly_summary):
    species_taxid = assembly_summary['species_taxid'].unique()
    
    stats = {
        'reference genome': 0,
        'representative genome': 0,
        'most recent': 0,
        'most recent paired with GenBank': 0,
        'most recent not paired with GenBank': 0,
    }
    stain_indices = []
    for specie_taxid in species_taxid:
        df = assembly_summary[assembly_summary['species_taxid'] == specie_taxid]
        ix = df.index
        for row_ix in ix:
            row = assembly_summary.loc[row_ix]
            refseq_category = row['refseq_category']
            stain_idx = None
            if refseq_category == 'reference genome':
                stats['reference genome'] += 1
                stain_idx = row_ix
                break
            elif refseq_category == 'representative genome':
                stats['representative genome'] += 1
                stain_idx = row_ix
                break
                
        if stain_idx is None:
            stats['most recent'] += 1
            sorted_df = df.sort_values('seq_rel_date_processed', ascending=False)
            
            for row_ix in sorted_df.index:
                row = assembly_summary.loc[row_ix]
                gbrs_paired_asm = row['gbrs_paired_asm']
                paired_asm_comp = row['paired_asm_comp']
                if (
                    not pd.isnull(gbrs_paired_asm) and 
                    not pd.isnull(paired_asm_comp) and 
                    paired_asm_comp == 'identical'
                ):
                    stain_idx = row_ix
                    stats['most recent paired with GenBank'] += 1
                    break
                    
            if stain_idx is None:
                stain_idx = sorted_df.index[0]
                stats['most recent not paired with GenBank'] += 1
            
        stain_indices.append(stain_idx)

    return assembly_summary.loc[stain_indices].reset_index(drop=True), stats


def make_download_url(row):
    ftp_path = row['ftp_path'].strip().replace('(', '_').replace(')', '_')
    assembly_accession = row['assembly_accession'].strip().replace('(', '_').replace(')', '_')
    asm_name = row['asm_name'].strip().replace('/', '_').replace('#', '_').replace('(', '_').replace(')', '_')
    download_url_base = f"{ftp_path}/{assembly_accession}_{asm_name}".replace(' ', '_')
    return download_url_base


if __name__ == '__main__':
    main()
