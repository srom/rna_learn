import argparse
import os
import logging
import time
from pathlib import Path
from multiprocessing import Process
from contextlib import closing
import socket
import urllib.request as urllib_request
import urllib.error as urllib_error

import pandas as pd


SEQUENCES_TO_DOWNLOAD = 'data/NCBI_selected_assemblies.csv'
OUTPUT_PATH = 'data/sequences'
OUTPUT_CSV = 'data/NCBI_assemblies_final.csv'

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--n_processes', type=int, default=4)
    args = parser.parse_args()

    input_csv = args.input_csv
    output_folder = args.output_folder
    n_processes = args.n_processes

    if input_csv is None:
        input_csv = os.path.join(os.getcwd(), SEQUENCES_TO_DOWNLOAD)
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), OUTPUT_PATH)

    output_csv = os.path.join(os.getcwd(), OUTPUT_CSV)

    logger.info('Loading data')

    ncbi_metadata = pd.read_csv(input_csv)

    strain_assemblies = ncbi_metadata['assembly_accession'].values.tolist()

    logger.info(f'Launching {n_processes} processes')

    n_strains = len(strain_assemblies)
    n_strains_per_job = int(round(n_strains / n_processes))
    processes = []
    for i in range(n_processes):
        start = i * n_strains_per_job
        end = start + n_strains_per_job
        if i + 1 < n_processes:
            job_assemblies = strain_assemblies[start:end]
        else:
            job_assemblies = strain_assemblies[start:]

        p = Process(
            target=worker_main, 
            args=(
                f'P{i+1}', 
                job_assemblies, 
                input_csv, 
                output_folder,
            ),
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    logger.info('Fetching completed')

    logger.info('Checking data integrity')
    _, details_df = check_for_missing_files(ncbi_metadata, output_folder)

    assemblies_to_discard = details_df[
        details_df['is_missing']
    ]['assembly_accession'].unique()

    logger.info(f'Discarding {len(assemblies_to_discard)} assemblies')
    assembly_final_summary = ncbi_metadata[
        ~ncbi_metadata['assembly_accession'].isin(assemblies_to_discard)
    ].reset_index(drop=True)

    logger.info(f'Final strains count: {len(assembly_final_summary)}')
    logger.info(f'Writing final assembly summary dataset to {output_csv}')
    assembly_final_summary.to_csv(output_csv, index=False)

    logger.info('DONE')


def worker_main(job_id, job_assemblies, input_csv, output_folder):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    timeout_in_seconds = 60
    socket.setdefaulttimeout(timeout_in_seconds)

    logger.info(f'Process {job_id} | Process ID: {os.getpid()}')

    ncbi_metadata = pd.read_csv(input_csv)

    job_data = ncbi_metadata[
        ncbi_metadata['assembly_accession'].isin(job_assemblies)
    ].reset_index(drop=True)

    logger.info(f'Process {job_id} | {len(job_data)} strains | Process ID: {os.getpid()}')

    for row_ix in job_data.index:
        if (row_ix + 1) % 10 == 0:
            logger.info(f'Process {job_id} | Strain {row_ix + 1} / {len(job_data)}')

        row = job_data.loc[row_ix]
        fetch_sequences(row, output_folder)

    logger.info(f'Process {job_id}: DONE')


def fetch_sequences(metadata, output_folder, sleep_time=0.4):
    assembly_accession = metadata['assembly_accession']
    download_url_base = metadata['download_url_base']

    save_folder = os.path.join(output_folder, f'{assembly_accession}')

    # Create folder if it doesn't exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    sequences_to_fetch = [
        (
            f'{download_url_base}_genomic.fna.gz', 
            os.path.join(save_folder, f'{assembly_accession}_genomic.fna.gz'),
        ),
        (
            f'{download_url_base}_genomic.gff.gz', 
            os.path.join(save_folder, f'{assembly_accession}_genomic.gff.gz'),
        ),
        (
            f'{download_url_base}_protein.faa.gz', 
            os.path.join(save_folder, f'{assembly_accession}_protein.faa.gz'),
        ),
        (
            f'{download_url_base}_cds_from_genomic.fna.gz', 
            os.path.join(save_folder, f'{assembly_accession}_cds_from_genomic.fna.gz'),
        ),
        (
            f'{download_url_base}_rna_from_genomic.fna.gz', 
            os.path.join(save_folder, f'{assembly_accession}_rna_from_genomic.fna.gz'),
        ),
    ]

    for ftp_url, save_path in sequences_to_fetch:
        if Path(save_path).is_file():
            continue

        start_time_s = time.time()

        download_file_with_error_handling(ftp_url, save_path, sleep_time=sleep_time)

        elapsed_s = time.time() - start_time_s

        # Don't fire requests too often
        if elapsed_s < sleep_time:
            time.sleep(sleep_time - elapsed_s)


def download_file_with_error_handling(ftp_url, save_path, max_tries=5, sleep_time=0.4):
    try_number = 0
    while True:
        try_number += 1
        error_message = ''

        try:
            download_file(ftp_url, save_path)
            return

        except urllib_error.ContentTooShortError:
            error_message = 'ContentTooShortError exception'

        except urllib_error.URLError as e:
            error_message = e.reason

            # NCBI FTP server returns a meaningful error when content does not exist.
            # In this case we simply move on.
            if (
                isinstance(error_message, str) and 
                'No such file or directory' in error_message
            ):
                return

        if try_number >= max_tries:
            logger.error(f'Too many errors for url {ftp_url} | Error: {error_message}')
            return
        else:
            if try_number == 2:
                logger.warning(f'Second error for url {ftp_url} | Error: {error_message}')

            # Sleep for longer as consecutive errors creep in.
            # This is so we can recover from cases where NCBI notifies
            # us that too many requests have been fired.
            sleep_time_seconds = try_number * sleep_time
            time.sleep(sleep_time_seconds)
            continue


def download_file(ftp_url, save_path):
    with closing(urllib_request.urlopen(ftp_url)) as r:
        with open(save_path, 'wb') as f:
            f.write(r.read())


def check_for_missing_files(ncbi_records, data_folder):
    stats_missing = {
        'dna (fna)': 0,
        'annotations (gff)': 0,
        'protein (faa)': 0,
        'CDS (fna)': 0,
        'RNA (fna)': 0,
    }
    
    details = []
    for row_ix in ncbi_records.index:
        row = ncbi_records.loc[row_ix]
        assembly_accession = row['assembly_accession']
        files = [
            ('dna (fna)', os.path.join(data_folder, f'{assembly_accession}', f'{assembly_accession}_genomic.fna.gz')),
            ('annotations (gff)', os.path.join(data_folder, f'{assembly_accession}', f'{assembly_accession}_genomic.gff.gz')),
            ('protein (faa)', os.path.join(data_folder, f'{assembly_accession}', f'{assembly_accession}_protein.faa.gz')),
            ('CDS (fna)', os.path.join(data_folder, f'{assembly_accession}', f'{assembly_accession}_cds_from_genomic.fna.gz')),
            ('RNA (fna)', os.path.join(data_folder, f'{assembly_accession}', f'{assembly_accession}_rna_from_genomic.fna.gz')),
        ]
        for category, filepath in files:
            is_missing = False
            if not Path(filepath).is_file():
                stats_missing[category] += 1
                is_missing = True
                
            details.append([assembly_accession, category, is_missing])
    
    details_df = pd.DataFrame(details, columns=['assembly_accession', 'content', 'is_missing'])
    
    return stats_missing, details_df


if __name__ == '__main__':
    main()
