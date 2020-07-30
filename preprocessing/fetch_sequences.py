#############################################################
## Project: mRNA thermal adaptation
## Script purpose: Assemble a large database of procaryotic
## sequences by fetching data from NCBI's ftp server.
## Date: July 2020
## Author: Adapted in Python by Romain Strock from an R 
## script developped by Antoine Hocher.
#############################################################

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


SEQUENCES_TO_DOWNLOAD = 'data/condensed_traits/ncbi_species_final.csv'
OUTPUT_PATH = 'data/condensed_traits/sequences'

logger = logging.getLogger(__name__)


def main():
    """
    Species to download should be specified in a comma separated file (CSV) in utf-8 format
    with at least the following two columns:
        - species_taxid: NCBI species taxonomy ID
        - download_url_base: base ftp url of the sequences to be fetched 
          (e.g. everything before "_genomic.fna.gz")
    """
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
        input_csv = os.path.join(os.getcwd(), input_csv)
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), OUTPUT_PATH)

    logger.info('Loading data')

    ncbi_metadata = pd.read_csv(input_csv)

    species_taxid = ncbi_metadata['species_taxid'].values.tolist()

    logger.info(f'Launching {n_processes} processes')

    n_species = len(species_taxid)
    n_species_per_job = int(round(n_species / n_processes))
    processes = []
    for i in range(n_processes):
        start = i * n_species_per_job
        end = start + n_species_per_job
        if i + 1 < n_processes:
            job_species_taxid = species_taxid[start:end]
        else:
            job_species_taxid = species_taxid[start:]

        p = Process(
            target=worker_main, 
            args=(
                f'P{i+1}', 
                job_species_taxid, 
                input_csv, 
                output_folder,
            ),
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    logger.info('DONE')


def worker_main(job_id, species_taxid, input_csv, output_folder):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    timeout_in_seconds = 60
    socket.setdefaulttimeout(timeout_in_seconds)

    logger.info(f'Process {job_id} | Process ID: {os.getpid()}')

    ncbi_metadata = pd.read_csv(input_csv)

    job_data = ncbi_metadata[
        ncbi_metadata['species_taxid'].isin(species_taxid)
    ].reset_index(drop=True)

    logger.info(f'Process {job_id} | {len(job_data)} species | Process ID: {os.getpid()}')

    for row_ix in job_data.index:
        if (row_ix + 1) % 10 == 0:
            logger.info(f'Process {job_id} | Specie {row_ix + 1} / {len(job_data)}')

        row = job_data.loc[row_ix]
        fetch_sequences(row, output_folder)

    logger.info(f'Process {job_id}: DONE')


def fetch_sequences(metadata, output_folder):
    specie_taxid = metadata['species_taxid']
    download_url_base = metadata['download_url_base']

    save_folder = os.path.join(output_folder, f'{specie_taxid}')

    # Create folder if it doesn't exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    sequences_to_fetch = [
        (
            f'{download_url_base}_genomic.fna.gz', 
            os.path.join(save_folder, f'{specie_taxid}_genomic.fna.gz'),
        ),
        (
            f'{download_url_base}_genomic.gff.gz', 
            os.path.join(save_folder, f'{specie_taxid}_genomic.gff.gz'),
        ),
        (
            f'{download_url_base}_protein.faa.gz', 
            os.path.join(save_folder, f'{specie_taxid}_protein.faa.gz'),
        ),
        (
            f'{download_url_base}_cds_from_genomic.fna.gz', 
            os.path.join(save_folder, f'{specie_taxid}_cds_from_genomic.fna.gz'),
        ),
        (
            f'{download_url_base}_rna_from_genomic.fna.gz', 
            os.path.join(save_folder, f'{specie_taxid}_rna_from_genomic.fna.gz'),
        ),
    ]

    for ftp_url, save_path in sequences_to_fetch:
        if Path(save_path).is_file():
            continue

        start_time_s = time.time()

        download_file_with_error_handling(ftp_url, save_path)

        elapsed_s = time.time() - start_time_s

        # Don't fire requests too often
        if elapsed_s < 0.4:
            time.sleep(0.4 - elapsed_s)


def download_file_with_error_handling(ftp_url, save_path, max_tries=5):
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

            # NCBI FTP server returns a meaningful message when content does not exist.
            # In this case we simply move on.
            if isinstance(error_message, str) and 'No such file or directory' in error_message:
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
            sleep_time_seconds = try_number * 0.4
            time.sleep(sleep_time_seconds)
            continue


def download_file(ftp_url, save_path):
    with closing(urllib_request.urlopen(ftp_url)) as r:
        with open(save_path, 'wb') as f:
            f.write(r.read())


if __name__ == '__main__':
    main()
