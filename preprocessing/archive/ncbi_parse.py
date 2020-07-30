import argparse
import logging
import os
import re
import time
import zlib
import urllib.request as urllib_request
from contextlib import closing
from multiprocessing import Process, get_logger

from Bio import Entrez
from bs4 import BeautifulSoup
import pandas as pd
import requests


SPECIES_CSV = 'data/bac_dive/species.csv'
OUTPUT_FOLDER = 'data/ncbi/raw/'


logger = logging.getLogger(__name__)


def main():
    """
    Fetch genome sequence and annotations from a list of species name.

    - Retrieve genome ID from Entrez.esearch
    - Parse page https://www.ncbi.nlm.nih.gov/genome/<genome_id>
    - Retrieve urls to genome (fna) and annotations (gff) files
    - Download files
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('email')
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()

    email = args.email
    api_key = args.api_key
    n_jobs = args.n_jobs

    Entrez.email = email

    if api_key is not None:
        Entrez.api_key = api_key

    specied_csv_path = os.path.join(os.getcwd(), SPECIES_CSV)
    specie_names = get_specie_names(specied_csv_path)

    n_species = len(specie_names)
    n_species_per_job = int(round(n_species / 4))
    processes = []
    for i in range(n_jobs):
        start = i * n_species_per_job
        end = start + n_species_per_job
        if i + 1 < n_jobs:
            job_specie_names = specie_names[start:end]
        else:
            job_specie_names = specie_names[start:]

        p = Process(target=process_worker, args=(f'P{i+1}', job_specie_names))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    logger.info('DONE')


def process_worker(job_id, specie_names):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    logger.info(f'Process {job_id} | Process ID: {os.getpid()}')
    min_time_between_req_s = 0.5

    n_species = len(specie_names)
    prev_time = time.time()
    for i, specie_name in enumerate(specie_names):
        elapsed = time.time() - prev_time
        if i > 0 and elapsed < min_time_between_req_s:
            time.sleep(min_time_between_req_s - elapsed)

        prev_time = time.time()

        genome_id = None
        try:
            genome_id = fetch_genome_id(specie_name)
            if genome_id is None:
                logger.error(f'Process {job_id} | {i+1} / {n_species} | {specie_name} | Not Found')
                continue

            fasta_url, gff_url = fetch_download_urls(genome_id)

            if fasta_url is None:
                logger.error(f'Process {job_id} | {i+1} / {n_species} | {specie_name} | {genome_id} | FASTA URL missing')
                continue
            elif gff_url is None:
                logger.error(f'Process {job_id} | {i+1} / {n_species} | {specie_name} | {genome_id} | GFF URL missing')
                continue

            download_files(specie_name, fasta_url, gff_url)

            logger.info(f'Process {job_id} | {i+1} / {n_species} | {specie_name} | {genome_id} | OK')

        except (KeyboardInterrupt, SystemExit):
            raise
        except requests.exceptions.RequestException:
            logger.exception(f'Process {job_id} | {i+1} / {n_species} | {specie_name} | {genome_id} | RequestException')
        except BaseException:
            logger.exception(f'Process {job_id} | {i+1} / {n_species} | {specie_name} | {genome_id} | Exception')

    logger.info(f'Process {job_id}: DONE')


def download_files(specie_name, fasta_url, gff_url):
    specie_name_ = specie_name.replace(' ', '_').lower()
    output_fasta = os.path.join(os.getcwd(), OUTPUT_FOLDER, f'{specie_name_}.fasta')
    output_gff = os.path.join(os.getcwd(), OUTPUT_FOLDER, f'{specie_name_}.gff')

    download_file(fasta_url, output_fasta)
    download_file(gff_url, output_gff)


def download_file(url, output_path):
    with closing(urllib_request.urlopen(url)) as r:
        data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
        with open(output_path, 'wb') as f:
            f.write(data)


def fetch_download_urls(genome_id):
    content = fetch_page(genome_id)

    soup = BeautifulSoup(content, 'html.parser')

    href_regexp_fasta = re.compile(r'^ftp://ftp\.ncbi\.nlm\.nih\.gov/genomes/.+\.fna\.gz$')
    href_regexp_gff = re.compile(r'^ftp://ftp\.ncbi\.nlm\.nih\.gov/genomes/.+\.gff\.gz$')

    fasta_a = soup.find('a', href=href_regexp_fasta)
    gff_a = soup.find('a', href=href_regexp_gff)

    fasta_url = fasta_a.attrs['href'] if fasta_a is not None else None
    gff_url = gff_a.attrs['href'] if gff_a is not None else None

    return fasta_url, gff_url


def fetch_genome_id(specie_name):
    search_term = f'"{specie_name}"[Organism]'
    with Entrez.esearch(db='genome', term=search_term) as handle:
        record = Entrez.read(handle)

    if not record or not 'IdList' in record or len(record['IdList']) == 0:
        return None
    else:
        return record['IdList'][0]


def get_specie_names(csv_path):
    return sorted(pd.read_csv(csv_path)['specie_name'].values.tolist())


def fetch_page(genome_id):
    timeout_seconds = 30
    url = f'https://www.ncbi.nlm.nih.gov/genome/{genome_id}'
    req = requests.get(url, timeout=timeout_seconds)
    req.raise_for_status()
    return req.text


class FastaDownloadError(Exception):
    pass


class GFFDownloadError(Exception):
    pass


if __name__ == '__main__':
    main()
