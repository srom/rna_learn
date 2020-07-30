import logging
import os
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd


logger = logging.getLogger(__name__)


def main():
    """
    Parser for BacDive pages of the form https://bacdive.dsmz.de/strain/<strain_id>
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    strain_ids_path = os.path.join(os.getcwd(), 'data/bac_dive/bacdive_strain_ids_all_but_mesophiles.csv')
    whitelist_path = os.path.join(os.getcwd(), 'data/bac_dive/whitelist_species.csv')
    output_path = os.path.join(os.getcwd(), 'data/bac_dive/species_no_mesophiles.csv')

    white_list_species = {s.lower().strip() for s in pd.read_csv(whitelist_path)['specie_name'].values}

    flush_every = 50
    min_time_between_req_s = 0.5
    strain_ids = load_strain_ids(strain_ids_path)

    columns = [
        'bacdive_id',
        'domain',
        'specie_name',
        'temperature',
        'temperature_range',
    ]

    n_strains = len(strain_ids)
    output_data_batch = []
    first_write = True
    prev_time = time.time()
    seen_species = set()
    for i, strain_id in enumerate(strain_ids):
        logger.info(f'\t{i+1} / {n_strains} | {strain_id}')

        elapsed = time.time() - prev_time
        if elapsed < min_time_between_req_s:
            time.sleep(min_time_between_req_s - elapsed)

        prev_time = time.time()

        try:
            domain, specie_name, temperature, temperature_range = parse_strain(strain_id)
        except (KeyboardInterrupt, SystemExit):
            raise
        except requests.exceptions.RequestException:
            logger.exception(f'Exception raised while fetching')
            continue
        except Exception:
            logger.exception(f'Exception raised while parsing')
            continue

        if specie_name is None:
            continue
        elif specie_name in seen_species:
            continue
        elif specie_name.lower() not in white_list_species:
            continue
        else:
            seen_species.add(specie_name)

        output_data_batch.append([
            strain_id,
            domain,
            specie_name,
            temperature,
            temperature_range,
        ])

        logger.info(f'{domain} | {specie_name} | {temperature_range} | {temperature}°C')

        if len(output_data_batch) >= flush_every:
            flush_data(output_data_batch, output_path, columns, include_header=first_write)
            output_data_batch = []
            first_write = False

    if len(output_data_batch) > 0:
        flush_data(output_data_batch, output_path, columns, include_header=first_write)

    logger.info('DONE')


def flush_data(output_data, output_path, columns, include_header=False):
    output_df = pd.DataFrame(output_data, columns=columns)
    output_df.to_csv(output_path, index=False, mode='a', header=include_header)


def load_strain_ids(path):
    return sorted(set(pd.read_csv(path).values.flatten()))


def parse_strain(strain_id):
    content = fetch_page(strain_id)

    soup = BeautifulSoup(content, 'html.parser')

    section = soup.find(
        'span', 
        string='Information on culture and growth conditions',
    ).find_parent(
        'div', 
        class_='section',
    )

    domain = soup.find('td', string='Domain').find_next_sibling('td').string
    specie_name = soup.find('td', string='Species').find_next_sibling('td').string
    temperature_str = section.find(
        'td', string='growth').find_next_sibling('td').string.replace('  ̊C', '')

    if domain is None or specie_name is None or temperature_str is None:
        return None, None, None, None
    else:
        domain = domain.strip()
        specie_name = specie_name.strip()
        temperature_str = temperature_str.strip()

    if '-' in temperature_str:
        t1, t2 = temperature_str.split('-')
        temperature = round((float(t1) + float(t2)) / 2, 1)
    else:
        temperature = round(float(temperature_str), 1)

    temperature_range = section.find('td', string='Temperature range').find_next_sibling('td').string.strip()

    return domain, specie_name, temperature, temperature_range


def fetch_page(strain_id):
    timeout_seconds = 5
    url = f'https://bacdive.dsmz.de/strain/{strain_id}'
    req = requests.get(url, timeout=timeout_seconds)
    req.raise_for_status()
    return req.text


if __name__ == '__main__':
    main()
