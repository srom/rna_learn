import argparse
import os
import logging
import gzip
import json
import re

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from Bio import SeqIO
from Bio.Seq import Seq

from .sequence_utils import (
    parse_location, 
    InvalidLocationError,
    parse_chromosome_id,
    parse_sequence_type,
    InvalidSequenceTypeError,
)


DB_PATH = 'data/condensed_traits/db/seq.db'
SEQUENCES_BASE_FOLDER = 'data/condensed_traits/sequences'

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--sequences_base_folder', type=str, default=None)
    args = parser.parse_args()

    db_path = args.db_path
    sequences_base_folder = args.sequences_base_folder

    if db_path is None:
        db_path = os.path.join(os.getcwd(), DB_PATH)
    if sequences_base_folder is None:
        sequences_base_folder = os.path.join(os.getcwd(), SEQUENCES_BASE_FOLDER)

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    rna_path_fmt = os.path.join(sequences_base_folder, '{0}/{0}_rna_from_genomic.fna.gz')

    species_taxid_query = 'select species_taxid from species_traits'
    species_taxids = pd.read_sql(species_taxid_query, engine)['species_taxid'].values.tolist()

    logger.info(f'Importing RNA for {len(species_taxids):,} species')

    for i, species_taxid in enumerate(species_taxids):
        if i == 0 or (i + 1) % 10 == 0:
            logger.info(f'Specie {i+1:,} / {len(species_taxids):,}')

        species_data_q = (
            'select species, phylum, superkingdom from species_traits '
            'where species_taxid = ?'
        )
        species_data = pd.read_sql(
            species_data_q, 
            engine,
            params=(species_taxid,),
        )
        superkingdom = species_data.iloc[0]['superkingdom']
        phylum = species_data.iloc[0]['phylum']
        species_name = species_data.iloc[0]['species']

        if superkingdom == 'Bacteria':
            q = (
                "select amino_acid, anticodon, coding_sequence from trna_reference "
                "where superkingdom = 'Bacteria'"
            )
            known_trnas = {
                tpl.coding_sequence: (tpl.amino_acid, tpl.anticodon)
                for tpl in pd.read_sql(q, engine).itertuples()
            }
        elif superkingdom == 'Archaea':
            q = (
                "select amino_acid, anticodon, coding_sequence from trna_reference "
                "where superkingdom = 'Archaea'"
            )
            known_trnas = {
                tpl.coding_sequence: (tpl.amino_acid, tpl.anticodon)
                for tpl in pd.read_sql(q, engine).itertuples()
            }
        else:
            raise ValueError(f'Unknown superkingdom {superkingdom} for specie {species_taxid}')

        sequence_records_to_import = []
        rna_fasta_path = rna_path_fmt.format(species_taxid)

        with gzip.open(rna_fasta_path, mode='rt') as f:
            rna_records = list(SeqIO.parse(f, "fasta"))

        import_sequences(engine, species_taxid, rna_records, superkingdom, known_trnas)


def import_sequences(engine, species_taxid, sequence_records, superkingdom, known_trnas):
    columns = [
        'sequence_id', 'species_taxid', 'sequence_type', 
        'chromosome_id', 'location_json', 'strand', 'length', 
        'description', 'metadata_json', 'sequence',
    ]

    data = []
    for sequence_record in sequence_records:
        seq_id = sequence_record.id
        sequence = sequence_record.seq
        chromosome_id = parse_chromosome_id(seq_id)

        try:
            location_list, strand = parse_location(sequence_record)
            location_json = json.dumps(location_list, sort_keys=True)
        except InvalidLocationError as e:
            logger.warning(f'{species_taxid} | Invalid location information: {e.message}')
            continue

        try:
            sequence_type = parse_sequence_type(sequence_record)
        except InvalidSequenceTypeError as e:
            logger.warning(f'{species_taxid} | Invalid sequence type information: {e.message}')
            continue

        # A handful of species have mRNA sequences reported.
        # These should be handled as CDS.
        if sequence_type == 'mRNA':
            continue

        metadata_json = None
        if sequence_type.lower() == 'trna':
            try:
                codon, anticodon, amino_acid = identify_trna_codon(
                    engine,
                    sequence_record, 
                    superkingdom,
                    known_trnas,
                )

                metadata_json = json.dumps(
                    {
                        'codon': codon,
                        'anticodon': anticodon,
                        'amino_acid': amino_acid,
                    },
                    sort_keys=True,
                )
            except RNAIdentificationError as e:
                pass

        row = [
            seq_id,
            species_taxid,
            sequence_type,
            chromosome_id,
            location_json,
            strand,
            len(sequence),
            sequence_record.description,
            metadata_json,
            str(sequence),
        ]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_sql(
        'sequences',
        engine,
        if_exists='append',
        method='multi',
        index=False,
    )


def identify_trna_codon(engine, sequence_record, superkingdom, known_trnas):
    sequence = str(sequence_record.seq).upper()
    description = sequence_record.description

    m = re.match(r'^.*\[gene=tRNA-([a-zA-Z]{3})_([ATGCatgc]{3})\].*$', description)
    if m is not None:
        amino_acid = m[1].strip().title()
        anticodon = m[2].strip().upper()
        codon = str(Seq(anticodon).reverse_complement())
        return codon, anticodon, amino_acid

    tpl = known_trnas.get(sequence)

    if tpl is None:
        seq = str(sequence_record.seq)[:-1].upper()
        tpl = known_trnas.get(seq)

    if tpl is None:
        seq = str(sequence_record.seq)[1:].upper()
        tpl = known_trnas.get(seq)

    if tpl is None:
        query = (
            "select amino_acid, anticodon from trna_reference "
            "where superkingdom = '{0}' and full_sequence like '%{1}%' "
            "limit 1"
        ).format(superkingdom, sequence)
        res = pd.read_sql(query, engine)
        if len(res) > 0:
            tpl = (res.iloc[0]['amino_acid'], res.iloc[0]['anticodon'])

    if tpl is None:
        raise RNAIdentificationError('Cannot identify tRNA codon')

    amino_acid = tpl[0].strip().title()
    anticodon = tpl[1].strip().upper()
    codon = str(Seq(anticodon).reverse_complement())

    return codon, anticodon, amino_acid


class RNAIdentificationError(Exception):

    def __init__(self, message, *args, **kwargs):
        self.message = message
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
	main()
