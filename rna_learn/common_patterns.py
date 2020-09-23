import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine

from .int_grads import (
    integrated_gradients_for_binary_features,
    integrated_gradients_for_binary_features_2,
)
from .load_sequences import load_batch_dataframe


logger = logging.getLogger(__name__)


# S can be G or C, see https://www.bioinformatics.org/sms/iupac.html
GC_SET = {'G', 'C', 'S'}

# Codons representative of amino acids I, V, Y, W, R, E, L
IVYWREL_codons = {
    'ATT',  # I
    'ATC',  # I
    'ATA',  # I
    'GTT',  # V
    'GTC',  # V
    'GTA',  # V
    'GTG',  # V
    'TAT',  # Y
    'TAC',  # Y
    'TGG',  # W
    'CGT',  # R
    'CGC',  # R
    'CGA',  # R
    'CGG',  # R
    'AGA',  # R
    'AGG',  # R
    'GAA',  # E
    'GAC',  # E
    'TTA',  # L
    'TTG',  # L
    'CTT',  # L
    'CTC',  # L
    'CTA',  # L
    'CTG',  # L
}


def compute_gc_content(nucleotide_sequence):
    """
    GC content is the ratio of the count of characters that are G or C
    over the total sequence length.
    """
    return compute_gc_count(nucleotide_sequence) / len(nucleotide_sequence)


def compute_gc_count(nucleotide_sequence):
    gc_count = 0
    for char in nucleotide_sequence:
        if char in GC_SET:
            gc_count += 1

    return gc_count


def compute_IVYWREL_content(cds_nucleotide_sequence):
    return compute_IVYWREL_count(cds_nucleotide_sequence) / len(cds_nucleotide_sequence)


def compute_IVYWREL_count(cds_nucleotide_sequence):
    assert len(cds_nucleotide_sequence) % 3 == 0

    IVYWREL_count = 0
    for i in range(0, len(cds_nucleotide_sequence), 3):
        codon = cds_nucleotide_sequence[i:i+3]
        if codon in IVYWREL_codons:
            IVYWREL_count += 1
    
    return IVYWREL_count


def compute_gc_and_IVYWREL_content_on_test_set(engine, chunksize=100, logging=True):
    total_rows_query = """
    select count(rowid) as c from sequences
    where species_taxid in (
        select species_taxid from train_test_split
        where in_test_set = 1
    )
    """
    n_sequences = pd.read_sql(total_rows_query, engine).iloc[0]['c']

    logger.info(f'Number of sequences: {n_sequences:,}')

    sequences_query = """
    select species_taxid, sequence, sequence_type from sequences
    where species_taxid in (
        select species_taxid from train_test_split
        where in_test_set = 1
    )
    """
    gc_acc = {}
    IVYWREL_acc = {}

    iterator = pd.read_sql(sequences_query, engine, chunksize=chunksize)

    count = 0
    for sequence_df in iterator:
        if logging and count % 100000 == 0:
            logger.info(f'{count:,} / {n_sequences:,}')

        for tpl in sequence_df.itertuples():
            species_taxid = tpl.species_taxid
            sequence = tpl.sequence
            sequence_type = tpl.sequence_type

            sequence_length = len(sequence)
            gc_count = compute_gc_count(sequence)

            if species_taxid not in gc_acc:
                gc_acc[species_taxid] = [gc_count, sequence_length]
            else:
                tpl = gc_acc[species_taxid]
                tpl[0] += gc_count
                tpl[1] += sequence_length

            if sequence_type == 'CDS':
                IVYWREL_count = compute_IVYWREL_count(sequence)

                if species_taxid not in IVYWREL_acc:
                    IVYWREL_acc[species_taxid] = [IVYWREL_count, sequence_length]
                else:
                    tpl = IVYWREL_acc[species_taxid]
                    tpl[0] += IVYWREL_count
                    tpl[1] += sequence_length

        count += chunksize

    specie_taxids = sorted(gc_acc.keys())
    output_data = []
    for species_taxid in specie_taxids:
        gc_count, gc_length = gc_acc[species_taxid]
        IVYWREL_count, IVYWREL_length = IVYWREL_acc[species_taxid]

        gc_content = gc_count / gc_length
        IVYWREL_content = IVYWREL_count / IVYWREL_length

        output_data.append([species_taxid, gc_content, IVYWREL_content])

    return pd.DataFrame(output_data, columns=['species_taxid', 'gc_content', 'IVYWREL_content'])


class AttributionHelper(object):

    def __init__(self, model):
        self.model = model
        self.attribution_fn = tf.function(
            integrated_gradients_for_binary_features_2,
            experimental_relax_shapes=True,
        )

    def compute_IVYWREL_average_attribution(self, species_seq, max_queue_size=50):
        queue = tf.keras.utils.OrderedEnqueuer(species_seq)
        try:
            queue.start(max_queue_size=max_queue_size)
            batch_generator = queue.get()
            return self._process_inner(
                species_seq,
                batch_generator,
            )
        finally:
            if queue.is_running():
                queue.stop()
    
    def _process_inner(self, species_seq, batch_generator):
        all_attributions = []
        all_attributions_2 = []
        all_attributions_3 = []
        max_length = species_seq.max_sequence_length
        for i in range(len(species_seq)):
            batch_x, batch_y, _ = next(batch_generator)
            baseline = np.ones(batch_x[0].shape, dtype='float32')
            target = batch_y[0]

            res = self.attribution_fn(
                self.model,
                batch_x,
                baseline,
                target,
            )

            attributions = res[0].numpy()
            attributions_baseline_mean = res[1].numpy()
            attributions_no_baseline = res[2].numpy()
            
            a = i * species_seq.batch_size
            b = (i + 1) * species_seq.batch_size
            batch_rowids = species_seq.rowids[a:b]
            
            batch_df = load_batch_dataframe(species_seq.engine, batch_rowids[:, 0])
            
            for j in range(len(attributions)):
                seq_attribution_padded = attributions[j]
                seq_attribution_padded_2 = attributions_baseline_mean[j]
                seq_attribution_padded_3 = attributions_no_baseline[j]

                rowid, part_ix = batch_rowids[j]
                seq_metadata = batch_df.loc[rowid]
                sequence_full = seq_metadata['sequence']
                length = int(seq_metadata['length'])
                sequence_type = seq_metadata['sequence_type']
                if sequence_type != 'CDS':
                    continue
                
                if length > max_length:
                    a = part_ix * max_length
                    b = (part_ix + 1) * max_length
                    if b > length:
                        length = length - part_ix * max_length
                    else:
                        length = max_length
                
                    sequence = sequence_full[a:b]
                else:
                    sequence = sequence_full
                    
                seq_attribution = seq_attribution_padded[:length]
                seq_attribution_2 = seq_attribution_padded_2[:length]
                seq_attribution_3 = seq_attribution_padded_3[:length]
                
                assert len(sequence) % 3 == 0
                assert len(sequence) == len(seq_attribution)
                
                for codon_start_ix in range(0, len(seq_attribution), 3):
                    codon_end_ix = codon_start_ix + 3
                    codon = sequence[codon_start_ix:codon_end_ix]
                    codon_attr = seq_attribution[codon_start_ix:codon_end_ix]
                    codon_attr_2 = seq_attribution_2[codon_start_ix:codon_end_ix]
                    codon_attr_3 = seq_attribution_3[codon_start_ix:codon_end_ix]

                    if codon in IVYWREL_codons:
                        all_attributions.append(codon_attr)
                        all_attributions_2.append(codon_attr_2)
                        all_attributions_3.append(codon_attr_3)
                    
        return (
            np.mean(all_attributions, axis=0),
            np.mean(all_attributions_2, axis=0),
            np.mean(all_attributions_3, axis=0),
        )


def main():
    import json
    from .alphabet import ALPHABET_DNA
    from .model import conv1d_densenet_regression_model
    from .load_sequences import (
        SpeciesSequence, 
        load_growth_temperatures,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    db_path = os.path.join(os.getcwd(), 'data/condensed_traits/db/seq.db')
    output_path = os.path.join(
        os.getcwd(), 'data/condensed_traits/IVYWREL_avg_attributions.csv')

    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    TEST_SET_QUERY = """
    select species_taxid from train_test_split
    where in_test_set = 1
    """
    test_set_species_df = pd.read_sql(TEST_SET_QUERY, engine)
    species_taxids = [
        int(s) 
        for s in test_set_species_df['species_taxid'].values
    ]

    run_id = 'run_yb64o'
    model_path = os.path.join(os.getcwd(), f'saved_models/{run_id}/model.h5')
    metadata_path = os.path.join(os.getcwd(), f'saved_models/{run_id}/metadata.json')

    with open(metadata_path) as f:
        metadata = json.load(f)

    model = conv1d_densenet_regression_model(
        alphabet_size=len(metadata['alphabet']), 
        growth_rate=metadata['growth_rate'],
        n_layers=metadata['n_layers'],
        kernel_sizes=metadata['kernel_sizes'],
        dilation_rates=metadata['dilation_rates'],
        l2_reg=metadata['l2_reg'],
        dropout=metadata['dropout'],
    )
    model.load_weights(model_path)

    logger.info('Computing IVYWREL average attribution')

    attr_helper = AttributionHelper(model)

    temperatures, mean, std = load_growth_temperatures(engine)

    batch_size = 64
    seed = metadata['seed']
    max_sequence_length = 5001  # divisible by 3

    data = []
    for i, species_taxid in enumerate(species_taxids):
        logger.info(f'Processing species {species_taxid} ({i+1} / {len(species_taxids)})')

        species_seq = SpeciesSequence(
            engine, 
            species_taxid=species_taxid,
            batch_size=batch_size, 
            temperatures=temperatures,
            mean=mean,
            std=std,
            alphabet=ALPHABET_DNA, 
            max_sequence_length=max_sequence_length,
            random_seed=seed,
        )

        attr_1, attr_2, attr_3 = attr_helper.compute_IVYWREL_average_attribution(species_seq)
        data.append([
            species_taxid,
            attr_1[0],
            attr_1[1],
            attr_1[2],
            np.mean(attr_1),
            attr_2[0],
            attr_2[1],
            attr_2[2],
            np.mean(attr_2),
            attr_3[0],
            attr_3[1],
            attr_3[2],
            np.mean(attr_3),
        ])

    output_df = pd.DataFrame(data, columns=[
        'species_taxid', 
        'attr_1_c1', 
        'attr_1_c2', 
        'attr_1_c3', 
        'mean_1',
        'attr_2_c1', 
        'attr_2_c2', 
        'attr_2_c3', 
        'mean_2',
        'attr_3_c1', 
        'attr_3_c2', 
        'attr_3_c3', 
        'mean_3',
    ])
    output_df.to_csv(output_path, index=False)

    logger.info('DONE')


if __name__ == '__main__':
    main()
