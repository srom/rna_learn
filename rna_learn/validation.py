import logging

import scipy
import numpy as np
import pandas as pd
import tensorflow as tf

from .alphabet import ALPHABET_DNA
from .transform import normalize
from .load_sequences import (
    SpeciesSequence,
    load_growth_temperatures,
)


TEST_SET_QUERY = """
select species_taxid from train_test_split
where in_test_set = 1
"""

GROWTH_TMP_QUERY = """
select growth_tmp from species_traits
where species_taxid = ?
"""

logger = logging.getLogger(__name__)


def validate_model_on_test_set(
    engine, 
    model,
    batch_size,
    max_queue_size=10,
    max_sequence_length=None,
):
    test_set_species_df = pd.read_sql(TEST_SET_QUERY, engine)
    species_taxids = [
        int(s) 
        for s in test_set_species_df['species_taxid'].values
    ]
    return validate_model_for_species(
        engine, 
        model, 
        species_taxids,
        batch_size,
        max_queue_size=max_queue_size,
        max_sequence_length=max_sequence_length,
    )


def validate_model_for_species(
    engine, 
    model, 
    species_taxids,
    batch_size,
    max_queue_size=10,
    max_sequence_length=None,
):
    data = []
    n_species = len(species_taxids)
    temperature_range = np.arange(-35, 145, 0.5)
    all_temperatures, mean, std = load_growth_temperatures(engine)
    for i, species_taxid in enumerate(species_taxids):
        logger.info(f'Validating specie {species_taxid} ({i+1:,} / {n_species:,})')
        species_data = process_species(
            engine, 
            model, 
            species_taxid,
            batch_size,
            temperature_range,
            all_temperatures,
            mean, 
            std,
            max_queue_size=max_queue_size,
            max_sequence_length=max_sequence_length,
        )
        data.append(species_data)

    columns = [
        'species_taxid',
        'growth_tmp_actual',
        'growth_tmp_prediction',
        'growth_tmp_std',
    ]
    output_df = pd.DataFrame(data, columns=columns)
    return output_df.set_index('species_taxid')


def process_species(
    engine, 
    model, 
    species_taxid,
    batch_size,
    temperature_range,
    all_temperatures,
    mean,
    std,
    max_queue_size=10,
    max_sequence_length=None,
):
    seq = SpeciesSequence(
        engine, 
        species_taxid=species_taxid,
        batch_size=batch_size, 
        alphabet=ALPHABET_DNA, 
        temperatures=all_temperatures,
        mean=mean, 
        std=std,
        max_sequence_length=max_sequence_length,
    )
    seq_length = len(seq)
    n_sequences = len(seq.rowids)
    queue = tf.keras.utils.OrderedEnqueuer(seq)
    try:
        queue.start(max_queue_size=max_queue_size)
        batch_generator = queue.get()
        return _process_specie_inner(
            engine, 
            model, 
            species_taxid, 
            temperature_range,
            batch_generator, 
            seq_length,
            n_sequences,
            mean,
            std,
        )
    finally:
        if queue.is_running():
            queue.stop()


def _process_specie_inner(
    engine, 
    model, 
    species_taxid, 
    temperature_range,
    batch_generator, 
    seq_length,
    n_sequences,
    mean,
    std,
):
    growth_tmp = pd.read_sql(
        GROWTH_TMP_QUERY, 
        engine,
        params=(species_taxid,),
    ).iloc[0]['growth_tmp']
    temperature_range_norm = normalize(temperature_range, mean, std)
    range_len = len(temperature_range)

    cursor = 0
    log_probabilities = np.zeros((n_sequences, range_len))
    for i, (x, _, _) in enumerate(batch_generator):
        if i == seq_length:
            break

        batch_size = len(x)
        a = cursor
        b = cursor + batch_size

        out_dist = model(x)
        for j, t in enumerate(temperature_range_norm):
            log_probs = out_dist.log_prob(t).numpy()
            log_probabilities[a:b, j] = log_probs

        cursor += batch_size

    probs_u = np.exp(scipy.special.logsumexp(log_probabilities, axis=0))
    probs = probs_u / np.sum(probs_u)

    prediction = np.average(temperature_range, weights=probs)
    variance = np.average(
        [(t - prediction)**2 for t in temperature_range], 
        weights=probs,
    )
    std = np.sqrt(variance)

    return [
        species_taxid,
        growth_tmp,
        np.round(prediction, 2),
        np.round(std, 2),
    ]


def main():
    import os
    from sqlalchemy import create_engine
    from .model import (
        conv1d_densenet_regression_model,
        compile_regression_model,
    )

    db_path = os.path.join(
        os.getcwd(), 
        'data/condensed_traits/db/seq.db',
    )
    engine = create_engine(f'sqlite+pysqlite:///{db_path}')

    tf.random.set_seed(444)

    alphabet = ALPHABET_DNA
    model = conv1d_densenet_regression_model(
        alphabet_size=len(alphabet),
        growth_rate=5,
        n_layers=2,
        kernel_sizes=[3, 2],
        masking=True,
    )
    compile_regression_model(model, learning_rate=1e-4)

    output_df = validate_model_for_species(
        engine, 
        model, 
        species_taxids=[7, 14],
        batch_size=64,
        max_sequence_length=5000,
        max_queue_size=20,
    )


if __name__ == '__main__':
    main()
