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


def validate_model_on_test_set(
    engine, 
    model,
    batch_size,
    max_queue_size=10,
    max_sequence_length=None,
):
    test_set_species_df = pd.read_sql(TEST_SET_QUERY, engine)
    species_taxids = test_set_species_df['species_taxid'].values
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
    max_sequence_length=None,
    max_queue_size=10,
):
    temperature_range = np.arange(-15, 125, 0.5)
    all_temperatures, mean, std = load_growth_temperatures(engine)
    data = []
    for species_taxid in species_taxids:
        species_data = process_species(
            engine, 
            model, 
            species_taxid,
            batch_size,
            temperature_range,
            all_temperatures,
            mean, 
            std,
            max_sequence_length=None,
            max_queue_size=10,
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
    max_sequence_length=None,
    max_queue_size=10,
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
        np.round(prediction, 1),
        np.round(std, 1),
    ]
