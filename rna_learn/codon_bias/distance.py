import numpy as np
from scipy.stats import entropy


CODON_BIAS_INDICES = [
    [0, 1],
    [2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19],
    [20, 21, 22],
    [23, 24],
    [25, 26],
    [27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
    [37, 38],
    [39, 40],
    [41, 42, 43, 44],
    [45, 46, 47, 48],
    [49, 50, 51, 52],
    [53, 54, 55],
    [56, 57],
    [58, 59],
    [60, 61],
]

CODONS_BIAS_CODONS = [
    'AAA', 'AAG',
    'AAT', 'AAC',
    'ACT', 'ACC', 'ACA', 'ACG',
    'CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG',
    'TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC',
    'ATT', 'ATC', 'ATA',
    'CAA', 'CAG',
    'CAC', 'CAT',
    'CCT', 'CCC', 'CCA', 'CCG',
    'CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG',
    'GAA', 'GAG',
    'GAT', 'GAC',
    'GCT', 'GCC', 'GCA', 'GCG',
    'GGT', 'GGC', 'GGA', 'GGG',
    'GTT', 'GTC', 'GTA', 'GTG',
    'TAA', 'TAG', 'TGA',
    'TAC', 'TAT',
    'TGC', 'TGT',
    'TTC', 'TTT',
]

CODON_BIAS_RATIOS = [
    f'{codon}_ratio' 
    for codon in CODONS_BIAS_CODONS
]

CODON_TO_AA_TABLE = { 
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W', 
}


def kl_divergence(p, q, axis=0):
    return entropy(p, q, axis=axis)


def jensen_shannon_distance(p, q, axis=0):
    m = (p + q) / 2
    a = kl_divergence(p, m, axis)
    b = kl_divergence(q, m, axis)
    return np.sqrt(
        np.clip(
            (a + b) / 2,
            a_min=0,
            a_max=None,
        )
    )


def compute_codon_bias_distance(X, y):
    """
    Compute distance between a target codon bias vector y 
    and a matrix of other codon biases.
    """
    if len(X.shape) == 1:
        X_m = X[np.newaxis, :]
        single_output = True
    else:
        X_m = X
        single_output = False

    if len(y.shape) == 1:
        Y = y[np.newaxis, :]
    elif y.shape[0] > 1:
        raise ValueError('Vector y must have shape (N,) or (1, N)')
    else:
        Y = y

    n_cds = len(X_m)

    distance_vector = np.zeros((n_cds,))

    cds_ix = list(range(n_cds))

    if n_cds > 1:
        Y_m = duplicate_rows(Y, 0, n_duplicates=n_cds - 1)
    else:
        Y_m = Y

    d = np.zeros((n_cds, len(CODON_BIAS_INDICES)))
    for aa_idx, idx in enumerate(CODON_BIAS_INDICES):
        p = X_m[:, idx]
        q = Y_m[:, idx]

        d[cds_ix, aa_idx] = jensen_shannon_distance(p, q, axis=1)

    distance_vector = np.mean(d, axis=1)

    if single_output:
        return distance_vector[0]
    else:
        return distance_vector


def duplicate_rows(a, ix, n_duplicates=1):
    return np.insert(a, [ix + 1] * n_duplicates, a[ix], axis=0)


def get_aa_indices():
    return CODON_BIAS_INDICES
