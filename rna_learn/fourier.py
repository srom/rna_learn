import numpy as np
import tensorflow as tf


def compute_autocorr_fft_on_signals(signals, n):
    """
    Compute the (fast) fourier transform on the autocorrelation
    of each 1D signals given as input.
    Return the n first elements of the real part of the fourier transform.
    """
    fft_shape = (n // 2) + 1 if n % 2 == 0 else (n + 1) // 2
    output_shape = (len(signals), fft_shape)
    output = np.zeros(output_shape, dtype='float64')
    for i, signal in enumerate(signals):
        output[i] = compute_autocorr_fft(signal, n)

    return output


def compute_autocorr_fft(signal, n):
    signal_autocorr = autocorr(signal)
    ft = np.fft.rfft(signal_autocorr, n=n, norm='ortho')
    return np.abs(ft)


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2 + 1:]
