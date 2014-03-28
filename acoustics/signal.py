"""
This module constains a function to perform a convolution of signal with a Linear Time-Variant system.

"""
from __future__ import division

import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    Butterworth bandpass filter.
    
    :param data: data
    :param lowcut: Lower cut-off frequency
    :param highcut: Upper cut-off frequency
    :param fs: Sample frequency
    :param order: Order
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def convolve(signal, ltv, mode='full'):
    """
    Perform convolution of signal with linear time-variant system ``ltv``.
    
    :param signal: Vector representing input signal :math:`u`.
    :param ltv: 2D array where each column represents an impulse response
    :param mode: 'full', 'valid', or 'same'. See :func:`np.convolve` for an explanation of the options.
    
    The convolution of two sequences is given by
    
    .. math:: \mathbf{y} = \mathbf{t} \\star \mathbf{u}
    
    This can be written as a matrix-vector multiplication
    
    .. math:: \mathbf{y} = \mathbf{T} \\cdot \mathbf{u}
   
    where :math:`T` is a Toeplitz matrix in which each column represents an impulse response. 
    In the case of a linear time-invariant (LTI) system, each column represents a time-shifted copy of the first column.
    In the time-variant case (LTV), every column can contain a unique impulse response, both in values as in size.
    
    This function assumes all impulse responses are of the same size. 
    The input matrix ``ltv`` thus represents the non-shifted version of the Toeplitz matrix.
    
    """
    
    assert(len(signal) == ltv.shape[1])
    
    n = ltv.shape[0] + len(signal) - 1                          # Length of output vector
    un = np.concatenate((signal, np.zeros(ltv.shape[0] - 1)))   # Resize input vector
    offsets = np.arange(0, -ltv.shape[0], -1)                   # Offsets for impulse responses
    Cs = spdiags(ltv, offsets, n, n)                            # Sparse representation of IR's.
    out = Cs.dot(un)                                            # Calculate dot product.

    if mode=='full':
        return out
    elif mode=='same':
        start = ltv.shape[0]/2 - 1 + ltv.shape[0]%2
        stop = len(signal) + ltv.shape[0]/2 - 1 + ltv.shape[0]%2
        return out[start:stop]
    elif mode=='valid':
        length = len(signal) - ltv.shape[0]
        start = ltv.shape[0] - 1
        stop = len(signal) 
        return out[start:stop]

