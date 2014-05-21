"""
This module constains a function to perform a convolution of signal with a Linear Time-Variant system.

"""
from __future__ import division

import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter

try:
    from pyfftw.interfaces.numpy_fft import rfft
except ImportError:
    from numpy.fft import rfft


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


def ir2fr(ir, fs, N=None):
    """
    Convert impulse response into frequency response. Returns single-sided RMS spectrum.
    
    :param ir: Impulser response
    :param fs: Sample frequency
    :param N: Blocks
    
    Calculates the positive frequencies using :func:`np.fft.rfft`.
    Corrections are then applied to obtain the single-sided spectrum.
    
    .. note:: Single-sided spectrum. Therefore, the amount of bins returned is either N/2 or N/2+1.
    
    """
    #ir = ir - np.mean(ir) # Remove DC component.
    
    N = N if N else ir.shape[-1]
    fr = rfft(ir, n=N) / N
    f = np.fft.rfftfreq(N, 1.0/fs)    #/ 2.0
    
    fr *= 2.0
    fr[..., 0] /= 2.0    # DC component should not be doubled.
    if not N%2: # if not uneven
        fr[..., -1] /= 2.0 # And neither should fs/2 be.
    
    #f = np.arange(0, N/2+1)*(fs/N)
    
    return f, fr


def decibel_to_neper(decibel):
    """
    Convert decibel to neper.
    
    :param decibel: Value in decibel (dB).
    
    The conversion is done according to
    
    .. math :: \\mathrm{dB} = \\frac{\\log{10}}{20} \\mathrm{Np}
    
    """
    return np.log(10.0) / 20.0  * decibel


def neper_to_decibel(neper):
    """
    Convert neper to decibel.
    
    :param neper: Value in neper (Np).
    
    The conversion is done according to

    .. math :: \\mathrm{Np} = \\frac{20}{\\log{10}} \\mathrm{dB}
    """
    return 20.0 / np.log(10.0) * neper