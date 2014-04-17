"""
This module constains a function to perform a convolution of signal with a Linear Time-Variant system.

"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
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


class Frequencies(object):
    
    def __init__(self, filterbank):
        
        self._filterbank = filterbank
        self.
        
    
    @property
    def center(self):
        return self._center

    @property.setter
    def center(self, x):
        self._center = x

    @property
    def lower(self):
        return self._upper
    
    @property
    def upper(self):
        return self._upper

class Filterbank(object):
    """
    Fractional-Octave filter bank.
    """
    
    def __init__(self, order=3, sample_frequency=44100, fraction=1, center_frequencies=None, f_ref=REFERENCE_FREQUENCY):
        
        self.fraction = fraction
        """
        Bands per octave.
        """
        
        self.order = order
        """
        Filter order.
        """
        
        self.sample_frequency = sample_frequency
        
        self.frequencies = Frequencies(self)
        
        
        self.frequencies.center = center_frequencies
        
    
    @property
    def sample_frequency(self):
        """
        Sample frequency.
        """
        return self._sample_frequency
    
    @sample_frequency.setter
    def sample_frequency(self, x):
        #if x <= self.center_frequencies.max():
            #raise ValueError("Sample frequency cannot be lower than the highest center frequency.")
        self._sample_frequency = x
        
    @property
    def center_frequencies(self):
        """
        Center frequencies.
        """
        if self._center_frequencies:
            return self._center_frequencies
        #else:
            raise NotImplementedError

    @center_frequencies.setter
    def center_frequencies(self, x):
        if not np.all(np.gradient(x) > 0):
            raise ValueError("Values are not in increasing order.")
        #if not (x.max() < self.sample_frequency):
            #raise ValueError("Center frequency cannot be higher than sample frequency.")
        self._center_frequencies = x
        
    @property
    def filters(self):
        """
        Filters this filterbank consists of.
        """
        order = self.order
        filters = list()
        for f in self.center_frequencies:
            filters.append(butter(order, [], btype='band')

    def filt(self, signal):
        """
        Filter signal with filterbank.
        Returns a list consisting of a filtered signal per filter.
        """
        filters = self.filters
        
        out = list()
        
        for f in filters:
            out.append( lfilter(f.b, f.a, signal) )
            
    def power(self, signal):
        """
        Power per band in signal.
        """
        filtered = self.filt(signal)
        return [(x**2.0).sum()/len(x) for x in filtered]
    
    def plot_power(self, signal, filename=None):
        """
        Plot power in signal.
        """
        
        f = self.center_frequencies
        p = self.power(signal)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = ax.bar(f, p)
        ax.set_xlabel('$f$ in Hz')
        ax.set_ylabel('$L$ in dB re. 1')
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
        
        
        
    
    