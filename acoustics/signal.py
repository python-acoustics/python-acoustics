"""
This module constains a function to perform a convolution of signal with a Linear Time-Variant system.

"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter, freqz, filtfilt

import acoustics.octave
from acoustics.octave import REFERENCE


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


class Band(object):
    """
    Frequency band object.
    """
    
    def __init__(self, center, lower, upper, bandwidth=None):
        
        self.center = center
        self.lower = lower
        self.upper = upper
        self.bandwidth = bandwidth if bandwidth is not None else self.upper - self.lower
    
    def __str__(self):
        return str(self.center)
    
    def __repr__(self):
        return "Band({})".format(str(self.center))
    
class Frequencies(object):
    """
    Object describing frequency bands.
    """
    
    def __init__(self, center, lower, upper, bandwidth=None):
        
        self.center = center
        """
        Center frequencies.
        """
        
        self.lower = lower
        """
        Lower frequencies.
        """
        
        self.upper = upper
        """
        Upper frequencies.
        """
        
        self.bandwidth = bandwidth if bandwidth is not None else self.upper - self.lower
        """
        Bandwidth.
        """
    
    def __iter__(self):
        for i in range(len(self.center)):
            yield Band(self.center[i], self.lower[i], self.upper[i], self.bandwidth[i])
    
    def __len__(self):
        return len(self.center)
    
    def __str__(self):
        return str(self.center)
    
    def __repr__(self):
        return "Frequencies({})".format(str(self.center))
    
    
class EqualBand(Frequencies):
    """
    Equal bandwidth spectrum. Generally used for narrowband data.
    """
    
    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, bandwidth=None):
        """
        Parameters.
        
        :param center: Vector of center frequencies.
        :param fstart: First center frequency.
        :param fstop: Last center frequency.
        :param nbands: Amount of frequency bands.
        :param bandwidth: Bandwidth of bands.
        
        """
        
        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
                nbands = 1

            u = np.unique(np.diff(center))
            if len(u)==1:
                bandwidth = u
            else:
                raise ValueError("Given center frequencies are not equally spaced.")
            fstart = center[0] #- bandwidth/2.0
            fstop = center[-1] #+ bandwidth/2.0
        elif fstart and fstop and nbands:
            bandwidth = (fstop - fstart) / (nbands-1)
        elif fstart and fstop and bandwidth:
            nbands = round((fstop - fstart) / bandwidth) + 1
        elif fstart and bandwidth and nbands:
            fstop = fstart + nbands * bandwidth
        elif fstop and bandwidth and nbands:
            fstart = fstop - (nbands-1) * bandwidth
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart, fstop, bandwidth.")
        
        center = fstart + np.arange(0, nbands) * bandwidth # + bandwidth/2.0
        upper  = fstart + np.arange(0, nbands) * bandwidth + bandwidth/2.0
        lower  = fstart + np.arange(0, nbands) * bandwidth - bandwidth/2.0
        
        super(EqualBand, self).__init__(center, lower, upper, bandwidth)
        
    
    def __repr__(self):
        return "EqualBand({})".format(str(self.center))
        

    
class OctaveBand(Frequencies):
    """
    Fractional-octave band spectrum.
    """
    
    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, fraction=1, reference=REFERENCE):
        
        
        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
                nbands = 1
            fstart = center[0]
            fstop = center[-1]
        
        if fstart and fstop:
            o = acoustics.octave.Octave(order=fraction, fmin=fstart, fmax=fstop, reference=reference)
            center = o.center
            nbands = len(center)
        
        if fstart and nbands:
            nstart = acoustics.octave.band_of_frequency(fstart, order=fraction, ref=reference)
            nstop = nstart + nbands-1
            fstop = acoustics.octave.frequency_of_band(nstop, order=fraction, ref=reference)
        elif fstop and nbands:
            nstop = acoustics.octave.band_of_frequency(fstop, order=fraction, ref=reference)
            nstart = nstop - nbands+1
            fstart = acoustics.octave.band_of_frequency(nstart, order=fraction, ref=reference)
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart and/or fstop.")    
        
        
        center = acoustics.octave.Octave(order=fraction, 
                                       fmin=fstart, 
                                       fmax=fstop, 
                                       reference=reference).center
    
        upper = acoustics.octave.upper_frequency(center, fraction)
        lower = acoustics.octave.lower_frequency(center, fraction)
        bandwidth = upper - lower

        super(OctaveBand, self).__init__(center, lower, upper, bandwidth)
        
        self.fraction = fraction
        """
        Fraction of fractional-octave filter.
        """
        
        self.reference = reference
        """
        Reference center frequency.
        """
        
    def __repr__(self):
        return "OctaveBand({})".format(str(self.center))
    
  

def integrate_bands(data, a, b):
    """
    Reduce frequency resolution of power spectrum. Merges frequency bands by integration.
    
    :param data: Vector with narrowband powers.
    :param a: Instance of :class:`Frequencies`.
    :param b: Instance of :class:`Frequencies`.
    
    .. note:: Needs rewriting so that the summation goes over axis=1.
    
    """
    
    try:
        if b.fraction%a.fraction:
            raise NotImplementedError("Non-integer ratio of fractional-octaves are not supported.")
    except AttributeError:
        pass
    
    lower, _ = np.meshgrid(b.lower, a.center)
    upper, _ = np.meshgrid(b.upper, a.center)
    _, center= np.meshgrid(b.center, a.center)

    return ((lower < center) * (center <= upper) * data[:,None]).sum(axis=0)



  
class Filterbank(object):
    """
    Fractional-Octave filter bank.
    
    
    .. note:: For high frequencies the filter coefficients are wrong for low frequencies. Therefore, to improve the response for lower frequencies the signal should be downsampled. Currently, there is no easy way to do so within the Filterbank.
    
    """
    
    def __init__(self, frequencies, sample_frequency=44100, order=3):
        
        
        self.frequencies = frequencies
        """
        Frequencies object.
        
        See also :class:`Frequencies` and subclasses.
        
        .. note:: A frequencies object should have the attributes center, lower and upper.
        
        """
        
        self.order = order
        """
        Filter order of Butterworth filter.
        """
        
        self.sample_frequency = sample_frequency
        """
        Sample frequency.
        """
    
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
    def filters(self):
        """
        Filters this filterbank consists of.
        """
        order = self.order
        filters = list()
        nyq = self.sample_frequency / 2.0
        return [ butter(order, [lower/nyq, upper/nyq], btype='band', analog=False) for lower, upper in zip(self.frequencies.lower, self.frequencies.upper) ]

    def lfilter(self, signal):
        """
        Filter signal with filterbank.
        
        .. note:: This function uses :func:`scipy.signal.lfilter`.
        """
        return [ lfilter(f[0], f[1], signal) for f in self.filters ]

    def filtfilt(self, signal):
        """
        Filter signal with filterbank.
        Returns a list consisting of a filtered signal per filter.
        
        .. note:: This function uses :func:`scipy.signal.filtfilt` and therefore has a zero-phase response.
        """
        return [ filtfilt(f[0], f[1], signal) for f in self.filters ]
            
    def power(self, signal):
        """
        Power per band in signal.
        """
        filtered = self.filtfilt(signal)
        return np.array([(x**2.0).sum()/len(x) / bw for x, bw in zip(filtered, self.frequencies.bandwidth)])
    
    def plot_response(self, filename=None):
        """
        Plot frequency response.
        
        .. note:: The follow phase response is obtained in case :meth:`lfilter` is used. The method :meth:`filtfilt` results in a zero-phase response.
        """
        
        fs = self.sample_frequency
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for f, fc in zip(self.filters, self.frequencies.center):
            w, h = freqz(f[0], f[1], int(fs/2))#np.arange(fs/2.0))
            ax1.semilogx(w / (2.0*np.pi) * fs, 20.0 * np.log10(np.abs(h)), label=str(int(fc)))
            ax2.semilogx(w / (2.0*np.pi) * fs, np.angle(h), label=str(int(fc)))
        ax1.set_xlabel(r'$f$ in Hz')
        ax1.set_ylabel(r'$|H|$ in dB re. 1')
        ax2.set_xlabel(r'$f$ in Hz')
        ax2.set_ylabel(r'$\angle H$ in rad')
        ax1.legend(loc=5)
        ax2.legend(loc=5)
        ax1.set_ylim(-60.0, +10.0)
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def plot_power(self, signal, filename=None):
        """
        Plot power in signal.
        """
        
        f = self.frequencies.center
        p = self.power(signal)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = ax.bar(f, 20.0*np.log10(p))
        ax.set_xlabel('$f$ in Hz')
        ax.set_ylabel('$L$ in dB re. 1')
        ax.set_xscale('log')
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
        
        
#class FilterbankFFT(object):
    #"""
    #Filterbank to filter signal using FFT.
    #"""
    
    #def __init__(self, frequencies, sample_frequency=44100):
       
       #self.frequencies = frequencies
       #"""
       #Frequencies.
       
       #See also :class:`Frequencies` and subclasses.
       #"""
       #self.sample_frequency = sample_frequency
       
    
    #def power(self, signal):
        #pass
    
    #def plot_power(self, signal):
        #pass
        
        
