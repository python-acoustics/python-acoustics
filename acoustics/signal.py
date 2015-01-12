"""
Signal
======

The signal module constains all kinds of signal processing related functions.

.. inheritance-diagram:: acoustics.signal


Filtering
*********

.. autofunction:: butter_bandpass_filter
.. autofunction:: convolve


Windowing
*********

.. autofunction:: window_scaling_factor
.. autofunction:: apply_window

Spectra
*******

Different types of spectra exist.

.. autofunction:: amplitude_spectrum
.. autofunction:: auto_spectrum
.. autofunction:: power_spectrum
.. autofunction:: density_spectrum

Frequency bands
***************

.. autoclass:: Band
.. autoclass:: Frequencies
.. autoclass:: EqualBand
.. autoclass:: OctaveBand
    
.. autofunction:: integrate_bands
.. autofunction:: octaves
.. autofunction:: third_octaves




Conversion
**********

.. autofunction:: decibel_to_neper
.. autofunction:: neper_to_decibel


Other
*****

.. autoclass:: Filterbank
.. autofunction:: rms
.. autofunction:: ir2fr


"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter, freqz, filtfilt

import acoustics.octave
#from acoustics.octave import REFERENCE

import acoustics.bands


try:
    from pyfftw.interfaces.numpy_fft import rfft
except ImportError:
    from numpy.fft import rfft

def bandpass_filter(lowcut, highcut, fs, order=3):
    """Band-pass filter.
    
    :param lowcut: Lower cut-off frequency
    :param highcut: Upper cut-off frequency
    :param fs: Sample frequency
    :param order: Filter order
    
    A Butterworth filter (see :func:`scipy.signal.butter`) is used.
    
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass(signal, lowcut, highcut, fs, order=3):
    """Filter signal with band-pass filter.
    
    :param signal: Signal
    :param lowcut: Lower cut-off frequency
    :param highcut: Upper cut-off frequency
    :param fs: Sample frequency
    :param order: Filter order
    
    .. seealso:: :func:`bandpass_filter`
    
    """
    #nyq = 0.5 * fs
    #low = lowcut / nyq
    #high = highcut / nyq
    #b, a = butter(order, [low, high], btype='band')
    b, a = bandpass_filter(lowcut, highcut, fs, order)
    return lfilter(b, a, signal)
    
    
def lowpass(signal, cutoff, fs, order=3):
    """Filter signal with low-pass filter.
    
    :param signal: Signal
    :param fs: Sample frequency
    :param cutoff: Cut-off frequency
    :param order: Filter order
    
    A Butterworth filter (see :func:`scipy.signal.butter`) is used.
    
    """
    b, a = butter(order, cutoff/(fs/2.0), btype='low')
    return lfilter(b, a, signal)
    
    
def highpass(signal, cutoff, fs, order=3):
    """Filter signal with low-pass filter.
    
    :param signal: Signal
    :param fs: Sample frequency
    :param cutoff: Cut-off frequency
    :param order: Filter order
    
    A Butterworth filter (see :func:`scipy.signal.butter`) is used.
    
    """
    b, a = butter(order, cutoff/(fs/2.0), btype='high')
    return lfilter(b, a, signal)


def octave_filter(center, fs, fraction, order=3):
    """Fractional-octave band-pass filter.
    
    :param center: Centerfrequency of fractional-octave band.
    :param fs: Sample frequency
    :param fraction: Fraction of fractional-octave band.
    :param order: Filter order
    
    .. seealso:: :func:`bandpass_filter`
    
    """
    ob = OctaveBand(center=center, fraction=fraction)
    return bandpass_filter(ob.lower[0], ob.upper[0], fs, order)
    

def octavepass(signal, center, fs, fraction, order=3):
    """Filter signal with fractional-octave bandpass filter.
    
    :param signal: Signal
    :param center: Centerfrequency of fractional-octave band.
    :param fs: Sample frequency
    :param fraction: Fraction of fractional-octave band.
    :param order: Filter order
    
    .. seealso:: :func:`octave_filter`
    
    """
    b, a = octave_filter(center, fs, fraction, order)
    return lfilter(b, a, signal)
    

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
    :returns: Value in neper (Np).
    
    The conversion is done according to
    
    .. math :: \\mathrm{dB} = \\frac{\\log{10}}{20} \\mathrm{Np}
    
    """
    return np.log(10.0) / 20.0  * decibel


def neper_to_decibel(neper):
    """
    Convert neper to decibel.
    
    :param neper: Value in neper (Np).
    :returns: Value in decibel (dB).
    
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
        """
        Center frequency.
        """
        self.lower = lower
        """
        Lower frequency.
        """
        self.upper = upper
        """
        Upper frequency.
        """
        self.bandwidth = bandwidth if bandwidth is not None else self.upper - self.lower
        """
        Bandwidth.
        """
    
    def __str__(self):
        return str(self.center)
    
    def __repr__(self):
        return "Band({})".format(str(self.center))
    
    
class Frequencies(object):
    """
    Object describing frequency bands.
    """
    
    def __init__(self, center, lower, upper, bandwidth=None):
        
        self.center = np.asarray(center)
        """
        Center frequencies.
        """
        
        self.lower = np.asarray(lower)
        """
        Lower frequencies.
        """
        
        self.upper = np.asarray(upper)
        """
        Upper frequencies.
        """
        
        self.bandwidth = np.asarray(bandwidth) if bandwidth is not None else np.asarray(self.upper) - np.asarray(self.lower)
        """
        Bandwidth.
        """
    
    def __iter__(self):
        for i in range(len(self.center)):
            yield self[i]
            #yield Band(self.center[i], self.lower[i], self.upper[i], self.bandwidth[i])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [Band(self.center[i], self.lower[i], self.upper[i], self.bandwidth[i]) for i in range(*key.indices(len(self))) ]
        elif isinstance(key, np.ndarray):
            if key.dtype == 'bool':
                return [Band(self.center[i], self.lower[i], self.upper[i], self.bandwidth[i]) for i, v in enumerate(key) if v ]
            else:
                raise NotImplementedError
        else:    
            return Band(self.center[key], self.lower[key], self.upper[key], self.bandwidth[key])
    
    def __setitem__(self, key, value):
        self.center[key] = value.center
        self.lower[key] = value.lower
        self.upper[key] = value.upper
        self.bandwidth[key] = value.bandwidth
    
    def __len__(self):
        return len(self.center)
    
    def __str__(self):
        return str(self.center)
    
    def __repr__(self):
        return "Frequencies({})".format(str(self.center))
    
    def angular(self):
        """Angular center frequency in radians per second.
        """
        return 2.0 * np.pi * self.center
    
    @classmethod
    def from_bands(cls, bands):
        """Frequencies object from a list of bands.
        """
        n = len(bands)
        obj = cls(nbands=n, fstart=1000.0) # Fake values
        #center = np.empty(n)
        #lower = np.empty(n)
        #upper = np.empty(n)
        #bandwidth = np.empty(n)
        for i, band in enumerate(bands):
            obj[i] = band
            #center[i] = band.center
        return obj    
    
class EqualBand(Frequencies):
    """
    Equal bandwidth spectrum. Generally used for narrowband data.
    """
    
    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, bandwidth=None):
        """
        
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

            u = np.unique(np.diff(center).round(decimals=3))
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
    """Fractional-octave band spectrum.
    """
    
    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, fraction=1, reference=acoustics.octave.REFERENCE):
        
        
        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
                nbands = 1
            fstart = center[0]
            fstop = center[-1]
        
        if fstart and fstop:
            o = acoustics.octave.Octave(fraction=fraction, fmin=fstart, fmax=fstop, reference=reference)
            center = o.center
            nbands = len(center)
        
        if fstart and nbands:
            nstart = acoustics.octave.band_of_frequency(fstart, fraction=fraction, ref=reference)
            nstop = nstart + nbands-1
            fstop = acoustics.octave.frequency_of_band(nstop, fraction=fraction, ref=reference)
        elif fstop and nbands:
            nstop = acoustics.octave.band_of_frequency(fstop, fraction=fraction, ref=reference)
            nstart = nstop - nbands+1
            fstart = acoustics.octave.band_of_frequency(nstart, fraction=fraction, ref=reference)
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart and/or fstop.")    
        
        
        center = acoustics.octave.Octave(fraction=fraction, 
                                       fmin=fstart, 
                                       fmax=fstop, 
                                       reference=reference).center
    
        upper = acoustics.octave.upper_frequency(center, fraction)
        lower = acoustics.octave.lower_frequency(center, fraction)
        bandwidth = upper - lower

        super(OctaveBand, self).__init__(center, lower, upper, bandwidth)
        
        self.fraction = fraction
        """Fraction of fractional-octave filter.
        """
        
        self.reference = reference
        """Reference center frequency.
        """
        
    def __repr__(self):
        return "OctaveBand({})".format(str(self.center))


def ms(x):
    """Mean value of signal `x` squared.
    
    :param x: Dynamic quantity.
    :returns: Mean squared of `x`.
    
    """
    return (np.abs(x)**2.0).mean()
  
def rms(x):
    """Root mean squared of signal `x`.
    
    :param x: Dynamic quantity.
    
    .. math:: x_{rms} = lim_{T \\to \\infty} \\sqrt{\\frac{1}{T} \int_0^T |f(x)|^2 \\mathrm{d} t }
    
    :seealso: :func:`ms`.
    
    """
    return np.sqrt(ms(x))
    
def normalise(y, x=None):
    """Normalise power in y to a (standard normal) white noise signal.
    
    Optionally normalise to power in signal `x`.
    
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    #return y * np.sqrt( (np.abs(x)**2.0).mean() / (np.abs(y)**2.0).mean() )
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt( x / ms(y) )
    #return y * np.sqrt( 1.0 / (np.abs(y)**2.0).mean() )
    
    ## Broken? Caused correlation in auralizations....weird!
    
    


def window_scaling_factor(window, axis=-1):
    """
    Calculate window scaling factor.
    
    :param window: Window.
    
    When analysing broadband (filtered noise) signals it is common to normalise
    the windowed signal so that it has the same power as the un-windowed one.

    .. math:: S = \\sqrt{\\frac{\\sum_{i=0}^N w_i^2}{N}}
    
    """
    return np.sqrt((window*window).mean(axis=axis))


def apply_window(x, window):
    """
    Apply window to signal.
    
    :param x: Instantaneous signal :math:`x(t)`.
    :param window: Vector representing window.
    
    :returns: Signal with window applied to it.
    
    .. math:: x_s(t) = x(t) / S
    
    where :math:`S` is the window scaling factor. See also :func:`window_scaling_factor`.
    
    """
    s = window_scaling_factor(window) # Determine window scaling factor.
    n = len(window)
    windows = x//n  # Amount of windows.
    x = x[0:windows*n] # Truncate final part of signal that does not fit.
    #x = x.reshape(-1, len(window)) # Reshape so we can apply window.
    y = np.tile(window, windows)
    
    return x * y / s


def amplitude_spectrum(x, fs, N=None):
    """
    Amplitude spectrum of instantaneous signal :math:`x(t)`.
    
    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.
    
    The amplitude spectrum gives the amplitudes of the sinusoidal the signal is built 
    up from, and the RMS (root-mean-square) amplitudes can easily be found by dividing 
    these amplitudes with :math:`\\sqrt{2}`.
    
    The amplitude spectrum is double-sided.
    
    """
    N = N if N else x.shape[-1]
    fr = np.fft.fft(x, n=N) / N
    f = np.fft.fftfreq(N, 1.0/fs)
    return np.fft.fftshift(f), np.fft.fftshift(fr, axes=[-1])


def auto_spectrum(x, fs, N=None):
    """
    Auto-spectrum of instantaneous signal :math:`x(t)`.
    
    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.
    
    The auto-spectrum contains the squared amplitudes of the signal. Squared amplitudes
    are used when presenting data as it is a measure of the power/energy in the signal.

    .. math:: S_{xx} (f_n) = \\overline{X (f_n)} \\cdot X (f_n)

    The auto-spectrum is double-sided. For a single-sided autospectrum, see :func:`single_sided_auto_spectrum`.
    
    """
    f, a = amplitude_spectrum(x, fs, N=N)
    return f, (a*a.conj()).real


def power_spectrum(x, fs, N=None):
    """
    Power spectrum of instantaneous signal :math:`x(t)`.
    
    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.
    
    The power spectrum, or single-sided autospectrum, contains the squared RMS amplitudes of the signal.
    
    A power spectrum is a spectrum with squared RMS values. The power spectrum is 
    calculated from the autospectrum of the signal.
    """
    N = N if N else x.shape[-1]
    f, a = auto_spectrum(x, fs, N=N)
    a = a[..., N//2:]
    f = f[..., N//2:]
    a *= 2.0
    a[..., 0] /= 2.0    # DC component should not be doubled.
    if not N%2: # if not uneven
        a[..., -1] /= 2.0 # And neither should fs/2 be.
    return f, a
  

def phase_spectrum(x, fs, N=None):
    """
    Phase spectrum of instantaneous signal :math:`x(t)`.
    
    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.
    
    This function returns the phase of a single-sided (amplitude) spectrum.
    
    """
    f, a = amplitude_spectrum(x, fs, N)
    f = f[N//2:]
    a = np.angle(f[N//2:])
    return f, a

#def power_and_phase_spectrum(x, fs, N=None):
    #"""
    #Power spectrum and phase of instantaneous signal :math:`x(t)`.
    
    #:param x: Instantaneous signal :math:`x(t)`.
    #:param fs: Sample frequency :math:`f_s`.
    #:param N: Amount of FFT bins.
    
    #Often one is interested in both the power spectrum and the phase. This function returns the power and a single-sided phase spectrum.
    #For an explanation of the power spectrum, see :func:`power_spectrum`.
    #"""
    
    
    #returns f, power, phase
    

  
def density_spectrum(x, fs, N=None):
    """
    Density spectrum of instantaneous signal :math:`x(t)`.
    
    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.
    
    A density spectrum considers the amplitudes per unit frequency. 
    Density spectra are used to compare spectra with different frequency resolution as the 
    magnitudes are not influenced by the resolution because it is per Hertz. The amplitude 
    spectra on the other hand depend on the chosen frequency resolution.
    
    """
    N = N if N else x.shape[-1]
    fr = np.fft.fft(x, n=N) / fs
    f = np.fft.fftfreq(N, 1.0/fs)
    return np.fft.fftshift(f), np.fft.fftshift(fr)
    
#def auto_density_spectrum(x, fs, N=None):
    #"""
    #Auto density spectrum of instantaneous signal :math:`x(t)`.
    #"""
    #f, d = density_spectrum(x, fs, N=N)
    #return f, (d*d.conj()).real

#def power_density_spectrum(x, fs, N=None):
    #"""
    #Power density spectrum.
    #"""
    #N = N if N else x.shape[-1]
    #f, a = auto_density_spectrum(x, fs, N=N)
    #a = a[N//2:]
    #f = f[N//2:]
    #a *= 2.0
    #a[..., 0] /= 2.0    # DC component should not be doubled.
    #if not N%2: # if not uneven
        #a[..., -1] /= 2.0 # And neither should fs/2 be.
    #return f, a
    

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

    return ((lower < center) * (center <= upper) * data[...,None]).sum(axis=-2)


def octaves(p, fs, density=False):
    """Calculate level per 1/1-octave.
    
    :param p: Instantaneous pressure :math:`p(t)`.
    :param fs: Sample frequency.
    :param density: Power density instead of power.
    
    .. note:: Based on power spectrum (FFT)
    
    .. seealso:: :attr:`acoustics.bands.OCTAVE_CENTER_FREQUENCIES`
    
    """
    fob = OctaveBand(acoustics.bands.OCTAVE_CENTER_FREQUENCIES, fraction=1)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= (fob.bandwidth/fnb.bandwidth)
    level = 10.0*np.log10(power)
    return fob, level


def third_octaves(p, fs, density=False):
    """Calculate level per 1/3-octave.
    
    :param p: Instantaneous pressure :math:`p(t)`.
    :param fs: Sample frequency.
    :param density: Power density instead of power.
    
    .. note:: Based on power spectrum (FFT)
    
    .. seealso:: :attr:`acoustics.bands.THIRD_OCTAVE_CENTER_FREQUENCIES`
    
    """
    fob = OctaveBand(acoustics.bands.THIRD_OCTAVE_CENTER_FREQUENCIES, fraction=3)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= (fob.bandwidth/fnb.bandwidth)
    level = 10.0*np.log10(power)
    return fob, level


def fractional_octaves(p, fs, start=5.0, stop=16000.0, fraction=3, density=False):
    """Calculate level per 1/N-octave where N is `fraction`.
    
    .. note:: Based on power spectrum (FFT)
    
    """
    fob = OctaveBand(fstart=start, fstop=stop, fraction=fraction)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= (fob.bandwidth/fnb.bandwidth)
    level = 10.0*np.log10(power)
    return fob, level

#def plot_signal(x, fs, bands):
    #"""
    #Plot signal ``x`` with sample frequency ``fs`` and frequency bands as specified in ``bands``.

    
    #fig = plt.figure():

  
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
        fs = self.sample_frequency
        return ( bandpass_filter(lower, upper, fs, order=self.order) for lower, upper in zip(self.frequencies.lower, self.frequencies.upper) )
        
        #order = self.order
        #filters = list()
        #nyq = self.sample_frequency / 2.0
        #return ( butter(order, [lower/nyq, upper/nyq], btype='band', analog=False) for lower, upper in zip(self.frequencies.lower, self.frequencies.upper) )

    def lfilter(self, signal):
        """
        Filter signal with filterbank.
        
        .. note:: This function uses :func:`scipy.signal.lfilter`.
        """
        return ( lfilter(b, a, signal) for b, a in self.filters )

    def filtfilt(self, signal):
        """
        Filter signal with filterbank.
        Returns a list consisting of a filtered signal per filter.
        
        .. note:: This function uses :func:`scipy.signal.filtfilt` and therefore has a zero-phase response.
        """
        return ( filtfilt(b, a, signal) for b, a in self.filters )
            
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
        
        
  
def isolate(signals):
    """Isolate signals.
    
    :param signals: Array of shape N x M where N is the amount of samples and M the amount of signals. Thus, each column is a signal.
    :returns: Array of isolated signals. Each column is a signal.
    
    Isolate signals using Singular Value Decomposition.
    
    """
    x = np.asarray(signals)
    
    W, s, v = np.linalg.svd( (np.tile( (x*x).sum(axis=0), (len(x), 1) ) * x).dot(x.T) )
    return v.T  
  
  
def zero_crossings(data):
    """
    Determine the positions of zero crossings in `data`.
    
    :param data: Vector
    
    :returns: Vector with indices of samples *before* the zero crossing.
    
    """
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
