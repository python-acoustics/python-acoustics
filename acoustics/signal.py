"""
Signal
======

The signal module constains all kinds of signal processing related functions.

.. inheritance-diagram:: acoustics.signal


Filtering
*********

.. autoclass:: Filterbank
.. autofunction:: bandpass_filter
.. autofunction:: octave_filter
.. autofunction:: bandpass
.. autofunction:: lowpass
.. autofunction:: highpass
.. autofunction:: octavepass
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
.. autofunction:: angle_spectrum
.. autofunction:: phase_spectrum

Frequency bands
***************

.. autoclass:: Frequencies
.. autoclass:: EqualBand
.. autoclass:: OctaveBand

.. autofunction:: integrate_bands
.. autofunction:: octaves
.. autofunction:: third_octaves


Hilbert transform
*****************

.. autofunction:: amplitude_envelope
.. autofunction:: instantaneous_phase
.. autofunction:: instantaneous_frequency


Conversion
**********

.. autofunction:: decibel_to_neper
.. autofunction:: neper_to_decibel


Other
*****

.. autofunction:: isolate
.. autofunction:: zero_crossings
.. autofunction:: rms
.. autofunction:: ms
.. autofunction:: normalize
.. autofunction:: ir2fr
.. autofunction:: wvd

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter, freqz, filtfilt, sosfilt

import acoustics.octave
#from acoustics.octave import REFERENCE

import acoustics.bands
from scipy.signal import hilbert
from acoustics.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
from acoustics.standards.iec_61672_1_2013 import (NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                                                  NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)

try:
    from pyfftw.interfaces.numpy_fft import rfft
except ImportError:
    from numpy.fft import rfft


def bandpass_filter(lowcut, highcut, fs, order=8, output='sos'):
    """Band-pass filter.

    :param lowcut: Lower cut-off frequency
    :param highcut: Upper cut-off frequency
    :param fs: Sample frequency
    :param order: Filter order
    :param output: Output type. {'ba', 'zpk', 'sos'}. Default is 'sos'. See also :func:`scipy.signal.butter`.
    :returns: Returned value depends on `output`.

    A Butterworth filter is used.

    .. seealso:: :func:`scipy.signal.butter`.

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    output = butter(order / 2, [low, high], btype='band', output=output)
    return output


def bandpass(signal, lowcut, highcut, fs, order=8, zero_phase=False):
    """Filter signal with band-pass filter.

    :param signal: Signal
    :param lowcut: Lower cut-off frequency
    :param highcut: Upper cut-off frequency
    :param fs: Sample frequency
    :param order: Filter order
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    A Butterworth filter is used. Filtering is done with second-order sections.

    .. seealso:: :func:`bandpass_filter` for the filter that is used.

    """
    sos = bandpass_filter(lowcut, highcut, fs, order, output='sos')
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def bandstop(signal, lowcut, highcut, fs, order=8, zero_phase=False):
    """Filter signal with band-stop filter.

    :param signal: Signal
    :param lowcut: Lower cut-off frequency
    :param highcut: Upper cut-off frequency
    :param fs: Sample frequency
    :param order: Filter order
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    """
    return lowpass(signal, lowcut, fs, order=(order // 2), zero_phase=zero_phase) + highpass(
        signal, highcut, fs, order=(order // 2), zero_phase=zero_phase)


def lowpass(signal, cutoff, fs, order=4, zero_phase=False):
    """Filter signal with low-pass filter.

    :param signal: Signal
    :param fs: Sample frequency
    :param cutoff: Cut-off frequency
    :param order: Filter order
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    A Butterworth filter is used. Filtering is done with second-order sections.

    .. seealso:: :func:`scipy.signal.butter`.

    """
    sos = butter(order, cutoff / (fs / 2.0), btype='low', output='sos')
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def highpass(signal, cutoff, fs, order=4, zero_phase=False):
    """Filter signal with low-pass filter.

    :param signal: Signal
    :param fs: Sample frequency
    :param cutoff: Cut-off frequency
    :param order: Filter order
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    A Butterworth filter is used. Filtering is done with second-order sections.

    .. seealso:: :func:`scipy.signal.butter`.

    """
    sos = butter(order, cutoff / (fs / 2.0), btype='high', output='sos')
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def octave_filter(center, fs, fraction, order=8, output='sos'):
    """Fractional-octave band-pass filter.

    :param center: Centerfrequency of fractional-octave band.
    :param fs: Sample frequency
    :param fraction: Fraction of fractional-octave band.
    :param order: Filter order
    :param output: Output type. {'ba', 'zpk', 'sos'}. Default is 'sos'. See also :func:`scipy.signal.butter`.

    A Butterworth filter is used.

    .. seealso:: :func:`bandpass_filter`

    """
    ob = OctaveBand(center=center, fraction=fraction)
    return bandpass_filter(ob.lower[0], ob.upper[0], fs, order, output=output)


def octavepass(signal, center, fs, fraction, order=8, zero_phase=True):
    """Filter signal with fractional-octave bandpass filter.

    :param signal: Signal
    :param center: Centerfrequency of fractional-octave band.
    :param fs: Sample frequency
    :param fraction: Fraction of fractional-octave band.
    :param order: Filter order
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    A Butterworth filter is used. Filtering is done with second-order sections.

    .. seealso:: :func:`octave_filter`

    """
    sos = octave_filter(center, fs, fraction, order)
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def convolve(signal, ltv, mode='full'):
    r"""
    Perform convolution of signal with linear time-variant system ``ltv``.

    :param signal: Vector representing input signal :math:`u`.
    :param ltv: 2D array where each column represents an impulse response
    :param mode: 'full', 'valid', or 'same'. See :func:`np.convolve` for an explanation of the options.

    The convolution of two sequences is given by

    .. math:: \mathbf{y} = \mathbf{t} \star \mathbf{u}

    This can be written as a matrix-vector multiplication

    .. math:: \mathbf{y} = \mathbf{T} \cdot \mathbf{u}

    where :math:`T` is a Toeplitz matrix in which each column represents an impulse response.
    In the case of a linear time-invariant (LTI) system, each column represents a time-shifted copy of the first column.
    In the time-variant case (LTV), every column can contain a unique impulse response, both in values as in size.

    This function assumes all impulse responses are of the same size.
    The input matrix ``ltv`` thus represents the non-shifted version of the Toeplitz matrix.

    .. seealso:: :func:`np.convolve`, :func:`scipy.signal.convolve` and :func:`scipy.signal.fftconvolve` for convolution with LTI system.

    """
    assert (len(signal) == ltv.shape[1])

    n = ltv.shape[0] + len(signal) - 1  # Length of output vector
    un = np.concatenate((signal, np.zeros(ltv.shape[0] - 1)))  # Resize input vector
    offsets = np.arange(0, -ltv.shape[0], -1)  # Offsets for impulse responses
    Cs = spdiags(ltv, offsets, n, n)  # Sparse representation of IR's.
    out = Cs.dot(un)  # Calculate dot product.

    if mode == 'full':
        return out
    elif mode == 'same':
        start = ltv.shape[0] / 2 - 1 + ltv.shape[0] % 2
        stop = len(signal) + ltv.shape[0] / 2 - 1 + ltv.shape[0] % 2
        return out[start:stop]
    elif mode == 'valid':
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
    f = np.fft.rfftfreq(N, 1.0 / fs)  #/ 2.0

    fr *= 2.0
    fr[..., 0] /= 2.0  # DC component should not be doubled.
    if not N % 2:  # if not uneven
        fr[..., -1] /= 2.0  # And neither should fs/2 be.

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
    return np.log(10.0) / 20.0 * decibel


def neper_to_decibel(neper):
    """
    Convert neper to decibel.

    :param neper: Value in neper (Np).
    :returns: Value in decibel (dB).

    The conversion is done according to

    .. math :: \\mathrm{Np} = \\frac{20}{\\log{10}} \\mathrm{dB}
    """
    return 20.0 / np.log(10.0) * neper


class Frequencies:
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

        self.bandwidth = np.asarray(bandwidth) if bandwidth is not None else np.asarray(self.upper) - np.asarray(
            self.lower)
        """
        Bandwidth.
        """

    def __iter__(self):
        for i in range(len(self.center)):
            yield self[i]

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
            n = len(u)
            if n == 1:
                bandwidth = u
            elif n > 1:
                raise ValueError("Given center frequencies are not equally spaced.")
            else:
                pass
            fstart = center[0]  #- bandwidth/2.0
            fstop = center[-1]  #+ bandwidth/2.0
        elif fstart is not None and fstop is not None and nbands:
            bandwidth = (fstop - fstart) / (nbands - 1)
        elif fstart is not None and fstop is not None and bandwidth:
            nbands = round((fstop - fstart) / bandwidth) + 1
        elif fstart is not None and bandwidth and nbands:
            fstop = fstart + nbands * bandwidth
        elif fstop is not None and bandwidth and nbands:
            fstart = fstop - (nbands - 1) * bandwidth
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart, fstop, bandwidth.")

        center = fstart + np.arange(0, nbands) * bandwidth  # + bandwidth/2.0
        upper = fstart + np.arange(0, nbands) * bandwidth + bandwidth / 2.0
        lower = fstart + np.arange(0, nbands) * bandwidth - bandwidth / 2.0

        super(EqualBand, self).__init__(center, lower, upper, bandwidth)

    def __getitem__(self, key):
        return type(self)(center=self.center[key], bandwidth=self.bandwidth)

    def __repr__(self):
        return "EqualBand({})".format(str(self.center))


class OctaveBand(Frequencies):
    """Fractional-octave band spectrum.
    """

    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, fraction=1,
                 reference=acoustics.octave.REFERENCE):

        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
            center = np.asarray(center)
            indices = acoustics.octave.index_of_frequency(center, fraction=fraction, ref=reference)
        elif fstart is not None and fstop is not None:
            nstart = acoustics.octave.index_of_frequency(fstart, fraction=fraction, ref=reference)
            nstop = acoustics.octave.index_of_frequency(fstop, fraction=fraction, ref=reference)
            indices = np.arange(nstart, nstop + 1)
        elif fstart is not None and nbands is not None:
            nstart = acoustics.octave.index_of_frequency(fstart, fraction=fraction, ref=reference)
            indices = np.arange(nstart, nstart + nbands)
        elif fstop is not None and nbands is not None:
            nstop = acoustics.octave.index_of_frequency(fstop, fraction=fraction, ref=reference)
            indices = np.arange(nstop - nbands, nstop)
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart and/or fstop.")

        center = acoustics.octave.exact_center_frequency(None, fraction=fraction, n=indices, ref=reference)
        lower = acoustics.octave.lower_frequency(center, fraction=fraction)
        upper = acoustics.octave.upper_frequency(center, fraction=fraction)
        bandwidth = upper - lower
        nominal = acoustics.octave.nominal_center_frequency(None, fraction, indices)

        super(OctaveBand, self).__init__(center, lower, upper, bandwidth)

        self.fraction = fraction
        """Fraction of fractional-octave filter.
        """

        self.reference = reference
        """Reference center frequency.
        """

        self.nominal = nominal
        """Nominal center frequencies.
        """

    def __getitem__(self, key):
        return type(self)(center=self.center[key], fraction=self.fraction, reference=self.reference)

    def __repr__(self):
        return "OctaveBand({})".format(str(self.center))


def ms(x):
    """Mean value of signal `x` squared.

    :param x: Dynamic quantity.
    :returns: Mean squared of `x`.

    """
    return (np.abs(x)**2.0).mean()


def rms(x):
    r"""Root mean squared of signal `x`.

    :param x: Dynamic quantity.

    .. math:: x_{rms} = lim_{T \\to \\infty} \\sqrt{\\frac{1}{T} \int_0^T |f(x)|^2 \\mathrm{d} t }

    :seealso: :func:`ms`.

    """
    return np.sqrt(ms(x))


def normalize(y, x=None):
    """normalize power in y to a (standard normal) white noise signal.

    Optionally normalize to power in signal `x`.

    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    #return y * np.sqrt( (np.abs(x)**2.0).mean() / (np.abs(y)**2.0).mean() )
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt(x / ms(y))
    #return y * np.sqrt( 1.0 / (np.abs(y)**2.0).mean() )

    ## Broken? Caused correlation in auralizations....weird!


def window_scaling_factor(window, axis=-1):
    """
    Calculate window scaling factor.

    :param window: Window.

    When analysing broadband (filtered noise) signals it is common to normalize
    the windowed signal so that it has the same power as the un-windowed one.

    .. math:: S = \\sqrt{\\frac{\\sum_{i=0}^N w_i^2}{N}}

    """
    return np.sqrt((window * window).mean(axis=axis))


def apply_window(x, window):
    """
    Apply window to signal.

    :param x: Instantaneous signal :math:`x(t)`.
    :param window: Vector representing window.

    :returns: Signal with window applied to it.

    .. math:: x_s(t) = x(t) / S

    where :math:`S` is the window scaling factor.

    .. seealso:: :func:`window_scaling_factor`.

    """
    s = window_scaling_factor(window)  # Determine window scaling factor.
    n = len(window)
    windows = x // n  # Amount of windows.
    x = x[0:windows * n]  # Truncate final part of signal that does not fit.
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
    f = np.fft.fftfreq(N, 1.0 / fs)
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

    The auto-spectrum is double-sided.

    """
    f, a = amplitude_spectrum(x, fs, N=N)
    return f, (a * a.conj()).real


def power_spectrum(x, fs, N=None):
    """
    Power spectrum of instantaneous signal :math:`x(t)`.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.

    The power spectrum, or single-sided autospectrum, contains the squared RMS amplitudes of the signal.

    A power spectrum is a spectrum with squared RMS values. The power spectrum is
    calculated from the autospectrum of the signal.

    .. warning:: Does not include scaling to reference value!

    .. seealso:: :func:`auto_spectrum`

    """
    N = N if N else x.shape[-1]
    f, a = auto_spectrum(x, fs, N=N)
    a = a[..., N // 2:]
    f = f[..., N // 2:]
    a *= 2.0
    a[..., 0] /= 2.0  # DC component should not be doubled.
    if not N % 2:  # if not uneven
        a[..., -1] /= 2.0  # And neither should fs/2 be.
    return f, a


def angle_spectrum(x, fs, N=None):
    """
    Phase angle spectrum of instantaneous signal :math:`x(t)`.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.

    This function returns a single-sided wrapped phase angle spectrum.

    .. seealso:: :func:`phase_spectrum` for unwrapped phase spectrum.

    """
    N = N if N else x.shape[-1]
    f, a = amplitude_spectrum(x, fs, N)
    a = np.angle(a)
    a = a[..., N // 2:]
    f = f[..., N // 2:]
    return f, a


def phase_spectrum(x, fs, N=None):
    """
    Phase spectrum of instantaneous signal :math:`x(t)`.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency :math:`f_s`.
    :param N: Amount of FFT bins.

    This function returns a single-sided unwrapped phase spectrum.

    .. seealso:: :func:`angle_spectrum` for wrapped phase angle.

    """
    f, a = angle_spectrum(x, fs, N=None)
    return f, np.unwrap(a)


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
    f = np.fft.fftfreq(N, 1.0 / fs)
    return np.fft.fftshift(f), np.fft.fftshift(fr)


def integrate_bands(data, a, b):
    """
    Reduce frequency resolution of power spectrum. Merges frequency bands by integration.

    :param data: Vector with narrowband powers.
    :param a: Instance of :class:`Frequencies`.
    :param b: Instance of :class:`Frequencies`.

    .. note:: Needs rewriting so that the summation goes over axis=1.

    """

    try:
        if b.fraction % a.fraction:
            raise NotImplementedError("Non-integer ratio of fractional-octaves are not supported.")
    except AttributeError:
        pass

    lower, _ = np.meshgrid(b.lower, a.center)
    upper, _ = np.meshgrid(b.upper, a.center)
    _, center = np.meshgrid(b.center, a.center)

    return ((lower < center) * (center <= upper) * data[..., None]).sum(axis=-2)


def bandpass_frequencies(x, fs, frequencies, order=8, purge=False, zero_phase=False):
    """"Apply bandpass filters for frequencies

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param frequencies: Frequencies. Instance of :class:`Frequencies`.
    :param order: Filter order.
    :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.
    """
    if purge:
        frequencies = frequencies[frequencies.upper < fs / 2.0]
    return frequencies, np.array(
        [bandpass(x, band.lower, band.upper, fs, order, zero_phase=zero_phase) for band in frequencies])


def bandpass_octaves(x, fs, frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES, order=8, purge=False, zero_phase=False):
    """Apply 1/1-octave bandpass filters.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param frequencies: Frequencies.
    :param order: Filter order.
    :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

    .. seealso:: :func:`octavepass`
    """
    return bandpass_fractional_octaves(x, fs, frequencies, fraction=1, order=order, purge=purge, zero_phase=zero_phase)


def bandpass_third_octaves(x, fs, frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, order=8, purge=False,
                           zero_phase=False):
    """Apply 1/3-octave bandpass filters.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param frequencies: Frequencies.
    :param order: Filter order.
    :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

    .. seealso:: :func:`octavepass`
    """
    return bandpass_fractional_octaves(x, fs, frequencies, fraction=3, order=order, purge=purge, zero_phase=zero_phase)


def bandpass_fractional_octaves(x, fs, frequencies, fraction=None, order=8, purge=False, zero_phase=False):
    """Apply 1/N-octave bandpass filters.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param frequencies: Frequencies. Either instance of :class:`OctaveBand`, or array along with fs.
    :param order: Filter order.
    :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
    :param zero_phase: Prevent phase error by filtering in both directions (filtfilt)
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

    .. seealso:: :func:`octavepass`
    """
    if not isinstance(frequencies, Frequencies):
        frequencies = OctaveBand(center=frequencies, fraction=fraction)
    return bandpass_frequencies(x, fs, frequencies, order=order, purge=purge, zero_phase=zero_phase)


def third_octaves(p, fs, density=False, frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, ref=REFERENCE_PRESSURE):
    """Calculate level per 1/3-octave in frequency domain using the FFT.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param density: Power density instead of power.
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

    .. note:: Based on power spectrum (FFT)

    .. seealso:: :attr:`acoustics.bands.THIRD_OCTAVE_CENTER_FREQUENCIES`

    .. note:: Exact center frequencies are always calculated.

    """
    fob = OctaveBand(center=frequencies, fraction=3)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= (fob.bandwidth / fnb.bandwidth)
    level = 10.0 * np.log10(power / ref**2.0)
    return fob, level


def octaves(p, fs, density=False, frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES, ref=REFERENCE_PRESSURE):
    """Calculate level per 1/1-octave in frequency domain using the FFT.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param density: Power density instead of power.
    :param frequencies: Frequencies.
    :param ref: Reference value.
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

    .. note:: Based on power spectrum (FFT)

    .. seealso:: :attr:`acoustics.bands.OCTAVE_CENTER_FREQUENCIES`

    .. note:: Exact center frequencies are always calculated.

    """
    fob = OctaveBand(center=frequencies, fraction=1)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= (fob.bandwidth / fnb.bandwidth)
    level = 10.0 * np.log10(power / ref**2.0)
    return fob, level


def fractional_octaves(p, fs, start=5.0, stop=16000.0, fraction=3, density=False):
    """Calculate level per 1/N-octave in frequency domain using the FFT. N is `fraction`.

    :param x: Instantaneous signal :math:`x(t)`.
    :param fs: Sample frequency.
    :param density: Power density instead of power.
    :returns: Tuple. First element is an instance of :class:`OctaveBand`. The second element an array.

    .. note:: Based on power spectrum (FFT)

    .. note:: This function does *not* use nominal center frequencies.

    .. note:: Exact center frequencies are always calculated.
    """
    fob = OctaveBand(fstart=start, fstop=stop, fraction=fraction)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= (fob.bandwidth / fnb.bandwidth)
    level = 10.0 * np.log10(power)
    return fob, level


class Filterbank:
    """
    Fractional-Octave filter bank.


    .. warning:: For high frequencies the filter coefficients are wrong for low frequencies. Therefore, to improve the response for lower frequencies the signal should be downsampled. Currently, there is no easy way to do so within the Filterbank.

    """

    def __init__(self, frequencies, sample_frequency=44100, order=8):

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
        return (bandpass_filter(lower, upper, fs, order=self.order, output='sos')
                for lower, upper in zip(self.frequencies.lower, self.frequencies.upper))

        #order = self.order
        #filters = list()
        #nyq = self.sample_frequency / 2.0
        #return ( butter(order, [lower/nyq, upper/nyq], btype='band', analog=False) for lower, upper in zip(self.frequencies.lower, self.frequencies.upper) )

    def lfilter(self, signal):
        """
        Filter signal with filterbank.

        .. note:: This function uses :func:`scipy.signal.lfilter`.
        """
        return (sosfilt(sos, signal) for sos in self.filters)

    def filtfilt(self, signal):
        """
        Filter signal with filterbank.
        Returns a list consisting of a filtered signal per filter.

        .. note:: This function uses :func:`scipy.signal.filtfilt` and therefore has a zero-phase response.
        """
        return (_sosfiltfilt(sos, signal) for sos in self.filters)

    def power(self, signal):
        """
        Power per band in signal.
        """
        filtered = self.filtfilt(signal)
        return np.array([(x**2.0).sum() / len(x) / bw for x, bw in zip(filtered, self.frequencies.bandwidth)])

    def plot_response(self):
        """
        Plot frequency response.

        .. note:: The follow phase response is obtained in case :meth:`lfilter` is used. The method :meth:`filtfilt` results in a zero-phase response.
        """

        fs = self.sample_frequency
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for f, fc in zip(self.filters, self.frequencies.center):
            w, h = freqz(f[0], f[1], int(fs / 2))  #np.arange(fs/2.0))
            ax1.semilogx(w / (2.0 * np.pi) * fs, 20.0 * np.log10(np.abs(h)), label=str(int(fc)))
            ax2.semilogx(w / (2.0 * np.pi) * fs, np.angle(h), label=str(int(fc)))
        ax1.set_xlabel(r'$f$ in Hz')
        ax1.set_ylabel(r'$|H|$ in dB re. 1')
        ax2.set_xlabel(r'$f$ in Hz')
        ax2.set_ylabel(r'$\angle H$ in rad')
        ax1.legend(loc=5)
        ax2.legend(loc=5)
        ax1.set_ylim(-60.0, +10.0)

        return fig

    def plot_power(self, signal):
        """
        Plot power in signal.
        """

        f = self.frequencies.center
        p = self.power(signal)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = ax.bar(f, 20.0 * np.log10(p))
        ax.set_xlabel('$f$ in Hz')
        ax.set_ylabel('$L$ in dB re. 1')
        ax.set_xscale('log')

        return fig


def isolate(signals):
    """Isolate signals.

    :param signals: Array of shape N x M where N is the amount of samples and M the amount of signals. Thus, each column is a signal.
    :returns: Array of isolated signals. Each column is a signal.

    Isolate signals using Singular Value Decomposition.

    """
    x = np.asarray(signals)

    W, s, v = np.linalg.svd((np.tile((x * x).sum(axis=0), (len(x), 1)) * x).dot(x.T))
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


def amplitude_envelope(signal, fs, axis=-1):
    """Instantaneous amplitude of tone.

    The instantaneous amplitude is the magnitude of the analytic signal.

    :param signal: Signal.
    :param fs: Sample frequency.
    :param axis: Axis.
    :returns: Amplitude envelope of `signal`.

    .. seealso:: :func:`scipy.signal.hilbert`

    """
    return np.abs(hilbert(signal, axis=axis))


def instantaneous_phase(signal, fs, axis=-1):
    """Instantaneous phase of tone.

    :param signal: Signal.
    :param fs: Sample frequency.
    :param axis: Axis.
    :returns: Instantaneous phase of `signal`.

    The instantaneous phase is the angle of the analytic signal.
    This function returns a wrapped angle.

    .. seealso:: :func:`scipy.signal.hilbert`

    """
    return np.angle(hilbert(signal, axis=axis))


def instantaneous_frequency(signal, fs, axis=-1):
    """Determine instantaneous frequency of tone.

    :param signal: Signal.
    :param fs: Sample frequency.
    :param axis: Axis.
    :returns: Instantaneous frequency of `signal`.

    The instantaneous frequency can be obtained by differentiating the unwrapped instantaneous phase.

    .. seealso:: :func:`instantaneous_phase`

    """
    return np.diff(np.unwrap(instantaneous_phase(signal, fs, axis=axis), axis=axis), axis=axis) / (2.0 * np.pi) * fs


def wvd(signal, fs, analytic=True):
    """Wigner-Ville Distribution

    :param signal: Signal
    :param fs: Sample frequency
    :param analytic: Use the analytic signal, calculated using Hilbert transform.

    .. math:: W_z(n, \\omega) = 2 \\sum_k z^*[n-k]z[n+k] e^{-j\\omega 2kT}

    Includes positive and negative frequencies.

    """
    signal = np.asarray(signal)

    N = int(len(signal) + len(signal) % 2)
    length_FFT = N  # Take an even value of N

    #if N != len(signal):
    #    signal = np.concatenate(signal, [0])

    length_time = len(signal)

    if analytic:
        signal = hilbert(signal)
    s = np.concatenate((np.zeros(length_time), signal, np.zeros(length_time)))
    W = np.zeros((length_FFT, length_time))
    tau = np.arange(0, N // 2)

    R = np.zeros((N, length_time), dtype='float64')

    i = length_time
    for t in range(length_time):
        R[t, tau1] = (s[i + tau] * s[i - tau].conj())  # In one direction
        R[t, N - (tau + 1)] = R[t, tau + 1].conj()  # And the other direction
        i += 1
    W = np.fft.fft(R, length_FFT) / (2 * length_FFT)

    f = np.fft.fftfreq(N, 1. / fs)
    return f, W.T


def _sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):
    """Filtfilt version using Second Order sections. Code is taken from scipy.signal.filtfilt and adapted to make it work with SOS.
    Note that broadcasting does not work.
    """
    from scipy.signal import sosfilt_zi
    from scipy.signal._arraytools import odd_ext, axis_slice, axis_reverse
    x = np.asarray(x)

    if padlen is None:
        edge = 0
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be at least " "padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x

    # Get the steady state of the filter's step response.
    zi = sosfilt_zi(sos)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    #zi_shape = [1] * x.ndim
    #zi_shape[axis] = zi.size
    #zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)
    # Forward filter.
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y


from scipy.signal import lti, cheby1, firwin


def decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=False):
    """
    Downsample the signal by using a filter.

    By default, an order 8 Chebyshev type I filter is used.  A 30 point FIR
    filter with hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    x : ndarray
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor.
    n : int, optional
        The order of the filter (1 less than the length for 'fir').
    ftype : str {'iir', 'fir'}, optional
        The type of the lowpass filter.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool
        Prevent phase shift by filtering with ``filtfilt`` instead of ``lfilter``.
    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See also
    --------
    resample

    Notes
    -----
    The ``zero_phase`` keyword was added in 0.17.0.
    The possibility to use instances of ``lti`` as ``ftype`` was added in 0.17.0.

    """

    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if ftype == 'fir':
        if n is None:
            n = 30
        system = lti(firwin(n + 1, 1. / q, window='hamming'), 1.)

    elif ftype == 'iir':
        if n is None:
            n = 8
        system = lti(*cheby1(n, 0.05, 0.8 / q))
    else:
        system = ftype

    if zero_phase:
        y = filtfilt(system.num, system.den, x, axis=axis)
    else:
        y = lfilter(system.num, system.den, x, axis=axis)

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)
    return y[tuple(sl)]


def impulse_response_real_even(tf, ntaps):
    """The impulse response of a real and even frequency response is also real and even.

    :param tf: Real and even frequency response. Only positive frequencies.
    :param ntaps: Amount of taps.
    :returns: A real and even (double-sided) impulse response with length `ntaps`.

    A symmetric impulse response is needed. The center of symmetry determines the delay
    of the filter and thereby whether the filter is causal (delay>0, linear-phase) or
    non-causal (delay=0, linear-phase, zero-phase).

    Creating linear phase can be done by multiplying the magnitude with a complex
    exponential corresponding to the desired shift. Another method is to rotate the
    impulse response.

    https://ccrma.stanford.edu/~jos/filters/Zero_Phase_Filters_Even_Impulse.html
    """
    ir = np.fft.ifftshift(np.fft.irfft(tf, n=ntaps)).real
    return ir


def linear_phase(ntaps, steepness=1):
    """Compute linear phase delay for a single-sided spectrum.

    :param ntaps: Amount of filter taps.
    :param steepness: Steepness of phase delay. Default value is 1, corresponding to delay in samples of `ntaps//2`.
    :returns: Linear phase delay.

    A linear phase delay can be added to an impulse response using the function `np.fft.ifftshift`.
    Sometimes, however, you would like to add the linear phase delay to the frequency response instead.
    This function computes the linear phase delay which can be multiplied with a single-sided frequency response.
    """
    f = np.fft.rfftfreq(ntaps, 1.0)  # Frequencies normalized to Nyquist.
    alpha = ntaps // 2 * steepness
    return np.exp(-1j * 2. * np.pi * f * alpha)


__all__ = [
    'bandpass',
    'bandpass_frequencies',
    'bandpass_fractional_octaves',
    'bandpass_octaves',
    'bandpass_third_octaves',
    'lowpass',
    'highpass',
    'octavepass',
    'octave_filter',
    'bandpass_filter',
    'convolve',
    'ir2fr',
    'decibel_to_neper',
    'neper_to_decibel',
    'EqualBand',
    'OctaveBand',
    'ms',
    'rms',
    'normalize',
    'window_scaling_factor',
    'apply_window',
    'amplitude_spectrum',
    'auto_spectrum',
    'power_spectrum',
    'angle_spectrum',
    'phase_spectrum',
    'density_spectrum',
    'integrate_bands',
    'octaves',
    'third_octaves',
    'fractional_octaves',
    'Filterbank',
    'isolate',
    'zero_crossings',
    'amplitude_envelope',
    'instantaneous_phase',
    'instantaneous_frequency',
    'wvd',
    'decimate',
]
