import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import detrend, lfilter, bilinear, spectrogram, filtfilt, resample, fftconvolve
import acoustics

from acoustics.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
from acoustics.standards.iec_61672_1_2013 import WEIGHTING_SYSTEMS
from acoustics.standards.iec_61672_1_2013 import (NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                                                  NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)


class Signal(np.ndarray):
    """A signal consisting of samples (array) and a sample frequency (float).

    """

    def __new__(cls, data, fs):
        obj = np.asarray(data).view(cls)
        obj.fs = fs
        return obj

    def __array_prepare__(self, array, context=None):
        try:
            a = context[1][0]
            b = context[1][1]
        except IndexError:
            return array

        if hasattr(a, 'fs') and hasattr(b, 'fs'):
            if a.fs == b.fs:
                return array
            else:
                raise ValueError("Sample frequencies do not match.")
        else:
            return array

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.fs = getattr(obj, 'fs', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Signal, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fs, )
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.fs = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Signal, self).__setstate__(state[0:-1])

    def __repr__(self):
        return "Signal({})".format(str(self))

    def _construct(self, x):
        """Construct signal like x."""
        return Signal(x, self.fs)

    @property
    def samples(self):
        """Amount of samples in signal."""
        return self.shape[-1]

    @property
    def channels(self):
        """Amount of channels.
        """
        if self.ndim > 1:
            return self.shape[-2]
        else:
            return 1

    @property
    def duration(self):
        """Duration of signal in seconds.
        """
        return float(self.samples / self.fs)

    @property
    def values(self):
        """Return the values of this signal as an instance of :class:`np.ndarray`."""
        return np.array(self)

    def calibrate_to(self, decibel, inplace=False):
        """Calibrate signal to value `decibel`.

        :param decibel: Value to calibrate to.
        :param inplace: Whether to perform inplace or not.
        :returns: Calibrated signal.
        :rtype: :class:`Signal`

        Values of `decibel` are broadcasted. To set a value per channel, use `decibel[...,None]`.
        """
        decibel = decibel * np.ones(self.shape)
        gain = decibel - self.leq()[..., None]
        return self.gain(gain, inplace=inplace)

    def calibrate_with(self, other, decibel, inplace=False):
        """Calibrate signal with other signal.

        :param other: Other signal/array.
        :param decibel: Signal level of `other`.
        :param inplace: Whether to perform inplace or not.
        :returns: calibrated signal.
        :rtype: :class:`Signal`
        """
        if not isinstance(other, Signal):
            other = Signal(other, self.fs)
        gain = decibel - other.leq()
        return self.gain(gain, inplace=inplace)

    def decimate(self, factor, zero_phase=False, ftype='iir', order=None):
        """Decimate signal by integer `factor`. Before downsampling a low-pass filter is applied.

        :param factor: Downsampling factor.
        :param zero_phase: Prevent phase shift by filtering with ``filtfilt`` instead of ``lfilter``.
        :param ftype: Filter type.
        :param order: Filter order.
        :returns: Decimated signal.
        :rtype: :class:`Signal`

        .. seealso:: :func:`scipy.signal.decimate`
        .. seealso:: :meth:`resample`

        """
        return Signal(
            acoustics.signal.decimate(x=self, q=factor, n=order, ftype=ftype, zero_phase=zero_phase), self.fs / factor)

    def resample(self, nsamples, times=None, axis=-1, window=None):
        """Resample signal.

        :param samples: New amount of samples.
        :param times: Times corresponding to samples.
        :param axis: Axis.
        :param window: Window.

        .. seealso:: :func:`scipy.signal.resample`
        .. seealso:: :meth:`decimate`

        You might want to low-pass filter this signal before resampling.

        """
        return Signal(resample(self, nsamples, times, axis, window), nsamples / self.samples * self.fs)

    def upsample(self, factor, axis=-1):
        """Upsample signal with integer factor.

        :param factor: Upsample factor.
        :param axis: Axis.

        .. seealso:: :meth:`resample`
        """
        return self.resample(int(self.samples * factor), axis=axis)

    def gain(self, decibel, inplace=False):
        """Apply gain of `decibel` decibels.

        :param decibel: Decibels
        :param inplace: In place
        :returns: Amplified signal.
        :rtype: :class:`Signal`
        """
        factor = 10.0**(decibel / 20.0)
        if inplace:
            self *= factor
            return self
        else:
            return self * factor

    def pick(self, start=0.0, stop=None):
        """Get signal from start time to stop time.

        :param start: Start time.
        :type start: float
        :param stop: End time.
        :type stop: float
        :returns: Selected part of the signal.
        :rtype: :class:`Signal`

        """
        if start is not None:
            start = int(np.floor(start * self.fs))
        if stop is not None:
            stop = int(np.floor(stop * self.fs))
        return self[..., start:stop]

    def times(self):
        """Time vector.

        :returns: A vector with a timestamp for each sample.
        :rtype: :class:`np.ndarray`

        """
        return np.arange(0, self.samples) / self.fs

    def energy(self):
        """Signal energy.

        :returns: Total energy per channel.
        :rtype: :class:`np.ndarray`

        .. math:: E = \\sum_{n=0}^{N-1} |x_n|^2

        """
        return float((self * self).sum())

    def power(self):
        """Signal power.

        .. math:: P = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n|^2
        """
        return self.energy() / len(self)

    def ms(self):
        """Mean value squared of signal.

        .. seealso:: :func:`acoustics.signal.ms`

        """
        return acoustics.signal.ms(self)

    def rms(self):
        """Root mean squared of signal.

        .. seealso:: :func:`acoustics.signal.rms`

        """
        return acoustics.signal.rms(self)
        #return np.sqrt(self.power())

    def weigh(self, weighting='A', zero_phase=False):
        """Apply frequency-weighting. By default 'A'-weighting is applied.

        :param weighting: Frequency-weighting filter to apply.
            Valid options are 'A', 'C' and 'Z'. Default weighting is 'A'.
        :returns: Weighted signal.
        :rtype: :class:`Signal`.

        By default the weighting filter is applied using
        :func:`scipy.signal.lfilter` causing a frequency-dependent delay. In case a
        delay is undesired, the filter can be applied using :func:`scipy.signal.filtfilt`
        by setting `zero_phase=True`.

        """
        num, den = WEIGHTING_SYSTEMS[weighting]()
        b, a = bilinear(num, den, self.fs)
        func = filtfilt if zero_phase else lfilter
        return self._construct(func(b, a, self))

    def correlate(self, other=None, mode='full'):
        """Correlate signal with `other` signal. In case `other==None` this
        method returns the autocorrelation.

        :param other: Other signal.
        :param mode: Mode.

        .. seealso:: :func:`np.correlate`, :func:`scipy.signal.fftconvolve`

        """
        if other is None:
            other = self
        if self.fs != other.fs:
            raise ValueError("Cannot correlate. Sample frequencies are not the same.")
        if self.channels > 1 or other.channels > 1:
            raise ValueError("Cannot correlate. Not supported for multichannel signals.")
        return self._construct(fftconvolve(self, other[::-1], mode=mode))

    def amplitude_envelope(self):
        """Amplitude envelope.

        :returns: Amplitude envelope of signal.
        :rtype: :class:`Signal`

        .. seealso:: :func:`acoustics.signal.amplitude_envelope`

        """
        return self._construct(acoustics.signal.amplitude_envelope(self, self.fs))

    def instantaneous_frequency(self):
        """Instantaneous frequency.

        :returns: Instantaneous frequency of signal.
        :rtype: :class:`Signal`

        .. seealso:: :func:`acoustics.signal.instantaneous_frequency`

        """
        return self._construct(acoustics.signal.instantaneous_frequency(self, self.fs))

    def instantaneous_phase(self):
        """Instantaneous phase.

        :returns: Instantaneous phase of signal.
        :rtype: :class:`Signal`

        .. seealso:: :func:`acoustics.signal.instantaneous_phase`

        """
        return self._construct(acoustics.signal.instantaneous_phase(self, self.fs))

    def detrend(self, **kwargs):
        """Detrend signal.

        :returns: Detrended version of signal.
        :rtype: :class:`Signal`

        .. seealso:: :func:`scipy.signal.detrend`

        """
        return self._construct(detrend(self, **kwargs))

    def unwrap(self):
        """Unwrap signal in case the signal represents wrapped phase.

        :returns: Unwrapped signal.
        :rtype: :class:`Signal`

        .. seealso:: :func:`np.unwrap`

        """
        return self._construct(np.unwrap(self))

    def complex_cepstrum(self, N=None):
        """Complex cepstrum.

        :param N: Amount of bins.
        :returns: Quefrency, complex cepstrum and delay in amount of samples.

        .. seealso:: :func:`acoustics.cepstrum.complex_cepstrum`

        """
        if N is not None:
            times = np.linspace(0.0, self.duration, N, endpoint=False)
        else:
            times = self.times()
        cepstrum, ndelay = acoustics.cepstrum.complex_cepstrum(self, n=N)
        return times, cepstrum, ndelay

    def real_cepstrum(self, N=None):
        """Real cepstrum.

        :param N: Amount of bins.
        :returns: Quefrency and real cepstrum.

        .. seealso:: :func:`acoustics.cepstrum.real_cepstrum`

        """
        if N is not None:
            times = np.linspace(0.0, self.duration, N, endpoint=False)
        else:
            times = self.times()
        return times, acoustics.cepstrum.real_cepstrum(self, n=N)

    def power_spectrum(self, N=None):
        """Power spectrum.

        :param N: Amount of bins.

        .. seealso:: :func:`acoustics.signal.power_spectrum`

        """
        return acoustics.signal.power_spectrum(self, self.fs, N=N)

    def angle_spectrum(self, N=None):
        """Phase angle spectrum. Wrapped.

        :param N: amount of bins.

        .. seealso::

            :func:`acoustics.signal.angle_spectrum`, :func:`acoustics.signal.phase_spectrum`
            and :meth:`phase_spectrum`.

        """
        return acoustics.signal.angle_spectrum(self, self.fs, N=N)

    def phase_spectrum(self, N=None):
        """Phase spectrum. Unwrapped.

        :param N: Amount of bins.

        .. seealso::

            :func:`acoustics.signal.phase_spectrum`, :func:`acoustics.signal.angle_spectrum`
            and :meth:`angle_spectrum`.

        """
        return acoustics.signal.phase_spectrum(self, self.fs, N=N)

    def peak(self, axis=-1):
        """Peak sound pressure.

        :param axis: Axis.

        .. seealso::

            :func:`acoustic.standards.iso_tr_25417_2007.peak_sound_pressure`

        """
        return acoustics.standards.iso_tr_25417_2007.peak_sound_pressure(self, axis=axis)

    def peak_level(self, axis=-1):
        """Peak sound pressure level.

        :param axis: Axis.

        .. seealso::

            :func:`acoustics.standards.iso_tr_25417_2007.peak_sound_pressure_level`

        """
        return acoustics.standards.iso_tr_25417_2007.peak_sound_pressure_level(self, axis=axis)

    def min(self, axis=-1):
        """Return the minimum along a given axis.

        Refer to `np.amin` for full documentation.
        """
        return np.ndarray.min(self, axis=axis)

    def max(self, axis=-1):
        """Return the minimum along a given axis.

        Refer to `np.amax` for full documentation.
        """
        return np.ndarray.max(self, axis=axis)

    def max_level(self, axis=-1):
        """Maximum sound pressure level.

        :param axis: Axis.

        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.max_sound_pressure_level`

        """
        return acoustics.standards.iso_tr_25417_2007.max_sound_pressure_level(self, axis=axis)

    def sound_exposure(self, axis=-1):
        """Sound exposure.

        :param axis: Axis.

        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.sound_exposure`

        """
        return acoustics.standards.iso_tr_25417_2007.sound_exposure(self, self.fs, axis=axis)

    def sound_exposure_level(self, axis=-1):
        """Sound exposure level.

        :param axis: Axis.

        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.sound_exposure_level`

        """
        return acoustics.standards.iso_tr_25417_2007.sound_exposure_level(self, self.fs, axis=axis)

    def plot_complex_cepstrum(self, N=None, **kwargs):
        """Plot complex cepstrum of signal.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * frequency: Boolean indicating whether the x-axis should show time in seconds or quefrency
        * xlabel_frequency: Label in case frequency is shown.

        """
        params = {
            'xscale': 'linear',
            'yscale': 'linear',
            'xlabel': "$t$ in s",
            'ylabel': "$C$",
            'title': 'Complex cepstrum',
            'frequency': False,
            'xlabel_frequency': "$f$ in Hz",
        }
        params.update(kwargs)

        t, ceps, _ = self.complex_cepstrum(N=N)
        if params['frequency']:
            t = 1. / t
            params['xlabel'] = params['xlabel_frequency']
            t = t[::-1]
            ceps = ceps[::-1]
        return _base_plot(t, ceps, params)

    def plot_real_cepstrum(self, N=None, **kwargs):
        """Plot real cepstrum of signal.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * frequency: Boolean indicating whether the x-axis should show time in seconds or quefrency
        * xlabel_frequency: Label in case frequency is shown.

        """
        params = {
            'xscale': 'linear',
            'yscale': 'linear',
            'xlabel': "$t$ in s",
            'ylabel': "$C$",
            'title': 'Real cepstrum',
            'frequency': False,
            'xlabel_frequency': "$f$ in Hz",
        }
        params.update(kwargs)

        t, ceps = self.real_cepstrum(N=N)
        if params['frequency']:
            t = 1. / t
            params['xlabel'] = params['xlabel_frequency']
            t = t[::-1]
            ceps = ceps[::-1]
        return _base_plot(t, ceps, params)

    def plot_power_spectrum(self, N=None, **kwargs):  #filename=None, scale='log'):
        """Plot spectrum of signal.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * reference: Reference power

        .. seealso:: :meth:`power_spectrum`

        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlabel': "$f$ in Hz",
            'ylabel': "$L_{p}$ in dB",
            'title': 'SPL',
            'reference': REFERENCE_PRESSURE**2.0,
        }
        params.update(kwargs)

        f, o = self.power_spectrum(N=N)
        return _base_plot(f, 10.0 * np.log10(o / params['reference']), params)

    def plot_angle_spectrum(self, N=None, **kwargs):
        """Plot phase angle spectrum of signal. Wrapped.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * reference: Reference power

        """
        params = {
            'xscale': 'linear',
            'yscale': 'linear',
            'xlabel': "$f$ in Hz",
            'ylabel': r"$\angle \phi$",
            'title': 'Phase response (wrapped)',
        }
        params.update(kwargs)
        f, o = self.angle_spectrum(N=N)
        return _base_plot(f, o, params)

    def plot_phase_spectrum(self, N=None, **kwargs):
        """Plot phase spectrum of signal. Unwrapped.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * reference: Reference power

        """
        params = {
            'xscale': 'linear',
            'yscale': 'linear',
            'xlabel': "$f$ in Hz",
            'ylabel': r"$\angle \phi$",
            'title': 'Phase response (unwrapped)',
        }
        params.update(kwargs)
        f, o = self.phase_spectrum(N=N)
        return _base_plot(f, o, params)

    def spectrogram(self, **kwargs):
        """Spectrogram of signal.

        :returns: Spectrogram.

        See :func:`scipy.signal.spectrogram`. Some of the default values have been changed.
        The generated spectrogram consists by default of complex values.

        """
        params = {
            'nfft': 4096,
            'noverlap': 128,
            'mode': 'complex',
        }
        params.update(kwargs)

        t, s, P = spectrogram(self, fs=self.fs, **params)

        return t, s, P

    def plot_spectrogram(self, **kwargs):
        """
        Plot spectrogram of the signal.

        Valid kwargs:

        * xlim
        * ylim
        * clim
        .. note:: This method only works for a single channel.

        """
        # To do, use :meth:`spectrogram`.
        params = {
            'xlim': None,
            'ylim': None,
            'clim': None,
            'NFFT': 4096,
            'noverlap': 128,
            'title': 'Spectrogram',
            'xlabel': '$t$ in s',
            'ylabel': '$f$ in Hz',
            'clabel': 'SPL in dB',
            'colorbar': True,
        }
        params.update(kwargs)

        if self.channels > 1:
            raise ValueError("Cannot plot spectrogram of multichannel signal. Please select a single channel.")

        # Check if an axes object is passed in. Otherwise, create one.
        ax0 = params.get('ax', plt.figure().add_subplot(111))
        ax0.set_title(params['title'])

        data = np.squeeze(self)
        try:
            _, _, _, im = ax0.specgram(data, Fs=self.fs, noverlap=params['noverlap'], NFFT=params['NFFT'],
                                       mode='magnitude', scale_by_freq=False)
        except AttributeError:
            raise NotImplementedError(
                "Your version of matplotlib is incompatible due to lack of support of the mode keyword argument to matplotlib.mlab.specgram."
            )

        if params['colorbar']:
            cb = ax0.get_figure().colorbar(mappable=im)
            cb.set_label(params['clabel'])

        ax0.set_xlim(params['xlim'])
        ax0.set_ylim(params['ylim'])
        im.set_clim(params['clim'])

        ax0.set_xlabel(params['xlabel'])
        ax0.set_ylabel(params['ylabel'])

        return ax0

    def levels(self, time=0.125, method='average'):
        """Calculate sound pressure level as function of time.

        :param time: Averaging time or integration time constant. Default value is 0.125 corresponding to FAST.
        :param method: Use time `average` or time `weighting`. Default option is `average`.
        :returns: sound pressure level as function of time.

        .. seealso:: :func:`acoustics.standards.iec_61672_1_2013.time_averaged_sound_level`
        .. seealso:: :func:`acoustics.standards.iec_61672_1_2013.time_weighted_sound_level`

        """
        if method == 'average':
            return acoustics.standards.iec_61672_1_2013.time_averaged_sound_level(self.values, self.fs, time)
        elif method == 'weighting':
            return acoustics.standards.iec_61672_1_2013.time_weighted_sound_level(self.values, self.fs, time)
        else:
            raise ValueError("Invalid method")

    def leq(self):
        """Equivalent level. Single-value number.

        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.equivalent_sound_pressure_level`

        """
        return acoustics.standards.iso_tr_25417_2007.equivalent_sound_pressure_level(self.values)

    def plot_levels(self, **kwargs):
        """Plot sound pressure level as function of time.

        .. seealso:: :meth:`levels`

        """
        params = {
            'xscale': 'linear',
            'yscale': 'linear',
            'xlabel': '$t$ in s',
            'ylabel': '$L_{p,F}$ in dB',
            'title': 'SPL',
            'time': 0.125,
            'method': 'average',
            'labels': None,
        }
        params.update(kwargs)
        t, L = self.levels(params['time'], params['method'])
        L_masked = np.ma.masked_where(np.isinf(L), L)
        return _base_plot(t, L_masked, params)

    #def octave(self, frequency, fraction=1):
    #"""Determine fractional-octave `fraction` at `frequency`.

    #.. seealso:: :func:`acoustics.signal.fractional_octaves`

    #"""
    #return acoustics.signal.fractional_octaves(self, self.fs, frequency,
    #frequency, fraction, False)[1]

    def bandpass(self, lowcut, highcut, order=8, zero_phase=False):
        """Filter signal with band-pass filter.

        :param lowcut: Lower cornerfrequency.
        :param highcut: Upper cornerfrequency.
        :param order: Filter order.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        :returns: Band-pass filtered signal.
        :rtype: :class:`Signal`.

        .. seealso:: :func:`acoustics.signal.bandpass`
        """
        return type(self)(acoustics.signal.bandpass(self, lowcut, highcut, self.fs, order=order, zero_phase=zero_phase),
                          self.fs)

    def bandstop(self, lowcut, highcut, order=8, zero_phase=False):
        """Filter signal with band-stop filter.

        :param lowcut: Lower cornerfrequency.
        :param highcut: Upper cornerfrequency.
        :param order: Filter order.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        :returns: Band-pass filtered signal.
        :rtype: :class:`Signal`.

        .. seealso:: :func:`acoustics.signal.bandstop`
        """
        return type(self)(acoustics.signal.bandstop(self, lowcut, highcut, self.fs, order=order, zero_phase=zero_phase),
                          self.fs)

    def highpass(self, cutoff, order=4, zero_phase=False):
        """Filter signal with high-pass filter.

        :param cutoff: Cornerfrequency.
        :param order: Filter order.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: High-pass filtered signal.
        :rtype: :class:`Signal`.

        .. seealso:: :func:`acoustics.signal.highpass`
        """
        return type(self)(acoustics.signal.highpass(self, cutoff, self.fs, order=order, zero_phase=zero_phase), self.fs)

    def lowpass(self, cutoff, order=4, zero_phase=False):
        """Filter signal with low-pass filter.

        :param cutoff: Cornerfrequency.
        :param order: Filter order.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: Low-pass filtered signal.
        :rtype: :class:`Signal`.

        .. seealso:: :func:`acoustics.signal.lowpass`
        """
        return type(self)(acoustics.signal.lowpass(self, cutoff, self.fs, order=order, zero_phase=zero_phase), self.fs)

    def octavepass(self, center, fraction, order=8, zero_phase=False):
        """Filter signal with fractional-octave band-pass filter.

        :param center: Center frequency. Any value in the band will suffice.
        :param fraction: Band designator.
        :param order: Filter order.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: Band-pass filtered signal.
        :rtype: :class:`Signal`.

        .. seealso:: :func:`acoustics.signal.octavepass`
        """
        return type(self)(acoustics.signal.octavepass(self, center, self.fs, fraction=fraction, order=order,
                                                      zero_phase=zero_phase), self.fs)

    def bandpass_frequencies(self, frequencies, order=8, purge=True, zero_phase=False):
        """Apply bandpass filters for frequencies.

        :param frequencies: Band-pass filter frequencies.
        :type frequencies: Instance of :class:`acoustics.signal.Frequencies`
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: Frequencies and band-pass filtered signal.

        .. seealso:: :func:`acoustics.signal.bandpass_frequencies`
        """
        frequencies, filtered = acoustics.signal.bandpass_frequencies(self, self.fs, frequencies, order, purge,
                                                                      zero_phase=zero_phase)
        return frequencies, type(self)(filtered, self.fs)

    def octaves(self, frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES, order=8, purge=True, zero_phase=False):
        """Apply 1/1-octaves bandpass filters.

        :param frequencies: Band-pass filter frequencies.
        :type frequencies: :class:`np.ndarray` with (approximate) center-frequencies or an instance of :class:`acoustics.signal.Frequencies`
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: Frequencies and band-pass filtered signal.

        .. seealso:: :func:`acoustics.signal.bandpass_octaves`
        """
        frequencies, octaves = acoustics.signal.bandpass_octaves(self, self.fs, frequencies, order, purge,
                                                                 zero_phase=zero_phase)
        return frequencies, type(self)(octaves, self.fs)

    def third_octaves(self, frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, order=8, purge=True, zero_phase=False):
        """Apply 1/3-octaves bandpass filters.

        :param frequencies: Band-pass filter frequencies.
        :type frequencies: :class:`np.ndarray` with (approximate) center-frequencies or an instance of :class:`acoustics.signal.Frequencies`
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: Frequencies and band-pass filtered signal.

        .. seealso:: :func:`acoustics.signal.bandpass_third_octaves`
        """
        frequencies, octaves = acoustics.signal.bandpass_third_octaves(self, self.fs, frequencies, order, purge,
                                                                       zero_phase=zero_phase)
        return frequencies, type(self)(octaves, self.fs)

    def fractional_octaves(self, frequencies=None, fraction=1, order=8, purge=True, zero_phase=False):
        """Apply 1/N-octaves bandpass filters.

        :param frequencies: Band-pass filter frequencies.
        :type frequencies: Instance of :class:`acoustics.signal.Frequencies`
        :param fraction: Default band-designator of fractional-octaves.
        :param order: Filter order.
        :param purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        :param zero_phase: Prevent phase error by filtering in both directions (filtfilt).
        :returns: Frequencies and band-pass filtered signal.

        .. seealso:: :func:`acoustics.signal.bandpass_fractional_octaves`
        """
        if frequencies is None:
            frequencies = acoustics.signal.OctaveBand(fstart=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[0],
                                                      fstop=self.fs / 2.0, fraction=fraction)
        frequencies, octaves = acoustics.signal.bandpass_fractional_octaves(self, self.fs, frequencies, fraction, order,
                                                                            purge, zero_phase=zero_phase)
        return frequencies, type(self)(octaves, self.fs)

    def plot_octaves(self, **kwargs):
        """Plot octaves.

        .. seealso:: :meth:`octaves`

        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlabel': '$f$ in Hz',
            'ylabel': '$L_{p}$ in dB',
            'title': '1/1-Octaves SPL',
        }
        params.update(kwargs)
        f, o = self.octaves()
        print(len(f.center), len(o.leq()))
        return _base_plot(f.center, o.leq().T, params)

    def plot_third_octaves(self, **kwargs):
        """Plot 1/3-octaves.

        .. seealso:: :meth:`third_octaves`

        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlabel': '$f$ in Hz',
            'ylabel': '$L_{p}$ in dB',
            'title': '1/3-Octaves SPL',
        }
        params.update(kwargs)
        f, o = self.third_octaves()
        return _base_plot(f.center, o.leq().T, params)

    def plot_fractional_octaves(self, frequencies=None, fraction=1, order=8, purge=True, zero_phase=False, **kwargs):
        """Plot fractional octaves.
        """
        title = '1/{}-Octaves SPL'.format(fraction)

        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlabel': '$f$ in Hz',
            'ylabel': '$L_p$ in dB',
            'title': title,
        }
        params.update(kwargs)
        f, o = self.fractional_octaves(frequencies=frequencies, fraction=fraction, order=order, purge=purge,
                                       zero_phase=zero_phase)
        return _base_plot(f.center, o.leq().T, params)

    def plot(self, **kwargs):
        """Plot signal as function of time. By default the entire signal is plotted.

        :param filename: Name of file.
        :param start: First sample index.
        :type start: Start time in seconds from start of signal.
        :param stop: Last sample index.
        :type stop: Stop time in seconds. from stop of signal.
        """
        params = {
            'xscale': 'linear',
            'yscale': 'linear',
            'xlabel': '$t$ in s',
            'ylabel': '$x$ in -',
            'title': 'Signal',
        }
        params.update(kwargs)
        return _base_plot(self.times(), self, params)

    #def plot_scalo(self, filename=None):
    #"""
    #Plot scalogram
    #"""
    #from scipy.signal import ricker, cwt

    #wavelet = ricker
    #widths = np.logspace(-1, 3.5, 10)
    #x = cwt(self, wavelet, widths)

    #interpolation = 'nearest'

    #from matplotlib.ticker import LinearLocator, AutoLocator, MaxNLocator
    #majorLocator = LinearLocator()
    #majorLocator = MaxNLocator()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_title('Scaleogram')
    ##ax.set_xticks(np.arange(0, x.shape[1])*self.fs)
    ##ax.xaxis.set_major_locator(majorLocator)

    ##ax.imshow(10.0 * np.log10(x**2.0), interpolation=interpolation, aspect='auto', origin='lower')#, extent=[0, 1, 0, len(x)])
    #ax.pcolormesh(np.arange(0.0, x.shape[1])/self.fs, widths, 10.0*np.log(x**2.0))
    #if filename:
    #fig.savefig(filename)
    #else:
    #return fig

    #def plot_scaleogram(self, filename):
    #"""
    #Plot scaleogram
    #"""
    #import pywt

    #wavelet = 'dmey'
    #level = pywt.dwt_max_level(len(self), pywt.Wavelet(wavelet))
    #print level
    #level = 20
    #order = 'freq'
    #interpolation = 'nearest'

    #wp = pywt.WaveletPacket(self, wavelet, 'sym', maxlevel=level)
    #nodes = wp.get_level(level, order=order)
    #labels = [n.path for n in nodes]
    #values = np.abs(np.array([n.data for n in nodes], 'd'))

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_title('Scaleogram')
    #ax.imshow(values, interpolation=interpolation, aspect='auto', origin='lower', extent=[0, 1, 0, len(values)])
    ##ax.set_yticks(np.arange(0.5, len(labels) + 0.5))
    ##ax.set_yticklabels(labels)

    #fig.savefig(filename)

    def normalize(self, gap=6.0, inplace=False):
        """Normalize signal.

        :param gap: Gap between maximum value and ceiling in decibel.
        :param inplace: Normalize signal in place.

        The parameter `gap` can be understood as using `gap` decibels fewer for the dynamic range.
        By default a 6 decibel gap is used.

        """
        factor = (self.max() * 10.0**(gap / 20.0))
        if inplace:
            self /= factor
            return self
        else:
            return self / factor

    def to_wav(self, filename, depth=16):
        """Save signal as WAV file.

        :param filename: Name of file to save to.
        :param depth: If given, convert to integer with specified depth. Else, try to store using the original data type.

        By default, this function saves a normalized 16-bit version of the signal with at least 6 dB range till clipping occurs.

        """
        data = self
        dtype = data.dtype if not depth else 'int' + str(depth)
        if depth:
            data = (data * 2**(depth - 1) - 1).astype(dtype)
        wavfile.write(filename, int(self.fs), data.T)
        #wavfile.write(filename, int(self.fs), self._data/np.abs(self._data).max() *  0.5)
        #wavfile.write(filename, int(self.fs), np.int16(self._data/(np.abs(self._data).max()) * 32767) )

    @classmethod
    def from_wav(cls, filename):
        """
        Create an instance of `Signal` from a WAV file.

        :param filename: Filename

        """
        fs, data = wavfile.read(filename)
        data = data.astype(np.float32, copy=False).T
        data /= np.max(np.abs(data))
        return cls(data, fs=fs)


_PLOTTING_PARAMS = {
    'title': None,
    'xlabel': None,
    'ylabel': None,
    'xscale': 'linear',
    'yscale': 'linear',
    'xlim': (None, None),
    'ylim': (None, None),
    'labels': None,
    'linestyles': ['-', '-.', '--', ':'],
}


def _get_plotting_params():
    d = dict()
    d.update(_PLOTTING_PARAMS)
    return d


def _base_plot(x, y, given_params):
    """Common function for creating plots.

    :returns: Axes object.
    :rtype: :class:`matplotlib.Axes`
    """

    params = _get_plotting_params()
    params.update(given_params)

    linestyles = itertools.cycle(iter(params['linestyles']))

    # Check if an axes object is passed in. Otherwise, create one.
    ax0 = params.get('ax', plt.figure().add_subplot(111))

    ax0.set_title(params['title'])
    if y.ndim > 1:
        for channel in y:
            ax0.plot(x, channel, linestyle=next(linestyles))
    else:
        ax0.plot(x, y)
    ax0.set_xlabel(params['xlabel'])
    ax0.set_ylabel(params['ylabel'])
    ax0.set_xscale(params['xscale'])
    ax0.set_yscale(params['yscale'])
    ax0.set_xlim(params['xlim'])
    ax0.set_ylim(params['ylim'])

    if params['labels'] is None and y.ndim > 1:
        params['labels'] = np.arange(y.shape[-2]) + 1
    if params['labels'] is not None:
        ax0.legend(labels=params['labels'])

    return ax0


__all__ = ["Signal"]
