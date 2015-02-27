cimport cython
cimport numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import acoustics

class Signal(numpy.ndarray):
    """Container for signals.Signal
    
    `Signal` is a container for acoustics signals.
    """
    #fs = 0.0
    
    def __new__(cls, data, fs):
        
        #if np.asarray(data).ndim!=1:
            #raise ValueError("Incorrect amount of dimensions. One dimension is required.")
        
        obj = np.asarray(data).view(cls)
        #obj = np.atleast_2d(np.asarray(data)).view(cls)
        obj.fs = fs #if fs is not None else 1000
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
        #attr = getattr(obj, 'fs', None)
        #print(attr)
        #self.fs = attr if attr is not None else 44100.0
        self.fs = getattr(obj, 'fs', None)#44100.0)
        #self.fs = 1000
            
    def __repr__(self):
        return "Signal({})".format(str(self))
        
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
    
    def pick(self, start=0.0, stop=None):
        """Get signal from start time to stop time.
        """
        if start is not None:
            start = np.floor(start*self.fs)
        if stop is not None:
            stop  = np.floor(stop*self.fs)
        return self[..., start:stop]
        
    def times(self):
        """Time vector.
        
        Creates a vector with a timestamp for every sample.
        
        """
        return np.arange(0, self.samples) / self.fs
    
    def energy(self):
        """
        Signal energy.
        
        .. math:: E = \\sum_{n=0}^{N-1} |x_n|^2
        
        """
        return float((self*self).sum())
    
    def power(self):
        """Signal power.
        
        .. math:: P = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n|^2
        """
        return self.energy() / len(self)
    
    def ms(self):
        """Mean value squared of signal.
        
        .. seealso:: `acoustics.signal.ms`
        
        """
        return acoustics.signal.ms(self)
    
    def rms(self):
        """Root mean squared of signal.
        
        .. seealso:: `acoustics.signal.rms`
        
        """
        return acoustics.signal.rms(self)
        #return np.sqrt(self.power())
    
    def spectrum(self):
        """Power spectrum.
        
        .. seealso:: :func:`acoustics.signal.power_spectrum`
        
        """
        return acoustics.signal.power_spectrum(self, self.fs)
    
    def peak(self):
        """Peak sound pressure.
        
        .. seealso:: :func:`acoustic.standards.iso_tr_25417_2007.peak_sound_pressure`
        
        """
        return acoustics.standards.iso_tr_25417_2007.peak_sound_pressure(self)
    
    
    def peak_level(self):
        """Peak sound pressure level.
        
        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.peak_sound_pressure_level`
        
        """
        return acoustics.standards.iso_tr_25417_2007.peak_sound_pressure_level(self.peak())
            
    
    def sound_exposure(self):
        """Sound exposure.
        
        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.sound_exposure`
        
        """
        return acoustics.standards.iso_tr_25417_2007.sound_exposure(self)
    
    
    def sound_exposure_level(self):
        """Sound exposure level.
        
        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.sound_exposure_level`
        
        """
        return acoustics.standards.iso_tr_25417_2007.sound_exposure_level(self.sound_exposure)
        
    
    def plot_spectrum(self, **kwargs):#filename=None, scale='log'):
        """Plot spectrum of signal.
        
        Valid kwargs:
        
        - xscale
        - yscale
        - xlim
        - ylim
        - filename
        
        .. seealso:: :meth:`spectrum`
        
        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlabel': "$f$ in Hz",
            'ylabel': "$L_{p}$ in dB",
            'title' : 'SPL',
            }
        params.update(kwargs)
        
        f, o = self.spectrum()
        return _base_plot(f, 10.0*np.log10(o.T), params)

    def spectrogram(self, **kwargs):
        """
        Plot spectrograms of the signals.
        
        Valid kwargs:
        
        - xlim
        - ylim
        - clim
        - filename
        
        """
        params = {
            'xlim' : None,
            'ylim' : None,
            'clim' : None,
            'filename' : None,
            }
        params.update(kwargs)
        
        if self.channels > 1:
            raise ValueError("Cannot plot spectrogram of multichannel signal. Please select a single channel.")
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Spectrogram')
        #f = ax0.specgram(self, Fs=self.fs)
        data = np.squeeze(self)
        _, _, _, im = ax0.specgram(data, Fs=self.fs, noverlap=128, NFFT=4096, mode='magnitude', scale_by_freq=False)#, vmin=self._data.min(), vmax=self._data.max())
        cb = fig.colorbar(mappable=im)
        cb.set_label('SPL in dB')
        
        #ax0.set_xlim(params['xlim'])
        ax0.set_ylim(params['ylim'])
        im.set_clim(params['clim'])

        ax0.set_xlabel(r'$t$ in s')
        ax0.set_ylabel(r'$f$ in Hz')
        
        if params['filename'] is not None:
            fig.savefig(params['filename'])
        else:
            return fig
    
    
    def levels(self, time=0.125, method='average'):
        """Calculate sound pressure level as function of time.
        
        :param time: Averaging time or integration time constant. Default value is 0.125 corresponding to FAST.
        :param method: Use time `average` or time `weighting`. Default option is `average`.
        :returns: sound pressure level as function of time.
        
        .. seealso:: :func:`acoustics.standards.iec_61672_1_2013.time_averaged_sound_level`
        .. seealso:: :func:`acoustics.standards.iec_61672_1_2013.time_weighted_sound_level`
        
        """
        if method=='average':
            return np.asarray(acoustics.standards.iec_61672_1_2013.time_averaged_sound_level(self, self.fs, time))
        elif method=='weighting':
            return np.asarray(acoustics.standards.iec_61672_1_2013.time_weighted_sound_level(self, self.fs, time))
        else:
            raise ValueError("Invalid method")
    
    def leq(self):
        """Equivalent level. Single-value number.
        
        .. seealso:: :func:`acoustics.standards.iso_tr_25417_2007.equivalent_sound_pressure_level`
        
        """
        return np.asarray(acoustics.standards.iso_tr_25417_2007.equivalent_sound_pressure_level(self))

    def plot_levels(self, **kwargs):
        """Plot sound pressure level as function of time.
        
        .. seealso:: :meth:`levels`
        
        """
        params = {
            'xscale'    :   'linear',
            'yscale'    :   'linear',
            'xlabel'    :   '$t$ in s',
            'ylabel'    :   '$L_{p,F}$ in dB',
            'title'     :   'SPL',   
            'time'      :   0.125,
            'method'    :   'average',
            'labels'    :   None,
            }
        params.update(kwargs)
        t, L = self.levels(params['time'], params['method'])
        L_masked = np.ma.masked_where(np.isinf(L), L)
        return _base_plot(t, L_masked.T, params)

    def octave(self, frequency, fraction=1):
        """Determine fractional-octave `fraction` at `frequency`.
        
        .. seealso:: :func:`acoustics.signal.fractional_octaves`
        
        """
        return acoustics.signal.fractional_octaves(self, self.fs, frequency, 
                                                  frequency, fraction, False)[1]

    def octaves(self):
        """
        Calculate time-series of octaves.
        """
        return acoustics.signal.octaves(self, self.fs)
    
    def plot_octaves(self, **kwargs):
        """Plot octaves.
        
        .. seealso:: :meth:`octaves`
        
        """
        params = {
            'xscale'    :   'log',
            'yscale'    :   'linear',
            'xlabel'    :   '$f$ in Hz',
            'ylabel'    :   '$L_{p}$ in dB',
            'title'     :   '1/1-Octaves SPL',
            }
        params.update(kwargs)
        f, o = self.octaves()
        return _base_plot(f.center, o.T, params)

    def third_octaves(self):
        """Calculate time-series of 1/3-octaves.
        
        .. seealso:: :func:`acoustics.signal.third_octaves`
        
        """
        return acoustics.signal.third_octaves(self, self.fs)
    
    def plot_third_octaves(self, **kwargs):
        """Plot 1/3-octaves.
        
        .. seealso:: :meth:`third_octaves`
        
        """
        params = {
            'xscale'    :   'log',
            'yscale'    :   'linear',
            'xlabel'    :   '$f$ in Hz',
            'ylabel'    :   '$L_{p}$ in dB',
            'title'     :   '1/3-Octaves SPL',
            }
        params.update(kwargs)
        f, o = self.third_octaves()
        return _base_plot(f.center, o.T, params)

    def fractional_octaves(self, fraction):
        """Fractional octaves.
        """
        return acoustics.signal.fractional_octaves(self, self.fs, fraction=fraction)
    
    def plot_fractional_octaves(self, fraction, **kwargs):
        """Plot fractional octaves.
        """
        title = '1/{}-Octaves SPL'.format(fraction)
        
        params = {
            'xscale'    :   'log',
            'yscale'    :   'linear',
            'xlabel'    :   '$f$ in Hz',
            'ylabel'    :   '$L_p$ in dB',
            'title'     :   title,
        }
        params.update(kwargs)
        f, o = self.fractional_octaves(fraction)
        return _base_plot(f.center, o.T, params)

    def plot(self, filename=None, start=0, stop=None, channels=None, **kwargs):
        """Plot signal as function of time. By default the entire signal is plotted.
        
        :param filename: Name of file.
        :param start: First sample index.
        :type start: Start time in seconds from start of signal.
        :param stop: Last sample index.
        :type stop: Stop time in seconds. from stop of signal.
        """
        params = {
            'xscale'    :   'linear',
            'yscale'    :   'linear',
            'xlabel'    :   '$t$ in s',
            'ylabel'    :   '$x$ in -',
            'title'     :   'Signal',
            }
        params.update(kwargs)
        signal = self.pick(start, stop)
        return _base_plot(signal.times, signal, params)

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
    
    def to_wav(self, filename, normalize=True, depth=16):
        """
        """Save signal as WAV file.
        
        :param filename: Name of file to save to.
        :param normalize: Normalize signal. Furthermore, the normalized signal is multiplied by 0.5 leaving a 6 dB gap till clipping occurs.
        :param depth: If given, convert to integer with specified depth. Else, try to store using the original data type.

        By default, this function saves a normalized 16-bit version of the signal with at least 6 dB range till clipping occurs.

        """
        data = self
        dtype = data.dtype if not depth else 'int'+str(depth)
        if normalize:
            data = data / data.max() * 0.5
        if depth:
            data = (data * 2**(depth-1)-1).astype(dtype)
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
        data = data.astype(np.float32, copy=False)
        data /= np.max(np.abs(data))
        return cls(data, fs=fs)


_base_params = {
    'title'     :   None,
    'xlabel'    :   None,
    'ylabel'    :   None,
    'xscale'    :   'linear',
    'yscale'    :   'linear',
    'xlim'      :   (None, None),
    'ylim'      :   (None, None),
    'labels'    :   None,
    }

def _base_plot(x, y, given_params):
    
    params = dict()
    params.update(_base_params)
    params.update(given_params)
    
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.set_title(params['title'])
    ax0.plot(x, y)
    ax0.set_xlabel(params['xlabel'])
    ax0.set_ylabel(params['ylabel'])
    ax0.set_xscale(params['xscale'])
    ax0.set_yscale(params['yscale'])
    ax0.set_xlim(params['xlim'])
    ax0.set_ylim(params['ylim'])
    
    if params['labels'] is None and y.ndim > 1:
        params['labels'] = np.arange(y.shape[-2])
    if params['labels'] is not None:
        ax0.legend(labels=params['labels'])
    
    return fig
