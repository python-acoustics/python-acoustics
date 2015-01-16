cimport cython
cimport numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import acoustics
from .signal import power_spectrum

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
        
   
    #def __add__(self, other):
        #return op('__add__', self, other)

    #def __sub__(self, other):
        #return op('__sub__', self, other)
    
    #def __mul__(self, other):
        #return op('__mul__', self, other)
    
    #def __div__(self, other):
        #return op('__div__', self, other)
    
    #def __mod__(self, other):
        #return op('__mod__', self, other)
    
    #def __iadd__(self, other):
        #return op('__iadd__', self, other)

    #def __isub__(self, other):
        #return op('__isub__', self, other)
    
    #def __imul__(self, other):
        #return op('__imul__', self, other)
    
    #def __idiv__(self, other):
        #return op('__idiv__', self, other)
    
    #def __imod__(self, other):
        #return op('__imod__', self, other)
    

    
###class Signal(object):
    ###"""
    ###Class for containing a signal in time-domain.
    
    ###This class provides methods for plotting the signal in time- and/or frequency-domain.
    ###Signals can be created from a WAV file or written to one.
    ###"""
    
    
    ####@property
    ####def data(self):
        ####return self._data
    
    ####@data.setter
    ####def data(self, data):
        ####if data.ndim==1:
            ####self._data = data
        ####else:
            ####raise ValueError("Wrong shape.")

    ###def __init__(self, data, fs=44100):
        ###"""
        ###Constructor
        
        ###:param input_array: Array describing the time data.
        ###:param fs: Sample frequency :math:`f_s`
        ###:type fs: int
        ###"""
        
        
        ###try:
            ###data = data._data
        ###except AttributeError:
            ###pass
        ###if data.ndim==1:
            ###self._data = data
        ###else:
            ###raise ValueError("Wrong shape.")
        
        ###self.fs = fs
        ###"""Sample frequency"""
    
    ###def __add__(self, other):
        ###return op('__add__', self, other)

    ###def __sub__(self, other):
        ###return op('__sub__', self, other)
    
    ###def __mul__(self, other):
        ###return op('__mul__', self, other)
    
    ###def __div__(self, other):
        ###return op('__div__', self, other)
    
    #def __mod__(self, other):
        #if isinstance(other, Signal):
            #return self.__class__(self._data % other._data, fs=self.fs)
        #else:
            #return self.__class__(self._data % other, fs=self.fs)
    
    #def __iadd__(self, other):
        #if isinstance(other, Signal):
            #if self.fs == other.fs:
                #self._data += other._data
        #else:
            #self._data += other
        #return self
    
    #def __isub__(self, other):
        #if isinstance(other, Signal):
            #if self.fs == other.fs:
                #self._data -= other._data
        #else:
            #self._data -= other
        #return self
    
    #def __imul__(self, other):
        #if isinstance(other, Signal):
            #if self.fs == other.fs:
                #self._data *= other._data
        #else:
            #self._data *= other
        #return self
    
    #def __idiv__(self, other):
        #if isinstance(other, Signal):
            #if self.fs == other.fs:
                #self._data /= other._data
        #else:
            #self._data /= other
        #return self
    
    #def __abs__(self):
        #return self * self
    
    #def __pos__(self):
        #return self.__class__(+self._data, fs=self.fs)
    
    #def __neg__(self):
        #return self.__class__(-self._data, fs=self.fs)
    
    #def __len__(self):
        #return len(self._data)
    
    #def __getitem__(self, key):
        #return self._data[key]
        ##return self.__class__(self._data[key], self.fs)
        
    #def __setitem__(self, key, value):
        #self._data[key] = value
    
    def __repr__(self):
        return "Signal({})".format(str(self))
    
    #def __str__(self):
        #return self._data.__str__()
    
    #def __iter__(self):
        #return self._data.__iter__()
    
    #@property
    #def real(self):
        #return self.__class__(self._data.real, self.fs)
    
    #@property
    #def imag(self):
        #return self.__class__(self._data.imag, self.fs)
    
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
    
    #def min(self):
        #"""Minimum value."""
        #return self._data.min()
    
    #def max(self):
        #"""Maximum value."""
        #return self._data.max()
    
    #def argmin(self):
        #"""Index of minimum value."""
        #return self._data.argmin()
    
    #def argmax(self):
        #"""Index of maximum value."""
        #return self._data.argmax()
    
    #def conjugate(self):
        #"""Complex conjugate."""
        #return self._data.conjugate()
    
    #def mean(self):
        #"""
        #Signal mean value.
        
        #.. math:: \\mu = \\frac{1}{N} \\sum_{n=0}^{N-1} x_n
        
        #"""
        #return self._data.mean()
    
    #@property
    #def fs(self):
        #return self.fs
    
    def times(self):
        """Time vector.
        """
        return np.arange(0, self.samples) / self.fs
    
    def energy(self):
        """
        Signal energy.
        
        .. math:: E = \\sum_{n=0}^{N-1} |x_n|^2
        
        """
        return (self*self).sum()
    
    def power(self):
        """
        Signal power.
        
        .. math:: P = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n|^2
        """
        return self.energy() / len(self)
    
    def rms(self):
        """
        RMS signal power.
        
        .. math:: P_{RMS} = \\sqrt{P}
        
        """
        return np.sqrt(self.power())

    #def std(self):
        #"""
        #Standard deviation.
        #"""
        #return self._data.std()
    
    #def var(self):
        #"""
        #Signal variance.
        
        #.. math:: \\sigma^2 = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n - \\mu |^2
        
        #"""
        #return self._data.var()
    
    def spectrum(self):
        """
        Create spectrum.
        """
        return power_spectrum(self, self.fs)
    
    def plot_spectrum(self, **kwargs):#filename=None, scale='log'):
        """
        Plot spectrum of signal.
        
        Valid kwargs:
        
        - xscale
        - yscale
        - xlim
        - ylim
        - filename
        
        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlim'  : None,
            'ylim'  : None,
            'xlabel': "$f$ in Hz",
            'ylabel': "$L_{p}$ in dB",
            'title' : 'SPL',
            'filename' : None,
            }
        params.update(kwargs)
        
        
        f, o = self.spectrum()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('SPL')
        #if linear:
        ax0.plot(f, 10.0*np.log10(o.T))
        ax0.set_xscale(params['xscale'])
        ax0.set_yscale(params['yscale'])
        ax0.set_ylabel(params['ylabel'])
        ax0.set_xlabel(params['xlabel'])
        ax0.set_xlim(params['xlim'])
        ax0.set_ylim(params['ylim'])
        
        if params['filename'] is not None:
            fig.savefig(params['filename'])
        else:
            return fig
    
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
    
    
    def leq(self):
        """Equivalent level.
        """
        return acoustics.standards.iec_61672_1_2013.fast_level(self, self.fs)
    
    
    def plot_leq(self, **kwargs):
        """
        Plot equivalent level.
        """
        params = {
            #'xscale': 'linear',
            #'yscale': 'linear',
            'xlim'  : None,
            'ylim'  : None,
            'filename' : None,
            }
        params.update(kwargs)
        
        
        t, L = self.leq()
        L_masked = np.ma.masked_where(np.isinf(L), L)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Sound Pressure Level')
        ax.plot(t, L_masked.T)
        ax.set_xlabel(r'$t$ in s')
        ax.set_ylabel(r'$L_{p,F}$ in dB')
        ax.legend(np.arange(self.channels))
        ax.set_xlim(params['xlim'])
        ax.set_ylim(params['ylim'])
        
        if 'filename' in params is not None:
            fig.savefig(params['filename'])
        else:
            return fig
    
    def octaves(self):#, fraction=1):
        """
        Calculate time-series of octaves.
        """
        #if fraction==1:
            #fob = OctaveBand(acoustics.bands.OCTAVE_CENTER_FREQUENCIES, fraction=1)
        #elif fraction==3:
            #fob = OctaveBand(acoustics.bands.OCTAVE_CENTER_FREQUENCIES, fraction=3)
        #return acoustics.signal.fractional_octaves(self, self.fs
        return acoustics.signal.octaves(self, self.fs)
    
    def plot_octaves(self, **kwargs):
        """
        Plot octaves.
        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlim'  : None,
            'ylim'  : None,
            'filename' : None,
            }
        params.update(kwargs)
        
        f, o = self.octaves()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('1/1-Octaves SPL')
        ax0.semilogx(f.center, o.T)
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        ax0.legend(np.arange(self.channels))
        ax0.set_xscale(params['xscale'])
        ax0.set_yscale(params['yscale'])
        ax0.set_xlim(params['xlim'])
        ax0.set_ylim(params['ylim'])
        
        if params['filename'] is not None:
            fig.savefig(params['filename'])
        else:
            return fig
    
    def third_octaves(self):
        """
        Calculate time-series of octaves.
        """
        return acoustics.signal.third_octaves(self, self.fs)
    
    def plot_third_octaves(self, **kwargs):
        """
        Plot octaves.
        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlim'  : None,
            'ylim'  : None,
            'filename' : None,
            }
        params.update(kwargs)
        
        f, o = self.third_octaves()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('1/3-Octaves SPL')
        ax0.semilogx(f.center, o.T)
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        ax0.legend(np.arange(self.channels))
        ax0.set_xscale(params['xscale'])
        ax0.set_yscale(params['yscale'])
        ax0.set_xlim(params['xlim'])
        ax0.set_ylim(params['ylim'])
        
        if params['filename'] is not None:
            fig.savefig(params['filename'])
        else:
            return fig
    
    def fractional_octaves(self, fraction):
        """Fractional octaves.
        """
        return acoustics.signal.fractional_octaves(self, self.fs, fraction=fraction)
    
    def plot_fractional_octaves(self, fraction, **kwargs):
        """Plot fractional octaves.
        """
        params = {
            'xscale': 'log',
            'yscale': 'linear',
            'xlim'  : None,
            'ylim'  : None,
            'filename' : None,
        }
        params.update(kwargs)
        
        f, o = self.fractional_octaves(fraction)
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('1/{}-Octaves SPL'.format(fraction))
        ax0.semilogx(f.center, o.T)
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        ax0.legend(np.arange(self.channels))
        ax0.set_xscale(params['xscale'])
        ax0.set_yscale(params['yscale'])
        ax0.set_xlim(params['xlim'])
        ax0.set_ylim(params['ylim'])
        
        if params['filename'] is not None:
            fig.savefig(params['filename'])
        else:
            return fig
    
    
    def plot(self, filename=None, start=None, stop=None, channels=None):
        """Plot signal as function of time. By default the entire signal is plotted.
        
        :param filename: Name of file.
        :param start: First sample index.
        :type start: Start time in seconds from start of signal.
        :param stop: Last sample index.
        :type stop: Stop time in seconds. from stop of signal.
        """
        start = int(start*self.fs)
        stop = int(stop*self.fs)
        
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Signal')
        ax0.plot(self[channels, start:stop])
        ax0.set_xlabel(r'$t$ in n')
        ax0.set_ylabel(r'$x$ in -') 
        if filename:
            fig.savefig(filename)
        else:
            return fig
        
    #def plot_scalo(self, filename=None):
        #"""
        #Plot scalogram 
        #"""
        #from scipy.signal import ricker
        
        #wavelet = ricker
        #widths = np.logspace(-1, 3.5, 100)
        #x = cwt(self._signal, wavelet, widths)
        
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
        Save signal as WAV file.
        
        .. warning:: The WAV file will have 16-bit depth!
        
        :param filename: Filename
        """
        data = self
        dtype = data.dtype if not depth else 'int'+str(depth)
        if normalize:
            data = data / data.max() * 0.5
        #if depth:
            #data = (data * 2.0**depth).astype(dtype)
        #print(data)
        wavfile.write(filename, int(self.fs), data.T)
        #wavfile.write(filename, int(self.fs), self._data/np.abs(self._data).max() *  0.5)
        #wavfile.write(filename, int(self.fs), np.int16(self._data/(np.abs(self._data).max()) * 32767) )
    
    @classmethod
    def from_wav(cls, filename):
        """
        Create signal from WAV file.
        
        :param filename: Filename
        """
        fs, data = wavfile.read(filename)  
        data = data.astype(np.float32, copy=False)
        data /= np.max(np.abs(data))
        return cls(data, fs=fs)
    
    #def to_mat(filename):
        #"""
        #Save signal to MAT file.
        #"""
        #raise NotImplementedError
    
    #@classmethod
    #def from_mat(cls, filename):
        #"""
        #Load signal from MAT file.
        #"""
        #raise NotImplementedError
